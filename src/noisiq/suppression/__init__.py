"""Noise suppression: dynamical decoupling (DD).

DD fills idle windows with pulse sequences whose product is the identity (up to a
global phase), so the ideal circuit is unchanged while slowly varying noise — such as
quasi-static detuning (``NoiseModel.add_quasistatic_detuning``) — is echoed away.

=========  ====================  ================================================
sequence   pulses                refocuses
=========  ====================  ================================================
``X2``     X X                   Z (dephasing) noise — Hahn/CPMG echo
``XY4``    X Y X Y               dephasing and amplitude-type (X/Y) noise
``XY8``    XY4 + reversed XY4    as XY4, robust to pulse errors
``YY``     Y Y                   dephasing (Y-axis echo)
=========  ====================  ================================================

Example:
    >>> import noisiq as nq
    >>> c = nq.Circuit(2).h(0).delay(0, 4e-6).h(0)       # long idle on qubit 0
    >>> protected = nq.suppression.apply_dd(c, "XY4")
"""

from __future__ import annotations

from ..ir import gates as G
from ..ir.circuit import Circuit
from ..ir.instructions import Barrier, Delay, Measure, Operation, Reset

__all__ = ["apply_dd", "list_sequences", "SEQUENCES"]

SEQUENCES: dict[str, list[G.Gate]] = {
    "X2": [G.X, G.X],
    "XY4": [G.X, G.Y, G.X, G.Y],
    "XY8": [G.X, G.Y, G.X, G.Y, G.Y, G.X, G.Y, G.X],
    "YY": [G.Y, G.Y],
}
_ALIASES = {"CPMG": "X2", "HAHN": "X2", "XY-4": "XY4", "XY-8": "XY8"}


def list_sequences() -> list[str]:
    """Names of the available DD sequences."""
    return sorted(SEQUENCES)


def apply_dd(
    circuit: Circuit,
    sequence: str = "XY4",
    *,
    qubits: list[int] | None = None,
    min_idle_layers: int | None = None,
    split_delays: bool = True,
) -> Circuit:
    """Return a copy of ``circuit`` with DD pulses in idle windows.

    An *idle window* on a qubit is a run of consecutive layers between two of its
    instructions in which it does nothing (or only waits in a :class:`~noisiq.ir.Delay`).
    Windows with at least ``len(sequence)`` free layers receive one evenly spaced copy of
    the sequence; windows made of a single ``Delay`` are split into equal delays around
    the pulses when ``split_delays`` is True (so the total time is preserved).

    Args:
        circuit: Source circuit (not modified).
        sequence: ``"X2"`` (alias ``"hahn"``/``"CPMG"``), ``"XY4"``, ``"XY8"`` or ``"YY"``.
        qubits: Only decouple these qubits (default: all).
        min_idle_layers: Minimum free layers to fill (default: the sequence length).
        split_delays: Insert pulses inside long ``Delay`` instructions.
    """
    key = _ALIASES.get(sequence.upper().replace("_", "-"), sequence.upper().replace("-", ""))
    if key not in SEQUENCES:
        raise ValueError(f"Unknown DD sequence {sequence!r}; choose from {list_sequences()}.")
    pulses = SEQUENCES[key]
    k = len(pulses)
    min_free = max(k, min_idle_layers or k)
    targets = set(range(circuit.n_qubits) if qubits is None else qubits)

    out = Circuit(circuit.n_qubits, circuit.n_clbits, circuit.name and f"{circuit.name}_dd_{key}")
    out.registers = list(circuit.registers)
    plan: dict[int, list] = {}  # instruction index -> replacement list (for delays)
    extra: list[tuple[int, int, G.Gate]] = []  # (layer, qubit, gate) pulses in free layers

    busy: dict[int, list[tuple[int, int]]] = {q: [] for q in range(circuit.n_qubits)}
    for idx, inst in circuit.ordered():
        if isinstance(inst, Barrier):
            continue
        for q in inst.qubits:
            busy[q].append((inst.layer, idx))

    for q in sorted(targets):
        events = sorted(busy[q])
        for (t0, _i0), (t1, _i1) in zip(events, events[1:]):
            free = t1 - t0 - 1
            if free >= min_free:
                for j, g in enumerate(pulses):
                    layer = t0 + 1 + round((j + 0.5) * free / k - 0.5)
                    extra.append((layer, q, g))
        if split_delays:
            for _, idx in events:
                inst = circuit[idx]
                if isinstance(inst, Delay) and inst.duration > 0:
                    tau = inst.duration / (2 * k)
                    seq: list = [("delay", tau)]
                    for j, g in enumerate(pulses):
                        seq.append(("gate", g))
                        seq.append(("delay", 2 * tau if j < k - 1 else tau))
                    plan[idx] = seq

    # Rebuild: keep original layers, shifting later layers when delays are expanded.
    expansion = {}  # layer -> extra layers inserted after it
    for idx, seq in plan.items():
        layer = circuit[idx].layer
        expansion[layer] = max(expansion.get(layer, 0), len(seq) - 1)

    def shift(layer: int) -> int:
        return layer + sum(v for t, v in expansion.items() if t < layer)

    for idx, inst in circuit.ordered():
        lay = shift(inst.layer)
        if idx in plan:
            for j, (kind, val) in enumerate(plan[idx]):
                if kind == "delay":
                    out.delay(inst.qubit, val, layer=lay + j)
                else:
                    out.append(val, [inst.qubit], layer=lay + j, tag="dd")
            continue
        if isinstance(inst, Operation):
            out.append(inst.gate, inst.qubits, layer=lay, condition=inst.condition, tag=inst.tag)
        elif isinstance(inst, Measure):
            out.measure(inst.qubit, inst.clbit, basis=inst.basis, reset=inst.reset, layer=lay)
        elif isinstance(inst, Reset):
            out.reset(inst.qubit, layer=lay)
        elif isinstance(inst, Delay):
            out.delay(inst.qubit, inst.duration, layer=lay)
        elif isinstance(inst, Barrier):
            out.barrier(list(inst.qubits))
    for layer, q, g in extra:
        out.append(g, [q], layer=shift(layer), tag="dd")
    return out
