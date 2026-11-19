"""Deterministic Pauli-error propagation through Clifford circuits.

``propagate(circuit, "X", qubit=0, after=2)`` answers the classic teaching question
"if an X error hits qubit 0 right after instruction 2, where does it end up?" exactly,
using the same frame engine as the stabilizer backend (no noise, no sampling).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from ..backends.frame import IncompatibleCircuitError, _Run
from ..backends.program import GateStep, LayerEnd, MeasureStep, ResetStep, compile_program
from ..ir.circuit import Circuit
from ..results.trace import FaultEvent, PropagationTrace

__all__ = ["propagate", "fault_spread", "conjugate_pauli"]

_BITS = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def _check_clifford(program) -> None:
    bad = sorted({s.gate.display_label for s in program.steps if isinstance(s, GateStep) and not s.gate.is_clifford})
    if bad:
        raise IncompatibleCircuitError(
            "Exact Pauli propagation needs Clifford gates; this circuit contains "
            f"{', '.join(bad[:6])}. A Pauli error does not stay a Pauli error through these gates. "
            "Use nq.simulate(..., backend='statevector', trace_shots=...) for a deviation trace instead."
        )
    bad_c = sorted(
        {s.gate.display_label for s in program.steps if isinstance(s, GateStep) and s.condition and not s.gate.is_pauli}
    )
    if bad_c:
        raise IncompatibleCircuitError(f"Conditional non-Pauli gates are not supported: {', '.join(bad_c)}.")


def propagate(
    circuit: Circuit,
    pauli: str,
    qubit: Union[int, Sequence[int], None] = None,
    *,
    after: int | None = None,
    layer: int | None = None,
) -> PropagationTrace:
    """Inject a Pauli error and trace it through the rest of the circuit.

    Args:
        circuit: A Clifford circuit (measurements, resets and Pauli feed-forward allowed).
        pauli: A Pauli string, e.g. ``"X"`` (with ``qubit``) or a full-width string such as
            ``"IXZ"`` (then ``qubit`` must be omitted).
        qubit: Qubit(s) the Pauli acts on.
        after: Inject right after instruction ``after`` (an index into ``circuit``).
        layer: Alternatively, inject at the end of layer ``layer`` (``-1`` = before
            the first layer). Exactly one of ``after``/``layer`` may be given; default
            is before the first layer.

    Returns:
        A one-shot :class:`~noisiq.results.PropagationTrace` (``kind="pauli"``).
    """
    pauli = pauli.upper()
    n = circuit.n_qubits
    if qubit is None:
        if len(pauli) != n:
            raise ValueError(f"Give `qubit`, or a Pauli string of length {n}.")
        qubits = list(range(n))
    else:
        qubits = [qubit] if isinstance(qubit, (int, np.integer)) else list(qubit)
        if len(pauli) != len(qubits):
            raise ValueError("Pauli string length must match the number of qubits.")
    for q in qubits:
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} out of range.")
    if any(ch not in _BITS for ch in pauli):
        raise ValueError(f"Invalid Pauli string {pauli!r}.")
    if after is not None and layer is not None:
        raise ValueError("Give either `after` or `layer`, not both.")

    program = compile_program(circuit, None)
    _check_clifford(program)
    steps = program.steps
    if after is not None:
        if not 0 <= after < len(circuit):
            raise ValueError(f"after={after} is not a valid instruction index (0..{len(circuit) - 1}).")
        pos = (
            next(
                i
                for i, s in enumerate(steps)
                if getattr(s, "index", None) == after and isinstance(s, (GateStep, MeasureStep, ResetStep))
            )
            + 1
        )
        inj_layer = circuit[after].layer
    elif layer is not None and layer >= 0:
        # inject just before the end-of-layer snapshot, so frame `layer + 1` shows it
        pos = next((i for i, s in enumerate(steps) if isinstance(s, LayerEnd) and s.layer == layer), None)
        if pos is None:
            raise ValueError(f"layer={layer} does not exist (circuit depth {circuit.depth}).")
        inj_layer = layer
    else:
        pos, inj_layer = 0, 0

    def inject(p, run: _Run) -> None:
        if p == pos:
            for ch, q in zip(pauli, qubits):
                x, z = _BITS[ch]
                run.Ex[q, 0] ^= x
                run.Ez[q, 0] ^= z

    run = _Run(
        program,
        1,
        np.random.default_rng(0),
        randomize=False,
        approximate=False,
        apply_noise=False,
        trace_shots=1,
        record=False,
        inject=inject,
    )
    run.run()
    fx = np.array(run.frames_x, dtype=np.uint8)
    fz = np.array(run.frames_z, dtype=np.uint8)
    if pos == 0:  # injected before the first layer: show it in the initial frame too
        for ch, q in zip(pauli, qubits):
            x, z = _BITS[ch]
            fx[0, q, 0] ^= x
            fz[0, q, 0] ^= z
    events = [FaultEvent(0, inj_layer, after, q, ch, "injected") for ch, q in zip(pauli, qubits) if ch != "I"]
    return PropagationTrace(
        circuit=circuit,
        kind="pauli",
        n_shots=1,
        events=events,
        frames_x=fx,
        frames_z=fz,
        measurement_flips=(run.rec ^ run.rnd).T[:1].astype(bool),
        description=f"{pauli} on {qubits} "
        + (
            f"after instruction {after}"
            if after is not None
            else f"after layer {layer}"
            if layer is not None
            else "at start"
        ),
    )


def conjugate_pauli(circuit: Circuit, pauli: str) -> str:
    """Return ``U P U†`` for a full-width Pauli string through a unitary Clifford circuit."""
    trace = propagate(circuit, pauli)
    return trace.pauli_string(trace.n_layers)


def fault_spread(circuit: Circuit) -> tuple[np.ndarray, np.ndarray]:
    """Weight of every single-qubit Pauli fault after propagation to the end.

    Returns:
        ``(w_inst, w_idle)`` with shapes ``(n_instructions, n_qubits, 3)`` and
        ``(n_layers, n_qubits, 3)`` (columns X, Y, Z): the number of measured bits the
        fault flips plus its residual weight on qubits not measured afterwards
        (see ``nq.explain("error_burden")``).
    """
    program = compile_program(circuit, None)
    _check_clifford(program)
    n = circuit.n_qubits
    n_inst = len(circuit)
    n_layers = program.n_layers
    cols: list[tuple[int, int, int, str, int, int]] = []  # (pos, q, pauli_idx, kind, a, b)
    for pos, s in enumerate(program.steps):
        if isinstance(s, GateStep):
            for q in s.qubits:
                for k in range(3):
                    cols.append((pos + 1, q, k, "inst", s.index, q))
        elif isinstance(s, MeasureStep) and s.index >= 0:
            for k in range(3):
                cols.append((pos + 1, s.qubit, k, "inst", s.index, s.qubit))
        elif isinstance(s, ResetStep):
            for k in range(3):
                cols.append((pos + 1, s.qubit, k, "inst", s.index, s.qubit))
        elif isinstance(s, LayerEnd):
            for q in range(n):
                for k in range(3):
                    cols.append((pos, q, k, "idle", s.layer, q))
    w_inst = np.zeros((n_inst, n, 3))
    w_idle = np.zeros((max(n_layers, 1), n, 3))
    if not cols:
        return w_inst, w_idle[:n_layers]
    by_pos: dict[int, list[int]] = {}
    for c, (pos, *_rest) in enumerate(cols):
        by_pos.setdefault(pos, []).append(c)
    xs = {0: 1, 1: 1, 2: 0}
    zs = {0: 0, 1: 1, 2: 1}

    def inject(p, run: _Run) -> None:
        for c in by_pos.get(p, ()):
            _, q, k, *_ = cols[c]
            run.Ex[q, c] ^= xs[k]
            run.Ez[q, c] ^= zs[k]

    run = _Run(
        program,
        len(cols),
        np.random.default_rng(0),
        randomize=False,
        approximate=False,
        apply_noise=False,
        trace_shots=0,
        record=False,
        inject=inject,
    )
    run.run()
    flips = (run.rec ^ run.rnd).sum(axis=0)  # measured bits flipped, per column
    # residual weight on qubits never measured after the end of the circuit
    measured_last = np.zeros(n, bool)
    last_meas_pos = {}
    last_dyn_pos = {}
    for pos, s in enumerate(program.steps):
        if isinstance(s, MeasureStep):
            last_meas_pos[s.qubit] = pos
        if isinstance(s, (GateStep, ResetStep)):
            for q in s.qubits if isinstance(s, GateStep) else (s.qubit,):
                last_dyn_pos[q] = pos
    for q in range(n):
        measured_last[q] = q in last_meas_pos and last_meas_pos[q] > last_dyn_pos.get(q, -1)
    residual = ((run.Ex | run.Ez).astype(bool) & ~measured_last[:, None]).sum(axis=0)
    weight = flips + residual
    for c, (_, _q, k, kind, a, b) in enumerate(cols):
        if kind == "inst":
            w_inst[a, b, k] = weight[c]
        else:
            w_idle[a, b, k] = weight[c]
    return w_inst, w_idle[:n_layers]
