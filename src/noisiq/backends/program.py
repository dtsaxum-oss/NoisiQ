"""Compile a circuit plus noise model into a flat list of simulation steps.

All backends execute the same :class:`Program`, so scheduling, gate durations, idle
noise and fault attribution are defined in exactly one place.

Step order within a layer: each instruction (in insertion order) followed by its
faults; then idle faults for qubits that were (partly) idle in that layer; then
quasi-static detuning for that layer's duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from ..ir.circuit import Circuit
from ..ir.gates import Gate
from ..ir.instructions import Condition, Delay, Measure, Operation, Reset
from ..noise.channels import Channel, ReadoutError
from ..noise.model import Durations, NoiseModel

__all__ = ["Program", "GateStep", "FaultStep", "MeasureStep", "ResetStep", "DetuneStep", "LayerEnd", "compile_program"]

# Fault tallies use these columns: X, Y, Z, other (non-Pauli / coherent / classical).
PAULI_COLUMNS = ("X", "Y", "Z", "other")


@dataclass(frozen=True)
class GateStep:
    index: int
    layer: int
    gate: Gate
    qubits: tuple[int, ...]
    condition: Condition | None


@dataclass(frozen=True)
class FaultStep:
    """Apply ``channel`` to ``qubits``. ``index`` is the instruction it belongs to
    (``None`` for idle noise, which is attributed to ``(layer, qubit)``)."""

    index: int | None
    layer: int
    channel: Channel
    qubits: tuple[int, ...]
    kind: str  # "gate" | "idle" | "reset" | "explicit"
    condition: Condition | None = None  # only fires if the parent gate fired


@dataclass(frozen=True)
class MeasureStep:
    index: int  # -1 for implicit end-of-circuit measurements
    layer: int
    qubit: int
    clbit: int
    basis: str
    reset: bool
    readout: ReadoutError | None


@dataclass(frozen=True)
class ResetStep:
    index: int
    layer: int
    qubit: int


@dataclass(frozen=True)
class DetuneStep:
    layer: int
    qubit: int
    duration: float


@dataclass(frozen=True)
class LayerEnd:
    layer: int


Step = Union[GateStep, FaultStep, MeasureStep, ResetStep, DetuneStep, LayerEnd]


@dataclass
class Program:
    circuit: Circuit  # the user's circuit (unchanged)
    steps: list[Step]
    n_qubits: int
    n_clbits: int
    n_layers: int
    n_instructions: int
    implicit_measurement: bool  # True if measurements were added automatically
    snapshot_position: int  # step index before terminal measurements
    layer_durations: list[float]
    detuning_sigma: list[float] = field(default_factory=list)
    noise: NoiseModel | None = None

    # ---------------------------------------------------------- inspection
    def channels(self) -> list[Channel]:
        return [s.channel for s in self.steps if isinstance(s, FaultStep)]

    @property
    def all_pauli(self) -> bool:
        return all(ch.is_pauli for ch in self.channels())

    @property
    def all_clifford(self) -> bool:
        return all(s.gate.is_clifford for s in self.steps if isinstance(s, GateStep))

    @property
    def has_detuning(self) -> bool:
        return any(isinstance(s, DetuneStep) for s in self.steps)

    @property
    def conditional_gates_are_pauli(self) -> bool:
        return all(s.gate.is_pauli for s in self.steps if isinstance(s, GateStep) and s.condition)

    @property
    def has_conditionals(self) -> bool:
        return any(isinstance(s, GateStep) and s.condition for s in self.steps)

    @property
    def has_mid_circuit_measurement(self) -> bool:
        return any(isinstance(s, (MeasureStep, ResetStep)) for s in self.steps[: self.snapshot_position])


def compile_program(
    circuit: Circuit,
    noise: NoiseModel | None = None,
    *,
    implicit_measure: bool = True,
    idle_after_last_use: bool = False,
) -> Program:
    """Flatten ``circuit`` + ``noise`` into a :class:`Program`.

    Args:
        circuit: Circuit to compile (not modified).
        noise: Optional noise model.
        implicit_measure: If the circuit has no measurements, measure every qubit in
            the Z basis at the end (clbit k ← qubit k).
        idle_after_last_use: Apply idle noise to a qubit after its last instruction too.
    """
    noise = None if (noise is not None and noise.is_empty) else noise
    durations = noise.durations if noise is not None else Durations()
    layers = circuit.layers()
    n_layers = len(layers)
    n = circuit.n_qubits

    last_use = [-1] * n
    for t, insts in enumerate(layers):
        for _, inst in insts:
            for q in inst.qubits:
                last_use[q] = max(last_use[q], t)
    sigma = [noise.detuning_sigma(q) if noise is not None else 0.0 for q in range(n)]

    steps: list[Step] = []
    layer_durations: list[float] = []
    for t, insts in enumerate(layers):
        dur_layer = max((durations.of(inst) for _, inst in insts), default=0.0)
        layer_durations.append(dur_layer)
        busy: dict[int, float] = {}
        for idx, inst in insts:
            d = durations.of(inst)
            for q in inst.qubits:
                busy[q] = d
            if isinstance(inst, Operation):
                steps.append(GateStep(idx, t, inst.gate, inst.qubits, inst.condition))
                if noise is not None:
                    for f in noise.gate_faults(idx, inst, d):
                        steps.append(FaultStep(idx, t, f.channel, f.qubits, "gate", inst.condition))
            elif isinstance(inst, Measure):
                ro = noise.readout_error(inst.qubit) if noise is not None else None
                steps.append(MeasureStep(idx, t, inst.qubit, inst.clbit, inst.basis, inst.reset, ro))
                if noise is not None:
                    if inst.reset:
                        for f in noise.reset_faults(inst.qubit):
                            steps.append(FaultStep(idx, t, f.channel, f.qubits, "reset"))
                    for f in noise.explicit_faults(idx, inst, d):
                        steps.append(FaultStep(idx, t, f.channel, f.qubits, "explicit"))
            elif isinstance(inst, Reset):
                steps.append(ResetStep(idx, t, inst.qubit))
                if noise is not None:
                    for f in noise.reset_faults(inst.qubit):
                        steps.append(FaultStep(idx, t, f.channel, f.qubits, "reset"))
                    for f in noise.explicit_faults(idx, inst, d):
                        steps.append(FaultStep(idx, t, f.channel, f.qubits, "explicit"))
            elif isinstance(inst, Delay):
                if noise is not None:
                    for f in noise.idle_faults(inst.qubit, inst.duration):
                        steps.append(FaultStep(None, t, f.channel, f.qubits, "idle"))
                    for f in noise.explicit_faults(idx, inst, d):
                        steps.append(FaultStep(idx, t, f.channel, f.qubits, "explicit"))
        if noise is not None and dur_layer > 0:
            for q in range(n):
                if not idle_after_last_use and t > last_use[q]:
                    continue
                if q in busy and isinstance(_inst_on(insts, q), Delay):
                    continue  # the delay already received its idle noise
                idle = dur_layer - busy.get(q, 0.0)
                if idle > 1e-18:
                    for f in noise.idle_faults(q, idle):
                        steps.append(FaultStep(None, t, f.channel, f.qubits, "idle"))
            for q in range(n):
                if sigma[q] > 0 and (idle_after_last_use or t <= last_use[q]):
                    steps.append(DetuneStep(t, q, dur_layer))
        steps.append(LayerEnd(t))

    n_clbits = circuit.n_clbits
    implicit = False
    if implicit_measure and not circuit.has_measurements:
        implicit = True
        n_clbits = n
        t = n_layers
        for q in range(n):
            ro = noise.readout_error(q) if noise is not None else None
            steps.append(MeasureStep(-1, t, q, q, "Z", False, ro))

    last_dynamic = max(
        (i for i, s in enumerate(steps) if isinstance(s, (GateStep, ResetStep, DetuneStep))),
        default=-1,
    )
    snapshot = len(steps)
    for i in range(last_dynamic + 1, len(steps)):
        if isinstance(steps[i], MeasureStep):
            snapshot = i
            break

    return Program(
        circuit=circuit,
        steps=steps,
        n_qubits=n,
        n_clbits=n_clbits,
        n_layers=n_layers,
        n_instructions=len(circuit),
        implicit_measurement=implicit,
        snapshot_position=snapshot,
        layer_durations=layer_durations,
        detuning_sigma=sigma,
        noise=noise,
    )


def _inst_on(insts, q):
    for _, inst in insts:
        if q in inst.qubits:
            return inst
    return None
