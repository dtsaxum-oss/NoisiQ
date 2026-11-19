"""Step-by-step error-propagation traces (the data behind animations)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ir.circuit import Circuit

__all__ = ["FaultEvent", "PropagationTrace"]

_PAULI_CHAR = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


@dataclass(frozen=True)
class FaultEvent:
    """One sampled fault in one shot.

    Attributes:
        shot: Shot index within the trace.
        layer: Layer in which it happened.
        index: Instruction index (``None`` for idle noise).
        qubit: Affected qubit.
        pauli: ``"X"``, ``"Y"``, ``"Z"`` or ``"other"`` (non-Pauli / readout).
        kind: ``"gate"``, ``"idle"``, ``"reset"``, ``"readout"``, ``"explicit"`` or ``"injected"``.
    """

    shot: int
    layer: int
    index: int | None
    qubit: int
    pauli: str
    kind: str


@dataclass
class PropagationTrace:
    """How errors move through a circuit, layer by layer, for a few shots.

    For Clifford circuits ("pauli" traces) ``frames_x``/``frames_z`` hold the exact
    Pauli error frame *after* each layer: shape ``(n_layers + 1, n_qubits, n_shots)``
    where index 0 is the state before the first layer.

    For general circuits ("deviation" traces, statevector backend) ``deviation`` holds,
    after each layer, ``1 - F(ρ_q, ρ_q^ideal)`` per qubit (how far each qubit's
    reduced state is from the ideal one): shape ``(n_layers + 1, n_qubits, n_shots)``.
    """

    circuit: Circuit
    kind: str  # "pauli" or "deviation"
    n_shots: int
    events: list[FaultEvent] = field(default_factory=list)
    frames_x: np.ndarray | None = None
    frames_z: np.ndarray | None = None
    deviation: np.ndarray | None = None
    measurement_flips: np.ndarray | None = None  # (n_shots, n_clbits) bool
    description: str = ""

    @property
    def n_layers(self) -> int:
        arr = self.frames_x if self.kind == "pauli" else self.deviation
        return 0 if arr is None else arr.shape[0] - 1

    @property
    def n_qubits(self) -> int:
        return self.circuit.n_qubits

    def pauli_string(self, step: int, shot: int = 0) -> str:
        """Error frame after ``step`` layers (``step=0``: before any gate), e.g. ``"IXZ"``."""
        if self.kind != "pauli":
            raise ValueError("pauli_string() is only available for Pauli (Clifford) traces.")
        x = self.frames_x[step, :, shot]
        z = self.frames_z[step, :, shot]
        return "".join(_PAULI_CHAR[(int(a), int(b))] for a, b in zip(x, z))

    def pauli_strings(self, shot: int = 0) -> list[str]:
        """Error frame after every layer for one shot."""
        return [self.pauli_string(s, shot) for s in range(self.n_layers + 1)]

    def weight(self, step: int, shot: int = 0) -> int:
        """Number of qubits carrying an error after ``step`` layers."""
        if self.kind == "pauli":
            return int(np.count_nonzero(self.frames_x[step, :, shot] | self.frames_z[step, :, shot]))
        return int(np.count_nonzero(self.deviation[step, :, shot] > 1e-6))

    def events_for(self, shot: int = 0) -> list[FaultEvent]:
        return [e for e in self.events if e.shot == shot]

    def first_faulty_shot(self, harmful: bool = False) -> int | None:
        """Index of the first shot in which any fault occurred (or ``None``).

        With ``harmful=True``, only shots in which a measured bit ended up flipped count.
        """
        shots = sorted({e.shot for e in self.events})
        if harmful:
            if self.measurement_flips is None:
                return None
            shots = [s for s in shots if s < len(self.measurement_flips) and self.measurement_flips[s].any()]
        return shots[0] if shots else None

    def __repr__(self) -> str:
        return (
            f"PropagationTrace(kind={self.kind!r}, shots={self.n_shots}, layers={self.n_layers}, "
            f"events={len(self.events)})"
        )
