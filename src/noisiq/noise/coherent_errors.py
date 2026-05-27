"""
Coherent (unitary) error channels.

A coherent error is a small unitary U_err = exp(-i * ε * H) applied alongside
an intended gate, where H is a Pauli-string Hamiltonian. Unlike stochastic
errors, coherent errors preserve state purity and accumulate in amplitude:
N gates each with the same ε miscalibration produce an Nε-amplitude error,
whose bad-outcome probability scales as N²ε². A stochastic channel with the
same per-gate probability would scale as Nε² — linear not quadratic. This
is the physical reason coherent errors can exceed Pauli-twirled predictions
by 8–10× in deep circuits (Hines et al., arXiv:2603.18457).

Pauli-twirling discards the quadratic accumulation. The two-mode design in
NoisiQ exposes this gap directly:
  - mode='coherent'      → TrajectoryBackend, exact dynamics, ≤13 qubits
  - mode='pauli_twirl'   → STIM/Qiskit, fast, scalable, underestimates deep circuits

Classes:
    CoherentRotation: A unitary error channel parameterized by a Pauli-string
        axis and a rotation angle ε. Extends KrausChannel so it plugs into
        TrajectoryBackend without any backend changes.
"""

from __future__ import annotations

import numpy as np

from .kraus_channels import KrausChannel


# Pauli matrix definitions used to build multi-qubit generators
_PAULI: dict[str, np.ndarray] = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_string_matrix(pauli_string: str) -> np.ndarray:
    """Build the matrix for a multi-qubit Pauli string, e.g. 'ZZ' or 'IXZ'."""
    matrix = _PAULI[pauli_string[0]]
    for ch in pauli_string[1:]:
        matrix = np.kron(matrix, _PAULI[ch])
    return matrix


class CoherentRotation(KrausChannel):
    """
    A coherent (unitary) error channel: U_err = exp(-i * epsilon * H).

    H is specified as a Pauli string (e.g. 'Z' for single-qubit dephasing,
    'ZZ' for two-qubit ZZ crosstalk, 'IX' for off-resonant driving). The
    channel is a single Kraus operator equal to the error unitary, so the
    state stays pure and errors accumulate coherently across shots.

    Compatible with TrajectoryBackend (deterministic application, no sampling
    needed for a single Kraus operator). For Clifford backends, call
    to_pauli_error() to convert to the Pauli-twirled approximation.

    Args:
        axis:    Pauli string (e.g. 'X', 'Z', 'ZZ', 'IZ') specifying the
                 Hamiltonian generator. Each character must be one of I/X/Y/Z.
                 Length sets the number of qubits this channel acts on.
        epsilon: Rotation angle in radians. Positive ε rotates in the
                 direction of H; negative ε rotates the opposite way.

    Example:
        # Single-qubit Z over-rotation on a gate by 0.05 rad (~3°):
        err = CoherentRotation(axis='Z', epsilon=0.05)

        # Accidental ZZ Hamiltonian term during a 2q gate:
        zz_err = CoherentRotation(axis='ZZ', epsilon=0.02)

        # Convert to Pauli-twirled form for STIM/Qiskit:
        pauli_approx = zz_err.to_pauli_error()

    Raises:
        ValueError: If axis contains characters outside I/X/Y/Z.
    """

    def __init__(self, axis: str, epsilon: float) -> None:
        if not axis or any(ch not in _PAULI for ch in axis):
            raise ValueError(
                f"axis must be a non-empty string of I/X/Y/Z characters, got {axis!r}"
            )
        self.axis = axis
        self.epsilon = epsilon
        self.num_qubits = len(axis)

        # U_err = exp(-i * ε * H) = cos(ε)*I − i*sin(ε)*H
        # This identity holds because H is a Pauli string, which is Hermitian
        # and squares to I, so the matrix exponential reduces to this form.
        dim = 2 ** self.num_qubits
        H = _pauli_string_matrix(axis)
        I_mat = np.eye(dim, dtype=complex)
        U_err = np.cos(epsilon) * I_mat - 1j * np.sin(epsilon) * H

        # Single Kraus operator — KrausChannel validates U†U = I automatically.
        super().__init__([U_err])

    def describe(self) -> dict:
        return {
            "type": "CoherentRotation",
            "axis": self.axis,
            "epsilon_rad": self.epsilon,
            "num_qubits": self.num_qubits,
        }

    def to_pauli_error(self):
        """Return the Pauli-twirled approximation of this coherent error.

        For a rotation exp(-i*ε*P) around a single Pauli P, twirling gives a
        Pauli channel that applies P with probability sin²(ε) and I otherwise.

        For multi-qubit axes, returns a CorrelatedPauliError mapping the axis
        Pauli string to sin²(ε).

        This is the bridge to STIM/Qiskit: calling this method converts coherent
        accumulation into a single-shot stochastic approximation, which is fast
        and scalable but underestimates error in deep circuits.

        Returns:
            PauliError  for single-qubit axes (axis length 1).
            CorrelatedPauliError for multi-qubit axes (axis length ≥ 2).
        """
        from .pauli_error import PauliError
        from .correlated_errors import CorrelatedPauliError

        p = float(np.sin(self.epsilon) ** 2)

        if self.num_qubits == 1:
            ax = self.axis
            return PauliError(
                p_x=p if ax == 'X' else 0.0,
                p_y=p if ax == 'Y' else 0.0,
                p_z=p if ax == 'Z' else 0.0,
            )

        return CorrelatedPauliError({self.axis: p})

    def __repr__(self) -> str:
        return (
            f"CoherentRotation(axis={self.axis!r}, epsilon={self.epsilon:.4f} rad, "
            f"num_qubits={self.num_qubits})"
        )
