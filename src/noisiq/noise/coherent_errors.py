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
  - mode='pauli_twirl'   → STIM/TSIM, fast, scalable, underestimates deep circuits

Classes:
    CoherentRotation: A unitary error channel parameterized by a Pauli-string
        axis and a rotation angle ε. Extends KrausChannel so it plugs into
        TrajectoryBackend without any backend changes.
    StochasticCoherentRotation: A quasi-static dephasing channel that draws a
        fresh ε ~ Normal(0, std_dev) per shot and applies exp(-i*ε*H). Unlike
        CoherentRotation, the rotation angle varies between shots, so the
        averaged density matrix loses purity — exactly as low-frequency (1/f)
        noise behaves on real hardware. Used by TrajectoryBackend only;
        incompatible with Pauli-frame backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .kraus_channels import KrausChannel

if TYPE_CHECKING:
    import numpy.random


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

        # Convert to Pauli-twirled form for STIM/TSIM:
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

        This is the bridge to STIM/TSIM: calling this method converts coherent
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


class StochasticCoherentRotation:
    """
    Quasi-static dephasing channel for TrajectoryBackend.

    Each shot draws ε ~ Normal(0, std_dev) once and applies
    U_err = exp(-i * ε * H) to the statevector, where H is the Pauli-string
    Hamiltonian. Because ε differs between shots, the averaged density matrix
    loses purity — faithfully modelling low-frequency (1/f) noise that is
    static within a shot but varies shot to shot.

    This is distinct from CoherentRotation, which uses a single fixed ε on
    every shot and therefore contributes zero purity loss.

    Only compatible with TrajectoryBackend. For Pauli-frame backends, call
    to_pauli_error() to get the corresponding stochastic approximation.

    Args:
        axis:    Pauli string (e.g. 'Z', 'ZZ') specifying the Hamiltonian
                 generator. Each character must be one of I/X/Y/Z.
        std_dev: Standard deviation of the per-shot rotation angle in radians.
                 Typical value for a 50–60 ns idle slot on superconducting
                 hardware: 0.05–0.10 rad (~1–2 MHz quasi-static Z detuning).
    """

    def __init__(self, axis: str, std_dev: float) -> None:
        if not axis or any(ch not in _PAULI for ch in axis):
            raise ValueError(
                f"axis must be a non-empty string of I/X/Y/Z characters, got {axis!r}"
            )
        if std_dev < 0:
            raise ValueError(f"std_dev must be >= 0, got {std_dev}")
        self.axis = axis
        self.std_dev = std_dev
        self.num_qubits = len(axis)
        # Pre-compute the Hamiltonian matrix H so each sample only needs one
        # scalar multiply rather than a full matrix construction.
        self._H = _pauli_string_matrix(axis)
        dim = 2 ** self.num_qubits
        self._I = np.eye(dim, dtype=complex)

    def sample(self, rng: "numpy.random.Generator") -> np.ndarray:
        """Draw ε ~ Normal(0, std_dev) and return U_err = exp(-i*ε*H)."""
        epsilon = float(rng.normal(0.0, self.std_dev))
        return np.cos(epsilon) * self._I - 1j * np.sin(epsilon) * self._H

    def to_pauli_error(self):
        """Return a PauliError approximation suitable for Pauli-frame backends.

        Uses sin²(std_dev) as the effective Pauli error probability, which
        matches the first-order expansion of the variance-averaged rotation.

        Returns:
            PauliError  for single-qubit axes.
            CorrelatedPauliError for multi-qubit axes.
        """
        from .pauli_error import PauliError
        from .correlated_errors import CorrelatedPauliError

        p = float(np.sin(self.std_dev) ** 2)

        if self.num_qubits == 1:
            ax = self.axis
            return PauliError(
                p_x=p if ax == 'X' else 0.0,
                p_y=p if ax == 'Y' else 0.0,
                p_z=p if ax == 'Z' else 0.0,
            )

        return CorrelatedPauliError({self.axis: p})

    def describe(self) -> dict:
        return {
            "type": "StochasticCoherentRotation",
            "axis": self.axis,
            "std_dev_rad": self.std_dev,
            "num_qubits": self.num_qubits,
        }

    def __repr__(self) -> str:
        return (
            f"StochasticCoherentRotation(axis={self.axis!r}, "
            f"std_dev={self.std_dev:.4f} rad, num_qubits={self.num_qubits})"
        )
