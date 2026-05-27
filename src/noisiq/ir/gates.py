from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def is_unitary(m: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Check if a matrix is unitary (U†U = I) within a given tolerance.

    Args:
        m: The matrix to check.
        tol: The tolerance for floating-point comparisons.

    Returns:
        True if the matrix is unitary, False otherwise.
    """
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False  # Must be a square matrix
    identity = np.eye(m.shape[0])
    return np.allclose(m.conj().T @ m, identity, atol=tol)


@dataclass(frozen=True)
class Gate:
    """
    A quantum gate, defined by its name, matrix representation, and qubit count.

    The gate's matrix is validated for unitarity upon creation to ensure it
    represents a valid physical operation.

    Args:
        name: The common name of the gate (e.g., "H", "CNOT").
        matrix: The unitary matrix representing the gate's operation.
        num_qubits: The number of qubits the gate acts on.

    Raises:
        ValueError: If the provided matrix is not unitary.
    """

    name: str
    matrix: np.ndarray
    num_qubits: int

    def __post_init__(self):
        if not is_unitary(self.matrix):
            raise ValueError(f"Gate '{self.name}' matrix is not unitary.")

    def __repr__(self) -> str:
        return f"Gate(name='{self.name}', num_qubits={self.num_qubits})"


# ==============================================================================
# Standard Gate Definitions
# ==============================================================================

# --- Single-Qubit Gates ---

I = Gate(
    name="I",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, 1]], dtype=complex),
)

# Same unitary as I but distinct object — noise-model builders branch on
# `op.gate is IDLE` to apply only T1/T2 decoherence, never gate error.
IDLE = Gate(
    name="IDLE",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, 1]], dtype=complex),
)

X = Gate(
    name="X",
    num_qubits=1,
    matrix=np.array([[0, 1], [1, 0]], dtype=complex),
)

Y = Gate(
    name="Y",
    num_qubits=1,
    matrix=np.array([[0, -1j], [1j, 0]], dtype=complex),
)

Z = Gate(
    name="Z",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, -1]], dtype=complex),
)

H = Gate(
    name="H",
    num_qubits=1,
    matrix=(1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
)

S = Gate(
    name="S",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, 1j]], dtype=complex),
)

S_DAG = Gate(
    name="S_DAG",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, -1j]], dtype=complex),
)

T = Gate(
    name="T",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
)

T_DAG = Gate(
    name="T_DAG",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
)


# --- Two-Qubit Gates ---

CNOT = Gate(
    name="CNOT",
    num_qubits=2,
    matrix=np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    ),
)
CX = CNOT

CZ = Gate(
    name="CZ",
    num_qubits=2,
    matrix=np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ],
        dtype=complex,
    ),
)

SWAP = Gate(
    name="SWAP",
    num_qubits=2,
    matrix=np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    ),
)


# ==============================================================================
# Parameterized Gate Factories
# ==============================================================================

def phase_gate(theta: float) -> Gate:
    """Return P(θ) = [[1, 0], [0, e^{iθ}]] — the standard phase gate.

    S  = phase_gate(π/2), T  = phase_gate(π/4), Z = phase_gate(π).
    These are NOT Clifford for arbitrary θ; the Clifford-only backends
    (StimTableauBackend, ManyShotRunner) will raise NonCliffordError.

    Args:
        theta: Phase angle in radians.

    Returns:
        Gate with name "P(θ)" rounded to 4 significant figures.
    """
    matrix = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    return Gate(name=f"P({theta:.4g})", num_qubits=1, matrix=matrix)


def rz_gate(theta: float) -> Gate:
    """Return RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]] — rotation about Z.

    Differs from P(θ) by a global phase: RZ(θ) = e^{-iθ/2} P(θ).
    Not Clifford for arbitrary θ.

    Args:
        theta: Rotation angle in radians.

    Returns:
        Gate with name "RZ(θ)" rounded to 4 significant figures.
    """
    matrix = np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )
    return Gate(name=f"RZ({theta:.4g})", num_qubits=1, matrix=matrix)

