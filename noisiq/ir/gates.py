from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _is_unitary(m: np.ndarray, tol: float = 1e-9) -> bool:
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
        if not _is_unitary(self.matrix):
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

T = Gate(
    name="T",
    num_qubits=1,
    matrix=np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
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