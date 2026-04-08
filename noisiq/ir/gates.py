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

# ==============================================================================
# Gate Expansion to Full Hilbert Space
# ==============================================================================

def _expand_multi_qubit_gate(
    gate: Gate,
    target_qubits: Tuple[int, ...],
    n_qubits: int
) -> np.ndarray:
    """
    Expand an arbitrary multi-qubit gate to full space.
    
    Works for any gate size: n-qubit.
    """
    num_gate_qubits = gate.num_qubits
    gate_dim = 2 ** num_gate_qubits  # e.g., 8 for CCZ (3-qubit)
    full_dim = 2 ** n_qubits
    
    result = np.zeros((full_dim, full_dim), dtype=complex)
    
    # Iterate over all basis states in the full Hilbert space
    for in_idx in range(full_dim):
        # Extract the bits for target qubits
        gate_in_bits = 0
        for i, q in enumerate(target_qubits):
            bit = (in_idx >> (n_qubits - 1 - q)) & 1
            gate_in_bits |= (bit << (num_gate_qubits - 1 - i))
        
        # Apply gate to these bits
        for gate_out_bits in range(gate_dim):
            if gate.matrix[gate_out_bits, gate_in_bits] == 0:
                continue
            
            # Construct output index by replacing target qubit bits
            out_idx = in_idx
            for i, q in enumerate(target_qubits):
                # Clear the bit
                out_idx &= ~(1 << (n_qubits - 1 - q))
                # Set new bit
                new_bit = (gate_out_bits >> (num_gate_qubits - 1 - i)) & 1
                out_idx |= (new_bit << (n_qubits - 1 - q))
            
            result[out_idx, in_idx] += gate.matrix[gate_out_bits, gate_in_bits]
    
    return result