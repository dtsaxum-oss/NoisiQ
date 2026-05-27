import numpy as np
import pytest

from noisiq.ir import gates


def test_gate_creation_unitary():
    """Tests that a Gate can be created with a valid unitary matrix."""
    h_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    gate = gates.Gate(name="H_test", matrix=h_matrix, num_qubits=1)
    assert gate.name == "H_test"
    assert gate.num_qubits == 1
    assert np.allclose(gate.matrix, h_matrix)


def test_gate_creation_non_unitary_fails():
    """Tests that creating a Gate with a non-unitary matrix raises a ValueError."""
    non_unitary_matrix = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="matrix is not unitary"):
        gates.Gate(name="NonUnitary", matrix=non_unitary_matrix, num_qubits=1)


def test_gate_creation_non_square_matrix_fails():
    """Tests that creating a Gate with a non-square matrix fails the unitarity check."""
    non_square_matrix = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="matrix is not unitary"):
        gates.Gate(name="NonSquare", matrix=non_square_matrix, num_qubits=1)


@pytest.mark.parametrize(
    "gate_instance, expected_name, expected_qubits, expected_shape",
    [
        (gates.I, "I", 1, (2, 2)),
        (gates.X, "X", 1, (2, 2)),
        (gates.Y, "Y", 1, (2, 2)),
        (gates.Z, "Z", 1, (2, 2)),
        (gates.H, "H", 1, (2, 2)),
        (gates.S, "S", 1, (2, 2)),
        (gates.S_DAG, "S_DAG", 1, (2, 2)),
        (gates.T, "T", 1, (2, 2)),
        (gates.T_DAG, "T_DAG", 1, (2, 2)),
        (gates.CNOT, "CNOT", 2, (4, 4)),
    ],
)
def test_standard_gates_are_valid(
    gate_instance, expected_name, expected_qubits, expected_shape
):
    """
    Tests that all standard gate definitions are valid Gates with correct properties.
    The Gate's __post_init__ already confirms unitarity.
    """
    assert isinstance(gate_instance, gates.Gate)
    assert gate_instance.name == expected_name
    assert gate_instance.num_qubits == expected_qubits
    assert gate_instance.matrix.shape == expected_shape


def test_is_unitary_helper():
    """Tests the internal is_unitary helper function."""
    # Unitary
    assert gates.is_unitary(gates.H.matrix)
    assert gates.is_unitary(gates.CNOT.matrix)

    # Non-unitary
    assert not gates.is_unitary(np.array([[1, 1], [1, 1]]))
    assert not gates.is_unitary(np.array([[1, 0], [0, 2]]))

    # Non-square
    assert not gates.is_unitary(np.array([[1, 0, 0], [0, 1, 0]]))


@pytest.mark.parametrize(
    "gate_instance, expected_name, expected_qubits, expected_shape",
    [
        (gates.SWAP,   "SWAP",   2, (4, 4)),
    ],
)
def test_new_gates_are_valid(gate_instance, expected_name, expected_qubits, expected_shape):
    assert isinstance(gate_instance, gates.Gate)
    assert gate_instance.name == expected_name
    assert gate_instance.num_qubits == expected_qubits
    assert gate_instance.matrix.shape == expected_shape


def test_swap_matrix_correct():
    # SWAP|01⟩ = |10⟩ (MSB-first: index 01=1 → index 10=2)
    psi_01 = np.array([0, 1, 0, 0], dtype=complex)
    psi_10 = gates.SWAP.matrix @ psi_01
    assert np.allclose(psi_10, [0, 0, 1, 0])


def test_phase_gate_factory():
    gate = gates.phase_gate(np.pi / 2)
    assert gate.name == "P(1.571)"
    assert gate.num_qubits == 1
    # P(π/2)|1⟩ = i|1⟩
    psi_1 = np.array([0, 1], dtype=complex)
    assert np.allclose(gate.matrix @ psi_1, [0, 1j])


def test_rz_gate_factory():
    gate = gates.rz_gate(np.pi)
    assert gate.name == "RZ(3.142)"
    assert gate.num_qubits == 1
    # RZ(π) = -i * Z  (up to global phase); |RZ(π)| diag = [e^{-iπ/2}, e^{iπ/2}] = [-i, i]
    expected = np.array([[-1j, 0], [0, 1j]], dtype=complex)
    assert np.allclose(gate.matrix, expected)


