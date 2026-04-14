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
        (gates.T, "T", 1, (2, 2)),
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


def test_expand_gate_to_full_space_single_qubit():
    """Tests expanding a single-qubit gate (X) in a 2-qubit system."""
    # X on qubit 0 in 2-qubit system
    # Matrix should be X ⊗ I
    expanded = gates.expand_gate_to_full_space(gates.X, (0,), 2)
    expected = np.kron(gates.X.matrix, gates.I.matrix)
    assert np.allclose(expanded, expected)

    # X on qubit 1 in 2-qubit system
    # Matrix should be I ⊗ X
    expanded = gates.expand_gate_to_full_space(gates.X, (1,), 2)
    expected = np.kron(gates.I.matrix, gates.X.matrix)
    assert np.allclose(expanded, expected)


def test_expand_gate_to_full_space_multi_qubit():
    """Tests expanding a multi-qubit gate (CNOT) in a 3-qubit system."""
    # CNOT on qubits (0, 1) in a 3-qubit system
    # Matrix should be CNOT ⊗ I
    expanded = gates.expand_gate_to_full_space(gates.CNOT, (0, 1), 3)
    expected = np.kron(gates.CNOT.matrix, gates.I.matrix)
    assert np.allclose(expanded, expected)

    # CNOT on qubits (1, 2) in a 3-qubit system
    # Matrix should be I ⊗ CNOT
    expanded = gates.expand_gate_to_full_space(gates.CNOT, (1, 2), 3)
    expected = np.kron(gates.I.matrix, gates.CNOT.matrix)
    assert np.allclose(expanded, expected)


def test_expand_gate_to_full_space_non_adjacent():
    """Tests expanding a CNOT on non-adjacent qubits (0, 2) in 3-qubit system."""
    # This is a good test for the index-mapping logic
    expanded = gates.expand_gate_to_full_space(gates.CNOT, (0, 2), 3)

    # |100> (index 4) should map to |101> (index 5)
    vec_4 = np.zeros(8)
    vec_4[4] = 1.0
    out_4 = expanded @ vec_4
    assert np.isclose(out_4[5], 1.0)

    # |000> (index 0) should map to |000> (index 0)
    vec_0 = np.zeros(8)
    vec_0[0] = 1.0
    out_0 = expanded @ vec_0
    assert np.isclose(out_0[0], 1.0)