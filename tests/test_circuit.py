import pytest

from noisiq.ir import gates
from noisiq.ir.circuit import Circuit, Operation


def test_circuit_creation():
    """Tests basic Circuit creation with default values."""
    c = Circuit(n_qubits=2)
    assert c.n_qubits == 2
    assert c.operations == []
    assert c.name is None
    assert c.metadata is None


def test_circuit_creation_with_name_and_meta():
    """Tests Circuit creation with optional name and metadata."""
    meta = {"author": "tester"}
    c = Circuit(n_qubits=1, name="test_circ", metadata=meta)
    assert c.n_qubits == 1
    assert c.name == "test_circ"
    assert c.metadata == meta


def test_add_gate_to_circuit():
    """Tests adding a valid gate to a circuit."""
    c = Circuit(n_qubits=2)
    c.add_gate(gates.H, qubits=[0])
    c.add_gate(gates.CNOT, qubits=(0, 1))

    assert len(c.operations) == 2
    op1, op2 = c.operations

    assert isinstance(op1, Operation)
    assert op1.gate == gates.H
    assert op1.qubits == (0,)
    assert op1.t == 0

    assert isinstance(op2, Operation)
    assert op2.gate == gates.CNOT
    assert op2.qubits == (0, 1)
    assert op2.t == 1


def test_add_gate_qubit_out_of_bounds_fails():
    """Tests that adding a gate to a non-existent qubit raises a ValueError."""
    c = Circuit(n_qubits=1)
    with pytest.raises(ValueError, match="Qubit index 1 is out of bounds"):
        c.add_gate(gates.H, qubits=[1])


def test_add_gate_wrong_qubit_count_fails():
    """Tests that adding a gate with the wrong number of qubits raises a ValueError."""
    c = Circuit(n_qubits=2)
    with pytest.raises(ValueError, match="acts on 1 qubit.*but was applied to 2"):
        c.add_gate(gates.H, qubits=(0, 1))

    with pytest.raises(ValueError, match="acts on 2 qubit.*but was applied to 1"):
        c.add_gate(gates.CNOT, qubits=[0])


def test_circuit_validate_method():
    """Tests the circuit's validate() method for correctness."""
    # Valid circuit should pass
    c_valid = Circuit(n_qubits=2)
    c_valid.add_gate(gates.H, qubits=[0])
    c_valid.validate()  # Should not raise

    # Invalid n_qubits
    c_invalid_qubits = Circuit(n_qubits=0)
    with pytest.raises(ValueError, match="n_qubits must be > 0"):
        c_invalid_qubits.validate()

    # Manually create an operation with an out-of-bounds qubit
    c_out_of_bounds = Circuit(n_qubits=1)
    c_out_of_bounds.operations.append(Operation(gate=gates.X, qubits=(1,), t=0))
    with pytest.raises(ValueError, match="Qubit index out of range: 1"):
        c_out_of_bounds.validate()