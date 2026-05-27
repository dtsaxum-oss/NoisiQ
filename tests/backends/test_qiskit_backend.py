import pytest
import numpy as np

from noisiq.ir.circuit import Circuit
from noisiq.ir import gates
from noisiq.backends.qiskit_backend import QiskitAerBackend

def test_qiskit_backend_all_gates():
    circuit = Circuit(2)
    # Add all supported gates to ensure they don't throw errors
    circuit.add_gate(gates.I, [0])
    circuit.add_gate(gates.IDLE, [0])
    circuit.add_gate(gates.X, [0])
    circuit.add_gate(gates.Y, [0])
    circuit.add_gate(gates.Z, [0])
    circuit.add_gate(gates.H, [0])
    circuit.add_gate(gates.S, [0])
    circuit.add_gate(gates.S_DAG, [0])
    circuit.add_gate(gates.T, [0])
    circuit.add_gate(gates.T_DAG, [0])
    circuit.add_gate(gates.phase_gate(np.pi/4), [0])
    circuit.add_gate(gates.rz_gate(np.pi/4), [0])
    circuit.add_gate(gates.CNOT, [0, 1])
    circuit.add_gate(gates.CZ, [0, 1])
    circuit.add_gate(gates.SWAP, [0, 1])
    
    # Custom gate
    custom_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    custom_gate = gates.Gate(name="CUSTOM", matrix=custom_matrix, num_qubits=1)
    circuit.add_gate(custom_gate, [0])

    backend = QiskitAerBackend()
    result = backend.run(circuit, n_shots=10)
    
    assert result.final_state is not None
    assert result.final_state.shape == (4, 4)

def test_qiskit_backend_with_noise():
    from noisiq.noise.pauli_error import PauliError
    
    circuit = Circuit(1)
    circuit.add_gate(gates.X, [0])
    
    noise = PauliError(p_x=0.1, p_y=0.0, p_z=0.0)
    
    backend = QiskitAerBackend()
    result = backend.run(circuit, noise_model=noise, n_shots=10)
    
    assert result.final_state is not None
    assert result.final_state.shape == (2, 2)
