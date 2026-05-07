import pytest
import numpy as np
import stim
from noisiq.visualization.pauli_frame_tracker import PauliFrame

def test_randomized_clifford_correctness():
    """
    Generate a random Clifford circuit, inject a random Pauli error,
    and verify that PauliFrame correctly tracks the error propagation
    by comparing it against stim.TableauSimulator.
    """
    n_qubits = 5
    n_gates = 100
    
    rng = np.random.default_rng(42)
    
    gates = ['H', 'S', 'X', 'Y', 'Z', 'CX', 'CZ']
    
    frame = PauliFrame(n_qubits)
    sim = stim.TableauSimulator()
    
    # Inject a random error
    error_qubit = int(rng.integers(0, n_qubits))
    error_pauli = rng.choice(['X', 'Y', 'Z'])
    
    frame.inject_error(error_qubit, error_pauli)
    
    stim_circuit = stim.Circuit()
    
    for _ in range(n_gates):
        gate = rng.choice(gates)
        if gate in ['H', 'S', 'X', 'Y', 'Z']:
            q = int(rng.integers(0, n_qubits))
            frame.apply_gate(gate, [q])
            stim_circuit.append(gate, [q])
        elif gate in ['CX', 'CZ']:
            q1, q2 = rng.choice(n_qubits, size=2, replace=False)
            q1, q2 = int(q1), int(q2)
            frame.apply_gate(gate, [q1, q2])
            stim_circuit.append(gate, [q1, q2])
            
    # To verify the error propagation, we can use stim.Tableau.
    # The error E at the beginning propagates to U E U^\dagger at the end.
    # In stim, we can find this by creating a Tableau for the circuit.
    tableau = stim.Tableau.from_circuit(stim_circuit)
    
    # tableau.x_output(k) gives U X_k U^\dagger
    # tableau.z_output(k) gives U Z_k U^\dagger
    # tableau.y_output(k) gives U Y_k U^\dagger
    
    if error_pauli == 'X':
        expected_pauli = tableau.x_output(error_qubit)
    elif error_pauli == 'Y':
        expected_pauli = tableau.y_output(error_qubit)
    elif error_pauli == 'Z':
        expected_pauli = tableau.z_output(error_qubit)
        
    # The expected_pauli is a stim.PauliString.
    # PauliFrame ignores phases, so we just compare the characters.
    expected_str = ""
    for i in range(n_qubits):
        expected_str += "_XYZ"[expected_pauli[i]]
    expected_str = expected_str.replace("_", "I")
    
    assert frame.get_pauli_string() == expected_str
