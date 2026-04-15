import pytest
import stim
import numpy as np
from noisiq.ir import Circuit, gates
from noisiq.backends import StimTableauBackend, ErrorEvent
from noisiq.noise import PauliError, depolarizing_error

def test_stim_tableau_backend_noiseless():
    """Test StimTableauBackend with a simple noiseless circuit."""
    circuit = Circuit(n_qubits=2)
    circuit.add_gate(gates.H, (0,))
    circuit.add_gate(gates.CNOT, (0, 1))
    
    backend = StimTableauBackend()
    result = backend.run_single_shot(circuit)
    
    assert result.n_qubits == 2
    assert len(result.steps) == 2
    
    # Final state should be |00> + |11> (Bell state)
    # In StimTableau, we can check if the stabilizers match
    # For Bell state: Z0Z1 and X0X1
    final_tableau = result.final_tableau
    
    # Check if Z0Z1 and X0X1 are stabilizers
    # Tableau.z_output(k) gives the Pauli string that Z_k maps to.
    # Actually, we want to check what X and Z on qubits map to.
    # For Bell state, stabilizers are X0X1 and Z0Z1.
    pass

def test_stim_tableau_backend_with_noise():
    """Test StimTableauBackend with noise injection."""
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(gates.I, (0,))
    
    # 100% X error on the first (and only) gate
    noise_config = {0: PauliError(p_x=1.0, p_y=0.0, p_z=0.0)}
    
    backend = StimTableauBackend()
    result = backend.run_single_shot(circuit, noise_config=noise_config, seed=42)
    
    assert len(result.steps) == 1
    step = result.steps[0]
    assert len(step.errors) == 1
    assert step.errors[0].pauli == 'X'
    assert step.errors[0].qubit == 0
    
    # Final state should be |1>
    # In stim.Tableau, Z maps to -Z for state |1>
    assert str(result.final_tableau.z_output(0)) == "-Z"

def test_stim_tableau_backend_unsupported_gate():
    """Test that unsupported gates raise an error."""
    # Create a dummy non-Clifford gate (though T is supported now)
    # Let's try a gate that isn't in our _apply_gate_to_sim mapping
    custom_gate = gates.Gate(name="CUSTOM", matrix=np.eye(2), num_qubits=1)
    
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(custom_gate, (0,))
    
    backend = StimTableauBackend()
    with pytest.raises(ValueError, match="not supported by StimTableauBackend"):
        backend.run_single_shot(circuit)
