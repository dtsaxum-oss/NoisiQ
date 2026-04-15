import pytest
import numpy as np
import stim
from noisiq.noise import PauliError, bit_flip_error, depolarizing_error, dephasing_error
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.ir import Circuit, gates

def test_pauli_error_validation():
    """Test that PauliError validates probabilities correctly."""
    # Valid
    PauliError(p_x=0.1, p_y=0.2, p_z=0.3)
    
    # Negative probabilities
    with pytest.raises(ValueError, match="non-negative"):
        PauliError(p_x=-0.1, p_y=0, p_z=0)
    
    # Sum > 1
    with pytest.raises(ValueError, match="Sum of error probabilities must be ≤ 1"):
        PauliError(p_x=0.5, p_y=0.5, p_z=0.1)

def test_pauli_error_sampling():
    """Test that sampling works and follows distribution (statistically)."""
    error = PauliError(p_x=0.2, p_y=0.0, p_z=0.0)
    rng = np.random.default_rng(42)
    
    samples = [error.sample(rng) for _ in range(1000)]
    x_count = samples.count('X')
    i_count = samples.count('I')
    
    # Expected ~200 X's, ~800 I's. 
    assert 150 < x_count < 250
    assert 750 < i_count < 850
    assert 'Y' not in samples
    assert 'Z' not in samples

def test_convenience_functions():
    """Test helper functions for creating errors."""
    bit_flip = bit_flip_error(0.1)
    assert bit_flip.p_x == 0.1
    assert bit_flip.p_y == 0.0
    assert bit_flip.p_z == 0.0
    
    depol = depolarizing_error(0.3)
    assert np.isclose(depol.p_x, 0.1)
    assert np.isclose(depol.p_y, 0.1)
    assert np.isclose(depol.p_z, 0.1)
    
    with pytest.raises(ValueError):
        bit_flip_error(1.1)

def test_backend_noiseless_bell_state():
    """Test StimTableauBackend with a noiseless Bell state."""
    circuit = Circuit(n_qubits=2)
    circuit.add_gate(gates.H, (0,))
    circuit.add_gate(gates.CNOT, (0, 1))
    
    backend = StimTableauBackend()
    result = backend.run_single_shot(circuit)
    
    # For Bell state, stabilizers are X0X1 and Z0Z1.
    # stim.Tableau.z_output(0) should be Z0 if it was just |00>
    # After H(0) and CNOT(0,1):
    # |00> -> 1/sqrt(2)(|0>+|1>)|0> -> 1/sqrt(2)(|00>+|11>)
    # Z0 -> X0X1, Z1 -> Z0Z1
    # Let's check the tableau.
    tableau = result.final_tableau
    assert str(tableau.z_output(0)) == "+XX"
    assert str(tableau.z_output(1)) == "+ZZ"

def test_backend_with_deterministic_noise():
    """Test that noise is applied correctly when p=1."""
    circuit = Circuit(n_qubits=1)
    # Start in |0>. Apply X gate -> should be |1>. 
    # Then apply X error (p=1) -> should be back to |0>.
    circuit.add_gate(gates.X, (0,))
    
    # Error on gate 0 (X gate)
    error = bit_flip_error(1.0)
    noise_config = {0: error}
    
    backend = StimTableauBackend()
    result = backend.run_single_shot(circuit, noise_config=noise_config, seed=42)
    
    # Expected state: |0> (Z0 maps to +Z0)
    assert str(result.final_tableau.z_output(0)) == "+Z"
    assert len(result.steps[0].errors) == 1
    assert result.steps[0].errors[0].pauli == 'X'
    assert result.steps[0].errors[0].qubit == 0

def test_intermediate_steps():
    """Test tracking of intermediate steps."""
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(gates.X, (0,))
    
    backend = StimTableauBackend()
    result = backend.run_single_shot(circuit)
    
    # steps length should be 1
    assert len(result.steps) == 1
    # State after X gate: |1> (Z0 maps to -Z0)
    assert str(result.steps[0].tableau.z_output(0)) == "-Z"
