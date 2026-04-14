import pytest
import numpy as np
from noisiq.noise import PauliError, bit_flip_error, depolarizing_error, dephasing_error
from noisiq.backends.pauli_frame import PauliFrameBackend
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
    # With seed 42, we should get exactly some number, but we'll check range.
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
    """Test PauliFrameBackend with a noiseless Bell state."""
    circuit = Circuit(n_qubits=2)
    circuit.add_gate(gates.H, (0,))
    circuit.add_gate(gates.CNOT, (0, 1))
    
    backend = PauliFrameBackend()
    result = backend.run_single_shot(circuit)
    
    # |Bell> = 1/sqrt(2) (|00> + |11>)
    # indices: 00 -> 0, 11 -> 3
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1/np.sqrt(2)
    expected[3] = 1/np.sqrt(2)
    
    # State might have global phase? No, should be exact.
    assert np.allclose(result.final_state, expected)
    assert len(result.error_history) == 0

def test_backend_with_deterministic_noise():
    """Test that noise is applied correctly when p=1."""
    circuit = Circuit(n_qubits=1)
    # Start in |0>. Apply X gate -> should be |1>. 
    # Then apply X error (p=1) -> should be back to |0>.
    circuit.add_gate(gates.X, (0,))
    
    # Error on gate 0 (X gate)
    error = bit_flip_error(1.0)
    noise_config = {0: error}
    
    backend = PauliFrameBackend()
    result = backend.run_single_shot(circuit, noise_config=noise_config, seed=42)
    
    # Expected state: |0> (index 0 = 1.0)
    assert np.isclose(result.final_state[0], 1.0)
    assert len(result.error_history) == 1
    assert result.error_history[0].pauli == 'X'
    assert result.error_history[0].qubit == 0

def test_intermediate_states():
    """Test tracking of intermediate states."""
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(gates.X, (0,))
    
    backend = PauliFrameBackend()
    result = backend.run_single_shot(circuit, track_history=True)
    
    # history_length should be 1 (after gate 0)
    # Actually, in the code:
    # state = apply_gate
    # if track_history: history.append(state.copy())
    # if op_idx in noise: ... apply error ... if track_history: history.append(state.copy())
    
    assert len(result.intermediate_states) == 1
    # State after X gate: |1>
    assert np.isclose(result.intermediate_states[0][1], 1.0)
