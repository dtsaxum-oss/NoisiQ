import pytest
import numpy as np
from noisiq.ir import Circuit
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.backends.trajectory_backend import TrajectoryBackend

from noisiq.noise.kraus_channels import KrausChannel

def test_t1_decay_curve():
    T1 = 1.0
    times = [0.0, 0.5, 1.0, 2.0]
    shots = 500
    
    backend = TrajectoryBackend()
    
    for t in times:
        # Prepare |1> state
        c = Circuit(1)
        c.x(0)
        
        noise = AmplitudeDamping(T1=T1, t=t)
        res = backend.run(c, noise_model=noise, n_shots=shots, seed=42)
        
        # Expected probability of measuring 1 is exp(-t/T1)
        expected_p1 = np.exp(-t / T1)

        # Read P(|1⟩) directly from the density matrix via partial trace
        p1 = res.excited_state_probability(0)

        # Allow 5% margin of error due to Monte Carlo sampling
        assert abs(p1 - expected_p1) < 0.05

def test_trajectory_underflow():
    # Test that the zero-probability fallback works in Kraus sampling
    c = Circuit(1)
    c.h(0)

    # Create an unphysical Kraus channel with 0 matrices to force sum_p = 0
    k0 = np.zeros((2, 2), dtype=complex)
    k1 = np.zeros((2, 2), dtype=complex)

    class ZeroChannel(KrausChannel):
        def describe(self) -> dict:
            return {"channel": "zero"}

    import unittest.mock
    with unittest.mock.patch.object(KrausChannel, '_validate_operators'):
        noise = ZeroChannel(operators=[k0, k1])

    backend = TrajectoryBackend()
    res = backend.run(c, noise_model=noise, n_shots=10, seed=42)

    # Should not crash, and should return a density matrix
    assert res.density_matrix is not None
    assert res.n_shots == 10
