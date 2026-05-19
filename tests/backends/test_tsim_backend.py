import pytest
from noisiq.ir import Circuit
from noisiq.backends.tsim_backend import TsimBackend

from noisiq.noise.amplitude_damping import AmplitudeDamping

def test_tsim_backend_basic():
    c = Circuit(1)
    c.h(0)
    c.tgate(0)
    
    backend = TsimBackend()
    res = backend.run(c, seed=42, n_shots=1)
    # Tsim returns a numpy array of booleans for the measurements.
    # In our wrapper we return SimulationResult.
    assert res.counts is not None
    assert sum(res.counts.values()) == 1

def test_tsim_unsupported_gate():
    # Make a dummy gate that is not in the map
    from noisiq.ir import Gate
    import numpy as np
    dummy = Gate("DUMMY", np.eye(2), 1)
    
    c = Circuit(1)
    c.add_gate(dummy, (0,))
    
    backend = TsimBackend()
    with pytest.raises(ValueError, match="not supported by TsimBackend"):
        backend.run(c)

def test_tsim_kraus_error():
    c = Circuit(1)
    c.h(0)
    
    noise = AmplitudeDamping(T1=1.0, t=1.0)
    backend = TsimBackend()
    with pytest.raises(TypeError, match="does not support non-Pauli noise models"):
        backend.run(c, noise_model=noise)
