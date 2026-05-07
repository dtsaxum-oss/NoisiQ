import pytest
from noisiq.ir import Circuit
from noisiq.noise import depolarizing_error
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.backends.backend_selector import BackendSelector
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.backends.tsim_backend import TsimBackend
from noisiq.backends.trajectory_backend import TrajectoryBackend

def test_selector_clifford_pauli():
    c = Circuit(1)
    c.h(0)
    noise = depolarizing_error(0.1)
    backend = BackendSelector.select(c, noise)
    assert isinstance(backend, StimTableauBackend)

def test_selector_nonclifford_pauli():
    c = Circuit(1)
    c.tgate(0)
    noise = depolarizing_error(0.1)
    backend = BackendSelector.select(c, noise)
    assert isinstance(backend, TsimBackend)

def test_selector_nonpauli():
    c = Circuit(1)
    c.h(0)
    noise = AmplitudeDamping(T1=1.0, t=1.0)
    backend = BackendSelector.select(c, noise)
    assert isinstance(backend, TrajectoryBackend)
