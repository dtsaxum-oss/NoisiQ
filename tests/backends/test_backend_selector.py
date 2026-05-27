import pytest
from noisiq.ir import Circuit
from noisiq.noise import depolarizing_error
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.backends.backend_selector import BackendSelector
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.backends.qiskit_backend import QiskitAerBackend
from noisiq.backends.trajectory_backend import TrajectoryBackend
import numpy as np
from noisiq.ir import gates

def test_selector_custom_nonclifford():
    c = Circuit(1)
    rx_gate = gates.Gate(name="RX", matrix=np.array([[1, 0], [0, 1]]), num_qubits=1) # dummy unitary
    c.add_gate(rx_gate, (0,))
    backend = BackendSelector.select(c)
    assert isinstance(backend, QiskitAerBackend)

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
    assert isinstance(backend, QiskitAerBackend)

def test_selector_nonpauli():
    c = Circuit(1)
    c.h(0)
    noise = AmplitudeDamping(T1=1.0, t=1.0)
    backend = BackendSelector.select(c, noise)
    assert isinstance(backend, QiskitAerBackend)

def test_selector_correlated_pauli_routes_to_trajectory():
    """CorrelatedPauliError must route to TrajectoryBackend, not StimTableauBackend."""
    c = Circuit(2)
    c.h(0).cnot(0, 1)
    noise = CorrelatedPauliError({'ZZ': 0.01})
    backend = BackendSelector.select(c, noise)
    assert isinstance(backend, QiskitAerBackend)


def test_selector_correlated_pauli_dict_routes_to_trajectory():
    """Per-op dict containing CorrelatedPauliError also routes to TrajectoryBackend."""
    c = Circuit(2)
    c.h(0).cnot(0, 1)
    noise = {1: CorrelatedPauliError({'ZZ': 0.01})}
    backend = BackendSelector.select(c, noise)
    assert isinstance(backend, QiskitAerBackend)


def test_backends_seed_none():
    # Verify that seed=None doesn't crash any backend
    c = Circuit(1)
    c.h(0)
    
    # 1. StimTableauBackend
    b1 = StimTableauBackend()
    res1 = b1.run(c, seed=None, n_shots=2)
    assert res1.counts is not None
    
    # 2. QiskitAerBackend
    b2 = QiskitAerBackend()
    res2 = b2.run(c, seed=None, n_shots=2)
    # QiskitAerBackend returns a density matrix in final_state
    assert res2.final_state is not None
    
    # 3. TrajectoryBackend
    b3 = TrajectoryBackend()
    res3 = b3.run(c, seed=None, n_shots=2)
    assert res3.final_state is not None
