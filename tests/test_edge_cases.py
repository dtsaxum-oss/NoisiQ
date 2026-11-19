"""Adversarial and edge-case inputs: the package must give a correct answer or a clear error."""

import numpy as np
import pytest

import noisiq as nq
from noisiq.backends import IncompatibleCircuitError
from noisiq.noise import NoiseModel, bit_flip, depolarizing
from tests.conftest import random_circuit, tvd

BACKENDS = ["frame", "statevector", "density_matrix"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_circuit(backend):
    r = nq.simulate(nq.Circuit(2), shots=10, backend=backend)
    assert r.counts == {"00": 10}


@pytest.mark.parametrize("backend", BACKENDS)
def test_only_barrier_and_identity(backend):
    r = nq.simulate(nq.Circuit(2).barrier().i(0), shots=10, backend=backend)
    assert r.counts == {"00": 10}


@pytest.mark.parametrize("backend", BACKENDS)
def test_unmeasured_clbits_and_repeated_measurement(backend):
    c = nq.Circuit(1, n_clbits=3)
    c.x(0).measure(0, 1).measure(0, 1)
    r = nq.simulate(c, shots=10, backend=backend)
    assert r.counts == {"010": 10}


@pytest.mark.parametrize("backend", BACKENDS)
def test_condition_on_unwritten_bit_reads_zero(backend):
    c = nq.Circuit(1, n_clbits=1)
    c.c_if(0, 0).x(0)
    c.measure(0, 0)
    assert nq.simulate(c, shots=10, backend=backend).counts == {"1": 10}


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_layers_out_of_insertion_order(backend):
    c = nq.Circuit(1).h(0, layer=2).x(0, layer=0)  # X runs first, then H
    c.measure(0, basis="X")
    assert nq.simulate(c, shots=50, backend=backend, seed=1).counts == {"1": 50}


def test_certain_errors_are_deterministic():
    c = nq.Circuit(2).i(0).i(1).measure_all()
    nm = NoiseModel().add_gate_error(bit_flip(1.0))
    for b in BACKENDS:
        assert nq.simulate(c, nm, shots=20, backend=b, seed=0).counts == {"11": 20}
    nm = NoiseModel().add_readout_error(1.0)
    for b in BACKENDS:
        assert nq.simulate(c, nm, shots=20, backend=b, seed=0).counts == {"11": 20}


def test_zero_probability_noise_is_noiseless():
    nm = NoiseModel.uniform_depolarizing(0.0, 0.0).add_readout_error(0.0)
    r = nq.simulate(nq.algorithms.ghz(3), nm, shots=100, seed=1)
    assert set(r.counts) == {"000", "111"}
    assert r.fault_free_fraction.value == 1.0


def test_single_qubit_and_large_frame_circuit():
    assert nq.simulate(nq.Circuit(1).h(0), shots=1000, seed=1).counts.keys() == {"0", "1"}
    c = nq.Circuit(64).h(range(64)).measure_all(basis="X")
    r = nq.simulate(c, NoiseModel.uniform_depolarizing(0.001), shots=200, seed=1)
    assert r.memory.shape == (200, 64)


def test_statevector_qubit_limit():
    with pytest.raises(IncompatibleCircuitError):
        nq.simulate(nq.Circuit(30).t(0), backend="statevector")
    with pytest.raises(IncompatibleCircuitError):
        nq.simulate(nq.Circuit(11).h(0), backend="density_matrix")


def test_quasistatic_detuning_rejected_where_unsupported():
    nm = NoiseModel().add_quasistatic_detuning(1e6)
    c = nq.Circuit(1).h(0).delay(0, 1e-6).measure(0)
    for b in ("frame", "density_matrix"):
        with pytest.raises(IncompatibleCircuitError):
            nq.simulate(c, nm, backend=b)
    assert nq.simulate(c, nm, shots=10).backend == "statevector"


def test_multi_bit_condition_feedforward():
    c = nq.Circuit(3)
    c.x(0).measure(0).measure(1)  # bits = (1, 0) → value 1
    c.c_if([0, 1], 1).x(2)
    c.c_if([0, 1], 2).z(2)
    c.measure(2)
    for b in BACKENDS:
        assert nq.simulate(c, shots=20, backend=b).counts == {"101": 20}, b


def test_three_qubit_correlated_channel_on_toffoli():
    c = nq.Circuit(3).ccx(0, 1, 2)
    nm = NoiseModel().add_gate_error(depolarizing(0.3, 3), gates="CCX")
    p_exact = nq.simulate(c, nm, shots=1, backend="density_matrix").probabilities()
    r = nq.simulate(c, nm, shots=6000, seed=1, backend="statevector")
    assert tvd(r.probabilities(), p_exact) < 0.04


def test_custom_kraus_channel_end_to_end():
    K0 = np.sqrt(0.9) * np.eye(2)
    K1 = np.sqrt(0.1) * np.array([[0, 1], [1, 0]])
    nm = NoiseModel().add_gate_error(nq.noise.KrausChannel([K0, K1], name="my_flip"))
    r = nq.simulate(nq.Circuit(1).i(0), nm, shots=10000, seed=1)
    assert r.backend == "statevector"
    assert r.probabilities()["1"] == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_random_programs_all_backends_agree(seed):
    """Random circuits mixing everything; frame/statevector must match the exact reference."""
    rng = np.random.default_rng(1000 + seed)
    clifford = bool(seed % 2)
    c = random_circuit(
        int(rng.integers(1, 4)),
        int(rng.integers(1, 25)),
        rng,
        clifford=clifford,
        mid_measure=True,
        conditionals=clifford,
        resets=True,
        bases=True,
    )
    nm = NoiseModel()
    nm.add_gate_error(depolarizing(float(rng.uniform(0, 0.1))), num_qubits=1)
    nm.add_gate_error(depolarizing(float(rng.uniform(0, 0.1)), 2), num_qubits=2)
    nm.add_readout_error(float(rng.uniform(0, 0.1)), float(rng.uniform(0, 0.1)))
    nm.add_reset_error(float(rng.uniform(0, 0.05)))
    if not clifford:
        nm.add_thermal_relaxation(30e-6, 40e-6)
    p_exact = nq.simulate(c, nm, shots=1, backend="density_matrix").probabilities()
    backends = ["statevector"] + (["frame"] if clifford else [])
    for b in backends:
        r = nq.simulate(c, nm, shots=5000, backend=b, seed=seed)
        assert tvd(r.probabilities(), p_exact) < 0.05, (b, seed)
