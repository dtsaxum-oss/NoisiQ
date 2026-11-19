"""Backend correctness: noiseless exactness, cross-backend agreement, reproducibility."""

import numpy as np
import pytest

import noisiq as nq
from noisiq.backends import IncompatibleCircuitError
from noisiq.noise import AmplitudeDamping, CoherentRotation, NoiseModel, bit_flip, depolarizing, phase_flip
from tests.conftest import random_circuit, tvd

SHOTS = 6000
TOL = 0.045  # TVD tolerance at 6000 shots for small outcome spaces


def exact(circuit, noise=None):
    return nq.simulate(circuit, noise, shots=10, backend="density_matrix", seed=0).probabilities()


# ----------------------------------------------------------------------------- basics
@pytest.mark.parametrize("backend", ["frame", "statevector", "density_matrix"])
def test_noiseless_bell(backend):
    r = nq.simulate(nq.algorithms.bell(), shots=2000, backend=backend, seed=1)
    assert set(r.counts) == {"00", "11"}
    assert r.counts["00"] == pytest.approx(1000, abs=150)


@pytest.mark.parametrize("backend", ["frame", "statevector", "density_matrix"])
def test_deterministic_x_and_implicit_measurement(backend):
    r = nq.simulate(nq.Circuit(3).x(0).x(2), shots=50, backend=backend, seed=1)
    assert r.counts == {"101": 50}
    assert r.implicit_measurement


def test_auto_backend_selection_reasons():
    assert nq.select_backend(nq.algorithms.ghz(5))[0] == "frame"
    assert nq.select_backend(nq.Circuit(1).t(0))[0] == "statevector"
    nm = NoiseModel().add_gate_error(AmplitudeDamping(0.1))
    assert nq.select_backend(nq.algorithms.ghz(3), nm)[0] == "statevector"
    assert nq.select_backend(nq.algorithms.ghz(3), nm.twirled())[0] == "frame"
    with pytest.raises(IncompatibleCircuitError):
        nq.select_backend(nq.Circuit(40).t(range(40)))
    r = nq.simulate(nq.algorithms.ghz(3), shots=5)
    assert "Clifford" in r.metadata["backend_reason"]


def test_frame_rejects_incompatible_inputs_with_clear_message():
    with pytest.raises(IncompatibleCircuitError, match="non-Clifford"):
        nq.simulate(nq.Circuit(1).t(0), backend="frame")
    nm = NoiseModel().add_gate_error(AmplitudeDamping(0.1))
    with pytest.raises(IncompatibleCircuitError, match="non-Pauli"):
        nq.simulate(nq.Circuit(1).h(0), nm, backend="frame")
    r = nq.simulate(nq.Circuit(1).h(0), nm, backend="frame", approximate=True, shots=10)
    assert "twirled" in r.noise_name
    c = nq.Circuit(2).h(0).measure(0)
    c.c_if(0).h(1)
    with pytest.raises(IncompatibleCircuitError, match="classically-controlled"):
        nq.simulate(c, backend="frame")


def test_unknown_backend_and_bad_args():
    with pytest.raises(ValueError):
        nq.simulate(nq.Circuit(1), backend="quantum_magic")
    with pytest.raises(TypeError):
        nq.simulate("not a circuit")
    with pytest.raises(ValueError):
        nq.simulate(nq.Circuit(1), shots=0, backend="frame")


@pytest.mark.parametrize("backend", ["frame", "statevector", "density_matrix"])
def test_seeded_runs_are_reproducible(backend):
    c = nq.algorithms.ghz(3, measure=True)
    nm = NoiseModel.uniform_depolarizing(0.05, readout=0.02)
    a = nq.simulate(c, nm, shots=300, seed=42, backend=backend)
    b = nq.simulate(c, nm, shots=300, seed=42, backend=backend)
    assert a.counts == b.counts


# ----------------------------------------------------------------------------- cross-checks
@pytest.mark.parametrize("seed", range(8))
def test_random_clifford_circuits_frame_matches_exact(seed):
    rng = np.random.default_rng(seed)
    c = random_circuit(3, 14, rng)
    nm = NoiseModel.uniform_depolarizing(0.03, 0.08, readout=0.03)
    nm.add_idle_error(phase_flip(0.02))
    p_exact = exact(c, nm)
    r = nq.simulate(c, nm, shots=SHOTS, backend="frame", seed=seed)
    assert tvd(r.probabilities(), p_exact) < TOL


@pytest.mark.parametrize("seed", range(6))
def test_random_universal_circuits_statevector_matches_exact(seed):
    rng = np.random.default_rng(100 + seed)
    c = random_circuit(3, 10, rng, clifford=False)
    nm = NoiseModel()
    nm.add_thermal_relaxation(20e-6, 25e-6)
    nm.add_gate_error(CoherentRotation("X", 0.15), num_qubits=1)
    nm.add_gate_error(depolarizing(0.05, 2), num_qubits=2)
    nm.add_readout_error(0.02, 0.05)
    p_exact = exact(c, nm)
    r = nq.simulate(c, nm, shots=SHOTS, backend="statevector", seed=seed)
    assert tvd(r.probabilities(), p_exact) < TOL


@pytest.mark.parametrize("seed", range(6))
def test_mid_circuit_measurement_feedforward_all_backends_agree(seed):
    rng = np.random.default_rng(200 + seed)
    c = random_circuit(3, 18, rng, mid_measure=True, conditionals=True, resets=True, bases=True)
    if not c.has_measurements:
        c.measure(0)
    nm = NoiseModel.uniform_depolarizing(0.02, 0.05, readout=0.03).add_reset_error(0.02)
    p_exact = exact(c, nm)
    for backend in ("frame", "statevector"):
        r = nq.simulate(c, nm, shots=SHOTS, backend=backend, seed=seed)
        assert tvd(r.probabilities(), p_exact) < TOL, backend


def test_twirled_noise_on_frame_matches_twirled_exact():
    c = nq.algorithms.ghz(3, measure=True)
    nm = nq.get_hardware("ibm_eagle_r3").noise_model(representation="pauli")
    r = nq.simulate(c, nm, shots=SHOTS, backend="frame", seed=3)
    assert tvd(r.probabilities(), exact(c, nm)) < TOL


def test_coherent_error_accumulates_quadratically():
    """N identical over-rotations add in amplitude (statevector) but not when twirled."""
    eps, n = 0.05, 20
    c = nq.Circuit(1)
    for _ in range(n):
        c.i(0)
    nm = NoiseModel().add_gate_error(CoherentRotation("X", eps))
    p1_coherent = exact(c, nm).get("1", 0)
    p1_twirled = exact(c, nm.twirled()).get("1", 0)
    assert p1_coherent == pytest.approx(np.sin(n * eps / 2) ** 2, rel=1e-6)
    assert p1_twirled == pytest.approx((1 - (1 - 2 * np.sin(eps / 2) ** 2) ** n) / 2, rel=1e-6)
    assert p1_coherent > 5 * p1_twirled


@pytest.mark.parametrize("basis,prep,expect", [("X", "h", "0"), ("Y", "hs", "0"), ("Z", "x", "1"), ("X", "xh", "1")])
@pytest.mark.parametrize("backend", ["frame", "statevector", "density_matrix"])
def test_measurement_bases(basis, prep, expect, backend):
    c = nq.Circuit(1)
    for g in prep:
        getattr(c, g)(0)
    c.measure(0, basis=basis)
    r = nq.simulate(c, shots=200, backend=backend, seed=1)
    assert r.counts == {expect: 200}


@pytest.mark.parametrize("backend", ["frame", "statevector", "density_matrix"])
def test_reset_and_measure_reset(backend):
    c = nq.Circuit(2).x(0).h(1).reset(0).measure(1, reset=True).measure(0).measure(1)
    r = nq.simulate(c, shots=400, backend=backend, seed=5)
    assert r.marginal_counts([1, 2]) == {"00": 400}
    assert set(r.marginal_counts([0])) == {"0", "1"}


def test_readout_error_affects_feedforward():
    c = nq.Circuit(2).measure(0)
    c.c_if(0).x(1)
    c.measure(1)
    nm = NoiseModel().add_readout_error(0.25, qubits=[0])
    for backend in ("frame", "statevector", "density_matrix"):
        r = nq.simulate(c, nm, shots=8000, backend=backend, seed=2)
        p = r.probabilities()
        assert p.get("11", 0) == pytest.approx(0.25, abs=0.03), backend
        assert set(p) <= {"00", "11"}


def test_conditional_faults_only_fire_with_gate():
    c = nq.Circuit(2).measure(0)
    c.c_if(0).x(1)
    c.measure(1)
    nm = NoiseModel().add_gate_error(bit_flip(1.0), gates="X")
    for backend in ("frame", "statevector", "density_matrix"):
        r = nq.simulate(c, nm, shots=100, backend=backend, seed=0)
        assert r.counts == {"00": 100}, backend


# ----------------------------------------------------------------------------- results content
def test_fault_statistics_match_channel_probabilities():
    c = nq.Circuit(1)
    for _ in range(10):
        c.i(0)
    nm = NoiseModel().add_gate_error(nq.noise.pauli_channel(0.01, 0.02, 0.03))
    for backend in ("frame", "statevector"):
        r = nq.simulate(c, nm, shots=20000, backend=backend, seed=1)
        rates = r.fault_rates(by_pauli=True)[:, 0, :3].mean(axis=0)
        assert rates == pytest.approx([0.01, 0.02, 0.03], abs=0.003), backend
        assert r.fault_free_fraction.value == pytest.approx(0.94**10, abs=0.01)


def test_idle_noise_is_recorded_per_layer():
    c = nq.Circuit(2).h(0).h(0).h(0).cx(0, 1)
    nm = NoiseModel().add_idle_error(lambda d: phase_flip(0.1))
    r = nq.simulate(c, nm, shots=5000, backend="frame", seed=1)
    idle = r.idle_fault_rates()
    assert idle[:3, 1] == pytest.approx([0.1] * 3, abs=0.02)
    assert idle[:, 0].sum() == 0


def test_output_error_rate_counts_only_harmful_faults():
    # Z errors before a Z measurement never change the outcome.
    c = nq.Circuit(1).x(0).i(0).measure(0)
    nm = NoiseModel().add_gate_error(phase_flip(0.5), gates="I")
    r = nq.simulate(c, nm, shots=2000, seed=1)
    assert r.output_error_rate().value == 0
    assert r.fault_free_fraction.value == pytest.approx(0.5, abs=0.05)
    assert r.final_error_strings(5)  # recorded before the measurement


def test_statevector_fidelity_and_reduced_states():
    c = nq.algorithms.ghz(3)
    r = nq.simulate(c, shots=50, backend="statevector", seed=1)
    assert r.fidelity.value == pytest.approx(1.0)
    assert r.qubit_purities() == pytest.approx([0.5, 0.5, 0.5])
    assert nq.metrics.ghz_fidelity(r.density_matrix) == pytest.approx(1.0)
    nm = NoiseModel().add_gate_error(depolarizing(0.2), num_qubits=1)
    r2 = nq.simulate(c, nm, shots=3000, backend="statevector", seed=1)
    ex = nq.simulate(c, nm, shots=1, backend="density_matrix")
    assert r2.fidelity.value == pytest.approx(ex.fidelity.value, abs=0.03)


def test_statevector_handles_20_qubits_without_density_matrix():
    c = nq.Circuit(20).h(0)
    for i in range(19):
        c.cx(i, i + 1)
    c.t(5)
    r = nq.simulate(c, NoiseModel().add_gate_error(AmplitudeDamping(0.01), num_qubits=1), shots=3, seed=1)
    assert r.backend == "statevector" and r.density_matrix is None
    assert r.qubit_states.shape == (20, 2, 2)


@pytest.mark.slow
def test_frame_backend_scales_to_hundreds_of_qubits():
    c = nq.algorithms.ghz(300, layout="tree", measure=True)
    r = nq.simulate(c, NoiseModel.uniform_depolarizing(1e-3), shots=5000, seed=0)
    assert r.backend == "frame"
    assert 0 < r.output_error_rate().value < 1


@pytest.mark.qiskit
def test_aer_backend_matches_exact():
    pytest.importorskip("qiskit_aer")
    rng = np.random.default_rng(7)
    c = random_circuit(3, 10, rng, clifford=False)
    nm = NoiseModel()
    nm.add_thermal_relaxation(20e-6, 25e-6)
    nm.add_gate_error(depolarizing(0.03, 2), num_qubits=2)
    r = nq.simulate(c, nm, shots=SHOTS, backend="aer", seed=1)
    assert tvd(r.probabilities(), exact(c, nm)) < TOL
