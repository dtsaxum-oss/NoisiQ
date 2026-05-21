"""
Tests for noisiq/noise/hardware_noise.py

Covers:
  - HardwareProfile / GHZResult / GateTimes dataclasses
  - to_noise_model() output types and parameter correctness
  - register_hardware / get_hardware / list_hardware registry
  - state_fidelity function (pure-state and mixed-state paths)
  - plot_density_matrix / plot_purity_decay smoke tests
"""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless — no display required

import noisiq as nq
from noisiq.noise import (
    AmplitudeDamping,
    T2Dephasing,
    GHZResult,
    GateTimes,
    HardwareProfile,
    get_hardware,
    list_hardware,
    register_hardware,
)
from noisiq.backends import TrajectoryBackend
from noisiq.visualization.density_matrix import (
    state_fidelity,
    plot_density_matrix,
    plot_purity_decay,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_circuit():
    """3-qubit GHZ circuit."""
    c = nq.Circuit(n_qubits=3)
    c.h(0).cnot(0, 1).cnot(1, 2)
    return c


@pytest.fixture
def eagle_profile():
    return get_hardware("ibm_eagle_r3")


# ===========================================================================
# GHZResult
# ===========================================================================

def test_ghz_result_fields():
    r = GHZResult(n_qubits=127, fidelity=0.546, year=2023, source="arXiv:2101.08946")
    assert r.n_qubits == 127
    assert r.fidelity == pytest.approx(0.546)
    assert r.year == 2023
    assert r.notes == ""


def test_ghz_result_none_fidelity():
    r = GHZResult(n_qubits=50, fidelity=None, year=2024, source="press release")
    assert r.fidelity is None


# ===========================================================================
# GateTimes
# ===========================================================================

def test_gate_times_fields():
    g = GateTimes(single_qubit_ns=50.0, two_qubit_ns=200.0)
    assert g.single_qubit_ns == 50.0
    assert g.two_qubit_ns == 200.0


# ===========================================================================
# HardwareProfile
# ===========================================================================

def test_profile_repr(eagle_profile):
    r = repr(eagle_profile)
    assert "ibm_eagle_r3" in r
    assert "IBM" in r


def test_profile_describe(eagle_profile):
    d = eagle_profile.describe()
    assert d["name"] == "ibm_eagle_r3"
    assert d["t2_us"] == pytest.approx(177.0)
    assert "gate_times_ns" in d
    assert len(d["ghz_results"]) == 3


def test_profile_ghz_results_populated(eagle_profile):
    assert len(eagle_profile.ghz_results) >= 1
    assert all(r.fidelity is not None for r in eagle_profile.ghz_results)


# ===========================================================================
# to_noise_model
# ===========================================================================

def test_to_noise_model_t2_returns_correct_type(eagle_profile, simple_circuit):
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2")
    assert isinstance(noise, dict)
    assert len(noise) == len(simple_circuit.operations)
    assert all(isinstance(v, T2Dephasing) for v in noise.values())


def test_to_noise_model_t1_returns_amplitude_damping(eagle_profile, simple_circuit):
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t1")
    assert all(isinstance(v, AmplitudeDamping) for v in noise.values())


def test_to_noise_model_gate_times_vary_by_qubit_count(eagle_profile, simple_circuit):
    """Single-qubit ops (H) use single_qubit_ns; two-qubit ops (CNOT) use two_qubit_ns."""
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2")
    for op_idx, op in enumerate(simple_circuit.operations):
        channel = noise[op_idx]
        expected_t = (
            eagle_profile.gate_times.single_qubit_ns * 1e-9
            if op.gate.num_qubits == 1
            else eagle_profile.gate_times.two_qubit_ns * 1e-9
        )
        assert channel.t == pytest.approx(expected_t)


def test_to_noise_model_invalid_mode_raises(eagle_profile, simple_circuit):
    with pytest.raises(ValueError, match="mode must be"):
        eagle_profile.to_noise_model(simple_circuit, mode="both")


def test_to_noise_model_channels_are_valid(eagle_profile, simple_circuit):
    """All generated channels must pass trace-preservation check."""
    for channel in eagle_profile.to_noise_model(simple_circuit, mode="t2").values():
        channel.validate()
    for channel in eagle_profile.to_noise_model(simple_circuit, mode="t1").values():
        channel.validate()


# ===========================================================================
# Registry
# ===========================================================================

def test_list_hardware_returns_sorted():
    names = list_hardware()
    assert names == sorted(names)


def test_list_hardware_contains_preloaded():
    names = list_hardware()
    for expected in ["ibm_eagle_r3", "ibm_heron_r2", "ionq_forte", "quantinuum_h2"]:
        assert expected in names


def test_get_hardware_returns_correct_profile():
    p = get_hardware("ibm_heron_r2")
    assert p.vendor == "IBM"
    assert p.t1 == pytest.approx(300e-6)


def test_get_hardware_missing_raises_key_error():
    with pytest.raises(KeyError, match="no_such_device"):
        get_hardware("no_such_device")


def test_register_and_retrieve_custom_profile():
    custom = HardwareProfile(
        name="_test_custom_device",
        vendor="Test",
        system="Prototype",
        t1=50e-6,
        t2=40e-6,
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        spam_error=0.02,
        gate_times=GateTimes(single_qubit_ns=50.0, two_qubit_ns=400.0),
    )
    register_hardware(custom)
    retrieved = get_hardware("_test_custom_device")
    assert retrieved.vendor == "Test"
    assert retrieved.t2 == pytest.approx(40e-6)
    assert "_test_custom_device" in list_hardware()


# ===========================================================================
# state_fidelity
# ===========================================================================

def _ghz_statevector(n: int) -> np.ndarray:
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = psi[-1] = 1.0 / np.sqrt(2)
    return psi


def _density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def test_state_fidelity_identical_pure():
    psi = _ghz_statevector(2)
    rho = _density_matrix(psi)
    assert state_fidelity(psi, rho) == pytest.approx(1.0, abs=1e-10)


def test_state_fidelity_orthogonal_pure():
    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0], dtype=complex)
    rho1 = _density_matrix(psi1)
    assert state_fidelity(psi0, rho1) == pytest.approx(0.0, abs=1e-10)


def test_state_fidelity_plus_vs_zero():
    """F(|0⟩, |+⟩⟨+|) = 0.5."""
    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    rho_plus = _density_matrix(psi_plus)
    assert state_fidelity(psi0, rho_plus) == pytest.approx(0.5, abs=1e-10)


def test_state_fidelity_maximally_mixed():
    """F(|0⟩, I/2) = 0.5."""
    psi0 = np.array([1.0, 0.0], dtype=complex)
    rho_mixed = np.eye(2, dtype=complex) / 2
    assert state_fidelity(psi0, rho_mixed) == pytest.approx(0.5, abs=1e-10)


def test_state_fidelity_general_both_matrices():
    """General formula agrees with pure-state shortcut on pure inputs."""
    psi = _ghz_statevector(2)
    rho_ideal = _density_matrix(psi)
    rho_noisy = _density_matrix(psi)
    assert state_fidelity(rho_ideal, rho_noisy) == pytest.approx(1.0, abs=1e-8)


def test_state_fidelity_shape_mismatch_raises():
    psi = np.array([1.0, 0.0], dtype=complex)
    rho = np.eye(4, dtype=complex) / 4
    with pytest.raises(ValueError):
        state_fidelity(psi, rho)


def test_state_fidelity_in_range_after_trajectory(simple_circuit, eagle_profile):
    """Fidelity from noisy simulation must be in (0, 1]."""
    psi_ideal = _ghz_statevector(3)
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2")
    result = TrajectoryBackend().run(simple_circuit, noise_model=noise, n_shots=300, seed=0)
    F = state_fidelity(psi_ideal, result.density_matrix)
    assert 0.0 < F <= 1.0 + 1e-8


# ===========================================================================
# plot_density_matrix smoke tests
# ===========================================================================

def test_plot_density_matrix_returns_figure():
    import matplotlib.pyplot as plt
    rho = _density_matrix(_ghz_statevector(2))
    fig = plot_density_matrix(rho)
    assert fig is not None
    assert len(fig.axes) >= 2
    plt.close("all")


def test_plot_density_matrix_invalid_shape():
    with pytest.raises(ValueError):
        plot_density_matrix(np.eye(3))  # 3 is not a power of 2


def test_plot_density_matrix_custom_title():
    import matplotlib.pyplot as plt
    rho = _density_matrix(_ghz_statevector(1))
    fig = plot_density_matrix(rho, title="My title")
    assert fig is not None
    plt.close("all")


# ===========================================================================
# plot_purity_decay smoke tests
# ===========================================================================

def test_plot_purity_decay_returns_figure():
    import matplotlib.pyplot as plt
    circuit_1q = nq.Circuit(n_qubits=1).h(0)
    profile = get_hardware("ibm_eagle_r3")
    t_values = np.array([1e-9, 10e-9, 100e-9])
    results = []
    for t in t_values:
        noise_1q = {0: T2Dephasing(T2=profile.t2, t=float(t))}
        r = TrajectoryBackend().run(circuit_1q, noise_model=noise_1q, n_shots=100, seed=0)
        results.append(r)
    fig = plot_purity_decay(results, t_values, title="Purity test")
    assert fig is not None
    plt.close("all")


def test_plot_purity_decay_length_mismatch_raises():
    circuit_1q = nq.Circuit(n_qubits=1).h(0)
    profile = get_hardware("ibm_eagle_r3")
    noise_1q = {0: T2Dephasing(T2=profile.t2, t=1e-9)}
    r = TrajectoryBackend().run(circuit_1q, noise_model=noise_1q, n_shots=50, seed=0)
    with pytest.raises(ValueError, match="same length"):
        plot_purity_decay([r], np.array([1e-9, 2e-9]))
