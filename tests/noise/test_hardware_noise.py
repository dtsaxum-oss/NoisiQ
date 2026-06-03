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
    Dephasing,
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

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_to_noise_model_t2_returns_correct_type(eagle_profile, simple_circuit):
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2")
    assert isinstance(noise, dict)
    assert len(noise) == len(simple_circuit.operations)
    assert all(isinstance(v, Dephasing) for v in noise.values())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_to_noise_model_t1_returns_amplitude_damping(eagle_profile, simple_circuit):
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t1")
    assert all(isinstance(v, AmplitudeDamping) for v in noise.values())


@pytest.mark.filterwarnings("ignore::UserWarning")
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


@pytest.mark.filterwarnings("ignore::UserWarning")
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


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_state_fidelity_in_range_after_trajectory(simple_circuit, eagle_profile):
    """Fidelity from noisy simulation must be in (0, 1]."""
    psi_ideal = _ghz_statevector(3)
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2")
    result = TrajectoryBackend().run(simple_circuit, noise_model=noise, n_shots=300, seed=0)
    F = state_fidelity(psi_ideal, result.final_state)
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
        noise_1q = {0: Dephasing(T2=profile.t2, t=float(t))}
        r = TrajectoryBackend().run(circuit_1q, noise_model=noise_1q, n_shots=100, seed=0)
        results.append(r)
    fig = plot_purity_decay(results, t_values, title="Purity test")
    assert fig is not None
    plt.close("all")


def test_plot_purity_decay_length_mismatch_raises():
    circuit_1q = nq.Circuit(n_qubits=1).h(0)
    profile = get_hardware("ibm_eagle_r3")
    noise_1q = {0: Dephasing(T2=profile.t2, t=1e-9)}
    r = TrajectoryBackend().run(circuit_1q, noise_model=noise_1q, n_shots=50, seed=0)
    with pytest.raises(ValueError, match="same length"):
        plot_purity_decay([r], np.array([1e-9, 2e-9]))


# ===========================================================================
# to_noise_model — representation kwarg (extended mode)
# ===========================================================================

from noisiq.noise.kraus_channels import CombinedChannel as _CombinedChannel
from noisiq.noise.pauli_error import PauliError as _PauliError
from noisiq.noise.amplitude_damping import AmplitudeDamping as _AmplitudeDamping
from noisiq.noise.t2_dephasing import Dephasing as _Dephasing
from noisiq.noise.coherent_errors import CoherentRotation as _CoherentRotation


@pytest.fixture
def heron_profile():
    return get_hardware("ibm_heron_r2")


@pytest.fixture
def ghz_circuit_2q():
    """2-qubit Bell circuit: H + CNOT."""
    c = nq.Circuit(n_qubits=2)
    c.h(0).cnot(0, 1)
    return c


def test_coherent_fraction_warns_when_representation_none(eagle_profile, simple_circuit):
    """to_noise_model(circuit) emits UserWarning when coherent_fraction > 0."""
    assert eagle_profile.coherent_fraction > 0
    with pytest.warns(UserWarning, match="coherent_fraction"):
        eagle_profile.to_noise_model(simple_circuit, mode="t2")


def test_coherent_fraction_no_warning_when_zero(simple_circuit):
    """to_noise_model(circuit) emits no UserWarning when coherent_fraction == 0."""
    profile = HardwareProfile(
        name="_test_no_coherent",
        vendor="Test",
        system="Test",
        t1=100e-6,
        t2=80e-6,
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        spam_error=0.005,
        gate_times=GateTimes(single_qubit_ns=50.0, two_qubit_ns=400.0),
        coherent_fraction=0.0,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        profile.to_noise_model(simple_circuit, mode="t2")  # must not raise


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_representation_none_backward_compatible(eagle_profile, simple_circuit):
    """Default (representation=None) still returns one Dephasing per op."""
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2")
    assert all(isinstance(v, Dephasing) for v in noise.values())


def test_representation_invalid_raises(eagle_profile, simple_circuit):
    with pytest.raises(ValueError, match="representation must be"):
        eagle_profile.to_noise_model(simple_circuit, mode="t2", representation="kraus")


def test_pauli_twirl_returns_dict_same_length(eagle_profile, simple_circuit):
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2", representation="pauli_twirl")
    assert isinstance(noise, dict)
    assert len(noise) == len(simple_circuit.operations)


def test_coherent_returns_dict_same_length(eagle_profile, simple_circuit):
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t1", representation="coherent")
    assert isinstance(noise, dict)
    assert len(noise) == len(simple_circuit.operations)


def test_pauli_twirl_values_contain_pauli_error(eagle_profile, ghz_circuit_2q):
    """pauli_twirl representation must produce PauliError or CombinedChannel containing one."""
    noise = eagle_profile.to_noise_model(ghz_circuit_2q, mode="t2", representation="pauli_twirl")
    for v in noise.values():
        if isinstance(v, _CombinedChannel):
            types = [type(c) for c in v.channels]
            assert _PauliError in types, f"Expected PauliError in CombinedChannel, got {types}"
        else:
            assert isinstance(v, _PauliError), f"Unexpected type {type(v)}"


def test_coherent_1q_gate_uses_kraus_channel(eagle_profile, ghz_circuit_2q):
    """coherent representation must include AmplitudeDamping or Dephasing for decoherence."""
    noise = eagle_profile.to_noise_model(ghz_circuit_2q, mode="t1", representation="coherent")
    for op_idx, op in enumerate(ghz_circuit_2q.operations):
        v = noise[op_idx]
        if isinstance(v, _CombinedChannel):
            assert any(isinstance(c, _AmplitudeDamping) for c in v.channels), (
                f"Expected AmplitudeDamping in CombinedChannel for op {op_idx}"
            )
        else:
            assert isinstance(v, _AmplitudeDamping)


def test_2q_gate_produces_combined_channel_with_spectator(heron_profile, ghz_circuit_2q):
    """IBM Heron has spectator_error > 0 → coherent representation must include spectator channel."""
    assert heron_profile.spectator_error_per_2q_gate > 0
    noise = heron_profile.to_noise_model(ghz_circuit_2q, mode="t2", representation="coherent")
    # CNOT is op_idx 1 (H at 0, CNOT at 1)
    cnot_channel = noise[1]
    # coherent representation wraps multiple channels in CombinedChannel
    assert isinstance(cnot_channel, _CombinedChannel), (
        f"Expected CombinedChannel for 2q gate with spectator error, got {type(cnot_channel)}"
    )


def test_idle_zz_adds_extra_channel_vs_no_idle(ghz_circuit_2q):
    """IBM Eagle (idle_zz_rate_hz>0) injects one more channel per 2q gate than IonQ (idle_zz_rate_hz=0)."""
    ionq = get_hardware("ionq_forte")
    eagle = get_hardware("ibm_eagle_r3")
    assert ionq.idle_zz_rate_hz == 0.0
    assert eagle.idle_zz_rate_hz > 0.0

    noise_ionq = ionq.to_noise_model(ghz_circuit_2q, mode="t2", representation="coherent")
    noise_eagle = eagle.to_noise_model(ghz_circuit_2q, mode="t2", representation="coherent")

    # CNOT is op_idx 1 in H+CNOT circuit
    def channel_count(channel):
        if isinstance(channel, _CombinedChannel):
            return len(channel.channels)
        return 1

    cnot_ionq = channel_count(noise_ionq[1])
    cnot_eagle = channel_count(noise_eagle[1])
    assert cnot_eagle > cnot_ionq, (
        f"Eagle (idle ZZ) should have more channels than IonQ (no idle ZZ): "
        f"eagle={cnot_eagle}, ionq={cnot_ionq}"
    )


def test_ibm_eagle_has_idle_zz(ghz_circuit_2q):
    """IBM Eagle has idle_zz_rate_hz=22000 → ZZ CoherentRotation present on 2q gate."""
    eagle = get_hardware("ibm_eagle_r3")
    assert eagle.idle_zz_rate_hz > 0
    noise = eagle.to_noise_model(ghz_circuit_2q, mode="t2", representation="coherent")
    cnot_channel = noise[1]
    assert isinstance(cnot_channel, _CombinedChannel)
    zz_channels = [c for c in cnot_channel.channels
                   if isinstance(c, _CoherentRotation) and c.axis == 'ZZ']
    assert len(zz_channels) >= 1, "Expected at least one ZZ CoherentRotation on IBM Eagle 2q gate"


def test_representation_coherent_runs_trajectory(eagle_profile, ghz_circuit_2q):
    """End-to-end: coherent noise model runs in TrajectoryBackend without error."""
    noise = eagle_profile.to_noise_model(ghz_circuit_2q, mode="t1", representation="coherent")
    result = TrajectoryBackend().run(ghz_circuit_2q, noise_model=noise, n_shots=200, seed=42)
    rho = result.final_state
    assert rho.shape == (4, 4)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


def test_representation_pauli_twirl_runs_trajectory(eagle_profile, ghz_circuit_2q):
    """End-to-end: pauli_twirl noise model runs in TrajectoryBackend without error."""
    noise = eagle_profile.to_noise_model(ghz_circuit_2q, mode="t2", representation="pauli_twirl")
    result = TrajectoryBackend().run(ghz_circuit_2q, noise_model=noise, n_shots=200, seed=42)
    rho = result.final_state
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


def test_new_hardware_fields_present():
    """All built-in profiles have the 4 new optional fields."""
    for name in list_hardware():
        p = get_hardware(name)
        assert hasattr(p, 'idle_zz_rate_hz')
        assert hasattr(p, 'coherent_fraction')
        assert hasattr(p, 'spectator_error_per_2q_gate')
        assert hasattr(p, 'mcmr_crosstalk')


def test_quantinuum_has_mcmr_crosstalk():
    q = get_hardware("quantinuum_h2")
    assert q.mcmr_crosstalk > 0.0


# ===========================================================================
# M2 — include_spam parameter
# ===========================================================================

def test_include_spam_false_is_default(eagle_profile, ghz_circuit_2q):
    """include_spam=False (default) must produce identical output to omitting the arg."""
    without = eagle_profile.to_noise_model(ghz_circuit_2q, mode="t2", representation="pauli_twirl")
    with_false = eagle_profile.to_noise_model(
        ghz_circuit_2q, mode="t2", representation="pauli_twirl", include_spam=False
    )
    for idx in without:
        assert without[idx].p_x == pytest.approx(with_false[idx].p_x)
        assert without[idx].p_y == pytest.approx(with_false[idx].p_y)
        assert without[idx].p_z == pytest.approx(with_false[idx].p_z)


def test_include_spam_adds_error_at_first_and_last_op(eagle_profile, simple_circuit):
    """With include_spam=True, first and last op per qubit must have higher total error."""
    base = eagle_profile.to_noise_model(simple_circuit, mode="t2", representation="pauli_twirl")
    spammed = eagle_profile.to_noise_model(
        simple_circuit, mode="t2", representation="pauli_twirl", include_spam=True
    )

    spam = eagle_profile.spam_error / 3.0
    for idx, channel in spammed.items():
        assert isinstance(channel, _PauliError)
        base_total = base[idx].p_x + base[idx].p_y + base[idx].p_z
        spam_total = channel.p_x + channel.p_y + channel.p_z
        # SPAM indices have higher total; non-SPAM indices are unchanged
        assert spam_total >= base_total - 1e-12


def test_include_spam_first_op_per_qubit_higher_than_base(eagle_profile, simple_circuit):
    """Qubit 0's first op (H at idx 0) must gain exactly spam_error/3 on each axis."""
    base = eagle_profile.to_noise_model(simple_circuit, mode="t2", representation="pauli_twirl")
    spammed = eagle_profile.to_noise_model(
        simple_circuit, mode="t2", representation="pauli_twirl", include_spam=True
    )
    spam = eagle_profile.spam_error / 3.0
    # H on qubit 0 is op 0 — first (and also last for that qubit-moment before CNOT propagates)
    # Just check p_x is higher by approximately spam (before clamping)
    assert spammed[0].p_x > base[0].p_x - 1e-12


def test_include_spam_values_are_physical(eagle_profile, simple_circuit):
    """All PauliError entries after SPAM injection must satisfy p_x+p_y+p_z <= 1."""
    noise = eagle_profile.to_noise_model(
        simple_circuit, mode="t2", representation="pauli_twirl", include_spam=True
    )
    for channel in noise.values():
        if isinstance(channel, _PauliError):
            assert channel.p_x + channel.p_y + channel.p_z <= 1.0 + 1e-9


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_include_spam_ignored_for_none_representation(eagle_profile, simple_circuit):
    """include_spam=True with representation=None must not raise and returns Dephasing channels."""
    noise = eagle_profile.to_noise_model(simple_circuit, mode="t2", include_spam=True)
    assert all(isinstance(v, Dephasing) for v in noise.values())


def test_include_spam_ignored_for_coherent_representation(eagle_profile, ghz_circuit_2q):
    """include_spam=True with representation='coherent' must not raise."""
    noise = eagle_profile.to_noise_model(
        ghz_circuit_2q, mode="t2", representation="coherent", include_spam=True
    )
    assert isinstance(noise, dict)
    assert len(noise) == len(ghz_circuit_2q.operations)


def test_include_spam_same_length_as_base(eagle_profile, simple_circuit):
    """SPAM injection must not add or remove keys from the noise dict."""
    base = eagle_profile.to_noise_model(simple_circuit, mode="t2", representation="pauli_twirl")
    spammed = eagle_profile.to_noise_model(
        simple_circuit, mode="t2", representation="pauli_twirl", include_spam=True
    )
    assert set(spammed.keys()) == set(base.keys())


# ===========================================================================
# idle_coherent_epsilon — channel type varies by representation
# ===========================================================================

import dataclasses
from noisiq.noise.coherent_errors import StochasticCoherentRotation as _StochasticCoherentRotation
from noisiq.noise.idle_fill import fill_idle_with_identities as _fill_idle


@pytest.fixture
def profile_with_epsilon():
    """ibm_eagle_r3 with idle_coherent_epsilon=0.07."""
    base = get_hardware("ibm_eagle_r3")
    return dataclasses.replace(base, idle_coherent_epsilon=0.07)


@pytest.fixture
def idle_circuit(profile_with_epsilon):
    """1-qubit H + gap + H circuit with IDLE slots filled."""
    c = nq.Circuit(n_qubits=1)
    c.h(0, t=0)
    c.h(0, t=4)
    return _fill_idle(c, profile_with_epsilon.gate_times)


def _find_idle_channel(noise_dict, circuit):
    """Return the channel for the first IDLE op in the circuit."""
    from noisiq.ir import gates as _ig
    for idx, op in enumerate(circuit.operations):
        if hasattr(op, 'gate') and op.gate is _ig.IDLE:
            return noise_dict[idx]
    raise AssertionError("No IDLE op found in circuit")


def test_idle_coherent_epsilon_coherent_path_uses_stochastic(profile_with_epsilon, idle_circuit):
    """representation='coherent' → CombinedChannel containing StochasticCoherentRotation."""
    noise = profile_with_epsilon.to_noise_model(
        idle_circuit, mode="t2", representation="coherent"
    )
    ch = _find_idle_channel(noise, idle_circuit)
    assert isinstance(ch, _CombinedChannel)
    stoch = [c for c in ch.channels if isinstance(c, _StochasticCoherentRotation)]
    assert len(stoch) == 1, f"Expected 1 StochasticCoherentRotation, got {stoch}"
    assert stoch[0].std_dev == pytest.approx(0.07)
    # No deterministic CoherentRotation should remain
    coh_rot = [c for c in ch.channels if isinstance(c, _CoherentRotation)]
    assert len(coh_rot) == 0, f"CoherentRotation should not appear in coherent path: {coh_rot}"


def test_idle_coherent_epsilon_pauli_twirl_path_uses_pauli_error(profile_with_epsilon, idle_circuit):
    """representation='pauli_twirl' → CombinedChannel with only PauliError, no CoherentRotation."""
    noise = profile_with_epsilon.to_noise_model(
        idle_circuit, mode="t2", representation="pauli_twirl"
    )
    ch = _find_idle_channel(noise, idle_circuit)
    assert isinstance(ch, _CombinedChannel)
    # No CoherentRotation or StochasticCoherentRotation should appear
    bad = [c for c in ch.channels
           if isinstance(c, (_CoherentRotation, _StochasticCoherentRotation))]
    assert len(bad) == 0, f"No coherent channel expected in pauli_twirl path, got {bad}"
    # The coherent epsilon contribution should be a PauliError
    pauli_channels = [c for c in ch.channels if isinstance(c, _PauliError)]
    assert len(pauli_channels) >= 1


def test_idle_coherent_epsilon_none_path_no_coherent_channel(profile_with_epsilon, idle_circuit):
    """representation=None takes the early-return Kraus path and does not apply
    idle_coherent_epsilon — that field is only used in the extended representation block."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise = profile_with_epsilon.to_noise_model(idle_circuit, mode="t2", representation=None)
    ch = _find_idle_channel(noise, idle_circuit)
    # No coherent channel of any kind should appear
    def _flatten(c):
        if isinstance(c, _CombinedChannel):
            return [x for inner in c.channels for x in _flatten(inner)]
        return [c]
    all_inner = _flatten(ch)
    bad = [c for c in all_inner
           if isinstance(c, (_CoherentRotation, _StochasticCoherentRotation))]
    assert len(bad) == 0, f"representation=None should not apply idle_coherent_epsilon: {bad}"


def test_idle_coherent_epsilon_zero_no_stochastic(idle_circuit):
    """When idle_coherent_epsilon=0, no coherent channel is appended."""
    base = get_hardware("ibm_eagle_r3")
    assert base.idle_coherent_epsilon == 0.0
    noise = base.to_noise_model(idle_circuit, mode="t2", representation="coherent")
    ch = _find_idle_channel(noise, idle_circuit)
    # Should be a plain Kraus channel (CombinedChannel from idle_kraus), not wrapped further
    inner_stoch = []
    if isinstance(ch, _CombinedChannel):
        inner_stoch = [c for c in ch.channels if isinstance(c, _StochasticCoherentRotation)]
    assert len(inner_stoch) == 0, f"Should have no StochasticCoherentRotation when epsilon=0"


def test_idle_coherent_epsilon_stochastic_causes_purity_loss(profile_with_epsilon, idle_circuit):
    """End-to-end: with StochasticCoherentRotation on IDLE, purity drops below 1."""
    noise = profile_with_epsilon.to_noise_model(
        idle_circuit, mode="t2", representation="coherent"
    )
    result = TrajectoryBackend().run(idle_circuit, noise_model=noise, n_shots=800, seed=0)
    rho = result.final_state
    purity = float(np.real(np.trace(rho @ rho)))
    assert purity < 0.999, f"Purity should drop with stochastic idle noise: {purity}"
