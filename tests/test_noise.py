import numpy as np
import pytest

import noisiq as nq
from noisiq.ir import gates as G
from noisiq.noise import (
    AmplitudeDamping,
    CoherentRotation,
    ComposedChannel,
    KrausChannel,
    NoiseModel,
    PauliChannel,
    PhaseDamping,
    ReadoutError,
    ThermalRelaxation,
    UnitaryChannel,
    bit_flip,
    bit_phase_flip,
    depolarizing,
    pauli_channel,
    phase_flip,
)

CHANNELS = [
    pauli_channel(0.01, 0.02, 0.03),
    depolarizing(0.1),
    depolarizing(0.2, 2),
    bit_flip(0.3),
    phase_flip(0.2),
    bit_phase_flip(0.1),
    AmplitudeDamping(0.4),
    PhaseDamping(0.3),
    ThermalRelaxation(50e-6, 70e-6, 5e-6),
    CoherentRotation("X", 0.3),
    CoherentRotation("ZZ", 0.2),
    PauliChannel({"ZZ": 0.05, "XI": 0.01}),
    ComposedChannel([AmplitudeDamping(0.1), phase_flip(0.05)]),
]


@pytest.mark.parametrize("ch", CHANNELS, ids=repr)
def test_channels_are_cptp(ch):
    ks = ch.kraus()
    d = 2**ch.num_qubits
    assert np.allclose(sum(K.conj().T @ K for K in ks), np.eye(d), atol=1e-10)


@pytest.mark.parametrize("ch", CHANNELS, ids=repr)
def test_twirl_is_a_distribution_and_preserves_process_fidelity(ch):
    tw = ch.pauli_twirl()
    assert tw.is_pauli
    total = tw.p_identity + sum(tw.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    # process fidelity |Tr K|²/d² summed is invariant under twirling
    d = 2**ch.num_qubits
    f_pro = sum(abs(np.trace(K)) ** 2 for K in ch.kraus()) / d**2
    assert tw.p_identity == pytest.approx(f_pro, abs=1e-9)


def test_twirl_superoperator_matches_twirled_channel(rng):
    from noisiq.noise.channels import pauli_matrix

    ch = AmplitudeDamping(0.3)
    rho = np.array([[0.6, 0.2 - 0.1j], [0.2 + 0.1j, 0.4]])
    twirled = (
        sum(
            pauli_matrix(P) @ ch.superoperator_apply(pauli_matrix(P) @ rho @ pauli_matrix(P)) @ pauli_matrix(P)
            for P in "IXYZ"
        )
        / 4
    )
    assert np.allclose(twirled, ch.pauli_twirl().superoperator_apply(rho))


def test_amplitude_damping_closed_form():
    ch = AmplitudeDamping.from_t1(100e-6, 20e-6)
    g = 1 - np.exp(-0.2)
    assert ch.gamma == pytest.approx(g)
    out = ch.superoperator_apply(np.array([[0, 0], [0, 1]], complex))
    assert out[1, 1].real == pytest.approx(np.exp(-0.2))
    tw = ch.pauli_twirl().probabilities
    assert tw["X"] == pytest.approx(g / 4) and tw["Y"] == pytest.approx(g / 4)
    assert tw["Z"] == pytest.approx((1 - np.sqrt(1 - g)) ** 2 / 4)


@pytest.mark.parametrize("t1,t2,t", [(100e-6, 80e-6, 1e-6), (50e-6, 100e-6, 30e-6), (1.0, 0.3, 0.01)])
def test_thermal_relaxation_decay_rates(t1, t2, t):
    ch = ThermalRelaxation(t1, t2, t)
    plus = np.full((2, 2), 0.5, complex)
    one = np.array([[0, 0], [0, 1]], complex)
    assert 2 * ch.superoperator_apply(plus)[0, 1].real == pytest.approx(np.exp(-t / t2), rel=1e-9)
    assert ch.superoperator_apply(one)[1, 1].real == pytest.approx(np.exp(-t / t1), rel=1e-9)


def test_thermal_relaxation_rejects_unphysical_t2():
    with pytest.raises(ValueError):
        ThermalRelaxation(10e-6, 25e-6, 1e-6)


def test_phase_damping_equivalent_phase_flip():
    lam = 0.36
    pd = PhaseDamping(lam)
    p = (1 - np.sqrt(1 - lam)) / 2
    assert pd.pauli_twirl().probabilities["Z"] == pytest.approx(p)
    rho = np.full((2, 2), 0.5, complex)
    assert np.allclose(pd.superoperator_apply(rho), phase_flip(p).superoperator_apply(rho))


def test_coherent_rotation_twirl_and_unitarity():
    ch = CoherentRotation("Y", 0.4)
    assert ch.is_unitary
    assert ch.pauli_twirl().probabilities["Y"] == pytest.approx(np.sin(0.2) ** 2)
    with pytest.raises(ValueError):
        CoherentRotation("II", 0.1)


def test_depolarizing_conventions():
    ch = depolarizing(0.15, 2)
    assert len(ch.probabilities) == 15
    assert ch.error_probability == pytest.approx(0.15)
    r = 0.01
    assert NoiseModel.depolarizing_from_infidelity(r, 1).error_probability == pytest.approx(1.5 * r)
    assert NoiseModel.depolarizing_from_infidelity(r, 2).error_probability == pytest.approx(1.25 * r)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: pauli_channel(-0.1),
        lambda: pauli_channel(0.6, 0.6),
        lambda: PauliChannel({"Q": 0.1}),
        lambda: PauliChannel({"X": 0.1, "ZZ": 0.1}),
        lambda: depolarizing(1.5),
        lambda: AmplitudeDamping(2),
        lambda: KrausChannel([np.eye(2) * 2]),
        lambda: KrausChannel([]),
        lambda: UnitaryChannel(np.ones((2, 2))),
        lambda: ComposedChannel([bit_flip(0.1), depolarizing(0.1, 2)]),
        lambda: ReadoutError(1.2),
    ],
)
def test_invalid_channels_raise(bad):
    with pytest.raises((ValueError, TypeError)):
        bad()


def test_composed_pauli_channels_combine_exactly():
    c = ComposedChannel([bit_flip(0.1), bit_flip(0.2)]).pauli_twirl()
    assert c.probabilities["X"] == pytest.approx(0.1 * 0.8 + 0.9 * 0.2)


def test_pauli_sampling_statistics():
    ch = pauli_channel(0.1, 0.05, 0.2)
    rng = np.random.default_rng(0)
    hit, x, z = ch.sample_bits(rng, 200_000)
    px = np.mean((x[:, 0] == 1) & (z[:, 0] == 0))
    py = np.mean((x[:, 0] == 1) & (z[:, 0] == 1))
    pz = np.mean((x[:, 0] == 0) & (z[:, 0] == 1))
    assert (px, py, pz) == pytest.approx((0.1, 0.05, 0.2), abs=0.004)
    assert hit.mean() == pytest.approx(0.35, abs=0.004)


def test_readout_error_confusion_matrix():
    e = ReadoutError(0.1, 0.3)
    assert np.allclose(e.confusion_matrix.sum(axis=0), 1)
    assert e.error_probability == pytest.approx(0.2)


# ----------------------------------------------------------------------------- NoiseModel
def _faults(nm, circuit, idx, d=50e-9):
    return nm.gate_faults(idx, circuit[idx], d)


def test_gate_rules_match_by_name_object_qubits_arity_and_predicate():
    c = nq.Circuit(3).h(0).cx(0, 1).x(2).cx(1, 2)
    nm = NoiseModel()
    nm.add_gate_error(bit_flip(0.1), gates="cnot")  # alias
    nm.add_gate_error(phase_flip(0.1), gates=G.H)  # gate object
    nm.add_gate_error(bit_phase_flip(0.1), qubits=[2])  # only gates fully on qubit 2
    nm.add_gate_error(depolarizing(0.1, 2), num_qubits=2, where=lambda op: op.qubits == (1, 2))
    assert [f.channel.name for f in _faults(nm, c, 0)] == ["PhaseFlip"]
    assert [f.qubits for f in _faults(nm, c, 1)] == [(0,), (1,)]
    assert [f.channel.name for f in _faults(nm, c, 2)] == ["BitPhaseFlip"]
    assert {f.channel.name for f in _faults(nm, c, 3)} == {"BitFlip", "Depolarizing"}


def test_channel_arity_mismatch_raises():
    c = nq.Circuit(3).ccx(0, 1, 2)
    nm = NoiseModel().add_gate_error(depolarizing(0.1, 2))
    with pytest.raises(ValueError, match="2-qubit channel"):
        nm.gate_faults(0, c[0], 1e-7)


def test_noise_survives_circuit_edits():
    """Rules are resolved per gate, so inserting gates never shifts noise."""
    nm = NoiseModel().add_gate_error(bit_flip(1.0), gates="CX")
    c = nq.Circuit(2).cx(0, 1)
    c2 = nq.Circuit(2).h(0).h(1).cx(0, 1)
    for circ in (c, c2):
        r = nq.simulate(circ.copy().measure_all(), nm, shots=50, seed=1)
        assert r.fault_rates()[[i for i, x in enumerate(circ) if x.name == "CX"][0]].sum() == pytest.approx(2.0)


def test_thermal_relaxation_on_gates_and_idles_uses_durations():
    nm = NoiseModel(durations=nq.noise.Durations(single_qubit=1e-6, two_qubit=4e-6))
    nm.add_thermal_relaxation(t1=100e-6, t2=100e-6)
    c = nq.Circuit(2).h(0).cx(0, 1)
    f = nm.gate_faults(1, c[1], 4e-6)
    assert len(f) == 2 and all(x.channel.duration == 4e-6 for x in f)
    prog = nq.backends.compile_program(c, nm)
    idle = [s for s in prog.steps if getattr(s, "kind", None) == "idle"]
    assert len(idle) == 1 and idle[0].qubits == (1,) and idle[0].channel.duration == pytest.approx(1e-6)


def test_per_qubit_thermal_parameters():
    nm = NoiseModel().add_thermal_relaxation({0: 10e-6, 1: 20e-6}, {0: 5e-6, 1: 30e-6})
    c = nq.Circuit(2).h(0).h(1)
    assert nm.gate_faults(0, c[0], 1e-6)[0].channel.t1 == 10e-6
    assert nm.gate_faults(1, c[1], 1e-6)[0].channel.t1 == 20e-6


def test_readout_errors_compose_and_reset_errors():
    nm = NoiseModel().add_readout_error(0.1).add_readout_error(0.2, qubits=[1]).add_reset_error(0.05)
    assert nm.readout_error(0).p01 == pytest.approx(0.1)
    assert nm.readout_error(1).p01 == pytest.approx(0.1 + 0.2 - 2 * 0.02)
    assert nm.reset_faults(0)[0].channel.probabilities["X"] == 0.05


def test_twirled_model_is_all_pauli():
    nm = nq.get_hardware("ibm_eagle_r3").noise_model(coherent_fraction=0.5).copy()
    nm.add_quasistatic_detuning(1e5)
    c = nq.algorithms.ghz(4)
    prog = nq.backends.compile_program(c, nm.twirled())
    assert prog.all_pauli and not prog.has_detuning


def test_explicit_error_after_instruction():
    c = nq.Circuit(2).h(0).cx(0, 1)
    nm = NoiseModel().add_error_after(0, PauliChannel({"X": 1.0}))
    assert _faults(nm, c, 0)[0].qubits == (0,)
    assert _faults(nm, c, 1) == []


def test_empty_model_is_noiseless():
    assert NoiseModel().is_empty
    r = nq.simulate(nq.Circuit(1).x(0), NoiseModel(), shots=10)
    assert r.counts == {"1": 10}


def test_uniform_depolarizing_and_repr():
    nm = NoiseModel.uniform_depolarizing(0.001)
    assert "depolarizing" in repr(nm)
    c = nq.Circuit(3).h(0).cx(0, 1).ccx(0, 1, 2)
    assert nm.gate_faults(1, c[1], 1)[0].channel.num_qubits == 2
    assert len(nm.gate_faults(2, c[2], 1)) == 3


# ----------------------------------------------------------------------------- hardware
@pytest.mark.parametrize("name", nq.list_hardware())
def test_hardware_profiles_build_models(name):
    prof = nq.get_hardware(name)
    for rep in ("exact", "pauli"):
        nm = prof.noise_model(representation=rep)
        prog = nq.backends.compile_program(nq.algorithms.ghz(3), nm)
        assert prog.channels()
        if rep == "pauli":
            assert prog.all_pauli
    assert prof.describe()["name"] == name


def test_hardware_gate_error_matches_datasheet_infidelity():
    """Thermal + depolarizing contributions must add up to the quoted 2q infidelity."""
    prof = nq.get_hardware("ibm_eagle_r3")
    nm = prof.noise_model(idle=False, readout=False)
    c = nq.Circuit(2).cx(0, 1)
    faults = nm.gate_faults(0, c[0], prof.durations.two_qubit)
    f_pro = 1.0
    for f in faults:
        if f.channel.num_qubits == 1:
            f_pro *= 1 - f.channel.error_probability
        else:
            f_pro *= 1 - f.channel.error_probability
    r = (1 - f_pro) * 4 / 5
    assert r == pytest.approx(prof.two_qubit_error, rel=0.02)


def test_hardware_registry():
    prof = nq.get_hardware("ionq_aria").with_overrides(name="my_device", t1=10.0)
    nq.noise.register_hardware(prof, overwrite=True)
    assert nq.get_hardware("my_device").t1 == 10.0
    with pytest.raises(ValueError):
        nq.noise.register_hardware(prof)
    with pytest.raises(KeyError):
        nq.get_hardware("nope")
    with pytest.raises(ValueError):
        prof.with_overrides(t2=100.0)
