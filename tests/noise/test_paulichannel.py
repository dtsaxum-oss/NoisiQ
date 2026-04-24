import pytest
import numpy as np
from noisiq.noise import (
    PauliChannel,
    DepolarizingChannel,
    DephaseChannel,
    BitFlipChannel,
    PhaseFlipChannel,
)


def test_channel_validation():
    """Test that all channels reject probabilities outside [0, 1]."""
    for cls in [DepolarizingChannel, DephaseChannel, BitFlipChannel, PhaseFlipChannel]:
        cls(p=0.0)
        cls(p=1.0)
        cls(p=0.5)

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cls(p=-0.1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            cls(p=1.1)


def test_channel_repr():
    """Test __repr__ includes class name and probability."""
    ch = BitFlipChannel(p=0.3)
    assert "BitFlipChannel" in repr(ch)
    assert "0.3" in repr(ch)


def test_depolarizing_channel_to_pauli_error():
    """DepolarizingChannel distributes error equally across X, Y, Z."""
    ch = DepolarizingChannel(p=0.3)
    err = ch.to_pauli_error()
    assert np.isclose(err.p_x, 0.1)
    assert np.isclose(err.p_y, 0.1)
    assert np.isclose(err.p_z, 0.1)


def test_depolarizing_channel_describe():
    ch = DepolarizingChannel(p=0.3)
    d = ch.describe()
    assert d["channel"] == "depolarizing"
    assert d["p"] == 0.3
    assert np.isclose(d["p_x"], 0.1)
    assert np.isclose(d["p_y"], 0.1)
    assert np.isclose(d["p_z"], 0.1)


def test_dephase_channel_to_pauli_error():
    """DephaseChannel produces Z-only errors."""
    ch = DephaseChannel(p=0.4)
    err = ch.to_pauli_error()
    assert np.isclose(err.p_z, 0.4)
    assert err.p_x == 0.0
    assert err.p_y == 0.0


def test_dephase_channel_describe():
    ch = DephaseChannel(p=0.4)
    d = ch.describe()
    assert d["channel"] == "dephasing"
    assert d["p"] == 0.4
    assert d["p_x"] == 0.0
    assert d["p_y"] == 0.0
    assert np.isclose(d["p_z"], 0.4)


def test_bitflip_channel_to_pauli_error():
    """BitFlipChannel produces X-only errors."""
    ch = BitFlipChannel(p=0.2)
    err = ch.to_pauli_error()
    assert np.isclose(err.p_x, 0.2)
    assert err.p_y == 0.0
    assert err.p_z == 0.0


def test_bitflip_channel_describe():
    ch = BitFlipChannel(p=0.2)
    d = ch.describe()
    assert d["channel"] == "bit_flip"
    assert d["p"] == 0.2
    assert np.isclose(d["p_x"], 0.2)
    assert d["p_y"] == 0.0
    assert d["p_z"] == 0.0


def test_phaseflip_channel_to_pauli_error():
    """PhaseFlipChannel produces Z-only errors, identical to DephaseChannel."""
    ch = PhaseFlipChannel(p=0.15)
    err = ch.to_pauli_error()
    assert np.isclose(err.p_z, 0.15)
    assert err.p_x == 0.0
    assert err.p_y == 0.0


def test_phaseflip_channel_describe():
    ch = PhaseFlipChannel(p=0.15)
    d = ch.describe()
    assert d["channel"] == "phase_flip"
    assert d["p"] == 0.15
    assert d["p_x"] == 0.0
    assert d["p_y"] == 0.0
    assert np.isclose(d["p_z"], 0.15)


def test_phaseflip_and_dephase_produce_same_pauli_error():
    """PhaseFlipChannel and DephaseChannel are physically identical."""
    p = 0.25
    pf_err = PhaseFlipChannel(p=p).to_pauli_error()
    dp_err = DephaseChannel(p=p).to_pauli_error()
    assert np.isclose(pf_err.p_x, dp_err.p_x)
    assert np.isclose(pf_err.p_y, dp_err.p_y)
    assert np.isclose(pf_err.p_z, dp_err.p_z)


def test_channel_sampling_via_pauli_error():
    """Channels delegate sampling correctly through to_pauli_error."""
    ch = BitFlipChannel(p=0.2)
    err = ch.to_pauli_error()
    rng = np.random.default_rng(42)

    samples = [err.sample(rng) for _ in range(1000)]
    x_count = samples.count('X')
    i_count = samples.count('I')

    assert 150 < x_count < 250
    assert 750 < i_count < 850
    assert 'Y' not in samples
    assert 'Z' not in samples


def test_pauli_channel_is_abstract():
    """PauliChannel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PauliChannel(p=0.1)
