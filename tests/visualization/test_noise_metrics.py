"""Tests for visualization.noise_metrics helpers."""

import numpy as np
import pytest

from src.noisiq.noise.pauli_error import PauliError
from src.noisiq.noise.correlated_errors import CorrelatedPauliError
from src.noisiq.noise.coherent_errors import CoherentRotation, StochasticCoherentRotation
from src.noisiq.noise.amplitude_damping import AmplitudeDamping
from src.noisiq.noise.t2_dephasing import Dephasing
from src.noisiq.noise.kraus_channels import CombinedChannel
from src.noisiq.visualization.noise_metrics import (
    combine_independent_probabilities,
    channel_event_probability,
    qualitative_burden_label,
    normalize_heat_values,
)


# ---------------------------------------------------------------------------
# combine_independent_probabilities
# ---------------------------------------------------------------------------

def test_combine_single():
    assert combine_independent_probabilities([0.1]) == pytest.approx(0.1)


def test_combine_two_independent():
    result = combine_independent_probabilities([0.01, 0.02])
    expected = 1.0 - (1.0 - 0.01) * (1.0 - 0.02)
    assert result == pytest.approx(expected)


def test_combine_empty():
    assert combine_independent_probabilities([]) == pytest.approx(0.0)


def test_combine_clips_at_one():
    assert combine_independent_probabilities([0.6, 0.6, 0.6]) <= 1.0


# ---------------------------------------------------------------------------
# channel_event_probability — PauliError
# ---------------------------------------------------------------------------

def test_pauli_error_sum():
    ch = PauliError(p_x=0.01, p_y=0.02, p_z=0.03)
    assert channel_event_probability(ch) == pytest.approx(0.06)


def test_pauli_error_zero():
    ch = PauliError(p_x=0.0, p_y=0.0, p_z=0.0)
    assert channel_event_probability(ch) == pytest.approx(0.0)


def test_none_returns_zero():
    assert channel_event_probability(None) == 0.0


# ---------------------------------------------------------------------------
# channel_event_probability — CorrelatedPauliError
# ---------------------------------------------------------------------------

def test_correlated_pauli_error_sum():
    ch = CorrelatedPauliError({"ZZ": 0.005, "XX": 0.002})
    assert channel_event_probability(ch) == pytest.approx(0.007)


def test_correlated_pauli_excludes_all_identity():
    # "II" is all-identity so should contribute 0, but CorrelatedPauliError
    # requires length >= 2 and no all-I strings are normally listed; the helper
    # guards against them defensively.
    ch = CorrelatedPauliError({"ZZ": 0.005})
    assert channel_event_probability(ch) == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# channel_event_probability — CoherentRotation
# ---------------------------------------------------------------------------

def test_coherent_rotation_single_qubit():
    ch = CoherentRotation(axis="Z", epsilon=0.05)
    assert channel_event_probability(ch) == pytest.approx(np.sin(0.05) ** 2)


def test_coherent_rotation_multi_qubit():
    ch = CoherentRotation(axis="ZZ", epsilon=0.02)
    assert channel_event_probability(ch) == pytest.approx(np.sin(0.02) ** 2)


def test_stochastic_coherent_rotation():
    ch = StochasticCoherentRotation(axis="Z", std_dev=0.04)
    assert channel_event_probability(ch) == pytest.approx(np.sin(0.04) ** 2)


# ---------------------------------------------------------------------------
# channel_event_probability — CombinedChannel
# ---------------------------------------------------------------------------

def test_combined_channel_independent():
    ch = CombinedChannel([
        PauliError(p_x=0.01, p_y=0.0, p_z=0.0),
        PauliError(p_x=0.0, p_y=0.02, p_z=0.0),
    ])
    expected = 1.0 - (1.0 - 0.01) * (1.0 - 0.02)
    assert channel_event_probability(ch) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# qualitative_burden_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (0.0,    "None"),
    (5e-4,   "Trace"),
    (5e-3,   "Low"),
    (5e-2,   "Moderate"),
    (0.5,    "High"),
    (1.0,    "High"),
])
def test_qualitative_burden_label(value, expected):
    assert qualitative_burden_label(value) == expected


# ---------------------------------------------------------------------------
# normalize_heat_values
# ---------------------------------------------------------------------------

def test_normalize_all_zeros():
    out = normalize_heat_values([0.0, 0.0, 0.0])
    assert np.all(out == 0.0)


def test_normalize_relative_max_is_one():
    out = normalize_heat_values([0.0, 0.5, 1.0], mode="relative", floor=0.0, gamma=1.0)
    assert out[-1] == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.0)


def test_normalize_relative_zero_stays_zero():
    out = normalize_heat_values([0.0, 0.5], mode="relative", floor=0.0, gamma=1.0)
    assert out[0] == 0.0


def test_normalize_absolute_log_zero_stays_zero():
    out = normalize_heat_values([0.0, 1e-2], mode="absolute_log", vmin=1e-4, vmax=1e-1, floor=0.15, gamma=1.0)
    assert out[0] == 0.0
    assert out[1] > 0.0


def test_normalize_absolute_log_at_vmax():
    out = normalize_heat_values([1e-1], mode="absolute_log", vmin=1e-4, vmax=1e-1, floor=0.15, gamma=1.0)
    assert out[0] == pytest.approx(1.0)


def test_normalize_absolute_log_floor_applied():
    # Any positive value below vmin still gets floor intensity, not zero.
    out = normalize_heat_values([1e-10], mode="absolute_log", vmin=1e-4, vmax=1e-1, floor=0.15, gamma=1.0)
    assert out[0] == pytest.approx(0.15)


def test_normalize_absolute_invalid_bounds():
    with pytest.raises(ValueError, match="vmax > vmin"):
        normalize_heat_values([0.1], mode="absolute", vmin=0.5, vmax=0.1)


def test_normalize_absolute_log_invalid_bounds():
    with pytest.raises(ValueError, match="0 < vmin < vmax"):
        normalize_heat_values([0.1], mode="absolute_log", vmin=0.0, vmax=0.1)


def test_normalize_unknown_mode():
    with pytest.raises(ValueError, match="Unknown heat normalization mode"):
        normalize_heat_values([0.1], mode="bad_mode")
