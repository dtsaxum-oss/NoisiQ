"""Tests for CorrelatedPauliError channel."""

import numpy as np
import pytest

from noisiq.noise.correlated_errors import CorrelatedPauliError


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

def test_empty_probs_raises():
    with pytest.raises(ValueError, match="empty"):
        CorrelatedPauliError({})


def test_single_qubit_string_raises():
    with pytest.raises(ValueError, match="length ≥ 2"):
        CorrelatedPauliError({'Z': 0.01})


def test_inconsistent_string_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        CorrelatedPauliError({'ZZ': 0.01, 'IXY': 0.02})


def test_invalid_char_in_string_raises():
    with pytest.raises(ValueError, match="IXYZ"):
        CorrelatedPauliError({'ZA': 0.01})


def test_negative_probability_raises():
    with pytest.raises(ValueError, match="non-negative"):
        CorrelatedPauliError({'ZZ': -0.01})


def test_total_probability_exceeds_one_raises():
    with pytest.raises(ValueError, match="≤ 1"):
        CorrelatedPauliError({'ZZ': 0.6, 'XX': 0.5})


def test_valid_single_term():
    err = CorrelatedPauliError({'ZZ': 0.005})
    assert err.num_qubits == 2
    assert abs(err.probs['ZZ'] - 0.005) < 1e-12


def test_valid_multi_term():
    err = CorrelatedPauliError({'IX': 0.001, 'ZI': 0.002, 'ZZ': 0.0005})
    assert err.num_qubits == 2
    assert len(err.probs) == 3


def test_three_qubit_channel():
    err = CorrelatedPauliError({'IZZ': 0.003, 'ZIZ': 0.002})
    assert err.num_qubits == 3


def test_probability_sum_exactly_one_is_valid():
    err = CorrelatedPauliError({'ZZ': 1.0})
    assert err.num_qubits == 2


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------

def test_sample_returns_correct_length():
    rng = np.random.default_rng(0)
    err = CorrelatedPauliError({'ZZ': 0.3})
    s = err.sample(rng)
    assert len(s) == 2


def test_sample_only_returns_valid_strings():
    rng = np.random.default_rng(42)
    err = CorrelatedPauliError({'ZZ': 0.3, 'IX': 0.2})
    valid = {'ZZ', 'IX', 'II'}
    for _ in range(200):
        s = err.sample(rng)
        assert s in valid


def test_sample_deterministic_at_prob_one():
    rng = np.random.default_rng(0)
    err = CorrelatedPauliError({'ZZ': 1.0})
    for _ in range(50):
        assert err.sample(rng) == 'ZZ'


def test_sample_identity_dominant_at_small_prob():
    rng = np.random.default_rng(99)
    err = CorrelatedPauliError({'ZZ': 0.001})
    outcomes = [err.sample(rng) for _ in range(5000)]
    identity_count = outcomes.count('II')
    assert identity_count > 4900, f"Expected mostly 'II', got {identity_count}/5000"


def test_sample_frequency_matches_probability():
    rng = np.random.default_rng(7)
    p_zz = 0.3
    err = CorrelatedPauliError({'ZZ': p_zz})
    n = 10_000
    zz_count = sum(1 for _ in range(n) if err.sample(rng) == 'ZZ')
    # Should be within 3 std dev: std = sqrt(n*p*(1-p)) ≈ 46
    assert abs(zz_count / n - p_zz) < 0.03


# ---------------------------------------------------------------------------
# to_stim_correlated_error_directives()
# ---------------------------------------------------------------------------

def test_directives_basic():
    err = CorrelatedPauliError({'ZZ': 0.005})
    directives = err.to_stim_correlated_error_directives([3, 5])
    assert len(directives) == 1
    name, prob, targets = directives[0]
    assert name == 'CORRELATED_ERROR'
    assert abs(prob - 0.005) < 1e-12
    assert set(targets) == {('Z', 3), ('Z', 5)}


def test_directives_skip_identity_chars():
    err = CorrelatedPauliError({'IZ': 0.01})
    directives = err.to_stim_correlated_error_directives([0, 1])
    assert len(directives) == 1
    _, _, targets = directives[0]
    # Only qubit 1 (Z), qubit 0 (I) is skipped
    assert targets == [('Z', 1)]


def test_directives_zero_prob_omitted():
    # Zero-prob entries are added after construction via manual dict modification
    err = CorrelatedPauliError({'ZZ': 0.01})
    err.probs['XX'] = 0.0
    directives = err.to_stim_correlated_error_directives([0, 1])
    names = [d[0] for d in directives]
    # Only ZZ has non-zero probability
    assert len(directives) == 1


def test_directives_multiple_terms():
    err = CorrelatedPauliError({'ZZ': 0.005, 'XX': 0.003})
    directives = err.to_stim_correlated_error_directives([0, 2])
    assert len(directives) == 2
    probs = {d[0]: d[1] for d in directives}  # (name, prob) keyed by name... actually by order
    prob_values = sorted(d[1] for d in directives)
    assert abs(prob_values[0] - 0.003) < 1e-12
    assert abs(prob_values[1] - 0.005) < 1e-12


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_prob():
    err = CorrelatedPauliError({'ZZ': 0.005})
    r = repr(err)
    assert 'CorrelatedPauliError' in r
    assert 'ZZ' in r
