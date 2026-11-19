"""
Tests for GHZ density-matrix metrics in noisiq.results.ghz_metrics.

Covers:
- ideal GHZ state produces the expected values for all four metric functions
- classical GHZ mixture (no coherence) is correctly distinguished from a true GHZ state
- minus-phase GHZ: fidelity=0 vs plus target, but |coherence| still equals 0.5
- shape mismatch raises ValueError in all four functions
- MetricReport kinds are physically correct (not labeled as DENSITY_MATRIX_STATE_FIDELITY
  for population / coherence metrics)
"""

import pytest
import numpy as np

from noisiq.results.ghz_metrics import (
    ghz_state_fidelity,
    ghz_subspace_population,
    ghz_branch_coherence_real,
    ghz_branch_coherence_abs,
)
from noisiq.results.metrics import MetricKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ghz_plus(n: int) -> np.ndarray:
    """Density matrix of the n-qubit GHZ+ state (|0...0⟩ + |1...1⟩)/√2."""
    d = 1 << n
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 0.5
    rho[0, -1] = 0.5
    rho[-1, 0] = 0.5
    rho[-1, -1] = 0.5
    return rho


def _ghz_minus(n: int) -> np.ndarray:
    """Density matrix of the n-qubit GHZ- state (|0...0⟩ - |1...1⟩)/√2."""
    d = 1 << n
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 0.5
    rho[0, -1] = -0.5
    rho[-1, 0] = -0.5
    rho[-1, -1] = 0.5
    return rho


def _classical_ghz_mixture(n: int) -> np.ndarray:
    """Classical mixture 0.5|0...0⟩⟨0...0| + 0.5|1...1⟩⟨1...1|."""
    d = 1 << n
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 0.5
    rho[-1, -1] = 0.5
    return rho


# ---------------------------------------------------------------------------
# Ideal GHZ+ state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3])
def test_ghz_state_fidelity_ideal(n):
    rho = _ghz_plus(n)
    report = ghz_state_fidelity(rho, n)
    assert abs(report.value - 1.0) < 1e-10
    assert report.kind == MetricKind.DENSITY_MATRIX_STATE_FIDELITY


@pytest.mark.parametrize("n", [2, 3])
def test_ghz_subspace_population_ideal(n):
    rho = _ghz_plus(n)
    report = ghz_subspace_population(rho, n)
    assert abs(report.value - 1.0) < 1e-10
    assert report.kind == MetricKind.GHZ_SUBSPACE_POPULATION


@pytest.mark.parametrize("n", [2, 3])
def test_ghz_branch_coherence_real_ideal(n):
    rho = _ghz_plus(n)
    report = ghz_branch_coherence_real(rho, n)
    assert abs(report.value - 0.5) < 1e-10
    assert report.kind == MetricKind.GHZ_BRANCH_COHERENCE_REAL


@pytest.mark.parametrize("n", [2, 3])
def test_ghz_branch_coherence_abs_ideal(n):
    rho = _ghz_plus(n)
    report = ghz_branch_coherence_abs(rho, n)
    assert abs(report.value - 0.5) < 1e-10
    assert report.kind == MetricKind.GHZ_BRANCH_COHERENCE_ABS


# ---------------------------------------------------------------------------
# Classical GHZ mixture — no coherence, full subspace population
# ---------------------------------------------------------------------------

def test_classical_ghz_fidelity_is_half():
    rho = _classical_ghz_mixture(n=2)
    report = ghz_state_fidelity(rho, n=2)
    assert abs(report.value - 0.5) < 1e-10


def test_classical_ghz_subspace_population_is_one():
    rho = _classical_ghz_mixture(n=2)
    report = ghz_subspace_population(rho, n=2)
    assert abs(report.value - 1.0) < 1e-10


def test_classical_ghz_branch_coherence_real_is_zero():
    rho = _classical_ghz_mixture(n=2)
    report = ghz_branch_coherence_real(rho, n=2)
    assert abs(report.value) < 1e-10


# ---------------------------------------------------------------------------
# Minus-phase GHZ — distinguishable by sign of real coherence
# ---------------------------------------------------------------------------

def test_ghz_minus_fidelity_against_plus_target_is_zero():
    rho = _ghz_minus(n=2)
    report = ghz_state_fidelity(rho, n=2)
    assert abs(report.value) < 1e-10


def test_ghz_minus_coherence_abs_is_half():
    rho = _ghz_minus(n=2)
    report = ghz_branch_coherence_abs(rho, n=2)
    assert abs(report.value - 0.5) < 1e-10


def test_ghz_minus_coherence_real_is_minus_half():
    rho = _ghz_minus(n=2)
    report = ghz_branch_coherence_real(rho, n=2)
    assert abs(report.value - (-0.5)) < 1e-10


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [
    ghz_state_fidelity,
    ghz_subspace_population,
    ghz_branch_coherence_real,
    ghz_branch_coherence_abs,
])
def test_shape_mismatch_raises_value_error(func):
    # 8×8 matrix passed with n=2 expects 4×4
    rho = np.eye(8, dtype=complex) / 8
    with pytest.raises(ValueError, match="shape"):
        func(rho, n=2)


@pytest.mark.parametrize("func", [
    ghz_state_fidelity,
    ghz_subspace_population,
    ghz_branch_coherence_real,
    ghz_branch_coherence_abs,
])
def test_correct_shape_does_not_raise(func):
    rho = _ghz_plus(n=2)
    report = func(rho, n=2)
    assert report is not None
