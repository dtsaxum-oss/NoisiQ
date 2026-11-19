import math

from noisiq.protocols.mek_magic_state_distillation import (
    mek_acceptance_probability,
    mek_undetected_marginal_probability,
    mek_output_error_probability,
    mek_threshold,
)


def test_mek_formula_boundary_values():
    assert math.isclose(mek_acceptance_probability(0.0), 1.0)
    assert math.isclose(mek_undetected_marginal_probability(0.0), 0.0)
    assert math.isclose(mek_output_error_probability(0.0), 0.0)


def test_mek_formula_known_values():
    p = 0.01
    assert math.isclose(mek_acceptance_probability(p), 0.9056119460774467, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(mek_undetected_marginal_probability(p), 0.0008455746387232003, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(mek_output_error_probability(p), 0.0009337052612719046, rel_tol=0, abs_tol=1e-15)


def test_mek_small_p_scaling():
    p = 1e-5
    a = mek_acceptance_probability(p)
    e = mek_output_error_probability(p)
    assert abs(a - (1 - 10 * p)) < 1e-8
    assert abs(e / (p * p) - 9.0) < 1e-2


def test_mek_threshold_reference():
    assert 0.088 < mek_threshold() < 0.090
