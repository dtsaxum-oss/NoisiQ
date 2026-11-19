import numpy as np
import pytest

from noisiq import metrics as M
from tests.conftest import random_unitary


def test_state_fidelity_pure_and_mixed(rng):
    a = random_unitary(4, rng)[:, 0]
    b = random_unitary(4, rng)[:, 0]
    assert M.state_fidelity(a, a) == pytest.approx(1)
    assert M.state_fidelity(a, b) == pytest.approx(abs(np.vdot(a, b)) ** 2)
    rho = np.outer(b, b.conj())
    assert M.state_fidelity(a, rho) == pytest.approx(M.state_fidelity(rho, a))
    mixed = np.eye(4) / 4
    assert M.state_fidelity(mixed, rho) == pytest.approx(0.25)
    assert M.state_fidelity(mixed, mixed) == pytest.approx(1)


def test_purity_trace_distance_bloch():
    plus = np.array([1, 1]) / np.sqrt(2)
    assert M.purity(plus) == pytest.approx(1)
    assert M.purity(np.eye(2) / 2) == pytest.approx(0.5)
    assert M.trace_distance(plus, np.array([1, -1]) / np.sqrt(2)) == pytest.approx(1)
    assert M.bloch_vector(plus) == pytest.approx((1, 0, 0))
    assert M.bloch_vector(np.array([1, 1j]) / np.sqrt(2)) == pytest.approx((0, 1, 0))
    with pytest.raises(ValueError):
        M.bloch_vector(np.eye(4))
    with pytest.raises(ValueError):
        M.to_density_matrix(np.ones((2, 3)))


def test_reduced_density_matrix_vector_and_matrix_agree(rng):
    psi = random_unitary(8, rng)[:, 0]
    rho = np.outer(psi, psi.conj())
    for keep in ([0], [2], [0, 2], [1, 0]):
        a = M.reduced_density_matrix(psi, keep)
        b = M.reduced_density_matrix(rho, keep)
        assert np.allclose(a, b)
        assert np.trace(a) == pytest.approx(1)


def test_bell_reduced_state_is_maximally_mixed():
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(M.reduced_density_matrix(bell, [1]), np.eye(2) / 2)


def test_distribution_distances():
    p = {"00": 50, "11": 50}
    assert M.hellinger_fidelity(p, p) == pytest.approx(1)
    assert M.total_variation_distance(p, {"01": 1}) == pytest.approx(1)
    assert M.total_variation_distance(p, {"00": 1}) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        M.hellinger_fidelity({}, p)


def test_ghz_fidelity_and_stderr():
    ghz = np.zeros(8)
    ghz[[0, 7]] = 1 / np.sqrt(2)
    assert M.ghz_fidelity(ghz) == pytest.approx(1)
    assert M.ghz_fidelity(np.eye(8) / 8) == pytest.approx(1 / 8)
    assert M.binomial_stderr(0.5, 100) == pytest.approx(0.05)
    assert M.binomial_stderr(0.5, 0) == 0
