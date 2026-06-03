"""Tests for CoherentRotation and StochasticCoherentRotation channels."""

import numpy as np
import pytest

from noisiq.noise.coherent_errors import (
    CoherentRotation,
    StochasticCoherentRotation,
    _pauli_string_matrix,
)
from noisiq.noise.pauli_error import PauliError
from noisiq.noise.correlated_errors import CorrelatedPauliError


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

def test_invalid_axis_empty():
    with pytest.raises(ValueError, match="non-empty"):
        CoherentRotation(axis='', epsilon=0.1)


def test_invalid_axis_char():
    with pytest.raises(ValueError, match="I/X/Y/Z"):
        CoherentRotation(axis='A', epsilon=0.1)


def test_valid_single_qubit_axes():
    for ax in ('I', 'X', 'Y', 'Z'):
        ch = CoherentRotation(axis=ax, epsilon=0.05)
        assert ch.num_qubits == 1


def test_valid_two_qubit_axes():
    for ax in ('ZZ', 'IX', 'YI', 'XY'):
        ch = CoherentRotation(axis=ax, epsilon=0.05)
        assert ch.num_qubits == 2


def test_three_qubit_axis():
    ch = CoherentRotation(axis='IZZ', epsilon=0.01)
    assert ch.num_qubits == 3


def test_describe_keys():
    ch = CoherentRotation(axis='Z', epsilon=0.1)
    d = ch.describe()
    assert d['type'] == 'CoherentRotation'
    assert d['axis'] == 'Z'
    assert abs(d['epsilon_rad'] - 0.1) < 1e-12
    assert d['num_qubits'] == 1


# ---------------------------------------------------------------------------
# Single Kraus operator is unitary (validated by KrausChannel)
# ---------------------------------------------------------------------------

def test_single_kraus_operator():
    ch = CoherentRotation(axis='X', epsilon=0.3)
    assert len(ch.operators) == 1


def test_kraus_operator_is_unitary():
    ch = CoherentRotation(axis='ZZ', epsilon=0.2)
    U = ch.operators[0]
    UdU = U.conj().T @ U
    np.testing.assert_allclose(UdU, np.eye(4), atol=1e-12)


def test_zero_epsilon_is_identity():
    ch = CoherentRotation(axis='X', epsilon=0.0)
    U = ch.operators[0]
    np.testing.assert_allclose(U, np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# Known rotation angles
# ---------------------------------------------------------------------------

def test_x_rotation_half_pi_flips_state():
    """exp(-i*(π/2)*X) = -iX. Applied to |0⟩ gives -i|1⟩ = [0, -i].

    Note: exp(-i*π*X) = -I (global phase flip, NOT -iX). The half-angle
    formula is cos(ε)I - i*sin(ε)X; for ε=π/2 this gives -iX.
    """
    ch = CoherentRotation(axis='X', epsilon=np.pi / 2)
    U = ch.operators[0]
    state_0 = np.array([1.0, 0.0], dtype=complex)
    result = U @ state_0
    # -iX|0⟩ = -i|1⟩ = [0, -i]
    expected = np.array([0.0, -1j])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_z_rotation_half_pi():
    """exp(-i*(π/2)*Z)|+⟩ = -i|−⟩ (up to global phase)."""
    ch = CoherentRotation(axis='Z', epsilon=np.pi / 2)
    U = ch.operators[0]
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    result = U @ plus
    # -iZ|+⟩ = -i(|0⟩−|1⟩)/√2 = -i|−⟩
    minus = np.array([1.0, -1.0]) / np.sqrt(2)
    # Compare density matrices (global-phase invariant)
    rho_result = np.outer(result, result.conj())
    rho_expected = np.outer(minus, minus.conj())
    np.testing.assert_allclose(rho_result, rho_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# to_pauli_error — single qubit
# ---------------------------------------------------------------------------

def test_to_pauli_error_x_axis():
    epsilon = 0.1
    ch = CoherentRotation(axis='X', epsilon=epsilon)
    pe = ch.to_pauli_error()
    assert isinstance(pe, PauliError)
    expected_p = np.sin(epsilon) ** 2
    assert abs(pe.p_x - expected_p) < 1e-12
    assert pe.p_y == 0.0
    assert pe.p_z == 0.0


def test_to_pauli_error_y_axis():
    ch = CoherentRotation(axis='Y', epsilon=0.2)
    pe = ch.to_pauli_error()
    assert pe.p_x == 0.0
    assert abs(pe.p_y - np.sin(0.2) ** 2) < 1e-12
    assert pe.p_z == 0.0


def test_to_pauli_error_z_axis():
    ch = CoherentRotation(axis='Z', epsilon=0.05)
    pe = ch.to_pauli_error()
    assert pe.p_x == 0.0
    assert pe.p_y == 0.0
    assert abs(pe.p_z - np.sin(0.05) ** 2) < 1e-12


def test_to_pauli_error_identity_axis():
    ch = CoherentRotation(axis='I', epsilon=0.5)
    pe = ch.to_pauli_error()
    assert isinstance(pe, PauliError)
    assert pe.p_x == 0.0
    assert pe.p_y == 0.0
    assert pe.p_z == 0.0


# ---------------------------------------------------------------------------
# to_pauli_error — multi-qubit
# ---------------------------------------------------------------------------

def test_to_pauli_error_zz_axis():
    ch = CoherentRotation(axis='ZZ', epsilon=0.1)
    corr = ch.to_pauli_error()
    assert isinstance(corr, CorrelatedPauliError)
    assert 'ZZ' in corr.probs
    assert abs(corr.probs['ZZ'] - np.sin(0.1) ** 2) < 1e-12


def test_to_pauli_error_ix_axis():
    ch = CoherentRotation(axis='IX', epsilon=0.3)
    corr = ch.to_pauli_error()
    assert isinstance(corr, CorrelatedPauliError)
    assert 'IX' in corr.probs


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_axis():
    ch = CoherentRotation(axis='ZZ', epsilon=0.1)
    assert 'ZZ' in repr(ch)
    assert 'CoherentRotation' in repr(ch)


# ---------------------------------------------------------------------------
# _pauli_string_matrix helper
# ---------------------------------------------------------------------------

def test_pauli_string_matrix_z():
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    np.testing.assert_allclose(_pauli_string_matrix('Z'), Z, atol=1e-12)


def test_pauli_string_matrix_zz():
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    ZZ = np.kron(Z, Z)
    np.testing.assert_allclose(_pauli_string_matrix('ZZ'), ZZ, atol=1e-12)


def test_pauli_string_matrix_ix():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    IX = np.kron(I, X)
    np.testing.assert_allclose(_pauli_string_matrix('IX'), IX, atol=1e-12)


# ===========================================================================
# StochasticCoherentRotation
# ===========================================================================

class TestStochasticCoherentRotation:
    # -----------------------------------------------------------------------
    # Construction and validation
    # -----------------------------------------------------------------------

    def test_invalid_axis_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            StochasticCoherentRotation(axis='', std_dev=0.1)

    def test_invalid_axis_char(self):
        with pytest.raises(ValueError, match="I/X/Y/Z"):
            StochasticCoherentRotation(axis='Q', std_dev=0.1)

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="std_dev"):
            StochasticCoherentRotation(axis='Z', std_dev=-0.1)

    def test_zero_std_dev_allowed(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.0)
        assert ch.std_dev == 0.0

    def test_num_qubits_single(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.05)
        assert ch.num_qubits == 1

    def test_num_qubits_multi(self):
        ch = StochasticCoherentRotation(axis='ZZ', std_dev=0.02)
        assert ch.num_qubits == 2

    # -----------------------------------------------------------------------
    # sample() contracts
    # -----------------------------------------------------------------------

    def test_sample_returns_unitary(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.3)
        rng = np.random.default_rng(0)
        U = ch.sample(rng)
        assert U.shape == (2, 2)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-12)

    def test_sample_multi_qubit_returns_unitary(self):
        ch = StochasticCoherentRotation(axis='ZZ', std_dev=0.1)
        rng = np.random.default_rng(1)
        U = ch.sample(rng)
        assert U.shape == (4, 4)
        np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=1e-12)

    def test_consecutive_samples_differ(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.5)
        rng = np.random.default_rng(42)
        U1 = ch.sample(rng)
        U2 = ch.sample(rng)
        assert not np.allclose(U1, U2), "consecutive samples should differ"

    def test_zero_std_dev_returns_identity(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.0)
        rng = np.random.default_rng(7)
        for _ in range(5):
            U = ch.sample(rng)
            np.testing.assert_allclose(U, np.eye(2), atol=1e-12)

    # -----------------------------------------------------------------------
    # to_pauli_error
    # -----------------------------------------------------------------------

    def test_to_pauli_error_z_axis(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.07)
        pe = ch.to_pauli_error()
        assert isinstance(pe, PauliError)
        expected_p = np.sin(0.07) ** 2
        assert abs(pe.p_z - expected_p) < 1e-12
        assert pe.p_x == 0.0 and pe.p_y == 0.0

    def test_to_pauli_error_x_axis(self):
        ch = StochasticCoherentRotation(axis='X', std_dev=0.1)
        pe = ch.to_pauli_error()
        assert isinstance(pe, PauliError)
        assert abs(pe.p_x - np.sin(0.1) ** 2) < 1e-12
        assert pe.p_y == 0.0 and pe.p_z == 0.0

    def test_to_pauli_error_zz_returns_correlated(self):
        ch = StochasticCoherentRotation(axis='ZZ', std_dev=0.05)
        corr = ch.to_pauli_error()
        assert isinstance(corr, CorrelatedPauliError)
        assert 'ZZ' in corr.probs
        assert abs(corr.probs['ZZ'] - np.sin(0.05) ** 2) < 1e-12

    # -----------------------------------------------------------------------
    # describe and repr
    # -----------------------------------------------------------------------

    def test_describe_keys(self):
        ch = StochasticCoherentRotation(axis='Z', std_dev=0.07)
        d = ch.describe()
        assert d['type'] == 'StochasticCoherentRotation'
        assert d['axis'] == 'Z'
        assert abs(d['std_dev_rad'] - 0.07) < 1e-12
        assert d['num_qubits'] == 1

    def test_repr_contains_class_name_and_axis(self):
        ch = StochasticCoherentRotation(axis='ZZ', std_dev=0.1)
        r = repr(ch)
        assert 'StochasticCoherentRotation' in r
        assert 'ZZ' in r

    # -----------------------------------------------------------------------
    # Behavioral: purity effect in TrajectoryBackend
    # -----------------------------------------------------------------------

    def test_causes_purity_loss_unlike_coherent_rotation(self):
        """StochasticCoherentRotation should decohere; CoherentRotation should not."""
        from noisiq.ir import Circuit
        from noisiq.backends.trajectory_backend import TrajectoryBackend

        # H|0> = |+>: Z-rotations affect the relative phase → purity drops for stochastic
        c = Circuit(n_qubits=1)
        c.h(0, t=0)

        std_dev = 0.8  # large so the purity gap is unambiguous
        n_shots = 1500

        noise_det = {0: CoherentRotation(axis='Z', epsilon=std_dev)}
        rho_det = TrajectoryBackend().run(c, noise_model=noise_det, n_shots=n_shots, seed=0).final_state
        purity_det = float(np.real(np.trace(rho_det @ rho_det)))

        noise_sto = {0: StochasticCoherentRotation(axis='Z', std_dev=std_dev)}
        rho_sto = TrajectoryBackend().run(c, noise_model=noise_sto, n_shots=n_shots, seed=0).final_state
        purity_sto = float(np.real(np.trace(rho_sto @ rho_sto)))

        assert abs(purity_det - 1.0) < 1e-6, f"CoherentRotation should stay pure: {purity_det}"
        assert purity_sto < 0.90, f"StochasticCoherentRotation should decohere: {purity_sto}"
