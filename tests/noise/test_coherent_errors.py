"""Tests for CoherentRotation channel."""

import numpy as np
import pytest

from noisiq.noise.coherent_errors import CoherentRotation, _pauli_string_matrix
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
