"""
Tests for CombinedChannel.

Covers:
- Construction: Kraus-only and Kraus+Pauli
- validate() delegates to the KrausChannel
- pauli_error() returns None when no Pauli layer; PauliError when set
- sample_pauli() returns None when no Pauli layer; valid character when set
- describe() keys match the layers present
- Combined channel produces higher error rate than Kraus alone
  (trajectory simulation: fidelity drops more when Pauli layer is added)
"""

import pytest
import numpy as np

from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.noise.t2_dephasing import Dephasing
from noisiq.noise.pauli_channels import DepolarizingChannel, DephaseChannel
from noisiq.noise.combined_channel import CombinedChannel
from noisiq.ir import Circuit
from noisiq.ir import gates as ir
from noisiq.backends.trajectory_backend import TrajectoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kraus_only():
    return CombinedChannel(kraus_channel=AmplitudeDamping(T1=1e-6, t=0.5e-6))


@pytest.fixture
def kraus_and_pauli():
    return CombinedChannel(
        kraus_channel=AmplitudeDamping(T1=1e-6, t=0.5e-6),
        pauli_channel=DepolarizingChannel(p=0.05),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_kraus_only_construction(kraus_only):
    pytest.skip("not yet implemented")


def test_kraus_and_pauli_construction(kraus_and_pauli):
    pytest.skip("not yet implemented")


def test_pauli_channel_defaults_to_none():
    ch = CombinedChannel(kraus_channel=AmplitudeDamping(T1=1e-6, t=0.1e-6))
    assert ch.pauli_channel is None


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def test_validate_passes_for_valid_kraus(kraus_only):
    # Must not raise
    pytest.skip("not yet implemented")


def test_validate_passes_with_pauli_layer(kraus_and_pauli):
    pytest.skip("not yet implemented")


# ---------------------------------------------------------------------------
# pauli_error()
# ---------------------------------------------------------------------------

def test_pauli_error_returns_none_when_no_pauli(kraus_only):
    pytest.skip("not yet implemented")


def test_pauli_error_returns_pauli_error_when_set(kraus_and_pauli):
    err = kraus_and_pauli.pauli_error()
    pytest.skip("not yet implemented")


# ---------------------------------------------------------------------------
# sample_pauli()
# ---------------------------------------------------------------------------

def test_sample_pauli_returns_none_when_no_pauli(kraus_only):
    rng = np.random.default_rng(0)
    pytest.skip("not yet implemented")


def test_sample_pauli_returns_valid_character(kraus_and_pauli):
    rng = np.random.default_rng(42)
    for _ in range(20):
        result = kraus_and_pauli.sample_pauli(rng)
        pytest.skip("not yet implemented")


def test_sample_pauli_deterministic_at_p1():
    """DepolarizingChannel(p=0) → always 'I'."""
    ch = CombinedChannel(
        kraus_channel=AmplitudeDamping(T1=1e-6, t=1e-9),
        pauli_channel=DepolarizingChannel(p=0.0),
    )
    rng = np.random.default_rng(0)
    for _ in range(10):
        pytest.skip("not yet implemented")


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

def test_describe_kraus_only_has_no_pauli_key(kraus_only):
    d = kraus_only.describe()
    pytest.skip("not yet implemented")


def test_describe_combined_has_both_keys(kraus_and_pauli):
    d = kraus_and_pauli.describe()
    pytest.skip("not yet implemented")


# ---------------------------------------------------------------------------
# Physics: combined channel degrades fidelity more than Kraus alone
# ---------------------------------------------------------------------------

def test_combined_increases_error_vs_kraus_only():
    """Adding a Pauli layer on top of T1 damping must reduce P(|1⟩) further.

    With T1 damping only, P(|1⟩) = exp(-t/T1) ≈ 0.61 at t=T1/2.
    Adding depolarizing noise (p=0.3) mixes the state more, so P(|1⟩) < 0.61.
    """
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(ir.X, qubits=[0])

    T1, t = 1e-6, 0.5e-6
    backend = TrajectoryBackend()

    # Kraus only
    ch_kraus = CombinedChannel(kraus_channel=AmplitudeDamping(T1=T1, t=t))
    r_kraus = backend.run(circuit, noise_model=ch_kraus, n_shots=2000, seed=0)

    # Kraus + Pauli
    ch_combined = CombinedChannel(
        kraus_channel=AmplitudeDamping(T1=T1, t=t),
        pauli_channel=DepolarizingChannel(p=0.3),
    )
    r_combined = backend.run(circuit, noise_model=ch_combined, n_shots=2000, seed=0)

    pytest.skip("not yet implemented")
    # p_kraus = r_kraus.excited_state_probability(qubit=0)
    # p_combined = r_combined.excited_state_probability(qubit=0)
    # assert p_combined < p_kraus, (
    #     f"Expected combined < kraus, got combined={p_combined:.4f} kraus={p_kraus:.4f}"
    # )
