"""
Tests for CombinedChannel (noisiq.noise.kraus_channels).

Covers:
- Construction: single channel, multiple channels, empty raises
- .channels stores items in order
- .describe() structure
- __repr__ string
- Backend integration: sequential application via _dispatch_channel
"""

import pytest
import numpy as np

from noisiq.noise.kraus_channels import CombinedChannel
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.noise.t2_dephasing import Dephasing
from noisiq.noise.pauli_error import PauliError, depolarizing_error
from noisiq.noise.coherent_errors import CoherentRotation
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.ir import Circuit
from noisiq.ir import gates as ir
from noisiq.backends.trajectory_backend import TrajectoryBackend


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_single_channel_construction():
    ch = CombinedChannel([AmplitudeDamping(T1=1e-6, t=0.5e-6)])
    assert len(ch.channels) == 1


def test_multiple_channels_construction():
    ch = CombinedChannel([
        AmplitudeDamping(T1=1e-6, t=0.5e-6),
        Dephasing(T2=2e-6, t=0.5e-6),
    ])
    assert len(ch.channels) == 2


def test_empty_channels_raises():
    with pytest.raises(ValueError, match="at least one"):
        CombinedChannel([])


def test_channels_stored_in_order():
    ad = AmplitudeDamping(T1=1e-6, t=0.5e-6)
    dep = depolarizing_error(p=0.01)
    ch = CombinedChannel([ad, dep])
    assert ch.channels[0] is ad
    assert ch.channels[1] is dep


def test_heterogeneous_channels_accepted():
    """CombinedChannel accepts mixed types without raising."""
    ch = CombinedChannel([
        AmplitudeDamping(T1=1e-6, t=0.5e-6),
        depolarizing_error(p=0.01),
        CoherentRotation(axis='Z', epsilon=0.05),
        CorrelatedPauliError({'ZZ': 0.01}),
    ])
    assert len(ch.channels) == 4


# ---------------------------------------------------------------------------
# describe() / repr
# ---------------------------------------------------------------------------

def test_describe_type_key():
    ch = CombinedChannel([AmplitudeDamping(T1=1e-6, t=0.5e-6)])
    d = ch.describe()
    assert d["type"] == "CombinedChannel"


def test_describe_n_channels():
    channels = [AmplitudeDamping(T1=1e-6, t=0.5e-6), Dephasing(T2=2e-6, t=0.5e-6)]
    ch = CombinedChannel(channels)
    assert ch.describe()["n_channels"] == 2


def test_describe_channels_list_length():
    channels = [AmplitudeDamping(T1=1e-6, t=0.5e-6), Dephasing(T2=2e-6, t=0.5e-6)]
    ch = CombinedChannel(channels)
    assert len(ch.describe()["channels"]) == 2


def test_repr_contains_class_name():
    ch = CombinedChannel([AmplitudeDamping(T1=1e-6, t=0.5e-6)])
    assert "CombinedChannel" in repr(ch)


def test_repr_contains_inner_type():
    ch = CombinedChannel([AmplitudeDamping(T1=1e-6, t=0.5e-6)])
    assert "AmplitudeDamping" in repr(ch)


# ---------------------------------------------------------------------------
# Backend integration: CombinedChannel applies channels in order
# ---------------------------------------------------------------------------

def test_combined_applies_both_channels():
    """Adding depolarizing on top of T1 damping must reduce P(|1⟩) further.

    With T1 only: P(|1⟩) = exp(-t/T1) ≈ 0.61 at t=T1/2.
    With depolarizing(p=0.4) added: mixed more, P(|1⟩) < 0.61.
    """
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(ir.X, qubits=[0])

    T1, t = 1e-6, 0.5e-6
    backend = TrajectoryBackend()

    r_kraus = backend.run(
        circuit,
        noise_model=AmplitudeDamping(T1=T1, t=t),
        n_shots=4000,
        seed=0,
    )
    r_combined = backend.run(
        circuit,
        noise_model=CombinedChannel([
            AmplitudeDamping(T1=T1, t=t),
            depolarizing_error(p=0.4),
        ]),
        n_shots=4000,
        seed=0,
    )

    p_kraus = float(np.real(r_kraus.final_state[1, 1]))
    p_combined = float(np.real(r_combined.final_state[1, 1]))
    assert p_combined < p_kraus, (
        f"Expected combined < kraus-only: combined={p_combined:.4f} kraus={p_kraus:.4f}"
    )


def test_combined_with_correlated_pauli():
    """CombinedChannel containing CorrelatedPauliError runs without error."""
    circuit = Circuit(n_qubits=2)
    circuit.add_gate(ir.H, qubits=[0])
    circuit.add_gate(ir.CNOT, qubits=[0, 1])

    backend = TrajectoryBackend()
    ch = CombinedChannel([
        Dephasing(T2=2e-6, t=50e-9),
        CorrelatedPauliError({'ZZ': 0.01}),
    ])
    result = backend.run(circuit, noise_model=ch, n_shots=100, seed=42)
    rho = result.final_state
    assert rho.shape == (4, 4)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


def test_nested_combined_channel():
    """CombinedChannel inside CombinedChannel is dispatched recursively."""
    inner = CombinedChannel([AmplitudeDamping(T1=1e-6, t=0.1e-6)])
    outer = CombinedChannel([inner, depolarizing_error(p=0.01)])

    circuit = Circuit(n_qubits=1)
    circuit.add_gate(ir.X, qubits=[0])

    backend = TrajectoryBackend()
    result = backend.run(circuit, noise_model=outer, n_shots=200, seed=7)
    rho = result.final_state
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)
