import numpy as np
import pytest

from noisiq.ir import Circuit, gates as ir
from noisiq.noise import (
    fill_idle_with_identities,
    idle_kraus,
    idle_pauli_twirl,
    get_hardware,
)
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.noise.t2_dephasing import Dephasing
from noisiq.noise.kraus_channels import CombinedChannel


def test_fill_inserts_idle_in_empty_slots():
    # q0 active at t=0,2; q1 active at t=1 only
    # → q0 idle at t=1; q1 idle at t=0 and t=2
    c = Circuit(2)
    c.add_gate(ir.H, (0,), t=0)
    c.add_gate(ir.H, (1,), t=1)
    c.add_gate(ir.X, (0,), t=2)

    profile = get_hardware("ibm_eagle_r3")
    filled = fill_idle_with_identities(c, profile.gate_times)

    idle_ops = [op for op in filled.operations if op.gate is ir.IDLE]
    placements = sorted((op.qubits[0], op.t) for op in idle_ops)
    assert placements == [(0, 1), (1, 0), (1, 2)]


def test_idle_carries_duration():
    c = Circuit(2)
    c.add_gate(ir.H,    (0,),    t=0)
    c.add_gate(ir.CNOT, (0, 1),  t=1)
    c.add_gate(ir.X,    (0,),    t=2)

    profile = get_hardware("ibm_eagle_r3")
    filled = fill_idle_with_identities(c, profile.gate_times)

    # q1 has an IDLE at t=0, parallel to the single-qubit H
    idle_at_t0 = next(
        op for op in filled.operations
        if op.gate is ir.IDLE and op.t == 0
    )
    assert idle_at_t0.params["duration_ns"] == profile.gate_times.single_qubit_ns


def test_fill_does_not_modify_original():
    c = Circuit(2)
    c.add_gate(ir.H, (0,), t=0)
    c.add_gate(ir.X, (1,), t=1)

    profile = get_hardware("ibm_eagle_r3")
    original_len = len(c.operations)
    fill_idle_with_identities(c, profile.gate_times)

    assert len(c.operations) == original_len


def test_fill_trailing_disabled_by_default():
    c = Circuit(2)
    c.add_gate(ir.H, (0,), t=0)
    c.add_gate(ir.X, (0,), t=3)
    # q1 has no gates — should receive no IDLEs

    profile = get_hardware("ibm_eagle_r3")
    filled = fill_idle_with_identities(c, profile.gate_times)
    q1_idles = [op for op in filled.operations if op.gate is ir.IDLE and op.qubits[0] == 1]
    assert q1_idles == []


def test_idle_noise_is_decoherence_only():
    """to_noise_model(representation='pauli_twirl') on IDLE returns a PauliError with no depolarizing."""
    c = Circuit(2)
    c.add_gate(ir.H, (0,), t=0)
    c.add_gate(ir.IDLE, (1,), t=0, params={"duration_ns": 40.0})

    profile = get_hardware("ibm_eagle_r3")
    noise = profile.to_noise_model(c, representation="pauli_twirl")

    expected = idle_pauli_twirl(profile.t1, profile.t2, 40.0)
    actual = noise[1]
    assert np.isclose(actual.p_x, expected.p_x)
    assert np.isclose(actual.p_y, expected.p_y)
    assert np.isclose(actual.p_z, expected.p_z)


def test_idle_kraus_returns_combined_channel():
    ch = idle_kraus(t1=100e-6, t2=80e-6, duration_ns=100.0)
    assert isinstance(ch, CombinedChannel)
    assert any(isinstance(sub, AmplitudeDamping) for sub in ch.channels)
    assert any(isinstance(sub, Dephasing) for sub in ch.channels)


def test_idle_pauli_twirl_probabilities_are_physical():
    pe = idle_pauli_twirl(t1=100e-6, t2=80e-6, duration_ns=500.0)
    assert pe.p_x >= 0
    assert pe.p_y >= 0
    assert pe.p_z >= 0
    assert pe.p_x + pe.p_y + pe.p_z <= 1.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_to_noise_model_idle_branch_kraus():
    c = Circuit(1)
    c.add_gate(ir.H, (0,), t=0)
    c.add_gate(ir.IDLE, (0,), t=1, params={"duration_ns": 60.0})

    profile = get_hardware("ibm_eagle_r3")
    noise = profile.to_noise_model(c)
    assert isinstance(noise[1], CombinedChannel)


def test_idle_missing_duration_raises():
    c = Circuit(1)
    c.add_gate(ir.IDLE, (0,), t=0)  # no params

    profile = get_hardware("ibm_eagle_r3")
    with pytest.raises(ValueError, match="duration_ns"):
        profile.to_noise_model(c, representation="pauli_twirl")


def test_dd_then_fill_composes():
    """apply_dd then fill_idle_with_identities — no double-counting."""
    from noisiq.suppression import apply_dd

    c = Circuit(1)
    c.add_gate(ir.H, (0,), t=0)
    c.add_gate(ir.X, (0,), t=9)   # 8-layer idle window (t=1..8)

    dd = apply_dd(c, sequence="XY-4")
    profile = get_hardware("ibm_eagle_r3")
    final = fill_idle_with_identities(dd, profile.gate_times)

    n_idle = sum(1 for op in final.operations if op.gate is ir.IDLE)
    # XY-4 inserts 4 pulses; 8 - 4 = 4 remaining idle slots
    assert n_idle == 4
