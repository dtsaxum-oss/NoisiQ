"""Tests for dynamic decoupling pulse insertion."""

import numpy as np
import pytest

from noisiq.ir import Circuit
from noisiq.ir import gates as ir_gates
from noisiq.suppression import apply_dd, list_sequences


# ---------------------------------------------------------------------------
# list_sequences
# ---------------------------------------------------------------------------

def test_list_sequences_returns_all():
    seqs = list_sequences()
    assert set(seqs) == {"hahn", "CPMG", "XY-4", "XY-8"}


def test_list_sequences_sorted():
    seqs = list_sequences()
    assert seqs == sorted(seqs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ops_on_qubit(circuit: Circuit, qubit: int):
    """Return all operations on a given qubit, sorted by time."""
    return sorted(
        [op for op in circuit.operations if qubit in op.qubits],
        key=lambda o: o.t,
    )


def _dd_ops_on_qubit(original: Circuit, dd: Circuit, qubit: int):
    """Return only the newly-inserted DD operations for a qubit."""
    original_ts = {op.t for op in original.operations if qubit in op.qubits}
    return sorted(
        [op for op in dd.operations if qubit in op.qubits and op.t not in original_ts],
        key=lambda o: o.t,
    )


# ---------------------------------------------------------------------------
# apply_dd — invalid input
# ---------------------------------------------------------------------------

def test_unknown_sequence_raises():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.X, (0,), t=5)
    with pytest.raises(ValueError, match="Unknown DD sequence"):
        apply_dd(c, sequence="bad_seq")


# ---------------------------------------------------------------------------
# apply_dd — Hahn echo
# ---------------------------------------------------------------------------

def test_hahn_inserts_single_x_pulse():
    # q0 idle from t=1..4 (window_len=4), q1 idle throughout
    c = Circuit(2)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=5)

    dd = apply_dd(c, sequence="hahn", idle_threshold=1)

    # Original gates preserved
    assert len([op for op in dd.operations if op.t == 0 and op.gate == ir_gates.H]) == 1
    assert len([op for op in dd.operations if op.t == 5 and op.gate == ir_gates.CNOT]) == 1

    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(dd_ops) == 1
    assert dd_ops[0].gate == ir_gates.X
    # Pulse must land strictly between the two anchor times
    assert 0 < dd_ops[0].t < 5


def test_hahn_original_circuit_unchanged():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.X, (0,), t=5)
    original_len = len(c.operations)

    apply_dd(c, sequence="hahn")

    assert len(c.operations) == original_len


# ---------------------------------------------------------------------------
# apply_dd — XY-4
# ---------------------------------------------------------------------------

def test_xy4_pulse_count_and_sequence():
    # Window of length 4 — exactly fits XY-4
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=5)

    dd = apply_dd(c, sequence="XY-4", idle_threshold=1)

    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(dd_ops) == 4
    gate_seq = [op.gate for op in dd_ops]
    assert gate_seq == [ir_gates.X, ir_gates.Y, ir_gates.X, ir_gates.Y]


def test_xy4_pulses_at_distinct_times_in_window():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=5)

    dd = apply_dd(c, sequence="XY-4", idle_threshold=1)
    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)

    times = [op.t for op in dd_ops]
    assert len(set(times)) == len(times), "DD pulses must be at distinct times"
    assert all(0 < t < 5 for t in times), "Pulses must be strictly inside the window"


# ---------------------------------------------------------------------------
# apply_dd — XY-8
# ---------------------------------------------------------------------------

def test_xy8_requires_8_idle_slots():
    # Window of 7 — not enough for XY-8 (needs 8)
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=8)   # window_len = 7

    dd = apply_dd(c, sequence="XY-8", idle_threshold=1)
    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(dd_ops) == 0  # not enough room → no pulses


def test_xy8_inserted_with_sufficient_window():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=9)   # window_len = 8 — exactly fits

    dd = apply_dd(c, sequence="XY-8", idle_threshold=1)
    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(dd_ops) == 8


# ---------------------------------------------------------------------------
# apply_dd — idle_threshold
# ---------------------------------------------------------------------------

def test_threshold_blocks_insertion():
    # Window of 4, but threshold=5 → no insertion
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=5)

    dd = apply_dd(c, sequence="hahn", idle_threshold=5)
    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(dd_ops) == 0


def test_threshold_allows_insertion():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=5)

    dd = apply_dd(c, sequence="hahn", idle_threshold=4)
    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(dd_ops) == 1  # window_len=4 >= threshold=4


# ---------------------------------------------------------------------------
# apply_dd — multi-qubit circuit
# ---------------------------------------------------------------------------

def test_dd_on_idle_qubit_in_ghz_circuit():
    # 3-qubit GHZ: H q0@t0, CNOT(q0,q1)@t1, CNOT(q1,q2)@t2
    # q2 is idle at t=0 (window between start/first active), but no idle *between*
    # two active gates for q2 — only 1 active time, skip.
    # Let's give q0 a window: H@t0, another gate@t=10
    c = Circuit(3)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=1)
    c.add_gate(ir_gates.CNOT, (1, 2), t=2)
    # Add a later gate on q0 to create an idle window
    c.add_gate(ir_gates.Z, (0,), t=7)  # q0 idle at t=2..6 (window_len=5)

    dd = apply_dd(c, sequence="hahn", idle_threshold=1)

    # q0 should get a Hahn pulse between t=1 and t=7
    q0_dd = _dd_ops_on_qubit(c, dd, qubit=0)
    assert len(q0_dd) == 1
    assert 1 < q0_dd[0].t < 7
    assert q0_dd[0].gate == ir_gates.X


def test_dd_preserves_all_original_gates():
    c = Circuit(2)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=5)

    dd = apply_dd(c, sequence="XY-4", idle_threshold=1)

    orig_ops = {(op.gate.name, op.qubits, op.t) for op in c.operations}
    dd_ops = {(op.gate.name, op.qubits, op.t) for op in dd.operations}
    assert orig_ops.issubset(dd_ops)


# ---------------------------------------------------------------------------
# apply_dd — empty and trivial circuits
# ---------------------------------------------------------------------------

def test_empty_circuit_returns_copy():
    c = Circuit(3)
    dd = apply_dd(c, sequence="XY-4")
    assert dd.n_qubits == 3
    assert len(dd.operations) == 0


def test_single_gate_no_window():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    dd = apply_dd(c, sequence="hahn")
    assert len(dd.operations) == 1  # only the original H gate


def test_circuit_name_preserved():
    c = Circuit(1, name="test_circ")
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.X, (0,), t=5)
    dd = apply_dd(c, sequence="hahn")
    assert dd.name == "test_circ"


# ---------------------------------------------------------------------------
# apply_dd — CPMG
# ---------------------------------------------------------------------------

def test_cpmg_inserts_two_x_pulses():
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.Z, (0,), t=5)  # window_len=4 >= 2

    dd = apply_dd(c, sequence="CPMG", idle_threshold=1)
    dd_ops = _dd_ops_on_qubit(c, dd, qubit=0)

    assert len(dd_ops) == 2
    assert all(op.gate == ir_gates.X for op in dd_ops)
    times = [op.t for op in dd_ops]
    assert len(set(times)) == 2
    assert all(0 < t < 5 for t in times)


# ---------------------------------------------------------------------------
# Regression: time-ordering invariance (Bug 1a from noisiq_bug_report.md)
# ---------------------------------------------------------------------------

def test_time_ordering_invariant_trajectory():
    """Backends must iterate ops by (t, idx), not insertion order.

    Build a 1-qubit circuit with X at t=2 and H at t=0. Add them in
    reverse insertion order (X first, H second). The result must equal
    building them in forward order.
    """
    from noisiq.backends import TrajectoryBackend

    # Forward order: H@t0, X@t2
    c_fwd = Circuit(1)
    c_fwd.add_gate(ir_gates.H, (0,), t=0)
    c_fwd.add_gate(ir_gates.X, (0,), t=2)

    # Reverse insertion: X@t2 appended first, H@t0 appended second.
    # Before the fix, backends applied X then H (wrong); after the fix they
    # sort by t and apply H@t0 then X@t2 (correct).
    c_rev = Circuit(1)
    c_rev.add_gate(ir_gates.X, (0,), t=2)
    c_rev.add_gate(ir_gates.H, (0,), t=0)

    rho_fwd = TrajectoryBackend().run(c_fwd, noise_model=None, n_shots=1, seed=0).final_state
    rho_rev = TrajectoryBackend().run(c_rev, noise_model=None, n_shots=1, seed=0).final_state

    assert np.allclose(rho_fwd, rho_rev), (
        "TrajectoryBackend must be invariant to insertion order — only t matters."
    )


def test_dd_reduces_fidelity_loss_with_coherent_idle_noise():
    """DD must visibly improve round-trip fidelity when IDLE noise has a
    coherent (refocusable) component.  Catches regressions in apply_dd
    placement or backend time-ordering.
    """
    import dataclasses
    from noisiq.backends import TrajectoryBackend
    from noisiq.noise import fill_idle_with_identities, get_hardware

    N = 3
    STORAGE = 16
    END_ENT = N - 1
    END_STO = END_ENT + STORAGE

    def _build_sparse():
        c = Circuit(n_qubits=N)
        c.add_gate(ir_gates.H, (0,), t=0)
        c.add_gate(ir_gates.CNOT, (0, 1), t=1)
        c.add_gate(ir_gates.CNOT, (1, 2), t=2)
        c.add_gate(ir_gates.CNOT, (1, 2), t=END_STO + 1)
        c.add_gate(ir_gates.CNOT, (0, 1), t=END_STO + 2)
        c.add_gate(ir_gates.H, (0,), t=END_STO + 3)
        return c

    sparse = _build_sparse()
    sparse_dd = apply_dd(sparse, sequence="XY-4", idle_threshold=4)

    profile = get_hardware("ibm_eagle_r3")
    demo_profile = dataclasses.replace(profile, idle_coherent_epsilon=0.07)
    gate_times = profile.gate_times

    baseline = fill_idle_with_identities(sparse,    gate_times)
    with_dd  = fill_idle_with_identities(sparse_dd, gate_times)

    noise_b = demo_profile.to_noise_model(baseline, mode="t2", representation="pauli_twirl")
    noise_d = demo_profile.to_noise_model(with_dd,  mode="t2", representation="pauli_twirl")

    rho_b = TrajectoryBackend().run(baseline, noise_model=noise_b, n_shots=400, seed=42).final_state
    rho_d = TrajectoryBackend().run(with_dd,  noise_model=noise_d, n_shots=400, seed=42).final_state

    ideal = np.zeros(2**N, dtype=complex)
    ideal[0] = 1.0
    rho_ideal = np.outer(ideal, ideal.conj())

    F_b = float(np.real(np.trace(rho_ideal @ rho_b)))
    F_d = float(np.real(np.trace(rho_ideal @ rho_d)))

    assert F_d > F_b, (
        f"DD should improve fidelity under coherent IDLE noise, "
        f"but baseline={F_b:.4f} >= DD={F_d:.4f}."
    )


# ---------------------------------------------------------------------------
# DD + fill_idle_with_identities composition (ordering contract)
# ---------------------------------------------------------------------------

def test_dd_then_fill_no_double_counting():
    """apply_dd first, then fill_idle — DD pulses consume idle layers,
    filler handles what's left. No IDLE op should land on a DD-pulse layer."""
    from noisiq.noise import fill_idle_with_identities, get_hardware

    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.X, (0,), t=9)   # 8-layer idle window (t=1..8)

    dd = apply_dd(c, sequence="XY-4")
    profile = get_hardware("ibm_eagle_r3")
    final = fill_idle_with_identities(dd, profile.gate_times)

    # XY-4 inserts 4 pulses into the 8-slot window → 4 idle slots remain
    n_idle = sum(1 for op in final.operations if op.gate is ir_gates.IDLE)
    assert n_idle == 4

    # No IDLE should share a layer with a DD pulse
    dd_layers = {
        op.t for op in dd.operations
        if op.gate.name in ("X", "Y") and op not in c.operations
    }
    idle_layers = {op.t for op in final.operations if op.gate is ir_gates.IDLE}
    assert dd_layers.isdisjoint(idle_layers)
