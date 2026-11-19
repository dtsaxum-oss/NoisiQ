"""
Phase 4B — Physically-correct MCM fidelity via paired ideal/noisy simulation.

Test matrix from mcm_fidelity_implementation_plan.md §5:

  1. Measure-only, no noise → frame 'I', no divergence (every shot).
  2. Deterministic X before measurement → outcome flips, divergence flagged.
  3. Measure-and-reset, deterministic error AFTER reset → correct frame, no divergence.
  4. Feed-forward correction that succeeds (noiseless) → frame 'I', no divergence.
  5. Feed-forward where noise flips the syndrome → wrong branch → divergence flagged,
     even though the final states happen to be identical (Option A ≠ Option B demo).
  6. Repeated measurement with reset between → no spurious divergence on random outcomes.
  7. Regression: MCM circuit with no pre-measurement errors and no conditionals produces
     the same output-frame Pauli as the existing non-MCM run() path on the same seed.
  8. Statistical cross-check against TrajectoryBackend density-matrix fidelity (acceptance
     gate — runs last, validates the whole pipeline against an independent backend).
  9. Seed reproducibility: same seed must produce identical final_pauli_frames and
     branch_diverged across two calls to run_paired().
"""

import numpy as np
import pytest

from noisiq.ir import Circuit, gates
from noisiq.backends.pauli_frame import StimTableauBackend, PairedSimResult
from noisiq.noise.pauli_error import PauliError

# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _meas_only_circuit() -> Circuit:
    """1 qubit, no gates, measure q0 into c[0]."""
    c = Circuit(n_qubits=1)
    reg = c.add_classical_register('c', 1)
    c.measure(0, reg[0])
    return c


def _x_then_meas_circuit() -> Circuit:
    """1 qubit: X gate (op 0) → measure q0 (op 1).

    With p_x=1.0 on the X gate:
      ideal:  X applied → q0 in |1⟩ before measurement → outcome 1
      noisy:  X + X_noise = I → q0 in |0⟩ before measurement → outcome 0
    The outcomes diverge.
    """
    c = Circuit(n_qubits=1)
    reg = c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, reg[0])
    return c


def _meas_reset_then_x_circuit() -> Circuit:
    """1 qubit: measure+reset q0 (op 0) → X gate (op 1).

    With p_x=1.0 on the X gate (op 1):
      Both sims measure q0 in |0⟩ → outcome 0 → reset is a no-op in both.
      ideal:  X applied → |1⟩
      noisy:  X + X_noise = I → |0⟩
    No divergence; output Pauli = 'X'.
    """
    c = Circuit(n_qubits=1)
    reg = c.add_classical_register('c', 1)
    c.measure(0, reg[0], reset=True)
    c.add_gate(gates.X, (0,))
    return c


def _feedforward_correction_circuit() -> Circuit:
    """1 qubit bit-flip correction: X(q0) → measure → c_if(==1).X(q0).

    Noiseless: X → |1⟩ → measure → 1 → c_if fires → X → |0⟩.  Net = identity.
    With p_x=1.0 on X gate (op 0):
      noisy:  X + X_noise = I → |0⟩ → measure → 0 → c_if does NOT fire → |0⟩.
      ideal:  X → |1⟩ → measure → 1 → c_if fires → X → |0⟩.
    Branches diverge (different corrections applied) even though final states are
    both |0⟩.  This is the canonical Option A ≠ Option B demonstration.
    """
    c = Circuit(n_qubits=1)
    reg = c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, reg[0])
    c.c_if(reg[0], value=1).x(0)
    return c


def _repeated_meas_circuit() -> Circuit:
    """1 qubit: measure+reset(q0)→c[0], H(q0), measure(q0)→c[1].

    First measurement is deterministic (q0 in |0⟩ → outcome 0, no divergence).
    After reset+H the qubit is in |+⟩ (random second measurement).
    The random outcome is postselected onto the noisy outcome — never flagged
    as divergence (there is no 'correct' deterministic answer to disagree with).
    """
    c = Circuit(n_qubits=1)
    reg0 = c.add_classical_register('c0', 1)
    reg1 = c.add_classical_register('c1', 1)
    c.measure(0, reg0[0], reset=True)
    c.add_gate(gates.H, (0,))
    c.measure(0, reg1[0])
    return c


def _two_qubit_independent_meas_circuit() -> Circuit:
    """2 qubits: X(q0) → measure(q1)→c[0].  q0 and q1 are independent.

    q1 starts in |0⟩ → deterministic measurement outcome 0 in both sims.
    Noise on X(q0) affects only q0's output Pauli; the measurement of q1
    introduces no divergence.

    Used as the regression anchor (test 7): run_paired() on this MCM circuit
    must agree with run() on the non-MCM version (X(q0) only, no measurement).
    """
    c = Circuit(n_qubits=2)
    reg = c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))      # op 0: X on q0
    c.measure(1, reg[0])           # op 1: measure q1 (independent of q0)
    return c


# ---------------------------------------------------------------------------
# Test 1 — Measure-only, no noise → 'I' frame, no divergence
# ---------------------------------------------------------------------------

def test_no_noise_no_divergence():
    """No noise → both sims agree at every step → no divergence, trivial frame."""
    c = _meas_only_circuit()
    backend = StimTableauBackend()
    result = backend.run_paired(c, noise_config=None, n_shots=20, seed=0)

    assert len(result.branch_diverged) == 20
    assert len(result.final_pauli_frames) == 20
    assert not any(result.branch_diverged), "Expected zero divergences with no noise"
    assert all(pf == 'I' for pf in result.final_pauli_frames), (
        f"Expected all-identity frames, got: {result.final_pauli_frames}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Deterministic X before measurement → outcome flips, divergence set
# ---------------------------------------------------------------------------

def test_deterministic_x_before_meas_diverges():
    """p_x=1.0 on X gate cancels it in the noisy sim → outcome 0 vs ideal 1.

    Analytic truth:
      ideal:  X gate → q0=|1⟩ → measure → outcome 1
      noisy:  X gate + X_noise = I → q0=|0⟩ → measure → outcome 0
      peek_z on ideal = -1 (deterministic |1⟩), b_noisy = 0 → diverged = True.
    All 20 shots must diverge because the error is deterministic.
    """
    c = _x_then_meas_circuit()
    # op 0 is the X gate; deterministic X noise cancels the gate in the noisy sim.
    noise = {0: PauliError(p_x=1.0, p_y=0.0, p_z=0.0)}
    backend = StimTableauBackend()
    result = backend.run_paired(c, noise_config=noise, n_shots=20, seed=0)

    assert all(result.branch_diverged), (
        "Expected all shots to diverge when X noise deterministically flips the outcome"
    )


# ---------------------------------------------------------------------------
# Test 3 — Measure-and-reset, deterministic error after reset → correct frame
# ---------------------------------------------------------------------------

def test_meas_reset_error_after_reset_no_divergence():
    """Noise after reset does not cause divergence; it shows up in the output frame.

    Analytic truth:
      Both sims: q0 in |0⟩ → measure → outcome 0 (deterministic, both sims agree).
      reset is a no-op (outcome was 0).
      ideal:  X gate → |1⟩  (final tableau: X)
      noisy:  X gate + X_noise = I → |0⟩  (final tableau: identity)
      T_error = X_inv · I = X → output Pauli 'X'.
      No divergence because the measurement outcomes agreed.
    """
    c = _meas_reset_then_x_circuit()
    # op 0 = Measurement(reset=True); op 1 = X gate — noise on op 1 only.
    op_indices = {i: op for i, op in enumerate(c.operations)}
    x_op_idx = next(i for i, op in enumerate(c.operations)
                    if hasattr(op, 'gate') and op.gate.name.upper() == 'X')
    noise = {x_op_idx: PauliError(p_x=1.0, p_y=0.0, p_z=0.0)}
    backend = StimTableauBackend()
    result = backend.run_paired(c, noise_config=noise, n_shots=20, seed=0)

    assert not any(result.branch_diverged), (
        "Noise after reset should not cause branch divergence"
    )
    assert all(pf == 'X' for pf in result.final_pauli_frames), (
        f"Expected 'X' output frame for cancelled X gate, got: {result.final_pauli_frames}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Feed-forward correction succeeds (noiseless)
# ---------------------------------------------------------------------------

def test_feedforward_noiseless_no_divergence():
    """Noiseless bit-flip correction: X → measure → c_if.X restores |0⟩.

    Both sims follow the same branch (no noise → same gate applications).
    Final state is always |0⟩ regardless of random measurement outcomes.
    Expected: no divergence, frame 'I'.
    """
    c = _feedforward_correction_circuit()
    backend = StimTableauBackend()
    result = backend.run_paired(c, noise_config=None, n_shots=20, seed=0)

    assert not any(result.branch_diverged), (
        "Noiseless feed-forward should never diverge"
    )
    assert all(pf == 'I' for pf in result.final_pauli_frames), (
        f"Noiseless correction should give identity frame, got: {result.final_pauli_frames}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Feed-forward where noise flips the syndrome (Option A ≠ Option B)
# ---------------------------------------------------------------------------

def test_feedforward_wrong_branch_diverges_even_if_final_states_match():
    """p_x=1.0 on X gate cancels the gate → wrong branch in noisy sim.

    Option B (branch divergence): FAILS — noisy took the wrong branch.
    Option A (stabilizer state): PASSES — both sims end in |0⟩.

    This test demonstrates why Option B is needed: the feed-forward correction
    is logically wrong even though the final classical state matches.

    Analytic truth:
      ideal:  X → |1⟩ → measure → 1 → c_if fires → X → |0⟩
      noisy:  X + X_noise = I → |0⟩ → measure → 0 → c_if skipped → |0⟩
      ideal final = |0⟩, noisy final = |0⟩ → output Pauli 'I' (Option A passes!)
      branch_diverged = True for every shot (Option B flags the failure).
    """
    c = _feedforward_correction_circuit()
    # op 0 is the X gate; see _feedforward_correction_circuit() docstring.
    noise = {0: PauliError(p_x=1.0, p_y=0.0, p_z=0.0)}
    backend = StimTableauBackend()
    result = backend.run_paired(c, noise_config=noise, n_shots=20, seed=0)

    # Option B: branch diverged on every shot.
    assert all(result.branch_diverged), (
        "Expected all shots to diverge when noise flips the syndrome"
    )
    # Option A: final states are identical (both |0⟩) → frame is 'I'.
    assert all(pf == 'I' for pf in result.final_pauli_frames), (
        f"Expected identity frame (both end in |0⟩), got: {result.final_pauli_frames}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Repeated measurement with reset: no spurious divergence
# ---------------------------------------------------------------------------

def test_repeated_meas_with_reset_no_spurious_divergence():
    """Two measurements on the same qubit; the second is random (|+⟩ state).

    The paired-sim must not flag the random second measurement as divergence:
    a random ideal outcome has no 'correct' value, so postselecting the ideal
    onto the noisy outcome is the right call, not flagging it as an error.
    """
    c = _repeated_meas_circuit()
    backend = StimTableauBackend()
    result = backend.run_paired(c, noise_config=None, n_shots=100, seed=42)

    assert not any(result.branch_diverged), (
        "Random measurement outcomes (|+⟩ state) should not cause divergence; "
        "the ideal sim should be postselected onto the noisy outcome."
    )


# ---------------------------------------------------------------------------
# Test 7 — Regression: MCM frame matches non-MCM frame on same seed
# ---------------------------------------------------------------------------

def test_regression_mcm_frame_matches_non_mcm_path():
    """MCM circuit with independent measurement agrees with the non-MCM path.

    Circuit A (non-MCM): X(q0) with p_z=1.0 on q0.
    Circuit B (MCM):     X(q0) + measure(q1) with the same p_z=1.0 on X.

    The measurement of q1 is independent of q0 and always gives outcome 0
    (q1 starts in |0⟩, deterministic → no divergence in either path).
    The output-frame Pauli on q0 must be the same in both.

    Analytic truth:
      Z noise after X gate: X Z X† = X Z X = -Z ≡ Z error on q0.
      T_error = T_X_inv · (T_X · T_Z) = T_X_inv · T_X · T_Z = T_Z = Z.
      Output Pauli: 'ZI' (Z on q0, I on q1 after measurement collapses q1 to |0⟩).
    """
    seed = 7

    # ── Existing non-MCM path (run()) ───────────────────────────────────────
    non_mcm = Circuit(n_qubits=2)
    non_mcm.add_gate(gates.X, (0,))   # op 0
    noise = {0: PauliError(p_x=0.0, p_y=0.0, p_z=1.0)}
    backend = StimTableauBackend()
    old_result = backend.run(non_mcm, noise_model=noise, n_shots=1, seed=seed)
    old_frame = old_result.meta["stim_result"].final_pauli_frames[0]

    # ── New paired-sim path (run_paired()) ──────────────────────────────────
    mcm = _two_qubit_independent_meas_circuit()  # X(q0) + measure(q1)
    new_result = backend.run_paired(mcm, noise_config=noise, n_shots=1, seed=seed)
    new_frame = new_result.final_pauli_frames[0]

    assert not any(new_result.branch_diverged), (
        "Independent measurement on q1 should never diverge"
    )
    assert old_frame == new_frame, (
        f"run_paired frame {new_frame!r} does not match run() frame {old_frame!r} "
        "for the same circuit body and seed (regression failure)."
    )
    # Also verify the expected analytic value directly.
    assert new_frame == 'ZI', (
        f"Expected 'ZI' (Z error on q0, I on measured q1), got {new_frame!r}"
    )


# ---------------------------------------------------------------------------
# Test 8 — Statistical cross-check against TrajectoryBackend (acceptance gate)
# ---------------------------------------------------------------------------

def test_statistical_cross_check_against_trajectory_backend():
    """Acceptance gate: paired-sim branch-success rate agrees with trajectory fidelity.

    Circuit: teleportation-style bit-flip correction on 1 qubit.
      X(q0) → measure → c_if(==1).X(q0)
    Noise: depolarising-style X error on X gate at moderate rate p.

    Expected: branch_success_rate from run_paired ≈ trajectory fidelity within
    Monte Carlo error (±3σ for n_shots=2000 is roughly ±0.03 for p~0.1).

    This test is the ACCEPTANCE GATE for the whole Phase 4B implementation.
    It validates the paired-sim pipeline against an independent, physically
    trusted backend.  The test is intentionally loose (5σ tolerance) to avoid
    flakiness while still catching a systematic implementation error.
    """
    pytest.importorskip("noisiq.backends.trajectory_backend")
    from noisiq.backends.trajectory_backend import TrajectoryBackend

    p = 0.10
    n_shots = 2000
    seed = 99
    noise = {0: PauliError(p_x=p, p_y=0.0, p_z=0.0)}

    c = _feedforward_correction_circuit()

    # ── Paired-sim branch-success rate ──────────────────────────────────────
    backend = StimTableauBackend()
    paired = backend.run_paired(c, noise_config=noise, n_shots=n_shots, seed=seed)
    branch_success_rate = 1.0 - np.mean(paired.branch_diverged)

    # ── TrajectoryBackend statevector fidelity ───────────────────────────────
    # Fidelity of the output state |0⟩ vs the ideal: run n_shots single shots
    # and count how often the final state has zero error.
    traj = TrajectoryBackend()
    traj_result = traj.run_aggregate(c, noise_model=noise, n_shots=n_shots, seed=seed)
    # Zero-error fraction from trajectory: fraction of shots where the error
    # product from the counts_matrix is zero (no error events sampled).
    traj_zero_error_rate = float(traj_result.zero_error_shots.mean())

    # ── Comparison within 5σ Monte Carlo tolerance ──────────────────────────
    # For Bernoulli with p≈0.9, σ ≈ sqrt(0.9*0.1/2000) ≈ 0.0067.
    # 5σ ≈ 0.033. We use 0.05 to be safe while still catching large discrepancies.
    assert abs(branch_success_rate - traj_zero_error_rate) < 0.05, (
        f"Paired-sim branch_success_rate={branch_success_rate:.4f} differs from "
        f"trajectory zero_error_rate={traj_zero_error_rate:.4f} by more than 0.05. "
        "This indicates a systematic error in the paired-sim implementation."
    )


def test_run_paired_seed_reproducibility():
    """Same seed must produce identical final_pauli_frames and branch_diverged.

    Circuit: H → measure → c_if(cbit==1).Z  with p_z=1.0 noise on the Z gate.

    - When the H-state measurement samples 0: the conditional Z does not fire on
      either branch → output-frame error Pauli is 'I'.
    - When it samples 1: the conditional Z fires on both branches; the Z noise
      (p_z=1.0) then applies a second Z to the noisy sim only.  Ideal sim ends
      with Z applied; noisy sim ends with Z·Z = I applied → error frame is 'Z'.

    ~50 % of shots give 'I', ~50 % give 'Z', so the frames are a genuine mix.
    Without seeding stim.TableauSimulator the coin-flip for the |+⟩ measurement
    is non-deterministic: two runs with the same numpy seed would sample different
    measurement outcomes and therefore produce different frame sequences.
    """
    c = Circuit(n_qubits=1)
    reg = c.add_classical_register("c", 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, reg[0])
    c.c_if(reg[0], value=1).z(0)
    # ConditionalOp wrapping Z is the last operation.
    cond_z_idx = next(
        i for i, op in enumerate(c.operations)
        if isinstance(op, __import__("noisiq.ir.classical", fromlist=["ConditionalOp"]).ConditionalOp)
    )
    noise = {cond_z_idx: PauliError(p_x=0.0, p_y=0.0, p_z=1.0)}

    backend = StimTableauBackend()
    r1 = backend.run_paired(c, noise_config=noise, n_shots=200, seed=42)
    r2 = backend.run_paired(c, noise_config=noise, n_shots=200, seed=42)

    assert r1.final_pauli_frames == r2.final_pauli_frames, (
        "run_paired produced different final_pauli_frames for the same seed. "
        "stim.TableauSimulator is not being seeded per shot."
    )
    assert r1.branch_diverged == r2.branch_diverged, (
        "run_paired produced different branch_diverged for the same seed. "
        "stim.TableauSimulator is not being seeded per shot."
    )
    # Sanity: ~50 % 'I', ~50 % 'Z' — both values must appear.
    assert len(set(r1.final_pauli_frames)) > 1, (
        "All 200 shots produced the same Pauli frame — either the circuit is "
        "degenerate or all shot-seeds are identical."
    )
