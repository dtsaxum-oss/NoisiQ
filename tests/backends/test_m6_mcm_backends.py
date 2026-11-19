"""
M6 — Mid-circuit measurement backend support.

Tests cover:
  - _probability_of_one / _project_to_outcome helpers (trajectory_backend)
  - StimTableauBackend.run() with Measurement and ConditionalOp ops
  - TrajectoryBackend.run() with Measurement and ConditionalOp ops
  - TrajectoryBackend.run_aggregate() with MCM
  - SimulationResult.measurements / acceptance_rate fields
  - AggregateResult.measurements / acceptance_rate fields
  - StimTableauResult.shot_cbits field
  - measure-and-reset (reset=True)
  - Conditional correction that cancels an injected error (teleport-style)
"""

import numpy as np
import pytest

from noisiq.ir import Circuit, gates
from noisiq.ir.classical import ClassicalRegister
from noisiq.backends.pauli_frame import StimTableauBackend, StimTableauResult
from noisiq.backends.trajectory_backend import (
    TrajectoryBackend,
    _probability_of_one,
    _project_to_outcome,
)
from noisiq.backends.many_shot_runner import AggregateResult, ManyShotRunner
from noisiq.results.type import SimulationResult
from noisiq.noise.pauli_error import PauliError


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _bell_circuit() -> Circuit:
    """2-qubit Bell-state circuit: H(0) followed by CNOT(0,1)."""
    c = Circuit(n_qubits=2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    return c


def _mcm_circuit_simple() -> Circuit:
    """
    1 qubit: X → measure c[0] → c_if(c[0]==1).X
    Net result is always |0⟩ regardless of noise model (identity after X,M,X).
    """
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, c.classical_registers[0][0])
    c.c_if(c.classical_registers[0][0], value=1).x(0)
    return c


# ──────────────────────────────────────────────────────────────
# 1. _probability_of_one helper
# ──────────────────────────────────────────────────────────────

def test_prob_one_pure_zero():
    state = np.array([1.0, 0.0], dtype=complex)
    assert abs(_probability_of_one(state, 0, 1) - 0.0) < 1e-12


def test_prob_one_pure_one():
    state = np.array([0.0, 1.0], dtype=complex)
    assert abs(_probability_of_one(state, 0, 1) - 1.0) < 1e-12


def test_prob_one_superposition():
    state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    assert abs(_probability_of_one(state, 0, 1) - 0.5) < 1e-12


def test_prob_one_two_qubit_first():
    # state[2] = 1.0 → index 2 = binary 10 → with reshape([2,2]):
    #   axis 0 (qubit 0) = 1, axis 1 (qubit 1) = 0
    state = np.zeros(4, dtype=complex)
    state[2] = 1.0
    assert abs(_probability_of_one(state, 0, 2) - 1.0) < 1e-12  # qubit 0 is 1
    assert abs(_probability_of_one(state, 1, 2) - 0.0) < 1e-12  # qubit 1 is 0


def test_prob_one_two_qubit_bell():
    # Bell state: (|00⟩ + |11⟩)/√2
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[3] = 1.0 / np.sqrt(2)
    assert abs(_probability_of_one(state, 0, 2) - 0.5) < 1e-12
    assert abs(_probability_of_one(state, 1, 2) - 0.5) < 1e-12


# ──────────────────────────────────────────────────────────────
# 2. _project_to_outcome helper
# ──────────────────────────────────────────────────────────────

def test_project_to_zero_from_plus():
    state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    projected = _project_to_outcome(state, 0, 0, 1)
    assert abs(projected[0] - 1.0) < 1e-10
    assert abs(projected[1] - 0.0) < 1e-10


def test_project_to_one_from_plus():
    state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    projected = _project_to_outcome(state, 0, 1, 1)
    assert abs(projected[0] - 0.0) < 1e-10
    assert abs(projected[1] - 1.0) < 1e-10


def test_project_renormalized():
    # After projection the result should be normalized
    state = np.array([0.6, 0.8], dtype=complex)
    for outcome in (0, 1):
        projected = _project_to_outcome(state, 0, outcome, 1)
        norm = float(np.sqrt(np.real(np.dot(projected.conj(), projected))))
        assert abs(norm - 1.0) < 1e-10


def test_project_degenerate_returns_unchanged():
    # Projecting |0⟩ onto outcome=1 is degenerate (prob=0); should return unchanged
    state = np.array([1.0, 0.0], dtype=complex)
    result = _project_to_outcome(state, 0, 1, 1)
    assert np.allclose(result, state)


def test_project_two_qubit():
    # Bell state: measure qubit 0 to 0 → collapse to |00⟩
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0 / np.sqrt(2)
    state[3] = 1.0 / np.sqrt(2)
    projected = _project_to_outcome(state, 0, 0, 2)
    assert abs(projected[0] - 1.0) < 1e-10
    assert abs(projected[3] - 0.0) < 1e-10


# ──────────────────────────────────────────────────────────────
# 3. SimulationResult / AggregateResult new fields
# ──────────────────────────────────────────────────────────────

def test_simulation_result_has_measurements_field():
    res = SimulationResult(counts={'0': 5}, measurements={'c[0]': [0, 1, 0]})
    assert res.measurements == {'c[0]': [0, 1, 0]}


def test_simulation_result_measurements_default_none():
    res = SimulationResult(counts={'0': 1})
    assert res.measurements is None


def test_simulation_result_acceptance_rate_field():
    res = SimulationResult(counts={'0': 8}, acceptance_rate=0.8)
    assert abs(res.acceptance_rate - 0.8) < 1e-10


def test_simulation_result_acceptance_rate_default_none():
    res = SimulationResult(counts={'0': 1})
    assert res.acceptance_rate is None


def test_aggregate_result_has_measurements_field():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.I, (0,))
    counts = np.zeros((1, 1), dtype=np.int64)
    agg = AggregateResult(
        counts_matrix=counts,
        n_shots=10,
        circuit=c,
        zero_error_shots=np.ones(10, dtype=bool),
        measurements={'c[0]': [0] * 10},
    )
    assert agg.measurements == {'c[0]': [0] * 10}


def test_aggregate_result_measurements_default_none():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.I, (0,))
    counts = np.zeros((1, 1), dtype=np.int64)
    agg = AggregateResult(
        counts_matrix=counts,
        n_shots=1,
        circuit=c,
        zero_error_shots=np.ones(1, dtype=bool),
    )
    assert agg.measurements is None


def test_aggregate_result_acceptance_rate_default_none():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.I, (0,))
    counts = np.zeros((1, 1), dtype=np.int64)
    agg = AggregateResult(
        counts_matrix=counts,
        n_shots=1,
        circuit=c,
        zero_error_shots=np.ones(1, dtype=bool),
    )
    assert agg.acceptance_rate is None


# ──────────────────────────────────────────────────────────────
# 4. StimTableauResult.shot_cbits
# ──────────────────────────────────────────────────────────────

def test_stim_result_shot_cbits_no_mcm():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.I, (0,))
    b = StimTableauBackend()
    res = b.run(c, n_shots=1, seed=0)
    stim_res = res.meta['stim_result']
    assert stim_res.shot_cbits is None


def test_stim_result_shot_cbits_after_x():
    """Measure qubit after X: first-shot cbit should be 1."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, c.classical_registers[0][0])
    b = StimTableauBackend()
    res = b.run(c, n_shots=5, seed=0)
    stim_res = res.meta['stim_result']
    assert stim_res.shot_cbits == {0: 1}


def test_stim_result_shot_cbits_always_zero():
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))
    c.measure(0, c.classical_registers[0][0])
    b = StimTableauBackend()
    res = b.run(c, n_shots=5, seed=0)
    stim_res = res.meta['stim_result']
    assert stim_res.shot_cbits == {0: 0}


# ──────────────────────────────────────────────────────────────
# 5. StimTableauBackend MCM
# ──────────────────────────────────────────────────────────────

def test_stim_measure_always_zero():
    """Measuring |0⟩ in Z basis always returns 0."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))
    c.measure(0, c.classical_registers[0][0])
    b = StimTableauBackend()
    res = b.run(c, n_shots=20, seed=5)
    assert res.measurements is not None
    assert res.measurements['c[0]'] == [0] * 20


def test_stim_measure_always_one():
    """Measuring |1⟩ in Z basis always returns 1."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, c.classical_registers[0][0])
    b = StimTableauBackend()
    res = b.run(c, n_shots=20, seed=5)
    assert res.measurements == {'c[0]': [1] * 20}


def test_stim_measure_plus_state_random():
    """Measuring |+⟩ should give ~50% 0 and ~50% 1."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, c.classical_registers[0][0])
    b = StimTableauBackend()
    res = b.run(c, n_shots=1000, seed=13)
    meas = res.measurements['c[0]']
    rate = sum(meas) / len(meas)
    assert 0.42 < rate < 0.58, f"Expected ~0.5, got {rate}"


def test_stim_conditional_not_met_gate_skipped():
    """ConditionalOp with condition=0 on a qubit measured as 0 should not fire."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))  # qubit stays |0>
    c.measure(0, c.classical_registers[0][0])  # measure -> 0
    # condition=1 but cbit=0 -> gate should NOT fire
    c.c_if(c.classical_registers[0][0], value=1).x(0)
    b = StimTableauBackend()
    res = b.run(c, n_shots=20, seed=0)
    # qubit never flipped, final readout = 0 every shot
    assert res.counts.get('0', 0) == 20, f"Unexpected counts: {res.counts}"


def test_stim_conditional_met_gate_fires():
    """ConditionalOp fires when cbit matches the required value."""
    c = _mcm_circuit_simple()  # X, measure->1, c_if(1).X -> back to |0>
    b = StimTableauBackend()
    res = b.run(c, n_shots=50, seed=0)
    assert res.counts.get('0', 0) == 50, f"Expected all-zero: {res.counts}"
    assert res.measurements == {'c[0]': [1] * 50}


def test_stim_measure_and_reset_qubit_zero():
    """After measure-and-reset on |1⟩, the qubit is back to |0⟩."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, c.classical_registers[0][0], reset=True)
    c.add_gate(gates.I, (0,))
    b = StimTableauBackend()
    res = b.run(c, n_shots=20, seed=0)
    # After X then measure+reset, qubit is |0>, final readout = 0
    assert res.counts.get('0', 0) == 20
    assert res.measurements == {'c[0]': [1] * 20}


def test_stim_measure_and_reset_already_zero():
    """measure-and-reset on |0⟩ leaves qubit as |0⟩ (reset condition is false)."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))
    c.measure(0, c.classical_registers[0][0], reset=True)
    b = StimTableauBackend()
    res = b.run(c, n_shots=20, seed=0)
    assert res.counts.get('0', 0) == 20
    assert res.measurements == {'c[0]': [0] * 20}


def test_stim_measurements_absent_without_mcm():
    """Without Measurement ops, result.measurements should be None."""
    c = Circuit(n_qubits=1)
    c.add_gate(gates.H, (0,))
    b = StimTableauBackend()
    res = b.run(c, n_shots=10, seed=0)
    assert res.measurements is None


def test_stim_measurements_length_equals_n_shots():
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, c.classical_registers[0][0])
    b = StimTableauBackend()
    n = 37
    res = b.run(c, n_shots=n, seed=0)
    assert len(res.measurements['c[0]']) == n


def test_stim_multi_qubit_measurement():
    """Measure both qubits of a Bell state — outcomes should be perfectly correlated."""
    c = Circuit(n_qubits=2)
    c.add_classical_register('c', 2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    c.measure(0, c.classical_registers[0][0])
    c.measure(1, c.classical_registers[0][1])
    b = StimTableauBackend()
    res = b.run(c, n_shots=200, seed=7)
    m0 = res.measurements['c[0]']
    m1 = res.measurements['c[1]']
    assert len(m0) == 200
    # Bell state: outcomes always agree
    assert all(a == b_ for a, b_ in zip(m0, m1)), "Bell state measurements must be correlated"


# ──────────────────────────────────────────────────────────────
# 6. TrajectoryBackend MCM
# ──────────────────────────────────────────────────────────────

def test_trajectory_measure_always_zero():
    """TrajectoryBackend: measuring |0⟩ always yields 0."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))
    c.measure(0, c.classical_registers[0][0])
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=20, seed=0)
    assert res.measurements == {'c[0]': [0] * 20}


def test_trajectory_measure_always_one():
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, c.classical_registers[0][0])
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=20, seed=0)
    assert res.measurements == {'c[0]': [1] * 20}


def test_trajectory_measure_collapses_state():
    """After measuring |+⟩ and projecting, the density matrix should be mixed."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, c.classical_registers[0][0])
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=1000, seed=42)
    dm = res.final_state
    # Mixed state: rho ≈ 0.5|0><0| + 0.5|1><1| = diag(0.5, 0.5)
    assert abs(dm[0, 0] - 0.5) < 0.05
    assert abs(dm[1, 1] - 0.5) < 0.05
    assert abs(dm[0, 1]) < 0.05  # off-diagonals ~0 due to collapse


def test_trajectory_conditional_correction():
    """X → measure → c_if(1).X should always restore |0⟩."""
    c = _mcm_circuit_simple()
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=200, seed=42)
    dm = res.final_state
    assert abs(dm[0, 0] - 1.0) < 0.02
    assert abs(dm[1, 1] - 0.0) < 0.02
    assert res.measurements == {'c[0]': [1] * 200}


def test_trajectory_conditional_not_met():
    """Condition not met: gate is skipped, state evolves as if no conditional."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))  # qubit stays |0>
    c.measure(0, c.classical_registers[0][0])  # always 0
    c.c_if(c.classical_registers[0][0], value=1).x(0)  # c[0]=0, so this skips
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=100, seed=0)
    dm = res.final_state
    assert abs(dm[0, 0] - 1.0) < 0.01


def test_trajectory_measure_and_reset():
    """After measure-and-reset on |1⟩, qubit returns to |0⟩."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.X, (0,))
    c.measure(0, c.classical_registers[0][0], reset=True)
    c.add_gate(gates.I, (0,))
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=100, seed=1)
    dm = res.final_state
    assert abs(dm[0, 0] - 1.0) < 0.01
    assert res.measurements == {'c[0]': [1] * 100}


def test_trajectory_measurements_none_without_mcm():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.H, (0,))
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=10, seed=0)
    assert res.measurements is None


def test_trajectory_measurements_length_equals_n_shots():
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, c.classical_registers[0][0])
    tb = TrajectoryBackend()
    n = 43
    res = tb.run(c, n_shots=n, seed=0)
    assert len(res.measurements['c[0]']) == n


def test_trajectory_bell_measurement_correlated():
    """Bell state: measuring qubit 0 and applying conditional X on qubit 1 teleports |0⟩."""
    c = Circuit(n_qubits=2)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    c.measure(0, c.classical_registers[0][0])
    c.c_if(c.classical_registers[0][0], value=1).x(1)
    # Qubit 1 should always end in |0> (teleport-style correction applied)
    tb = TrajectoryBackend()
    res = tb.run(c, n_shots=200, seed=77)
    dm = res.final_state
    # Trace out qubit 0 by summing the 2x2 reduced density matrix of qubit 1
    # rho has shape (4,4); we want reduced_dm for qubit 1
    # Use the excited_state_probability helper
    p1_q1 = res.excited_state_probability(1)
    assert p1_q1 < 0.05, f"Qubit 1 should be in |0>, got P(1)={p1_q1:.3f}"


# ──────────────────────────────────────────────────────────────
# 7. TrajectoryBackend.run_aggregate() MCM
# ──────────────────────────────────────────────────────────────

def test_run_aggregate_mcm_no_crash():
    """run_aggregate should handle MCM ops without crashing (was the M5 TODO)."""
    c = _mcm_circuit_simple()
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=50, seed=0)
    assert isinstance(agg, AggregateResult)


def test_run_aggregate_mcm_measurements_populated():
    c = _mcm_circuit_simple()
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=50, seed=0)
    assert agg.measurements is not None
    assert 'c[0]' in agg.measurements
    assert len(agg.measurements['c[0]']) == 50


def test_run_aggregate_mcm_measurements_values_correct():
    """X → measure → c_if(1).X: cbit always 1 in run_aggregate."""
    c = _mcm_circuit_simple()
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=50, seed=42)
    assert all(v == 1 for v in agg.measurements['c[0]'])


def test_run_aggregate_no_mcm_measurements_none():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.H, (0,))
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=10, seed=0)
    assert agg.measurements is None


def test_run_aggregate_acceptance_rate_none_by_default():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.I, (0,))
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=10, seed=0)
    assert agg.acceptance_rate is None


def test_run_aggregate_counts_matrix_shape_with_mcm():
    """counts_matrix shape should reflect total circuit.operations including MCM ops."""
    c = _mcm_circuit_simple()  # 3 ops: X, Measurement, ConditionalOp
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=10, seed=0)
    assert agg.counts_matrix.shape == (1, 3)


# ──────────────────────────────────────────────────────────────
# 8. ManyShotRunner with MCM (goes through StimTableauBackend)
# ──────────────────────────────────────────────────────────────

def test_many_shot_runner_mcm_no_crash():
    c = _mcm_circuit_simple()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=30, seed=0)
    assert isinstance(agg, AggregateResult)


def test_many_shot_runner_mcm_zero_errors():
    """Noiseless MCM circuit should have zero error events."""
    c = _mcm_circuit_simple()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=50, seed=42)
    assert agg.zero_error_fraction == 1.0
    assert agg.counts_matrix.sum() == 0


# ──────────────────────────────────────────────────────────────
# 9. SimulationResult repr with new fields
# ──────────────────────────────────────────────────────────────

def test_simulation_result_repr_includes_measurements():
    res = SimulationResult(measurements={'c[0]': [0, 1, 0]})
    r = repr(res)
    assert 'measurements' in r


def test_simulation_result_repr_includes_acceptance_rate():
    res = SimulationResult(acceptance_rate=0.75)
    r = repr(res)
    assert 'acceptance_rate' in r


def test_simulation_result_repr_no_measurements_when_none():
    res = SimulationResult(counts={'0': 1})
    r = repr(res)
    assert 'measurements' not in r


# ──────────────────────────────────────────────────────────────
# 10. ManyShotRunner post-selection (Stim path)
# ──────────────────────────────────────────────────────────────

def _h_measure_circuit() -> Circuit:
    """H → measure: cbit outcome is ~50/50."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, c.classical_registers[0][0])
    return c


def _always_zero_circuit() -> Circuit:
    """I → measure: cbit is always 0."""
    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.I, (0,))
    c.measure(0, c.classical_registers[0][0])
    return c


def test_post_select_accepts_subset():
    """~50% acceptance on a fair-coin measurement."""
    c = _h_measure_circuit()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=400, seed=42, post_select={0: 1})
    assert 0 < agg.n_shots < 400
    assert agg.acceptance_rate is not None
    assert 0.0 < agg.acceptance_rate < 1.0


def test_post_select_measurements_are_accepted_only():
    """Measurements dict must contain only accepted shots, all with the selected value."""
    c = _h_measure_circuit()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=400, seed=7, post_select={0: 1})
    assert agg.measurements is not None
    assert len(agg.measurements['c[0]']) == agg.n_shots
    assert all(v == 1 for v in agg.measurements['c[0]'])


def test_post_select_no_arg_no_acceptance_rate():
    """Without post_select, acceptance_rate must be None."""
    c = _h_measure_circuit()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=50, seed=0)
    assert agg.acceptance_rate is None


def test_post_select_all_shots_always_accepted():
    """post_select on the guaranteed outcome accepts every shot."""
    c = _always_zero_circuit()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=50, seed=0, post_select={0: 0})
    assert agg.n_shots == 50
    assert abs(agg.acceptance_rate - 1.0) < 1e-9


def test_post_select_invalid_cbit_index_raises():
    c = _h_measure_circuit()
    runner = ManyShotRunner()
    with pytest.raises(ValueError, match="not allocated"):
        runner.run(c, n_shots=10, seed=0, post_select={99: 1})


def test_post_select_invalid_value_2_raises():
    c = _h_measure_circuit()
    runner = ManyShotRunner()
    with pytest.raises(ValueError, match="0 or 1"):
        runner.run(c, n_shots=10, seed=0, post_select={0: 2})


def test_post_select_invalid_value_minus1_raises():
    c = _h_measure_circuit()
    runner = ManyShotRunner()
    with pytest.raises(ValueError, match="0 or 1"):
        runner.run(c, n_shots=10, seed=0, post_select={0: -1})


def test_post_select_zero_accepted_raises():
    """Impossible condition must raise ValueError, not return empty result."""
    c = _always_zero_circuit()
    runner = ManyShotRunner()
    with pytest.raises(ValueError, match="rejected all"):
        runner.run(c, n_shots=30, seed=0, post_select={0: 1})


def test_post_select_zero_error_fraction_over_accepted_shots():
    """zero_error_fraction is computed only over accepted shots."""
    c = _always_zero_circuit()
    runner = ManyShotRunner()
    agg = runner.run(c, n_shots=50, seed=0, post_select={0: 0})
    assert agg.zero_error_fraction == 1.0  # noiseless I gate


# ──────────────────────────────────────────────────────────────
# 11. TrajectoryBackend.run_aggregate post-selection
# ──────────────────────────────────────────────────────────────

def test_trajectory_post_select_basic():
    """TrajectoryBackend.run_aggregate with post_select accepts a subset."""
    c = _h_measure_circuit()
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=400, seed=13, post_select={0: 1})
    assert isinstance(agg, AggregateResult)
    assert 0 < agg.n_shots < 400
    assert 0.0 < agg.acceptance_rate < 1.0


def test_trajectory_post_select_measurements_all_selected_value():
    c = _h_measure_circuit()
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=400, seed=99, post_select={0: 1})
    assert agg.measurements is not None
    assert all(v == 1 for v in agg.measurements['c[0]'])
    assert len(agg.measurements['c[0]']) == agg.n_shots


def test_trajectory_post_select_zero_accepted_raises():
    c = _always_zero_circuit()
    tb = TrajectoryBackend()
    with pytest.raises(ValueError, match="rejected all"):
        tb.run_aggregate(c, n_shots=20, seed=0, post_select={0: 1})


def test_trajectory_post_select_no_arg_no_acceptance_rate():
    c = _h_measure_circuit()
    tb = TrajectoryBackend()
    agg = tb.run_aggregate(c, n_shots=50, seed=0)
    assert agg.acceptance_rate is None


# ──────────────────────────────────────────────────────────────
# 12. ManyShotRunner → TrajectoryBackend fallback with post_select
# ──────────────────────────────────────────────────────────────

def test_many_shot_runner_post_select_trajectory_fallback():
    """Non-Pauli noise routes to Trajectory; post_select must still work."""
    from noisiq.noise.amplitude_damping import AmplitudeDamping

    # AmplitudeDamping is a KrausChannel subclass — forces Trajectory routing.
    adc = AmplitudeDamping(T1=1000.0, t=50.0)

    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))       # op 0 — will carry amplitude-damping noise
    c.measure(0, c.classical_registers[0][0])  # op 1

    noise = {0: adc}
    runner = ManyShotRunner()
    # post_select on cbit=1; AD slightly biases toward |0>, so some fraction accepted
    agg = runner.run(c, n_shots=300, seed=5, post_select={0: 1})
    assert isinstance(agg, AggregateResult)
    assert agg.acceptance_rate is not None
    assert 0 < agg.n_shots <= 300
    if agg.measurements:
        assert all(v == 1 for v in agg.measurements['c[0]'])


# ──────────────────────────────────────────────────────────────
# 13. ManyShotRunner ConditionalOp.inner Clifford check
# ──────────────────────────────────────────────────────────────

def test_many_shot_runner_rejects_non_clifford_inside_conditional():
    """A T gate hidden inside ConditionalOp must be rejected by ManyShotRunner."""
    from noisiq.ir.circuit import Operation
    from noisiq.ir import gates

    c = Circuit(n_qubits=1)
    c.add_classical_register('c', 1)
    c.add_gate(gates.H, (0,))
    c.measure(0, c.classical_registers[0][0])
    # Manually append a ConditionalOp wrapping a T gate (non-Clifford).
    from noisiq.ir.classical import ConditionalOp as COP
    inner_t = Operation(gate=gates.T, qubits=(0,), t=2)
    c.operations.append(COP(inner=inner_t, condition=c.classical_registers[0][0], value=1))

    runner = ManyShotRunner()
    with pytest.raises(NotImplementedError, match="non-Clifford"):
        runner.run(c, n_shots=10, seed=0)
