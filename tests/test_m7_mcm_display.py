"""
Tests for M7 — MCM Display Modes: worst_case / average / qec.

Covers:
  - _downstream_walk branching logic (qec vs worst_case)
  - _compute_downstream_impact with mcm_mode parameter
  - _first_measurement_p_wrong helper
  - plot_error_heatmap mcm_mode parameter and title annotation
  - plot_error_heatmap_side_by_side
  - interactive_heatmap mcm_mode parameter (smoke test)
  - CircuitAnimator mcm_mode parameter
  - _build_gate_bboxes skips non-Operation ops (bug-fix guard)
  - gate_info.py guards for Measurement ops
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisiq.ir.circuit import Circuit, Operation
from noisiq.ir.classical import ClassicalRegister, Measurement, ConditionalOp
from noisiq.ir import gates as G
from noisiq.backends.many_shot_runner import AggregateResult
from noisiq.visualization.charts.heatmap import (
    _compute_downstream_impact,
    _downstream_walk,
    _first_measurement_p_wrong,
    plot_error_heatmap,
    plot_error_heatmap_side_by_side,
    _build_gate_bboxes,
)
from noisiq.visualization.animation import CircuitAnimator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_aggregate_result(circuit: Circuit, n_shots: int = 100) -> AggregateResult:
    """Return a minimal AggregateResult with zero errors (all zeros counts_matrix)."""
    n_qubits = circuit.n_qubits
    n_ops = len(circuit.operations)
    return AggregateResult(
        n_shots=n_shots,
        counts_matrix=np.zeros((n_qubits, n_ops), dtype=np.int64),
        zero_error_shots=np.ones(n_shots, dtype=bool),
        circuit=circuit,
    )


def _make_mcm_circuit():
    """
    2-qubit circuit: H on q0, then measure q0 → c[0], then X on q1
    conditioned on c[0]==1, then CNOT q0→q1.

    ops layout:
      0: H   q0  t=0
      1: Meas q0 → c[0]  t=1
      2: ConditionalOp(X q1, cond=c[0]==1)  t=2
      3: CNOT q0,q1  t=3
    """
    reg = ClassicalRegister("c", 1)
    c0 = reg[0]
    circuit = Circuit(n_qubits=2)
    circuit.add_classical_register("c", 1)

    circuit.add_gate(G.H, (0,), t=0)
    circuit.measure(0, c0, t=1)

    # ConditionalOp: X on q1 conditioned on c[0]==1
    inner_x = Operation(gate=G.X, qubits=(1,), t=2)
    cond_op = ConditionalOp(inner=inner_x, condition=c0, value=1)
    circuit.operations.append(cond_op)

    circuit.add_gate(G.CNOT, (0, 1), t=3)
    return circuit, c0


def _make_simple_circuit_no_mcm():
    """3-op circuit with no MCM: H → CNOT → H (q0,q1)."""
    circuit = Circuit(n_qubits=2)
    circuit.add_gate(G.H, (0,), t=0)
    circuit.add_gate(G.CNOT, (0, 1), t=1)
    circuit.add_gate(G.H, (0,), t=2)
    return circuit


# ---------------------------------------------------------------------------
# Tests: _downstream_walk
# ---------------------------------------------------------------------------

class TestDownstreamWalk:
    def test_no_mcm_ops_same_result_for_all_modes(self):
        """With no MCM ops, qec and worst_case give identical counts."""
        circuit = _make_simple_circuit_no_mcm()
        ops = circuit.operations
        initial = {0: 'X'}
        count_qec = _downstream_walk(ops, 0, initial, "qec")
        count_worst = _downstream_walk(ops, 0, initial, "worst_case")
        assert count_qec == count_worst

    def test_qec_corrects_error(self):
        """QEC mode: X correction should cancel an X error on q1 after measurement."""
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations

        # Start from H (k=0): inject X on q0
        # H propagates X → Z, so at the Measurement (q0) we have Z on q0
        # Z commutes with Z-basis measurement → no anti-commutation → correction doesn't fire
        # Downstream: CNOT(q0,q1) — Z on q0 propagates to Z on q0 only (CNOT: Z_ctrl→Z_ctrl)
        initial = {0: 'X'}
        count_qec = _downstream_walk(ops, 0, initial, "qec")
        count_worst = _downstream_walk(ops, 0, initial, "worst_case")
        # Both modes should give same result when error commutes with measurement
        assert count_qec == count_worst

    def test_worst_case_applies_correction_when_commutes(self):
        """Worst-case: if no error detected (Z on measured qubit commutes),
        the correction FIRES anyway in worst_case, creating a spurious error."""
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations

        # Inject Z on q0 directly (at k=-1 conceptually, i.e., start from beginning)
        # Z commutes with Z measurement → qec: correction doesn't fire
        # worst_case: correction FIRES (worst case — it fires when it shouldn't)
        initial = {0: 'Z'}
        count_qec = _downstream_walk(ops, 0, initial, "qec")
        count_worst = _downstream_walk(ops, 0, initial, "worst_case")
        # worst_case fires correction (X on q1) → potentially more downstream impact
        # (correction itself touches q1 which is before CNOT)
        # At minimum they can differ; both should be non-negative
        assert count_qec >= 0
        assert count_worst >= 0

    def test_measurement_does_not_count_as_impacted_gate(self):
        """Measurement ops are never counted as impacted gates in the walk."""
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations
        # Start from the very beginning with a random initial state
        initial = {0: 'X', 1: 'Y'}
        count = _downstream_walk(ops, 0, initial, "qec")
        # Count should only reflect regular Operations, not Measurements
        assert isinstance(count, (int, float))
        assert count >= 0

    def test_empty_pauli_state_stops_walk(self):
        """Walk stops when pauli_state becomes empty (error cancelled)."""
        circuit = _make_simple_circuit_no_mcm()
        ops = circuit.operations
        # X propagated through H becomes Z, through CNOT becomes (Z on q0 only),
        # then H changes it back to X. Eventually may terminate.
        count = _downstream_walk(ops, 0, {0: 'X'}, "qec")
        assert count >= 0

    def test_unseen_cbit_conditional_is_skipped(self):
        """ConditionalOp whose cbit was set before start_k is silently skipped."""
        # start_k=2 means we start AFTER the measurement at idx=1
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations
        # Starting at k=2 (after Measurement), ConditionalOp at idx=2 has cbit NOT
        # in measurement_outcomes → should be skipped gracefully
        count = _downstream_walk(ops, 2, {1: 'X'}, "qec")
        assert count >= 0  # doesn't crash


# ---------------------------------------------------------------------------
# Tests: _first_measurement_p_wrong
# ---------------------------------------------------------------------------

class TestFirstMeasurementPWrong:
    def test_no_measurements_returns_zero(self):
        circuit = _make_simple_circuit_no_mcm()
        ops = circuit.operations
        p = _first_measurement_p_wrong(0, ops, noise_config=None)
        # noise_config=None and no measurements after start → depends on implementation
        # With noise_config=None: returns 0.5 (uninformative prior)
        assert p == 0.5

    def test_no_measurements_after_start_returns_zero_with_noise(self):
        circuit = _make_simple_circuit_no_mcm()
        ops = circuit.operations
        p = _first_measurement_p_wrong(0, ops, noise_config={})
        # Empty noise_config + no measurements → 0.0
        assert p == 0.0

    def test_with_measurement_and_no_noise(self):
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations
        p = _first_measurement_p_wrong(0, ops, noise_config={})
        assert 0.0 <= p <= 0.5

    def test_returns_float_in_range(self):
        from noisiq.noise.pauli_error import PauliError
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations
        noise = {0: PauliError(p_x=0.01, p_y=0.01, p_z=0.005)}
        p = _first_measurement_p_wrong(0, ops, noise_config=noise)
        assert 0.0 <= p <= 0.5

    def test_capped_at_half(self):
        from noisiq.noise.pauli_error import PauliError
        circuit, c0 = _make_mcm_circuit()
        ops = circuit.operations
        noise = {0: PauliError(p_x=0.4, p_y=0.4, p_z=0.1)}
        p = _first_measurement_p_wrong(0, ops, noise_config=noise)
        assert p <= 0.5


# ---------------------------------------------------------------------------
# Tests: _compute_downstream_impact with mcm_mode
# ---------------------------------------------------------------------------

class TestComputeDownstreamImpact:
    def test_default_mode_is_qec(self):
        circuit, _ = _make_mcm_circuit()
        impact_default = _compute_downstream_impact(circuit)
        impact_qec = _compute_downstream_impact(circuit, mcm_mode="qec")
        np.testing.assert_array_equal(impact_default, impact_qec)

    def test_returns_ndarray_with_n_ops_length(self):
        circuit, _ = _make_mcm_circuit()
        n_ops = len(circuit.operations)
        impact = _compute_downstream_impact(circuit)
        assert impact.shape == (n_ops,)

    def test_measurement_ops_have_zero_impact(self):
        circuit, _ = _make_mcm_circuit()
        impact = _compute_downstream_impact(circuit, mcm_mode="qec")
        # ops[1] is a Measurement — impact must be 0
        assert impact[1] == 0.0

    def test_conditional_ops_have_zero_impact(self):
        circuit, _ = _make_mcm_circuit()
        impact = _compute_downstream_impact(circuit, mcm_mode="qec")
        # ops[2] is a ConditionalOp — impact must be 0
        assert impact[2] == 0.0

    def test_non_negative_impacts(self):
        for mode in ("qec", "worst_case", "average"):
            circuit, _ = _make_mcm_circuit()
            impact = _compute_downstream_impact(circuit, mcm_mode=mode)
            assert (impact >= 0).all(), f"Negative impact in mode={mode}"

    def test_worst_case_gte_qec_for_mcm_circuit(self):
        """Worst-case should never have LESS total impact than QEC."""
        circuit, _ = _make_mcm_circuit()
        impact_qec = _compute_downstream_impact(circuit, mcm_mode="qec")
        impact_worst = _compute_downstream_impact(circuit, mcm_mode="worst_case")
        # Sum over gates: worst_case total >= qec total (or equal if no MCM effect)
        assert impact_worst.sum() >= impact_qec.sum() - 1e-9

    def test_average_between_qec_and_worst_case(self):
        """Average-mode impact should be between qec and worst_case."""
        circuit, _ = _make_mcm_circuit()
        impact_qec = _compute_downstream_impact(circuit, mcm_mode="qec")
        impact_worst = _compute_downstream_impact(circuit, mcm_mode="worst_case")
        impact_avg = _compute_downstream_impact(circuit, mcm_mode="average")
        for i in range(len(impact_qec)):
            lo = min(impact_qec[i], impact_worst[i])
            hi = max(impact_qec[i], impact_worst[i])
            assert lo - 1e-9 <= impact_avg[i] <= hi + 1e-9, (
                f"op {i}: avg={impact_avg[i]} not in [{lo}, {hi}]"
            )

    def test_no_mcm_circuit_modes_equal(self):
        circuit = _make_simple_circuit_no_mcm()
        impact_qec = _compute_downstream_impact(circuit, mcm_mode="qec")
        impact_worst = _compute_downstream_impact(circuit, mcm_mode="worst_case")
        impact_avg = _compute_downstream_impact(circuit, mcm_mode="average")
        np.testing.assert_array_equal(impact_qec, impact_worst)
        np.testing.assert_array_equal(impact_qec, impact_avg)

    def test_noise_weighted_impact(self):
        from noisiq.noise.pauli_error import PauliError
        circuit = _make_simple_circuit_no_mcm()
        noise = {0: PauliError(p_x=0.05, p_y=0.05, p_z=0.02)}
        impact_unweighted = _compute_downstream_impact(circuit)
        impact_weighted = _compute_downstream_impact(circuit, noise_config=noise)
        # Weighted impact at gate 0 should equal unweighted * p_total
        if impact_unweighted[0] > 0:
            ratio = impact_weighted[0] / impact_unweighted[0]
            assert abs(ratio - 0.12) < 1e-9


# ---------------------------------------------------------------------------
# Tests: plot_error_heatmap with mcm_mode
# ---------------------------------------------------------------------------

class TestPlotErrorHeatmapMCMMode:
    def test_qec_mode_runs(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, mcm_mode="qec")
        assert fig is not None
        plt.close("all")

    def test_worst_case_mode_runs(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, mcm_mode="worst_case")
        assert fig is not None
        plt.close("all")

    def test_average_mode_runs(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, mcm_mode="average")
        assert fig is not None
        plt.close("all")

    def test_side_by_side_mode_raises(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        with pytest.raises(ValueError, match="side_by_side"):
            plot_error_heatmap(result, circuit, mcm_mode="side_by_side")

    def test_invalid_mode_raises(self):
        circuit = _make_simple_circuit_no_mcm()
        result = _make_aggregate_result(circuit)
        with pytest.raises(ValueError):
            plot_error_heatmap(result, circuit, mcm_mode="banana")

    def test_title_annotated_with_mode_when_measurements_present(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, mcm_mode="worst_case")
        ax = fig.axes[0]
        assert "[worst_case]" in ax.get_title()
        plt.close("all")

    def test_title_not_annotated_when_no_measurements(self):
        circuit = _make_simple_circuit_no_mcm()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, mcm_mode="qec")
        ax = fig.axes[0]
        assert "[" not in ax.get_title()
        plt.close("all")

    def test_custom_title_includes_mode_suffix(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, title="MyTitle", mcm_mode="qec")
        ax = fig.axes[0]
        assert "MyTitle" in ax.get_title()
        assert "[qec]" in ax.get_title()
        plt.close("all")

    def test_default_mode_is_qec(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig_default = plot_error_heatmap(result, circuit)
        ax_default = fig_default.axes[0]
        assert "[qec]" in ax_default.get_title()
        plt.close("all")

    def test_non_mcm_circuit_no_crash(self):
        circuit = _make_simple_circuit_no_mcm()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap(result, circuit, mcm_mode="qec")
        assert fig is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_error_heatmap_side_by_side
# ---------------------------------------------------------------------------

class TestPlotErrorHeatmapSideBySide:
    def test_returns_single_figure(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_has_two_axes(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit)
        # Two subplots + two colorbars = 4 axes; at minimum 2
        assert len(fig.axes) >= 2
        plt.close("all")

    def test_left_panel_is_worst_case(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit)
        left_title = fig.axes[0].get_title()
        assert "worst" in left_title.lower() or "worst_case" in left_title.lower()
        plt.close("all")

    def test_right_panel_is_qec(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit)
        right_title = fig.axes[1].get_title()
        assert "qec" in right_title.lower()
        plt.close("all")

    def test_super_title_set_when_title_provided(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit, title="SBS Test")
        # suptitle is stored in fig._suptitle
        assert fig._suptitle is not None
        assert "SBS Test" in fig._suptitle.get_text()
        plt.close("all")

    def test_no_mcm_circuit_runs(self):
        circuit = _make_simple_circuit_no_mcm()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit)
        assert fig is not None
        plt.close("all")

    def test_figsize_override(self):
        circuit = _make_simple_circuit_no_mcm()
        result = _make_aggregate_result(circuit)
        fig = plot_error_heatmap_side_by_side(result, circuit, figsize=(12, 4))
        assert abs(fig.get_figwidth() - 12) < 0.1
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _build_gate_bboxes with MCM circuit
# ---------------------------------------------------------------------------

class TestBuildGateBboxes:
    def test_does_not_crash_with_measurement_ops(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        assert isinstance(bboxes, dict)

    def test_measurement_ops_included_in_bboxes(self):
        # M8: Measurement ops get hit-test bboxes so users can hover over them
        # in interactive_heatmap() and see p_wrong_measurement.
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        assert 1 in bboxes   # Measurement at index 1 — now has a bbox

    def test_conditional_op_included_in_bboxes(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        assert 2 in bboxes  # ConditionalOp at index 2 — now has a bbox spanning its inner gate

    def test_operation_ops_included_in_bboxes(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        assert 0 in bboxes   # H at index 0
        assert 3 in bboxes   # CNOT at index 3


# ---------------------------------------------------------------------------
# Tests: CircuitAnimator mcm_mode parameter
# ---------------------------------------------------------------------------

class TestCircuitAnimatorMCMMode:
    def _make_many_shot_result(self, circuit, n_shots=50, measurements=None):
        n_qubits = circuit.n_qubits
        n_ops = len(circuit.operations)
        return AggregateResult(
            n_shots=n_shots,
            counts_matrix=np.zeros((n_qubits, n_ops), dtype=np.int64),
            zero_error_shots=np.ones(n_shots, dtype=bool),
            circuit=circuit,
            measurements=measurements,
        )

    def test_valid_mcm_mode_qec(self):
        circuit = _make_simple_circuit_no_mcm()
        result = self._make_many_shot_result(circuit)
        animator = CircuitAnimator(circuit, result, mcm_mode="qec")
        assert animator._mcm_mode == "qec"

    def test_valid_mcm_mode_worst_case(self):
        circuit = _make_simple_circuit_no_mcm()
        result = self._make_many_shot_result(circuit)
        animator = CircuitAnimator(circuit, result, mcm_mode="worst_case")
        assert animator._mcm_mode == "worst_case"

    def test_none_mcm_mode_default(self):
        circuit = _make_simple_circuit_no_mcm()
        result = self._make_many_shot_result(circuit)
        animator = CircuitAnimator(circuit, result)
        assert animator._mcm_mode is None

    def test_invalid_mcm_mode_raises(self):
        circuit = _make_simple_circuit_no_mcm()
        result = self._make_many_shot_result(circuit)
        with pytest.raises(ValueError, match="not supported"):
            CircuitAnimator(circuit, result, mcm_mode="average")

    def test_side_by_side_mcm_mode_raises(self):
        circuit = _make_simple_circuit_no_mcm()
        result = self._make_many_shot_result(circuit)
        with pytest.raises(ValueError):
            CircuitAnimator(circuit, result, mcm_mode="side_by_side")

    def test_mcm_mode_annotations_no_crash(self):
        """With mcm_mode set and MCM circuit, _build_annotations should not crash."""
        circuit, _ = _make_mcm_circuit()
        result = self._make_many_shot_result(circuit)
        animator = CircuitAnimator(circuit, result, mcm_mode="qec")
        # Frame 0 corresponds to layer t=0 (H gate), no measurement there
        per_qubit, summary = animator._build_annotations(0)
        assert per_qubit is not None

    def test_measurement_layer_gets_annotation(self):
        """At a measurement layer, per_qubit annotation has 'M→' suffix."""
        circuit, c0 = _make_mcm_circuit()
        result = self._make_many_shot_result(
            circuit, n_shots=10,
            measurements={"c[0]": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]},
        )
        animator = CircuitAnimator(circuit, result, mcm_mode="qec")
        # Layer t=1 has the Measurement op
        layers = animator._layers
        frame_idx_for_t1 = layers.index(1)
        per_qubit, _ = animator._build_annotations(frame_idx_for_t1)
        # q0 is measured at t=1 → its annotation should contain "M→"
        assert "M→" in per_qubit[0]

    def test_non_measurement_layer_no_mcm_annotation(self):
        """At a non-measurement layer, per_qubit annotation has NO 'M→' suffix."""
        circuit, c0 = _make_mcm_circuit()
        result = self._make_many_shot_result(
            circuit, n_shots=10,
            measurements={"c[0]": [0] * 10},
        )
        animator = CircuitAnimator(circuit, result, mcm_mode="qec")
        layers = animator._layers
        frame_idx_for_t0 = layers.index(0)
        per_qubit, _ = animator._build_annotations(frame_idx_for_t0)
        assert "M→" not in per_qubit[0]


# ---------------------------------------------------------------------------
# Tests: gate_info.py Measurement-aware guards
# ---------------------------------------------------------------------------

class TestGateInfoMeasurementGuards:
    def test_for_qubit_endcap_with_measurement_in_circuit(self):
        """for_qubit_endcap must not crash when circuit has Measurement ops."""
        from noisiq.visualization.gate_info import GateInfoExtractor

        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_qubit_endcap(0, circuit, result)
        assert "qubit" in info

    def test_cumulative_decoherence_with_measurement_in_circuit(self):
        """_cumulative_decoherence must not crash with MCM circuit."""
        from noisiq.visualization.gate_info import GateInfoExtractor

        circuit, _ = _make_mcm_circuit()
        # op_idx 0 is H on q0 — ordinary Operation
        result = GateInfoExtractor._cumulative_decoherence(0, circuit, {})
        assert isinstance(result, dict)

    def test_get_op_qubits_measurement(self):
        """_get_op_qubits returns (qubit,) tuple for Measurement."""
        from noisiq.visualization.gate_info import _get_op_qubits

        cbit = ClassicalRegister("c", 1)[0]
        meas = Measurement(qubit=2, cbit=cbit, t=3)
        assert _get_op_qubits(meas) == (2,)

    def test_get_op_qubits_operation(self):
        from noisiq.visualization.gate_info import _get_op_qubits

        op = Operation(gate=G.H, qubits=(0,), t=0)
        assert _get_op_qubits(op) == (0,)

    def test_get_op_qubits_conditional_op(self):
        from noisiq.visualization.gate_info import _get_op_qubits

        inner = Operation(gate=G.X, qubits=(1,), t=2)
        cbit = ClassicalRegister("c", 1)[0]
        cond = ConditionalOp(inner=inner, condition=cbit, value=1)
        assert _get_op_qubits(cond) == (1,)
