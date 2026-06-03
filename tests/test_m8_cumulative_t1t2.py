"""
Tests for M8 — Cumulative T1/T2 at Measurement + MCM Heatmap Integration.

Covers:
  - _cumulative_decoherence new PauliError-based fields
    (cumulative_x_error_prob, cumulative_z_error_prob, p_wrong_measurement)
  - for_gate Measurement dispatch → _measurement_info
  - format_full / format_compact / format_for_display measurement mode
  - _build_gate_bboxes includes Measurement ops (M8) / excludes ConditionalOp
  - _p_wrong_for_average helper
  - _compute_downstream_impact average mode uses new cumulative p_wrong
  - End-to-end: plot_error_heatmap doesn't crash with MCM + noise_config
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
from noisiq.noise.pauli_error import PauliError
from noisiq.visualization.gate_info import GateInfoExtractor, _get_op_qubits
from noisiq.visualization.charts.heatmap import (
    _build_gate_bboxes,
    _compute_downstream_impact,
    _first_measurement_p_wrong,
    _p_wrong_for_average,
    plot_error_heatmap,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mcm_circuit():
    """2-qubit MCM circuit mirroring the M7 fixture.

    ops:
      0: H    q0  t=0
      1: Meas q0 → c[0]  t=1
      2: ConditionalOp(X q1, c[0]==1)  t=2
      3: CNOT q0,q1  t=3
    """
    reg = ClassicalRegister("c", 1)
    c0 = reg[0]
    circuit = Circuit(n_qubits=2)
    circuit.add_classical_register("c", 1)
    circuit.add_gate(G.H, (0,), t=0)
    circuit.measure(0, c0, t=1)
    inner_x = Operation(gate=G.X, qubits=(1,), t=2)
    cond_op = ConditionalOp(inner=inner_x, condition=c0, value=1)
    circuit.operations.append(cond_op)
    circuit.add_gate(G.CNOT, (0, 1), t=3)
    return circuit, c0


def _make_simple_circuit():
    """3-qubit circuit with no MCM."""
    circuit = Circuit(n_qubits=2)
    circuit.add_gate(G.H, (0,), t=0)
    circuit.add_gate(G.CNOT, (0, 1), t=1)
    circuit.add_gate(G.H, (0,), t=2)
    return circuit


def _make_aggregate_result(circuit, n_shots=100):
    n_qubits = circuit.n_qubits
    n_ops = len(circuit.operations)
    return AggregateResult(
        n_shots=n_shots,
        counts_matrix=np.zeros((n_qubits, n_ops), dtype=np.int64),
        zero_error_shots=np.ones(n_shots, dtype=bool),
        circuit=circuit,
    )


# ---------------------------------------------------------------------------
# Tests: _cumulative_decoherence — new PauliError fields
# ---------------------------------------------------------------------------

class TestCumulativeDecoherencePauliFields:
    """The three new fields are always present, defaulting to 0 when absent."""

    def test_new_fields_present_in_output(self):
        circuit, _ = _make_mcm_circuit()
        # op 0 is H on q0
        out = GateInfoExtractor._cumulative_decoherence(0, circuit, {})
        assert "cumulative_x_error_prob" in out["q0"]
        assert "cumulative_z_error_prob" in out["q0"]
        assert "p_wrong_measurement" in out["q0"]

    def test_new_fields_zero_with_empty_noise(self):
        circuit, _ = _make_mcm_circuit()
        out = GateInfoExtractor._cumulative_decoherence(1, circuit, {})
        assert out["q0"]["cumulative_x_error_prob"] == 0.0
        assert out["q0"]["cumulative_z_error_prob"] == 0.0
        assert out["q0"]["p_wrong_measurement"] == 0.0

    def test_pauli_error_channels_accumulate_x_y(self):
        """p_x + p_y from a gate on the measured qubit is accumulated."""
        circuit, _ = _make_mcm_circuit()
        # Put PauliError on op 0 (H, q0)
        noise = {0: PauliError(p_x=0.02, p_y=0.03, p_z=0.01)}
        # Query at op 1 (Measurement on q0) — should see the H's noise
        out = GateInfoExtractor._cumulative_decoherence(1, circuit, noise)
        q0 = out["q0"]
        assert abs(q0["cumulative_x_error_prob"] - 0.05) < 1e-9
        assert abs(q0["cumulative_z_error_prob"] - 0.01) < 1e-9

    def test_p_wrong_measurement_equals_sum_px_py(self):
        circuit, _ = _make_mcm_circuit()
        noise = {0: PauliError(p_x=0.04, p_y=0.02, p_z=0.01)}
        out = GateInfoExtractor._cumulative_decoherence(1, circuit, noise)
        q0 = out["q0"]
        assert abs(q0["p_wrong_measurement"] - q0["cumulative_x_error_prob"]) < 1e-9

    def test_p_wrong_measurement_capped_at_half(self):
        circuit, _ = _make_mcm_circuit()
        # Deliberately high px+py that should be capped at 0.5
        noise = {0: PauliError(p_x=0.35, p_y=0.30, p_z=0.05)}
        out = GateInfoExtractor._cumulative_decoherence(1, circuit, noise)
        assert out["q0"]["p_wrong_measurement"] <= 0.5

    def test_pauli_noise_only_accumulates_for_qubit_on_op(self):
        """Noise on a gate that does NOT touch q0 should not affect q0's tally."""
        circuit = Circuit(n_qubits=2)
        circuit.add_gate(G.H, (0,), t=0)       # op 0: q0
        circuit.add_gate(G.X, (1,), t=0)        # op 1: q1 only
        circuit.add_gate(G.H, (0,), t=1)        # op 2: q0
        # Noise on op 1 (q1 only) should not touch q0's accumulator
        noise = {0: PauliError(p_x=0.01, p_y=0.0, p_z=0.0),
                 1: PauliError(p_x=0.99, p_y=0.0, p_z=0.0)}
        out = GateInfoExtractor._cumulative_decoherence(2, circuit, noise)
        # q0 should only see op 0's noise (0.01), not op 1's
        assert abs(out["q0"]["cumulative_x_error_prob"] - 0.01) < 1e-9

    def test_multiple_gates_accumulate_correctly(self):
        circuit = Circuit(n_qubits=1)
        circuit.add_gate(G.H, (0,), t=0)
        circuit.add_gate(G.X, (0,), t=1)
        circuit.add_gate(G.H, (0,), t=2)
        noise = {
            0: PauliError(p_x=0.01, p_y=0.01, p_z=0.005),
            1: PauliError(p_x=0.02, p_y=0.01, p_z=0.003),
        }
        out = GateInfoExtractor._cumulative_decoherence(2, circuit, noise)
        expected = 0.01 + 0.01 + 0.02 + 0.01  # sum px+py ops 0 and 1
        assert abs(out["q0"]["cumulative_x_error_prob"] - expected) < 1e-9

    def test_measurement_op_itself_contributes_no_noise(self):
        """A Measurement has no PauliError channel; querying at its index is safe."""
        circuit, _ = _make_mcm_circuit()
        noise = {0: PauliError(p_x=0.01, p_y=0.01, p_z=0.0)}
        # Query at measurement op (idx=1)
        out = GateInfoExtractor._cumulative_decoherence(1, circuit, noise)
        assert out["q0"]["cumulative_x_error_prob"] == pytest.approx(0.02, abs=1e-9)

    def test_existing_t1_t2_fields_still_present(self):
        """The new PauliError fields don't displace the Kraus-channel fields."""
        circuit, _ = _make_mcm_circuit()
        out = GateInfoExtractor._cumulative_decoherence(0, circuit, {})
        q0 = out["q0"]
        assert "cumulative_t1_leakage" in q0
        assert "cumulative_t2_phase_decay" in q0
        assert "total_t1_exposure_ns" in q0
        assert "total_t2_exposure_ns" in q0


# ---------------------------------------------------------------------------
# Tests: for_gate Measurement dispatch → _measurement_info
# ---------------------------------------------------------------------------

class TestForGateMeasurementDispatch:
    def test_returns_measurement_mode(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit)
        assert info["mode"] == "measurement"

    def test_returns_qubit_and_cbit(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit)
        assert info["qubit"] == 0
        assert info["cbit"] == "c[0]"

    def test_returns_basis_and_timestep(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit)
        assert info["basis"] == "Z"
        assert info["timestep"] == 1

    def test_reset_field_present(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit)
        assert "reset" in info
        assert info["reset"] is False

    def test_p_wrong_measurement_with_noise(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.005)}
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit, noise_config=noise)
        assert "p_wrong_measurement" in info
        assert abs(info["p_wrong_measurement"] - 0.03) < 1e-9

    def test_p_wrong_measurement_absent_when_no_noise(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit)
        assert "p_wrong_measurement" not in info

    def test_cumulative_field_present_with_noise(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.01, p_y=0.01, p_z=0.0)}
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit, noise_config=noise)
        assert "cumulative" in info
        assert isinstance(info["cumulative"], dict)

    def test_non_measurement_op_returns_gate_mode(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        # op 0 is H — should return many_shot mode, not measurement
        info = GateInfoExtractor.for_gate(0, result, circuit=circuit)
        assert info["mode"] == "many_shot"

    def test_for_gate_uses_result_circuit_when_circuit_kwarg_omitted(self):
        """AggregateResult.circuit is used as a fallback when circuit= is not passed."""
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        # Omit circuit= — should still dispatch correctly via result.circuit
        info = GateInfoExtractor.for_gate(1, result)
        assert info["mode"] == "measurement"

    def test_op_idx_field_present(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit)
        assert info["op_idx"] == 1


# ---------------------------------------------------------------------------
# Tests: format_full / format_compact / format_for_display measurement mode
# ---------------------------------------------------------------------------

class TestFormatMeasurementMode:
    def _make_meas_info(self, with_noise=False):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.005)} if with_noise else None
        return GateInfoExtractor.for_gate(1, result, circuit=circuit, noise_config=noise)

    def test_format_full_does_not_crash_no_noise(self):
        info = self._make_meas_info(with_noise=False)
        text = GateInfoExtractor.format_full(info)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_full_contains_qubit_and_cbit(self):
        info = self._make_meas_info(with_noise=True)
        text = GateInfoExtractor.format_full(info)
        assert "q0" in text
        assert "c[0]" in text

    def test_format_full_contains_p_wrong_when_noise(self):
        info = self._make_meas_info(with_noise=True)
        text = GateInfoExtractor.format_full(info)
        assert "P(wrong" in text or "p_wrong" in text.lower()

    def test_format_full_no_gate_type_key_error(self):
        """format_full must not crash even though measurement info has no gate_type."""
        info = self._make_meas_info(with_noise=False)
        assert "gate_type" not in info
        # Should not raise KeyError
        _ = GateInfoExtractor.format_full(info)

    def test_format_compact_does_not_crash(self):
        info = self._make_meas_info(with_noise=True)
        text = GateInfoExtractor.format_compact(info)
        assert isinstance(text, str)
        assert "q0" in text

    def test_format_compact_contains_p_wrong(self):
        info = self._make_meas_info(with_noise=True)
        text = GateInfoExtractor.format_compact(info)
        assert "P(wrong)" in text

    def test_format_for_display_does_not_crash(self):
        info = self._make_meas_info(with_noise=True)
        text = GateInfoExtractor.format_for_display(info)
        assert isinstance(text, str)
        assert "q0" in text

    def test_format_for_display_no_gate_type_key_error(self):
        info = self._make_meas_info(with_noise=False)
        _ = GateInfoExtractor.format_for_display(info)

    def test_format_full_regular_gate_unaffected(self):
        """format_full for a regular gate still works after M8 changes."""
        circuit = _make_simple_circuit()
        result = _make_aggregate_result(circuit)
        info = GateInfoExtractor.for_gate(0, result, circuit=circuit)
        text = GateInfoExtractor.format_full(info)
        assert "Gate" in text
        assert "H" in text


# ---------------------------------------------------------------------------
# Tests: _build_gate_bboxes — Measurement ops now have bboxes
# ---------------------------------------------------------------------------

class TestBuildGateBboxesMeasurement:
    def test_measurement_included_in_bboxes(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        # ops[1] is Measurement — M8 adds a bbox for it
        assert 1 in bboxes

    def test_measurement_bbox_is_single_qubit_sized(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        xmin, xmax, ymin, ymax = bboxes[1]
        assert xmin < xmax
        assert ymin < ymax
        # Should be centred at t=1, width ~GATE_HALF_W*2
        assert abs((xmin + xmax) / 2 - 1.0) < 1e-9

    def test_conditional_op_included_in_bboxes(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        assert 2 in bboxes  # ConditionalOp at index 2 — now has a bbox spanning its inner gate

    def test_regular_ops_still_have_bboxes(self):
        circuit, _ = _make_mcm_circuit()
        bboxes = _build_gate_bboxes(circuit)
        assert 0 in bboxes  # H
        assert 3 in bboxes  # CNOT


# ---------------------------------------------------------------------------
# Tests: _p_wrong_for_average helper
# ---------------------------------------------------------------------------

class TestPWrongForAverage:
    def test_none_noise_config_returns_half(self):
        circuit, _ = _make_mcm_circuit()
        ops = circuit.operations
        p = _p_wrong_for_average(0, ops, circuit, noise_config=None)
        assert p == 0.5

    def test_no_measurement_downstream_returns_zero(self):
        circuit = _make_simple_circuit()
        ops = circuit.operations
        p = _p_wrong_for_average(0, ops, circuit, noise_config={})
        assert p == 0.0

    def test_with_pauli_noise_returns_correct_p(self):
        circuit, _ = _make_mcm_circuit()
        ops = circuit.operations
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.005)}
        p = _p_wrong_for_average(0, ops, circuit, noise)
        assert abs(p - 0.03) < 1e-9

    def test_matches_first_measurement_p_wrong_with_noise(self):
        """_p_wrong_for_average and _first_measurement_p_wrong give the same value."""
        circuit, _ = _make_mcm_circuit()
        ops = circuit.operations
        noise = {0: PauliError(p_x=0.04, p_y=0.02, p_z=0.01)}
        p_new = _p_wrong_for_average(0, ops, circuit, noise)
        p_old = _first_measurement_p_wrong(0, ops, noise)
        assert abs(p_new - p_old) < 1e-9

    def test_capped_at_half(self):
        circuit, _ = _make_mcm_circuit()
        ops = circuit.operations
        noise = {0: PauliError(p_x=0.35, p_y=0.30, p_z=0.05)}
        p = _p_wrong_for_average(0, ops, circuit, noise)
        assert p <= 0.5


# ---------------------------------------------------------------------------
# Tests: _compute_downstream_impact average mode integration
# ---------------------------------------------------------------------------

class TestComputeDownstreamImpactM8:
    def test_average_mode_between_qec_and_worst_case(self):
        """Average-mode result remains between qec and worst_case after M8."""
        circuit, _ = _make_mcm_circuit()
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.005)}
        impact_qec = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="qec")
        impact_worst = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="worst_case")
        impact_avg = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="average")
        for i in range(len(impact_qec)):
            lo = min(impact_qec[i], impact_worst[i])
            hi = max(impact_qec[i], impact_worst[i])
            assert lo - 1e-9 <= impact_avg[i] <= hi + 1e-9, (
                f"op {i}: avg={impact_avg[i]:.6f} not in [{lo:.6f}, {hi:.6f}]"
            )

    def test_average_mode_no_noise_uses_uninformative_prior(self):
        """With noise_config=None, p_wrong=0.5 ⇒ average == 0.5*(qec+worst)."""
        circuit, _ = _make_mcm_circuit()
        impact_qec = _compute_downstream_impact(circuit, noise_config=None, mcm_mode="qec")
        impact_worst = _compute_downstream_impact(circuit, noise_config=None, mcm_mode="worst_case")
        impact_avg = _compute_downstream_impact(circuit, noise_config=None, mcm_mode="average")
        expected = 0.5 * impact_qec + 0.5 * impact_worst
        np.testing.assert_allclose(impact_avg, expected, atol=1e-9)

    def test_no_mcm_modes_equal(self):
        """No MCM ops → all three modes produce identical impact arrays."""
        circuit = _make_simple_circuit()
        noise = {0: PauliError(p_x=0.01, p_y=0.0, p_z=0.0)}
        impact_qec = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="qec")
        impact_avg = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="average")
        np.testing.assert_array_equal(impact_qec, impact_avg)

    def test_zero_noise_average_same_as_qec(self):
        """With an empty noise_config, p_wrong=0 → average collapses to qec."""
        circuit, _ = _make_mcm_circuit()
        noise = {}
        impact_qec = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="qec")
        impact_avg = _compute_downstream_impact(circuit, noise_config=noise, mcm_mode="average")
        np.testing.assert_array_equal(impact_qec, impact_avg)


# ---------------------------------------------------------------------------
# Tests: End-to-end smoke tests
# ---------------------------------------------------------------------------

class TestEndToEndM8:
    def test_plot_error_heatmap_mcm_circuit_with_pauli_noise(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.005)}
        fig = plot_error_heatmap(result, circuit, noise_config=noise, mcm_mode="average")
        assert fig is not None
        plt.close("all")

    def test_plot_error_heatmap_all_modes_no_crash(self):
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.01, p_y=0.01, p_z=0.005)}
        for mode in ("qec", "worst_case", "average"):
            fig = plot_error_heatmap(result, circuit, noise_config=noise, mcm_mode=mode)
            assert fig is not None
            plt.close("all")

    def test_for_gate_on_measurement_called_in_hover_context(self):
        """Simulate the interactive_heatmap hover path: for_gate + format_full."""
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.0)}
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit, noise_config=noise)
        info["op_idx"] = 1
        text = GateInfoExtractor.format_full(info)
        assert "Measurement" in text
        assert "q0" in text

    def test_format_compact_measurement_in_annotate_path(self):
        """Simulate the annotate-mode hover path: for_gate + format_compact."""
        circuit, _ = _make_mcm_circuit()
        result = _make_aggregate_result(circuit)
        noise = {0: PauliError(p_x=0.02, p_y=0.01, p_z=0.0)}
        info = GateInfoExtractor.for_gate(1, result, circuit=circuit, noise_config=noise)
        info["op_idx"] = 1
        text = GateInfoExtractor.format_compact(info)
        assert "Measurement" in text or "q0" in text
