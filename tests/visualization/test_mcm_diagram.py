"""Tests for Phase 9 MCM visualization — measurement boxes, classical wires,
and Pauli-frame measurement collapse.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from noisiq.ir import Circuit, gates
from noisiq.ir.circuit import Operation
from noisiq.ir.classical import Measurement, ConditionalOp
from noisiq.visualization.pauli_frame_tracker import PauliFrame
from noisiq.visualization.theme import draw_measurement, draw_classical_control_wire
from noisiq.visualization.circuit_diagram import draw_circuit
from noisiq.visualization.drawer import draw_circuit_with_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_mcm_circuit():
    """H on q0, measure q0 → c[0], conditional X on q1 if c[0]==1."""
    c = Circuit(n_qubits=2)
    reg = c.add_classical_register("c", 1)
    cbit = reg[0]
    c.add_gate(gates.H, (0,))
    c.measure(0, cbit)
    c.c_if(cbit, value=1).x(1)
    return c, cbit


# ---------------------------------------------------------------------------
# theme.py drawing primitives
# ---------------------------------------------------------------------------

class TestDrawMeasurementPrimitive:
    def test_runs_without_error(self):
        fig, ax = plt.subplots()
        draw_measurement(ax, x=1.0, y=0.0)
        plt.close(fig)

    def test_with_cbit_label(self):
        fig, ax = plt.subplots()
        draw_measurement(ax, x=1.0, y=0.0, cbit_label="c[0]")
        plt.close(fig)

    def test_adds_patch_to_axes(self):
        fig, ax = plt.subplots()
        before = len(ax.patches)
        draw_measurement(ax, x=0.0, y=0.0)
        assert len(ax.patches) > before
        plt.close(fig)


class TestDrawClassicalControlWire:
    def test_same_qubit_runs_without_error(self):
        fig, ax = plt.subplots()
        draw_classical_control_wire(ax, x_meas=0.0, y_meas=1.0,
                                    x_gate=2.0, y_gate=1.0)
        plt.close(fig)

    def test_different_qubits_runs_without_error(self):
        fig, ax = plt.subplots()
        draw_classical_control_wire(ax, x_meas=0.0, y_meas=2.0,
                                    x_gate=3.0, y_gate=0.0)
        plt.close(fig)

    def test_adds_lines_to_axes(self):
        fig, ax = plt.subplots()
        before = len(ax.lines)
        draw_classical_control_wire(ax, x_meas=0.0, y_meas=2.0,
                                    x_gate=3.0, y_gate=0.0)
        # Expect 3 line segments: vertical drop, horizontal run, vertical rise
        assert len(ax.lines) == before + 3
        plt.close(fig)


# ---------------------------------------------------------------------------
# PauliFrame.apply_measurement()
# ---------------------------------------------------------------------------

class TestApplyMeasurement:
    def test_x_error_flips_outcome(self):
        frame = PauliFrame(2)
        frame.inject_error(0, "X")
        flipped = frame.apply_measurement(0)
        assert flipped is True

    def test_y_error_flips_outcome(self):
        frame = PauliFrame(2)
        frame.inject_error(0, "Y")  # Y = XZ
        flipped = frame.apply_measurement(0)
        assert flipped is True

    def test_z_error_does_not_flip_outcome(self):
        frame = PauliFrame(2)
        frame.inject_error(0, "Z")
        flipped = frame.apply_measurement(0)
        assert flipped is False

    def test_clean_qubit_does_not_flip(self):
        frame = PauliFrame(2)
        flipped = frame.apply_measurement(0)
        assert flipped is False

    def test_x_component_cleared_after_measurement(self):
        frame = PauliFrame(2)
        frame.inject_error(0, "X")
        frame.apply_measurement(0)
        assert not frame.x[0]
        assert not frame.z[0]

    def test_z_component_cleared_after_measurement(self):
        """Z-type error also gets cleared — the qubit is projected regardless."""
        frame = PauliFrame(2)
        frame.inject_error(0, "Z")
        frame.apply_measurement(0)
        assert not frame.x[0]
        assert not frame.z[0]

    def test_other_qubit_unaffected(self):
        frame = PauliFrame(3)
        frame.inject_error(1, "X")
        frame.apply_measurement(0)
        assert frame.x[1]  # qubit 1 untouched


# ---------------------------------------------------------------------------
# circuit_diagram.draw_circuit() with MCM ops
# ---------------------------------------------------------------------------

class TestDrawCircuitMCM:
    def test_measurement_circuit_renders_without_error(self):
        c, _ = _simple_mcm_circuit()
        fig, ax = plt.subplots()
        draw_circuit(ax, c)
        plt.close(fig)

    def test_pure_measurement_circuit(self):
        """Single qubit, single measurement — no conditional gate."""
        c = Circuit(n_qubits=1)
        reg = c.add_classical_register("m", 1)
        c.add_gate(gates.H, (0,))
        c.measure(0, reg[0])

        fig, ax = plt.subplots()
        draw_circuit(ax, c)
        plt.close(fig)

    def test_conditional_op_circuit_renders_without_error(self):
        c, _ = _simple_mcm_circuit()
        fig, ax = plt.subplots()
        draw_circuit(ax, c)
        assert ax is not None
        plt.close(fig)

    def test_measurement_and_plain_gates_coexist(self):
        """Circuit with both Operation gates and Measurement ops."""
        c = Circuit(n_qubits=2)
        reg = c.add_classical_register("s", 1)
        c.add_gate(gates.H, (0,))
        c.add_gate(gates.CNOT, (0, 1))
        c.measure(0, reg[0])

        fig, ax = plt.subplots()
        draw_circuit(ax, c)
        plt.close(fig)

    def test_highlight_works_on_mcm_circuit(self):
        c, _ = _simple_mcm_circuit()
        fig, ax = plt.subplots()
        # highlight_t on a Measurement layer should not raise
        meas_t = next(op.t for op in c.operations if isinstance(op, Measurement))
        draw_circuit(ax, c, highlight_t=meas_t)
        plt.close(fig)


# ---------------------------------------------------------------------------
# drawer.draw_circuit_with_labels() with MCM ops
# ---------------------------------------------------------------------------

class TestDrawCircuitWithLabelsMCM:
    def test_measurement_circuit_renders_without_error(self):
        c, _ = _simple_mcm_circuit()
        fig, ax = plt.subplots()
        draw_circuit_with_labels(ax, c)
        plt.close(fig)

    def test_conditional_op_renders_without_error(self):
        c, _ = _simple_mcm_circuit()
        fig, ax = plt.subplots()
        draw_circuit_with_labels(ax, c)
        assert ax is not None
        plt.close(fig)
