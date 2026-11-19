"""
Upgraded static circuit diagram renderer for NoisiQ.

Draws a clean Quirk-style timeline with:
- Tighter, evenly-spaced gate columns (independent of op.t layer index)
- Gate boxes colored by category from theme.py
- Qubit labels on the left, timestep index labels on top
- CNOT and CZ multi-qubit gate rendering
- Optional per-qubit Pauli error labels riding the wires
- (NEW) Optional per-qubit annotations on the right margin showing
  cumulative final Pauli (single-shot) or error rate (many-shot),
  plus an overall summary box.

Usage::

    fig, ax = plt.subplots(figsize=(10, 3))
    draw_circuit(ax, circuit)
    plt.show()
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..ir import Circuit, Operation
from ..ir.classical import Measurement as _Measurement, ConditionalOp as _ConditionalOp
from .pauli_frame_tracker import PauliFrame
from .theme import (
    ACTIVE_COLUMN_COLOR,
    ACTIVE_COLUMN_PAD_X,
    ACTIVE_COLUMN_PAD_Y,
    ERROR_COLOR,
    ERROR_LABEL_FONT_SIZE,
    FONT_FAMILY,
    GATE_EDGE_COLOR,
    GATE_EDGE_WIDTH,
    GATE_HALF_H,
    GATE_HALF_W,
    GATE_HIGHLIGHT_EDGE_COLOR,
    GATE_HIGHLIGHT_EDGE_WIDTH,
    GATE_SIZE,
    MEASUREMENT_EDGE_COLOR,
    PAULI_ERROR_BG_COLOR,
    PAULI_ERROR_BOX_PAD,
    PAULI_ERROR_LINEWIDTH,
    PAULI_ERROR_X_OFFSET,
    PAULI_ERROR_Y_OFFSET,
    QUBIT_LABEL_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    TIMESTEP_LABEL_COLOR,
    TIMESTEP_LABEL_FONT_SIZE,
    WIRE_COLOR,
    WIRE_LINEWIDTH,
    gate_color,
    draw_single_gate,
    draw_cnot,
    draw_cz,
    draw_swap,
    draw_cs,
    draw_ccz,
    draw_measurement,
    draw_classical_control_wire,
    draw_ry,
)

# Horizontal pitch between layer columns (centre-to-centre)
_X_PITCH: float = 1.0
# Extra wire margin on either side of the diagram
_WIRE_LEFT_MARGIN: float = 0.65
# Vertical gap between the top qubit wire and the step-label row
_LABEL_ROW_Y_OFFSET: float = 0.55

# ---------------------------------------------------------------------------
# NEW: right-margin annotation sizing
# ---------------------------------------------------------------------------
# Width reserved on the right side for per-qubit annotations.
# Widened from 0.35 → 2.0 only when annotations are requested; the original
# 0.35 stays the default when there's no annotation data.
_RIGHT_ANNOTATION_MARGIN: float = 2.0
_WIRE_RIGHT_MARGIN_PLAIN: float = 0.35

# Where the annotation text sits, relative to wire-end x.
# Must clear the floating Pauli error circle which lands at
# x_max + PAULI_ERROR_X_OFFSET (= x_max + 0.4) on the last frame.
_ANNOTATION_X_OFFSET: float = 0.85
# Font size for per-qubit summary labels
_ANNOTATION_FONT_SIZE: int = 9
# Colors
_ANNOTATION_BG: str = "#f6f6fa"
_ANNOTATION_EDGE: str = "#888888"


def draw_circuit(
    ax: plt.Axes,
    circuit: Circuit,
    pauli_frame: Optional[PauliFrame] = None,
    highlight_t: Optional[int] = None,
    title: str = "",
    per_qubit_annotations: Optional[list[str]] = None,
    summary_text: Optional[str] = None,
) -> None:
    """
    Draw a clean static circuit diagram on *ax*.

    Parameters
    ----------
    ax          : Matplotlib Axes to draw on.
    circuit     : NoisiQ Circuit (IR).
    pauli_frame : Current error state; when provided, non-I errors are shown
                  as floating labels on the qubit wires to the right of each
                  gate column.
    highlight_t : Layer index (op.t) to highlight with a column box.
    title       : Optional axes title.
    per_qubit_annotations
                : Optional list of length n_qubits. Each entry is a string
                  shown at the end of that qubit's wire (right margin).
                  Typical contents:
                    - single-shot mode: final cumulative Pauli, e.g. "Z"
                    - many-shot mode:   error rate, e.g. "err: 18.7%"
                  Pass None to disable.
    summary_text
                : Optional multi-line string shown in a corner box below the
                  per-qubit annotations. Used for circuit-wide stats like
                  "Zero-error: 41.0%" or "F = 0.873".
    """
    n_qubits = circuit.n_qubits
    ops = circuit.operations

    # Decide how much right margin we need based on whether annotations
    # are requested. This keeps existing call sites visually unchanged.
    wire_right_margin = (
        _RIGHT_ANNOTATION_MARGIN
        if (per_qubit_annotations is not None or summary_text is not None)
        else _WIRE_RIGHT_MARGIN_PLAIN
    )

    # Unique sorted layer indices → map each to a tight x-column.
    # All CircuitOp variants (Operation, Measurement, ConditionalOp) expose .t.
    layer_indices = sorted(set(op.t for op in ops)) if ops else []

    def _preceding_meas(cbit_idx: int, before_t: int) -> "_Measurement | None":
        """Return the most recent Measurement writing cbit_idx at or before before_t."""
        candidates = [
            op for op in ops
            if isinstance(op, _Measurement) and op.cbit.index == cbit_idx and op.t <= before_t
        ]
        return max(candidates, key=lambda m: m.t, default=None)
    layer_to_x: dict[int, float] = {
        t: i * _X_PITCH for i, t in enumerate(layer_indices)
    }
    n_cols = len(layer_indices) if layer_indices else 1
    x_max = (n_cols - 1) * _X_PITCH

    # --- Qubit wires --------------------------------------------------------
    wire_x0 = -_WIRE_LEFT_MARGIN
    wire_x1 = x_max + wire_right_margin
    for q in range(n_qubits):
        y = _qubit_y(q, n_qubits)
        ax.plot(
            [wire_x0, wire_x1], [y, y],
            color=WIRE_COLOR, linewidth=WIRE_LINEWIDTH, zorder=1,
        )

    # --- Active column highlight (drawn behind gates) -----------------------
    if highlight_t is not None and highlight_t in layer_to_x:
        cx = layer_to_x[highlight_t]
        col_patch = FancyBboxPatch(
            (cx - GATE_HALF_W - ACTIVE_COLUMN_PAD_X, -ACTIVE_COLUMN_PAD_Y),
            GATE_SIZE + 2 * ACTIVE_COLUMN_PAD_X,
            (n_qubits - 1) + 2 * ACTIVE_COLUMN_PAD_Y,
            boxstyle="round,pad=0.05",
            facecolor=ACTIVE_COLUMN_COLOR,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(col_patch)

    # --- Gates, Measurements, and Conditional Ops ---------------------------
    for op in ops:
        x = layer_to_x[op.t]
        is_hl = highlight_t is not None and op.t == highlight_t
        edge_lw = GATE_HIGHLIGHT_EDGE_WIDTH if is_hl else GATE_EDGE_WIDTH

        if isinstance(op, _Measurement):
            draw_measurement(ax, x, _qubit_y(op.qubit, n_qubits), op.cbit.name)
            continue

        if isinstance(op, _ConditionalOp):
            inner = op.inner
            inner_name = inner.gate.name.upper()
            fill = gate_color(inner_name)
            # Draw inner gate with the measurement edge color to signal it is conditional
            if inner_name in ("CNOT", "CX"):
                qc, qt = inner.qubits
                draw_cnot(ax, x, _qubit_y(qc, n_qubits), _qubit_y(qt, n_qubits), fill, edge_lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name == "CZ":
                q1, q2 = inner.qubits
                draw_cz(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), fill, edge_lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name == "SWAP":
                q1, q2 = inner.qubits
                draw_swap(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), fill, edge_lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name in ("CS", "CS_DAG"):
                q_ctrl, q_tgt = inner.qubits
                draw_cs(ax, x, _qubit_y(q_ctrl, n_qubits), _qubit_y(q_tgt, n_qubits), fill, MEASUREMENT_EDGE_COLOR, edge_lw)
            elif inner_name == "CCZ":
                q1, q2, q3 = inner.qubits
                draw_ccz(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), _qubit_y(q3, n_qubits), fill, edge_lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name.startswith("RY("):
                (q,) = inner.qubits
                draw_ry(ax, x, _qubit_y(q, n_qubits), inner_name, fill,
                        MEASUREMENT_EDGE_COLOR, edge_lw)
            else:
                (q,) = inner.qubits
                draw_single_gate(ax, x, _qubit_y(q, n_qubits), inner_name, fill,
                                 MEASUREMENT_EDGE_COLOR, edge_lw)
            # Draw classical control wire from the originating measurement
            meas = _preceding_meas(op.condition.index, op.t)
            if meas is not None:
                x_m = layer_to_x[meas.t]
                y_m = _qubit_y(meas.qubit, n_qubits)
                y_g = _qubit_y(inner.qubits[0], n_qubits)
                draw_classical_control_wire(ax, x_m, y_m, x, y_g)
            continue

        # Skip any unknown op type (e.g. test stubs, future IR extensions).
        if not isinstance(op, Operation):
            continue

        # Plain Operation
        name = op.gate.name.upper()
        edge_col = GATE_HIGHLIGHT_EDGE_COLOR if is_hl else GATE_EDGE_COLOR
        fill = gate_color(name)

        if name in ("CNOT", "CX"):
            q_ctrl, q_tgt = op.qubits
            draw_cnot(ax, x, _qubit_y(q_ctrl, n_qubits), _qubit_y(q_tgt, n_qubits), fill, edge_lw)
        elif name == "CZ":
            q1, q2 = op.qubits
            draw_cz(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), fill, edge_lw)
        elif name == "SWAP":
            q1, q2 = op.qubits
            draw_swap(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), fill, edge_lw)
        elif name in ("CS", "CS_DAG"):
            q_ctrl, q_tgt = op.qubits
            draw_cs(ax, x, _qubit_y(q_ctrl, n_qubits), _qubit_y(q_tgt, n_qubits), fill, edge_col, edge_lw)
        elif name == "CCZ":
            q1, q2, q3 = op.qubits
            draw_ccz(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), _qubit_y(q3, n_qubits), fill, edge_lw)
        elif name in ("I", "IDLE"):
            pass  # identity and idle are invisible; IDLE decoherence shown via wire halos
        elif name.startswith("RY("):
            (q,) = op.qubits
            draw_ry(ax, x, _qubit_y(q, n_qubits), name, fill, edge_col, edge_lw)
        else:
            (q,) = op.qubits
            draw_single_gate(ax, x, _qubit_y(q, n_qubits), name, fill, edge_col, edge_lw)

    # --- Qubit labels (left side) ------------------------------------------
    for q in range(n_qubits):
        ax.text(
            wire_x0 - 0.08, _qubit_y(q, n_qubits),
            f"q{q}",
            va="center", ha="right",
            fontsize=QUBIT_LABEL_FONT_SIZE,
            color=QUBIT_LABEL_COLOR,
            fontfamily=FONT_FAMILY,
        )

    # --- Timestep labels (top row) -----------------------------------------
    for t, x in layer_to_x.items():
        ax.text(
            x, (n_qubits - 1) + _LABEL_ROW_Y_OFFSET,
            f"t{t}",
            va="bottom", ha="center",
            fontsize=TIMESTEP_LABEL_FONT_SIZE, color=TIMESTEP_LABEL_COLOR,
            fontfamily=FONT_FAMILY,
        )

    # --- Pauli error labels floating on wires ------------------------------
    if pauli_frame is not None:
        pauli_str = pauli_frame.get_pauli_string()
        for q in range(n_qubits):
            label = pauli_str[q]
            if label == "I":
                continue
            # Error steps with the animation — always at the current column's x.
            if highlight_t is not None and highlight_t in layer_to_x:
                err_x = layer_to_x[highlight_t] + PAULI_ERROR_X_OFFSET
            else:
                err_x = x_max + wire_right_margin * 0.5

            ax.text(
                err_x, _qubit_y(q, n_qubits) + PAULI_ERROR_Y_OFFSET,
                label,
                va="center", ha="center",
                fontsize=ERROR_LABEL_FONT_SIZE, color=ERROR_COLOR, fontweight="bold",
                fontfamily=FONT_FAMILY,
                bbox=dict(
                    boxstyle=f"circle,pad={PAULI_ERROR_BOX_PAD}",
                    facecolor=PAULI_ERROR_BG_COLOR,
                    edgecolor=ERROR_COLOR,
                    linewidth=PAULI_ERROR_LINEWIDTH,
                ),
                zorder=5,
            )

    # --- NEW: per-qubit annotations on right margin ------------------------
    # One small text box per qubit, just past the end of each wire. Shows
    # either the cumulative final Pauli (single-shot) or the per-qubit
    # error rate (many-shot). Set by the caller via per_qubit_annotations.
    if per_qubit_annotations is not None:
        if len(per_qubit_annotations) != n_qubits:
            raise ValueError(
                f"per_qubit_annotations length ({len(per_qubit_annotations)}) "
                f"must match n_qubits ({n_qubits})"
            )
        annotation_x = x_max + _ANNOTATION_X_OFFSET
        for q, text in enumerate(per_qubit_annotations):
            if not text:
                continue
            ax.text(
                annotation_x, _qubit_y(q, n_qubits),
                text,
                va="center", ha="left",
                fontsize=_ANNOTATION_FONT_SIZE,
                color="#333333",
                fontfamily=FONT_FAMILY,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=_ANNOTATION_BG,
                    edgecolor=_ANNOTATION_EDGE,
                    linewidth=0.8,
                ),
                zorder=4,
            )

    # --- NEW: overall summary box (bottom-right corner) --------------------
    if summary_text:
        # Place the summary box below the bottom wire, right-aligned with
        # the per-qubit annotations.
        summary_x = x_max + _ANNOTATION_X_OFFSET
        summary_y = -0.4   # below the lowest qubit (which sits at y=0)
        ax.text(
            summary_x, summary_y,
            summary_text,
            va="top", ha="left",
            fontsize=_ANNOTATION_FONT_SIZE,
            color="#222222",
            fontfamily=FONT_FAMILY,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=_ANNOTATION_BG,
                edgecolor=_ANNOTATION_EDGE,
                linewidth=1.0,
            ),
            zorder=4,
        )

    # --- Axes styling -------------------------------------------------------
    if title:
        ax.set_title(title, fontsize=12, fontfamily=FONT_FAMILY)

    ax.set_xlim(wire_x0 - 0.4, x_max + wire_right_margin + 0.15)
    ax.set_ylim(-0.8, (n_qubits - 1) + _LABEL_ROW_Y_OFFSET + 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _qubit_y(qubit: int, n_qubits: int) -> float:
    """Y-coordinate for a qubit wire (top qubit = highest y)."""
    return float(n_qubits - 1 - qubit)
