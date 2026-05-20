"""
Upgraded static circuit diagram renderer for NoisiQ.

Draws a clean Quirk-style timeline with:
- Tighter, evenly-spaced gate columns (independent of op.t layer index)
- Gate boxes colored by category from theme.py
- Qubit labels on the left, timestep index labels on top
- CNOT and CZ multi-qubit gate rendering
- Optional per-qubit Pauli error labels riding the wires

Usage::

    fig, ax = plt.subplots(figsize=(10, 3))
    draw_circuit(ax, circuit)
    plt.show()
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..ir import Circuit
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
    GATE_HALF_W,
    GATE_HIGHLIGHT_EDGE_COLOR,
    GATE_HIGHLIGHT_EDGE_WIDTH,
    GATE_SIZE,
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
)

# Horizontal pitch between layer columns (centre-to-centre)
_X_PITCH: float = 1.0
# Extra wire margin on either side of the diagram
_WIRE_LEFT_MARGIN: float = 0.65
_WIRE_RIGHT_MARGIN: float = 0.35
# Vertical gap between the top qubit wire and the step-label row
_LABEL_ROW_Y_OFFSET: float = 0.55


def draw_circuit(
    ax: plt.Axes,
    circuit: Circuit,
    pauli_frame: Optional[PauliFrame] = None,
    highlight_t: Optional[int] = None,
    title: str = "",
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
    """
    n_qubits = circuit.n_qubits
    ops = circuit.operations

    # Unique sorted layer indices → map each to a tight x-column
    layer_indices = sorted(set(op.t for op in ops)) if ops else []
    layer_to_x: dict[int, float] = {
        t: i * _X_PITCH for i, t in enumerate(layer_indices)
    }
    n_cols = len(layer_indices) if layer_indices else 1
    x_max = (n_cols - 1) * _X_PITCH

    # --- Qubit wires --------------------------------------------------------
    wire_x0 = -_WIRE_LEFT_MARGIN
    wire_x1 = x_max + _WIRE_RIGHT_MARGIN
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

    # --- Gates --------------------------------------------------------------
    for op in ops:
        x = layer_to_x[op.t]
        name = op.gate.name.upper()
        is_hl = highlight_t is not None and op.t == highlight_t
        edge_col = GATE_HIGHLIGHT_EDGE_COLOR if is_hl else GATE_EDGE_COLOR
        edge_lw = GATE_HIGHLIGHT_EDGE_WIDTH if is_hl else GATE_EDGE_WIDTH
        fill = gate_color(name)

        if name in ("CNOT", "CX"):
            q_ctrl, q_tgt = op.qubits
            draw_cnot(ax, x, _qubit_y(q_ctrl, n_qubits), _qubit_y(q_tgt, n_qubits), fill, edge_lw)
        elif name == "CZ":
            q1, q2 = op.qubits
            draw_cz(ax, x, _qubit_y(q1, n_qubits), _qubit_y(q2, n_qubits), fill, edge_lw)
        elif name == "I":
            pass
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
                err_x = x_max + _WIRE_RIGHT_MARGIN * 0.5

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

    # --- Axes styling -------------------------------------------------------
    if title:
        ax.set_title(title, fontsize=12, fontfamily=FONT_FAMILY)

    ax.set_xlim(wire_x0 - 0.4, x_max + _WIRE_RIGHT_MARGIN + 0.15)
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
