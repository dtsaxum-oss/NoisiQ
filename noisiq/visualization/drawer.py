import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Optional, List
from ..ir import Circuit
from .pauli_frame_tracker import PauliFrame
from .theme import (
    gate_color,
    WIRE_COLOR,
    WIRE_LINEWIDTH,
    GATE_HALF_W,
    GATE_HALF_H,
    GATE_SIZE,
    GATE_EDGE_COLOR,
    GATE_EDGE_WIDTH,
    GATE_LABEL_COLOR,
    GATE_LABEL_FONT_SIZE,
    QUBIT_LABEL_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    ERROR_COLOR,
    TARGET_CIRCLE_RADIUS,
    CONTROL_DOT_SIZE,
    ACTIVE_COLUMN_COLOR,
)

_COLUMN_PAD: float = 0.1   # extra space beyond gate edges inside the column highlight


def draw_circuit_with_labels(
    ax: plt.Axes,
    circuit: Circuit,
    pauli_frame: Optional[PauliFrame] = None,
    highlight_t: Optional[int] = None,
    note: str = "",
):
    """
    Draws a themed circuit timeline with Pauli error labels floating on wires.
    Gate positions use op.t (layer index) so parallel gates share a column.
    """
    n_qubits = circuit.n_qubits
    n_layers = (max(op.t for op in circuit.operations) + 1) if circuit.operations else 1

    # Wires
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot(
            [-0.5, n_layers - 0.5], [y, y],
            linewidth=WIRE_LINEWIDTH, color=WIRE_COLOR, zorder=1,
        )

    # Gates
    for op in circuit.operations:
        x = op.t
        name = op.gate.name.upper()
        fill = gate_color(name)
        is_highlighted = (highlight_t is not None and op.t == highlight_t)
        edge = GATE_EDGE_COLOR
        lw = 2.5 if is_highlighted else GATE_EDGE_WIDTH

        if name in ("CNOT", "CX"):
            q_ctrl, q_tgt = op.qubits
            y_ctrl = n_qubits - 1 - q_ctrl
            y_tgt = n_qubits - 1 - q_tgt
            _draw_cnot(ax, x, y_ctrl, y_tgt, fill, edge, lw)

        elif name == "CZ":
            q1, q2 = op.qubits
            y1 = n_qubits - 1 - q1
            y2 = n_qubits - 1 - q2
            _draw_cz(ax, x, y1, y2, fill, edge, lw)

        elif name == "I":
            pass

        else:
            (q,) = op.qubits
            y = n_qubits - 1 - q
            _draw_single_qubit_gate(ax, x, y, name, fill, edge, lw)

    # Active column highlight — drawn before gates so it sits behind them
    if highlight_t is not None:
        col_x = highlight_t - GATE_HALF_W - _COLUMN_PAD
        col_y = -_COLUMN_PAD
        col_w = GATE_SIZE + 2 * _COLUMN_PAD
        col_h = (n_qubits - 1) + 2 * _COLUMN_PAD
        col_box = FancyBboxPatch(
            (col_x, col_y), col_w, col_h,
            boxstyle="round,pad=0.05",
            facecolor=ACTIVE_COLUMN_COLOR,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(col_box)

    # Qubit labels and Pauli error labels on wires
    pauli_string = pauli_frame.get_pauli_string() if pauli_frame else "I" * n_qubits
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.text(
            -0.75, y, f"q{q}",
            va="center", ha="right",
            fontsize=QUBIT_LABEL_FONT_SIZE, color=QUBIT_LABEL_COLOR,
        )
        p_label = pauli_string[q]
        if p_label != "I" and highlight_t is not None:
            x_err = highlight_t + 0.55
            ax.text(
                x_err, y, p_label,
                va="center", ha="center",
                fontsize=10, color=ERROR_COLOR, fontweight="bold",
                bbox=dict(
                    boxstyle="circle,pad=0.15",
                    facecolor="#FFECEC",
                    edgecolor=ERROR_COLOR,
                    linewidth=1,
                ),
                zorder=5,
            )

    if note:
        ax.set_title(note, fontsize=12)

    ax.set_xlim(-1.2, n_layers - 0.2)
    ax.set_ylim(-1, n_qubits)
    ax.set_xticks(range(n_layers))
    ax.set_yticks([])
    ax.set_frame_on(False)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _draw_single_qubit_gate(ax, x, y, name, fill, edge, lw):
    box = FancyBboxPatch(
        (x - GATE_HALF_W, y - GATE_HALF_H),
        GATE_HALF_W * 2, GATE_HALF_H * 2,
        boxstyle="round,pad=0.02",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, name,
        ha="center", va="center",
        fontsize=GATE_LABEL_FONT_SIZE, color=GATE_LABEL_COLOR,
        fontweight="bold", zorder=4,
    )


def _draw_cnot(ax, x, y_ctrl, y_tgt, fill, edge, lw):
    ax.plot([x, x], [y_ctrl, y_tgt], linewidth=lw, color=fill, zorder=2)
    ax.scatter([x], [y_ctrl], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
    circle = plt.Circle(
        (x, y_tgt), TARGET_CIRCLE_RADIUS,
        fill=True, facecolor="white", edgecolor=fill, linewidth=lw, zorder=3,
    )
    ax.add_patch(circle)
    ax.plot([x, x], [y_tgt - TARGET_CIRCLE_RADIUS, y_tgt + TARGET_CIRCLE_RADIUS],
            linewidth=lw, color=fill, zorder=4)
    ax.plot([x - TARGET_CIRCLE_RADIUS, x + TARGET_CIRCLE_RADIUS], [y_tgt, y_tgt],
            linewidth=lw, color=fill, zorder=4)


def _draw_cz(ax, x, y1, y2, fill, edge, lw):
    ax.plot([x, x], [y1, y2], linewidth=lw, color=fill, zorder=2)
    ax.scatter([x, x], [y1, y2], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
