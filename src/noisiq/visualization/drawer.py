import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import offset_copy
from typing import Optional
from ..ir import Circuit
from ..ir.circuit import Operation
from ..ir.classical import Measurement as _Measurement, ConditionalOp as _ConditionalOp
from .pauli_frame_tracker import PauliFrame
from .theme import (
    gate_color,
    WIRE_COLOR,
    WIRE_LINEWIDTH,
    GATE_HALF_W,
    GATE_SIZE,
    GATE_EDGE_COLOR,
    GATE_EDGE_WIDTH,
    MEASUREMENT_EDGE_COLOR,
    METRIC_BOX_ALPHA,
    METRIC_BOX_EDGE_COLOR,
    METRIC_BOX_FACE_COLOR,
    METRIC_BOX_FONT_FAMILY,
    METRIC_BOX_FONT_SIZE,
    METRIC_BOX_PAD,
    METRIC_BOX_TEXT_COLOR,
    METRIC_BOX_X,
    METRIC_BOX_YOFFSET_PT,
    QUBIT_LABEL_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    ERROR_COLOR,
    ERROR_LABEL_FONT_SIZE,
    PAULI_ERROR_BG_COLOR,
    PAULI_ERROR_BOX_PAD,
    PAULI_ERROR_LINEWIDTH,
    PAULI_ERROR_X_OFFSET,
    PAULI_ERROR_Y_OFFSET,
    ACTIVE_COLUMN_COLOR,
    ACTIVE_COLUMN_PAD_X,
    ACTIVE_COLUMN_PAD_Y,
    draw_single_gate,
    draw_cnot,
    draw_cz,
    draw_swap,
    draw_cs,
    draw_ccz,
    draw_measurement,
    draw_classical_control_wire,
)


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
    all_ops = circuit.operations
    n_layers = (max(op.t for op in all_ops) + 1) if all_ops else 1

    def _preceding_meas(cbit_idx: int, before_t: int):
        """Return the most recent Measurement writing cbit_idx at or before before_t."""
        candidates = [
            op for op in all_ops
            if isinstance(op, _Measurement) and op.cbit.index == cbit_idx and op.t <= before_t
        ]
        return max(candidates, key=lambda m: m.t, default=None)

    # Wires
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot(
            [-0.5, n_layers - 0.5], [y, y],
            linewidth=WIRE_LINEWIDTH, color=WIRE_COLOR, zorder=1,
        )

    # Gates, Measurements, and Conditional Ops
    for op in all_ops:
        x = op.t
        is_highlighted = (highlight_t is not None and op.t == highlight_t)
        lw = 2.5 if is_highlighted else GATE_EDGE_WIDTH

        if isinstance(op, _Measurement):
            draw_measurement(ax, x, n_qubits - 1 - op.qubit, op.cbit.name)
            continue

        if isinstance(op, _ConditionalOp):
            inner = op.inner
            inner_name = inner.gate.name.upper()
            fill = gate_color(inner_name)
            if inner_name in ("CNOT", "CX"):
                qc, qt = inner.qubits
                draw_cnot(ax, x, n_qubits - 1 - qc, n_qubits - 1 - qt, fill, lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name == "CZ":
                q1, q2 = inner.qubits
                draw_cz(ax, x, n_qubits - 1 - q1, n_qubits - 1 - q2, fill, lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name == "SWAP":
                q1, q2 = inner.qubits
                draw_swap(ax, x, n_qubits - 1 - q1, n_qubits - 1 - q2, fill, lw, edge=MEASUREMENT_EDGE_COLOR)
            elif inner_name in ("CS", "CS_DAG"):
                q_ctrl, q_tgt = inner.qubits
                draw_cs(ax, x, n_qubits - 1 - q_ctrl, n_qubits - 1 - q_tgt, fill, MEASUREMENT_EDGE_COLOR, lw)
            elif inner_name == "CCZ":
                q1, q2, q3 = inner.qubits
                draw_ccz(ax, x, n_qubits - 1 - q1, n_qubits - 1 - q2, n_qubits - 1 - q3, fill, lw, edge=MEASUREMENT_EDGE_COLOR)
            else:
                (q,) = inner.qubits
                draw_single_gate(ax, x, n_qubits - 1 - q, inner_name, fill,
                                 MEASUREMENT_EDGE_COLOR, lw)
            meas = _preceding_meas(op.condition.index, op.t)
            if meas is not None:
                y_m = n_qubits - 1 - meas.qubit
                y_g = n_qubits - 1 - inner.qubits[0]
                draw_classical_control_wire(ax, meas.t, y_m, x, y_g)
            continue

        # Skip any unknown op type (e.g. test stubs, future IR extensions).
        if not isinstance(op, Operation):
            continue

        # Plain Operation
        name = op.gate.name.upper()
        fill = gate_color(name)
        edge = GATE_EDGE_COLOR

        if name in ("CNOT", "CX"):
            q_ctrl, q_tgt = op.qubits
            draw_cnot(ax, x, n_qubits - 1 - q_ctrl, n_qubits - 1 - q_tgt, fill, lw)
        elif name == "CZ":
            q1, q2 = op.qubits
            draw_cz(ax, x, n_qubits - 1 - q1, n_qubits - 1 - q2, fill, lw)
        elif name == "SWAP":
            q1, q2 = op.qubits
            draw_swap(ax, x, n_qubits - 1 - q1, n_qubits - 1 - q2, fill, lw)
        elif name in ("CS", "CS_DAG"):
            q_ctrl, q_tgt = op.qubits
            draw_cs(ax, x, n_qubits - 1 - q_ctrl, n_qubits - 1 - q_tgt, fill, edge, lw)
        elif name == "CCZ":
            q1, q2, q3 = op.qubits
            draw_ccz(ax, x, n_qubits - 1 - q1, n_qubits - 1 - q2, n_qubits - 1 - q3, fill, lw)
        elif name == "I":
            pass
        else:
            (q,) = op.qubits
            draw_single_gate(ax, x, n_qubits - 1 - q, name, fill, edge, lw)

    # Active column highlight
    if highlight_t is not None:
        col_x = highlight_t - GATE_HALF_W - ACTIVE_COLUMN_PAD_X
        col_y = -ACTIVE_COLUMN_PAD_Y
        col_w = GATE_SIZE + 2 * ACTIVE_COLUMN_PAD_X
        col_h = (n_qubits - 1) + 2 * ACTIVE_COLUMN_PAD_Y
        col_box = FancyBboxPatch(
            (col_x, col_y), col_w, col_h,
            boxstyle="round,pad=0.05",
            facecolor=ACTIVE_COLUMN_COLOR,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(col_box)

    # Qubit labels and Pauli error labels
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
            ax.text(
                float(highlight_t) + PAULI_ERROR_X_OFFSET, y + PAULI_ERROR_Y_OFFSET, p_label,
                va="center", ha="center",
                fontsize=ERROR_LABEL_FONT_SIZE, color=ERROR_COLOR, fontweight="bold",
                bbox=dict(
                    boxstyle=f"circle,pad={PAULI_ERROR_BOX_PAD}",
                    facecolor=PAULI_ERROR_BG_COLOR,
                    edgecolor=ERROR_COLOR,
                    linewidth=PAULI_ERROR_LINEWIDTH,
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


# ---------------------------------------------------------------------------
# Public: metric box overlay
# ---------------------------------------------------------------------------

def add_metric_box(
    ax: plt.Axes,
    *,
    fidelity: float | None = None,
    global_purity: float | None = None,
    avg_qubit_purity: float | None = None,
    zero_error: float | None = None,
    output_success: float | None = None,
    stabilizer_fidelity: float | None = None,
    ghz_subspace_population: float | None = None,
    ghz_branch_coherence_real: float | None = None,
    ghz_branch_coherence_abs: float | None = None,
    hardware_fidelity: float | None = None,
    hardware_n_qubits: int | None = None,
    hardware_delta: float | None = None,
    t1_us: float | None = None,
    t2_us: float | None = None,
    label: str | None = None,
    fidelity_label: str = "F target",
    hardware_fidelity_label: str | None = None,
) -> None:
    """Add a poster-style metric box to the lower-right corner of *ax*.

    All keyword arguments are optional; only non-None values appear in the box.
    Position and styling are controlled by the METRIC_BOX_* constants in theme.py.
    """
    lines: list[str] = []
    if label is not None:
        lines.append(label)
    if fidelity is not None:
        lines.append(f"{fidelity_label:<15}= {fidelity:.4f}")
    if global_purity is not None:
        lines.append(f"Tr(ρ²) global  = {global_purity:.4f}")
    if avg_qubit_purity is not None:
        lines.append(f"avg Tr(ρq²)    = {avg_qubit_purity:.4f}")
    if zero_error is not None:
        lines.append(f"zero-error     = {zero_error:.4f}")
    if output_success is not None:
        lines.append(f"parity success = {output_success:.4f}")
    if stabilizer_fidelity is not None:
        lines.append(f"stabilizer F   = {stabilizer_fidelity:.4f}")
    if ghz_subspace_population is not None:
        lines.append(f"GHZ subspace   = {ghz_subspace_population:.4f}")
    if ghz_branch_coherence_real is not None:
        lines.append(f"Re branch coh. = {ghz_branch_coherence_real:.4f}")
    if ghz_branch_coherence_abs is not None:
        lines.append(f"|branch coh.|  = {ghz_branch_coherence_abs:.4f}")
    if hardware_fidelity is not None:
        if hardware_fidelity_label is None:
            nq_tag = f"({hardware_n_qubits}q)" if hardware_n_qubits is not None else ""
            hardware_fidelity_label = f"hardware F{nq_tag}"
        lines.append(f"{hardware_fidelity_label:<15}= {hardware_fidelity:.4f}")
    if hardware_delta is not None:
        lines.append(f"Δ NoisiQ-hw    = {hardware_delta:+.4f}")
    if t1_us is not None:
        lines.append(f"T1 idle        = {t1_us:.0f} µs")
    if t2_us is not None:
        lines.append(f"T2 idle        = {t2_us:.0f} µs")

    if not lines:
        return

    # Shift METRIC_BOX_YOFFSET_PT points below the axes bottom edge so the box
    # always clears the x-tick labels at any figure size (points are absolute,
    # unlike axes-fraction coordinates which scale with the axes pixel height).
    trans = offset_copy(
        ax.transAxes,
        fig=ax.get_figure(),
        x=0,
        y=-METRIC_BOX_YOFFSET_PT,
        units="points",
    )
    ax.text(
        METRIC_BOX_X,
        0.0,  # axes bottom edge; transform shifts it down by METRIC_BOX_YOFFSET_PT
        "\n".join(lines),
        transform=trans,
        ha="right",
        va="top",
        fontsize=METRIC_BOX_FONT_SIZE,
        family=METRIC_BOX_FONT_FAMILY,
        color=METRIC_BOX_TEXT_COLOR,
        clip_on=False,
        bbox={
            "boxstyle": f"round,pad={METRIC_BOX_PAD}",
            "facecolor": METRIC_BOX_FACE_COLOR,
            "edgecolor": METRIC_BOX_EDGE_COLOR,
            "alpha": METRIC_BOX_ALPHA,
        },
        zorder=1000,
    )
