"""
Aggregate error heatmap overlaid on the circuit diagram.

Renders a circuit "print-out" with gate boxes and qubit wires, with a
halo-color effect around each gate box scaled by error intensity.

Week 4–6: halo intensity = error_rate_matrix (error count / n_shots).
Week 7+:  halo intensity = downstream_impact_matrix (true propagation count).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ...ir.circuit import Circuit
from ...backends.many_shot_runner import AggregateResult
from ..theme import (
    WIRE_COLOR,
    WIRE_LINEWIDTH,
    GATE_EDGE_COLOR,
    GATE_EDGE_WIDTH,
    GATE_LABEL_COLOR,
    GATE_LABEL_FONT_SIZE,
    QUBIT_LABEL_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    GATE_HALF_W,
    GATE_HALF_H,
    HALO_PAD,
    HALO_ALPHA,
    TARGET_CIRCLE_RADIUS,
    CONTROL_DOT_SIZE,
    gate_color,
    halo_color,
    get_halo_colormap,
)


_SCALE: float = 1.4   # inches per data unit — controls how large gates render


def plot_error_heatmap(
    result: AggregateResult,
    circuit: Circuit,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Draw the circuit with a halo-color effect around each gate box.

    Gate boxes are always square: the figure size is computed automatically
    from the circuit depth and qubit count so that 1 data unit = _SCALE inches
    in both axes.  Pass figsize explicitly to override.

    Parameters
    ----------
    result   : AggregateResult from ManyShotRunner.run()
    circuit  : The Circuit that was simulated
    title    : Optional figure title
    ax       : Existing Axes to draw on; creates a new figure if None
    figsize  : Figure size override.  Auto-computed when None.

    Returns
    -------
    The matplotlib Figure.
    """
    n_qubits = circuit.n_qubits
    n_ops = len(circuit.operations)
    n_layers = (max(op.t for op in circuit.operations) + 1) if circuit.operations else 1

    if ax is None:
        if figsize is None:
            # +1.5 on width for qubit labels + colorbar; +0.8 on height for title/ticks
            figsize = (n_layers * _SCALE + 1.5, n_qubits * _SCALE + 0.8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
    else:
        fig = ax.get_figure()

    # --- Compute normalised halo intensities (indexed by op_idx) ---
    gate_error_totals = result.counts_matrix.sum(axis=0).astype(float)
    max_total = gate_error_totals.max()
    if max_total > 0:
        intensities = gate_error_totals / max_total
    else:
        intensities = np.zeros(n_ops)

    # --- Qubit wires (span all layers) ---
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot(
            [-0.5, n_layers - 0.5],
            [y, y],
            linewidth=WIRE_LINEWIDTH,
            color=WIRE_COLOR,
            zorder=1,
        )

    # --- Gates (x-position from op.t, halo intensity from op_idx) ---
    for op_idx, op in enumerate(circuit.operations):
        x = op.t
        name = op.gate.name.upper()
        intensity = float(intensities[op_idx])
        fill = gate_color(name)
        halo = halo_color(intensity)

        if name in ("H", "S", "S†", "SDG", "X", "Y", "Z", "T", "T†", "T_DAG", "TDG"):
            (q,) = op.qubits
            y = n_qubits - 1 - q
            _draw_single_qubit_gate(ax, x, y, name, fill, halo, intensity)

        elif name in ("CX", "CNOT"):
            q_ctrl, q_tgt = op.qubits
            y_ctrl = n_qubits - 1 - q_ctrl
            y_tgt = n_qubits - 1 - q_tgt
            _draw_cnot(ax, x, y_ctrl, y_tgt, fill, halo, intensity)

        elif name == "CZ":
            q1, q2 = op.qubits
            y1 = n_qubits - 1 - q1
            y2 = n_qubits - 1 - q2
            _draw_cz(ax, x, y1, y2, fill, halo, intensity)

        elif name == "I":
            pass

    # --- Qubit labels ---
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.text(
            -0.75, y, f"q{q}",
            va="center", ha="right",
            fontsize=QUBIT_LABEL_FONT_SIZE,
            color=QUBIT_LABEL_COLOR,
        )

    # --- Colorbar ---
    cmap = get_halo_colormap()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative error intensity", fontsize=9)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["none", "mid", "max"])

    # --- Axes formatting ---
    ax.set_xlim(-1.1, n_layers - 0.3)
    ax.set_ylim(-0.8, n_qubits - 0.2)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"t={t}" for t in range(n_layers)], fontsize=9)
    ax.set_yticks([])
    ax.set_frame_on(False)

    if title:
        ax.set_title(title, fontsize=11, pad=10)
    else:
        ax.set_title(
            f"Error heatmap — {result.n_shots} shots",
            fontsize=11, pad=10,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------

def _draw_halo(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    intensity: float,
) -> None:
    """Draw a rounded halo rectangle behind a gate box."""
    pad = HALO_PAD
    color = halo_color(intensity)
    halo_rect = mpatches.FancyBboxPatch(
        (x - w / 2 - pad, y - h / 2 - pad),
        w + 2 * pad,
        h + 2 * pad,
        boxstyle="round,pad=0.04",
        linewidth=0,
        facecolor=color,
        alpha=HALO_ALPHA,
        zorder=2,
    )
    ax.add_patch(halo_rect)


def _draw_single_qubit_gate(
    ax: plt.Axes,
    x: float,
    y: float,
    name: str,
    fill: str,
    halo: tuple,
    intensity: float,
) -> None:
    _draw_halo(ax, x, y, GATE_HALF_W * 2, GATE_HALF_H * 2, intensity)
    rect = mpatches.FancyBboxPatch(
        (x - GATE_HALF_W, y - GATE_HALF_H),
        GATE_HALF_W * 2,
        GATE_HALF_H * 2,
        boxstyle="round,pad=0.02",
        linewidth=GATE_EDGE_WIDTH,
        edgecolor=GATE_EDGE_COLOR,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x, y, name,
        ha="center", va="center",
        fontsize=GATE_LABEL_FONT_SIZE,
        color=GATE_LABEL_COLOR,
        zorder=4,
    )


def _draw_cnot(
    ax: plt.Axes,
    x: float,
    y_ctrl: float,
    y_tgt: float,
    fill: str,
    halo: tuple,
    intensity: float,
) -> None:
    y_min = min(y_ctrl, y_tgt)
    y_max = max(y_ctrl, y_tgt)
    span = y_max - y_min
    _draw_halo(ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, span, intensity)

    # Vertical connector
    ax.plot([x, x], [y_ctrl, y_tgt], linewidth=GATE_EDGE_WIDTH, color=fill, zorder=3)
    # Control dot
    ax.scatter([x], [y_ctrl], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
    # Target ⊕
    circle = mpatches.Circle(
        (x, y_tgt), TARGET_CIRCLE_RADIUS,
        fill=True, facecolor="white",
        edgecolor=fill, linewidth=GATE_EDGE_WIDTH, zorder=4,
    )
    ax.add_patch(circle)
    ax.plot([x, x], [y_tgt - TARGET_CIRCLE_RADIUS, y_tgt + TARGET_CIRCLE_RADIUS],
            linewidth=GATE_EDGE_WIDTH, color=fill, zorder=5)
    ax.plot([x - TARGET_CIRCLE_RADIUS, x + TARGET_CIRCLE_RADIUS], [y_tgt, y_tgt],
            linewidth=GATE_EDGE_WIDTH, color=fill, zorder=5)


def _draw_cz(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    halo: tuple,
    intensity: float,
) -> None:
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    span = y_max - y_min
    _draw_halo(ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, span, intensity)

    ax.plot([x, x], [y1, y2], linewidth=GATE_EDGE_WIDTH, color=fill, zorder=3)
    ax.scatter([x, x], [y1, y2], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
