"""
Hardware comparison composite figure.

Combines the error propagation heatmap with a fidelity annotation panel
so that a single plot answers: "how noisy is this circuit under this
hardware's noise model, and how does that compare to published results?"

Functions:
    plot_hardware_comparison : Heatmap + fidelity comparison in one figure
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ...backends.many_shot_runner import AggregateResult
from ...ir.circuit import Circuit
from ...noise.hardware_noise import HardwareProfile
from .heatmap import plot_error_heatmap
from ..theme import (
    CHART_BAR_EDGE_WIDTH,
    CHART_BAR_HEIGHT,
    CHART_GRID_ALPHA,
    CHART_TITLE_FONT_SIZE,
    CLIFFORD_GATE_COLOR,
    ERROR_COLOR,
    HARDWARE_CMP_ASPECT_RATIO,
    HARDWARE_CMP_BAR_ALPHA,
    HARDWARE_CMP_FOOTER_FONT_SIZE,
    HARDWARE_CMP_FOOTER_Y,
    HARDWARE_CMP_HEATMAP_WIDTH_PAD,
    HARDWARE_CMP_HEIGHT_RATIOS,
    HARDWARE_CMP_NOTE_FONT_SIZE,
    HARDWARE_CMP_NOTE_X_OFFSET,
    HARDWARE_CMP_PANEL_TITLE_FONT_SIZE,
    HARDWARE_CMP_PANEL_TITLE_PAD,
    HARDWARE_CMP_REFLINE_LABEL_FONT_SIZE,
    HARDWARE_CMP_REFLINE_LABEL_Y_OFFSET,
    HARDWARE_CMP_SUBPLOT_BOTTOM,
    HARDWARE_CMP_SUBPLOT_HSPACE,
    HARDWARE_CMP_SUBPLOT_LEFT,
    HARDWARE_CMP_SUBPLOT_RIGHT,
    HARDWARE_CMP_SUBPLOT_TOP,
    HARDWARE_CMP_SUPTITLE_FONT_SIZE,
    HARDWARE_CMP_SUPTITLE_Y,
    HARDWARE_CMP_XLIM_MAX,
    HARDWARE_CMP_XLABEL_FONT_SIZE,
    HARDWARE_CMP_YLIM_BOTTOM_PAD,
    HARDWARE_CMP_YLIM_TOP_OFFSET,
    HARDWARE_CMP_YTICK_FONT_SIZE,
    CIRCUIT_WIDTH_PER_LAYER,
    CIRCUIT_MIN_WIDTH,
    WIRE_COLOR,
)


def plot_hardware_comparison(
    result: AggregateResult,
    circuit: Circuit,
    profile: HardwareProfile,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Composite figure: error heatmap + fidelity comparison panel.

    The top panel shows which gates accumulated the most errors across
    all shots, using the standard halo-color heatmap.

    The bottom panel shows:
    - Our simulated fidelity estimate (zero-error shot fraction)
    - Published GHZ fidelities for this hardware platform (if any)
    - A horizontal bar chart so differences are immediately visible
    - Key noise parameters used (T2, gate error rates)

    Args:
        result:   AggregateResult from ManyShotRunner.run() on this circuit.
        circuit:  The Circuit that was simulated.
        profile:  HardwareProfile whose noise was used.
        title:    Overall figure suptitle.  Defaults to the profile name.
        figsize:  Figure size override.  Auto-computed when None.

    Returns:
        matplotlib Figure.

    Example:
        noise  = profile.to_pauli_noise_model(circuit)
        result = ManyShotRunner().run(circuit, n_shots=2000, noise_config=noise)
        fig    = plot_hardware_comparison(result, circuit, profile)
        plt.show()
    """
    n_layers = max(op.t for op in circuit.operations) + 1
    n_qubits = circuit.n_qubits

    if figsize is None:
        heatmap_w = max(CIRCUIT_MIN_WIDTH, CIRCUIT_WIDTH_PER_LAYER * n_layers + HARDWARE_CMP_HEATMAP_WIDTH_PAD)
        figsize = (heatmap_w, heatmap_w * HARDWARE_CMP_ASPECT_RATIO)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(
        left=HARDWARE_CMP_SUBPLOT_LEFT,
        right=HARDWARE_CMP_SUBPLOT_RIGHT,
        top=HARDWARE_CMP_SUBPLOT_TOP,
        bottom=HARDWARE_CMP_SUBPLOT_BOTTOM,
        hspace=HARDWARE_CMP_SUBPLOT_HSPACE,
    )

    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=HARDWARE_CMP_HEIGHT_RATIOS)

    # ── Top: error heatmap ────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    plot_error_heatmap(
        result,
        circuit,
        title=f"{profile.system}  —  {result.n_shots} shots,  "
              f"{n_qubits}-qubit GHZ",
        ax=ax_heat,
    )

    # ── Bottom: fidelity comparison ───────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    _draw_fidelity_comparison(ax_bar, result, profile, circuit.n_qubits)

    fig.suptitle(
        title or f"NoisiQ  ·  {profile.vendor} {profile.system}  noise model",
        fontsize=HARDWARE_CMP_SUPTITLE_FONT_SIZE,
        y=HARDWARE_CMP_SUPTITLE_Y,
    )
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _draw_fidelity_comparison(
    ax: plt.Axes,
    result: AggregateResult,
    profile: HardwareProfile,
    n_qubits_simulated: int,
) -> None:
    """Draw a horizontal bar chart comparing NoisiQ fidelity to hardware data."""

    sim_fidelity = result.zero_error_fraction

    # Collect published fidelities (only those with a non-None value)
    pub = [r for r in profile.ghz_results if r.fidelity is not None]

    # Build bar entries: simulated first, then published
    bar_labels: list[str] = []
    bar_values: list[float] = []
    bar_colors: list[str] = []
    bar_notes:  list[str] = []

    bar_labels.append(f"NoisiQ  ({n_qubits_simulated}q)")
    bar_values.append(sim_fidelity)
    bar_colors.append(CLIFFORD_GATE_COLOR)
    bar_notes.append(f"{sim_fidelity:.3f}  ← zero-error shot fraction")

    for r in pub:
        bar_labels.append(f"Hardware  ({r.n_qubits}q,  {r.year})")
        bar_values.append(r.fidelity)
        bar_colors.append(ERROR_COLOR)
        bar_notes.append(f"{r.fidelity:.3f}  ← {r.source}")

    # ── Horizontal bars ───────────────────────────────────────────────────────
    y_pos = list(range(len(bar_labels)))
    bars = ax.barh(
        y_pos,
        bar_values,
        color=bar_colors,
        edgecolor=WIRE_COLOR,
        linewidth=CHART_BAR_EDGE_WIDTH,
        height=CHART_BAR_HEIGHT,
        alpha=HARDWARE_CMP_BAR_ALPHA,
    )

    # Value labels
    for bar, note in zip(bars, bar_notes):
        ax.text(
            bar.get_width() + HARDWARE_CMP_NOTE_X_OFFSET,
            bar.get_y() + bar.get_height() / 2,
            note,
            va="center",
            ha="left",
            fontsize=HARDWARE_CMP_NOTE_FONT_SIZE,
            color=WIRE_COLOR,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=HARDWARE_CMP_YTICK_FONT_SIZE)
    ax.set_xlabel("Fidelity estimate", fontsize=HARDWARE_CMP_XLABEL_FONT_SIZE)
    ax.set_xlim(0, HARDWARE_CMP_XLIM_MAX)
    ax.set_ylim(HARDWARE_CMP_YLIM_BOTTOM_PAD, len(bar_labels) + HARDWARE_CMP_YLIM_TOP_OFFSET)
    ax.invert_yaxis()
    ax.set_frame_on(False)
    ax.xaxis.grid(True, linestyle="--", alpha=CHART_GRID_ALPHA)
    ax.set_axisbelow(True)

    # Noise params footer
    t2_us = profile.t2 * 1e6
    t_2q_ns = profile.gate_times.two_qubit_ns
    lam = 1 - np.exp(-2 * t_2q_ns * 1e-9 / profile.t2)
    footer = (
        f"Noise params:  T2={t2_us:.0f} µs  ·  "
        f"2Q gate={t_2q_ns:.0f} ns  ·  "
        f"2Q gate error={profile.two_qubit_error:.4f}  ·  "
        f"T2 λ per 2Q gate={lam:.5f}"
    )
    ax.text(
        0.0, HARDWARE_CMP_FOOTER_Y,
        footer,
        transform=ax.transAxes,
        fontsize=HARDWARE_CMP_FOOTER_FONT_SIZE,
        color="gray",
        va="top",
    )

    ax.set_title("Fidelity: NoisiQ simulation vs published hardware",
                 fontsize=HARDWARE_CMP_PANEL_TITLE_FONT_SIZE,
                 pad=HARDWARE_CMP_PANEL_TITLE_PAD)

    # Dashed reference line at 0.5 (maximally mixed)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=CHART_BAR_EDGE_WIDTH, alpha=0.5)
    ax.text(0.5, len(bar_labels) + HARDWARE_CMP_REFLINE_LABEL_Y_OFFSET, "random\nguessing",
            ha="center", fontsize=HARDWARE_CMP_REFLINE_LABEL_FONT_SIZE, color="gray", va="top")
