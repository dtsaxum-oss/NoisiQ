"""
Standalone chart functions for aggregate simulation results.

Each function accepts a result object and returns a matplotlib Figure,
displaying automatically when called in a Jupyter notebook.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt

from ...backends.many_shot_runner import AggregateResult
from ..theme import (
    CLIFFORD_GATE_COLOR,
    ERROR_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    WIRE_COLOR,
)


def plot_qubit_error_bar(
    result: AggregateResult,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    """
    Horizontal bar chart of total error counts per qubit across all timesteps.

    Parameters
    ----------
    result  : AggregateResult from ManyShotRunner.run()
    title   : Optional figure title
    ax      : Existing Axes to draw on; creates a new figure if None
    figsize : Figure size when creating a new figure

    Returns
    -------
    The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Total errors per qubit: sum counts_matrix along timestep axis
    qubit_totals = result.counts_matrix.sum(axis=1)
    qubit_rates = qubit_totals / result.n_shots
    n_qubits = result.n_qubits
    qubit_labels = [f"q{q}" for q in range(n_qubits)]

    colors = [ERROR_COLOR if r > 0 else CLIFFORD_GATE_COLOR for r in qubit_rates]

    bars = ax.barh(
        qubit_labels,
        qubit_rates,
        color=colors,
        edgecolor=WIRE_COLOR,
        linewidth=0.8,
        height=0.55,
    )

    # Value labels on bars
    for bar, rate in zip(bars, qubit_rates):
        if rate > 0:
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.3f}",
                va="center", ha="left",
                fontsize=9, color=WIRE_COLOR,
            )

    ax.set_xlabel("Error rate  (errors / shot)", fontsize=10)
    ax.set_xlim(0, max(qubit_rates.max() * 1.25, 0.01))
    ax.invert_yaxis()
    ax.set_frame_on(False)
    ax.tick_params(axis="y", labelsize=QUBIT_LABEL_FONT_SIZE)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.set_title(
        title or f"Per-qubit error rate — {result.n_shots} shots",
        fontsize=11, pad=8,
    )

    fig.tight_layout()
    return fig


def plot_fidelity_decay(
    depth_results: List[AggregateResult] | Sequence[List[AggregateResult]],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 4),
) -> plt.Figure:
    """
    Gate depth vs. fidelity estimate chart.

    Fidelity proxy: fraction of shots with zero total errors at each depth.

    Parameters
    ----------
    depth_results : Either a single List[AggregateResult] (one curve) from
                    ManyShotRunner.run_depth_sweep(), or a list of such lists
                    (multiple curves, e.g. before/after suppression).
    labels        : Curve labels for the legend (required when multiple curves).
    title         : Optional figure title
    ax            : Existing Axes; creates a new figure if None
    figsize       : Figure size when creating a new figure

    Returns
    -------
    The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Normalise input: wrap single list in a list
    if depth_results and isinstance(depth_results[0], AggregateResult):
        curves = [depth_results]
    else:
        curves = list(depth_results)

    default_colors = [CLIFFORD_GATE_COLOR, ERROR_COLOR, "#4CAF50", "#FF9800", "#9C27B0"]
    default_styles = ["-", "--", "-.", ":", "-"]

    for i, curve in enumerate(curves):
        depths = list(range(1, len(curve) + 1))
        fidelities = [r.zero_error_fraction for r in curve]
        label = labels[i] if labels and i < len(labels) else f"Run {i + 1}"
        color = default_colors[i % len(default_colors)]
        linestyle = default_styles[i % len(default_styles)]

        ax.plot(
            depths, fidelities,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )

    ax.set_xlabel("Gate depth", fontsize=10)
    ax.set_ylabel("Fidelity estimate\n(zero-error shot fraction)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, max(len(c) for c in curves) + 0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_frame_on(False)

    if len(curves) > 1 or labels:
        ax.legend(fontsize=9, frameon=False)

    ax.set_title(title or "Fidelity decay vs. circuit depth", fontsize=11, pad=8)

    fig.tight_layout()
    return fig
