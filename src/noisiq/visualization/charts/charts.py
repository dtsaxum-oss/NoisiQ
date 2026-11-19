"""
Standalone chart functions for aggregate simulation results.

Each function accepts a result object and returns a matplotlib Figure,
displaying automatically when called in a Jupyter notebook.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ...backends.many_shot_runner import AggregateResult
from ..theme import (
    CHART_AXIS_LABEL_FONT_SIZE,
    CHART_BAR_EDGE_WIDTH,
    CHART_BAR_HEIGHT,
    CHART_BAR_VALUE_X_OFFSET,
    CHART_DEFAULT_FIGSIZE,
    CHART_GRID_ALPHA,
    CHART_LINE_STYLES,
    CHART_LINE_WIDTH,
    CHART_MARKER_SIZE,
    CHART_PROBABILITY_YLIM,
    CHART_SECONDARY_COLORS,
    CHART_TITLE_FONT_SIZE,
    CHART_TITLE_PAD,
    CHART_VALUE_LABEL_FONT_SIZE,
    CLIFFORD_GATE_COLOR,
    ERROR_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    WIRE_COLOR,
    get_qubit_error_colormap,
)


def plot_qubit_error_bar(
    result: AggregateResult,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 4),
    *,
    color_scale: Literal["relative", "absolute", "absolute_log"] = "absolute",
    vmax: float = 1.0,
    show_qualitative_labels: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of error events per shot for each qubit.

    The reported quantity is counts_matrix.sum(axis=1) / n_shots — error events
    accumulated over all timesteps, divided by the number of shots.  This can
    exceed 1 if multiple modeled events hit the same qubit across timesteps.

    Parameters
    ----------
    result                  : AggregateResult from ManyShotRunner.run()
    title                   : Optional figure title
    ax                      : Existing Axes to draw on; creates a new figure if None
    figsize                 : Figure size when creating a new figure
    color_scale             : Bar color gradient mode.
                              "absolute"     — linear 0–vmax scale (default; natural
                                              for error-events/shot which is bounded
                                              by n_ops per qubit, not 1).
                              "relative"     — max qubit in this run → darkest red.
                              "absolute_log" — log₁₀ scale between 1e-4 and vmax.
    vmax                    : Upper bound for "absolute" scaling (default 1.0).
    show_qualitative_labels : When True, append a qualitative label (Trace/Low/
                              Moderate/High) next to each bar's numeric value.

    Returns
    -------
    The matplotlib Figure.
    """
    from ..noise_metrics import normalize_heat_values, qualitative_burden_label

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Total error events per qubit across all timesteps, divided by shots.
    qubit_totals = result.counts_matrix.sum(axis=1)
    qubit_rates = qubit_totals / result.n_shots
    n_qubits = result.n_qubits
    qubit_labels = [f"q{q}" for q in range(n_qubits)]

    # Map each qubit's rate to a [0, 1] color intensity, then look up in the
    # sequential red colormap.  Qubits with zero events stay dark-gray.
    cmap = get_qubit_error_colormap()
    if color_scale == "absolute":
        denom = vmax if vmax > 0 else 1.0
        color_intensities = np.clip(qubit_rates / denom, 0.0, 1.0)
    elif color_scale == "relative":
        denom = float(qubit_rates.max()) if qubit_rates.max() > 0 else 1.0
        color_intensities = np.clip(qubit_rates / denom, 0.0, 1.0)
    elif color_scale == "absolute_log":
        color_intensities = normalize_heat_values(
            qubit_rates,
            mode="absolute_log",
            vmin=1e-4,
            vmax=max(vmax, 1e-4 + 1e-12),
            floor=0.0,
            gamma=1.0,
        )
    else:
        raise ValueError(f"Unknown color_scale={color_scale!r}")

    colors = [
        cmap(float(v)) if rate > 0 else CLIFFORD_GATE_COLOR
        for rate, v in zip(qubit_rates, color_intensities)
    ]

    bars = ax.barh(
        qubit_labels,
        qubit_rates,
        color=colors,
        edgecolor=WIRE_COLOR,
        linewidth=CHART_BAR_EDGE_WIDTH,
        height=CHART_BAR_HEIGHT,
    )

    # Value labels on bars
    for bar, rate in zip(bars, qubit_rates):
        if rate > 0:
            if show_qualitative_labels:
                label = f"{rate:.3g}  ({qualitative_burden_label(rate)})"
            else:
                label = f"{rate:.3g}"
            ax.text(
                bar.get_width() + CHART_BAR_VALUE_X_OFFSET,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha="left",
                fontsize=CHART_VALUE_LABEL_FONT_SIZE, color=WIRE_COLOR,
            )

    ax.set_xlabel("Error events / shot", fontsize=CHART_AXIS_LABEL_FONT_SIZE)
    ax.set_xlim(0, max(qubit_rates.max() * 1.25, 0.01))
    ax.invert_yaxis()
    ax.set_frame_on(False)
    ax.tick_params(axis="y", labelsize=QUBIT_LABEL_FONT_SIZE)
    ax.xaxis.grid(True, linestyle="--", alpha=CHART_GRID_ALPHA)
    ax.set_axisbelow(True)

    ax.set_title(
        title or f"Per-qubit error events / shot — {result.n_shots} shots",
        fontsize=CHART_TITLE_FONT_SIZE, pad=CHART_TITLE_PAD,
    )

    fig.tight_layout()
    return fig


def plot_zero_error_survival_decay(
    depth_results: List[AggregateResult] | Sequence[List[AggregateResult]],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = CHART_DEFAULT_FIGSIZE,
) -> plt.Figure:
    """
    Gate depth vs. zero-error survival probability chart.

    Zero-error survival: fraction of shots with zero sampled error events at each
    depth.  This is a noise-burden diagnostic, not a state or process fidelity.

    Parameters
    ----------
    depth_results : Either a single List[AggregateResult] (one curve) from
                    ManyShotRunner.run_depth_sweep(), or a list of such lists
                    (multiple curves, e.g. before/after suppression).
    labels        : Curve labels for the legend (required when multiple curves).
    title         : Figure title.  Defaults to "Zero-error survival vs. circuit depth".
    xlabel        : x-axis label.  Defaults to "Gate depth".
    ylabel        : y-axis label.  Defaults to metric description.
    ylim          : (bottom, top) y-axis limits.  Defaults to CHART_PROBABILITY_YLIM.
    ax            : Existing Axes; creates a new figure if None.
    figsize       : Figure size when creating a new figure.

    Returns
    -------
    The matplotlib Figure.
    """
    if not depth_results:
        raise ValueError("depth_results must not be empty")

    title = title or "Zero-error survival vs. circuit depth"
    xlabel = xlabel or "Gate depth"
    ylabel = ylabel or "Zero-error survival probability\n(zero-error shot fraction)"
    ylim = ylim or CHART_PROBABILITY_YLIM

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Normalise input: wrap single list in a list
    if depth_results and isinstance(depth_results[0], AggregateResult):
        curves = [depth_results]
    else:
        curves = list(depth_results)

    default_colors = [CLIFFORD_GATE_COLOR, ERROR_COLOR] + CHART_SECONDARY_COLORS
    default_styles = CHART_LINE_STYLES

    for i, curve in enumerate(curves):
        depths = list(range(1, len(curve) + 1))
        survival = [r.zero_error_fraction for r in curve]
        label = labels[i] if labels and i < len(labels) else f"Run {i + 1}"
        color = default_colors[i % len(default_colors)]
        linestyle = default_styles[i % len(default_styles)]

        ax.plot(
            depths, survival,
            color=color,
            linestyle=linestyle,
            linewidth=CHART_LINE_WIDTH,
            marker="o",
            markersize=CHART_MARKER_SIZE,
            label=label,
        )

    ax.set_xlabel(xlabel, fontsize=CHART_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=CHART_AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.5, max(len(c) for c in curves) + 0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=CHART_GRID_ALPHA)
    ax.yaxis.grid(True, linestyle="--", alpha=CHART_GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.set_frame_on(False)

    if len(curves) > 1 or labels:
        ax.legend(fontsize=CHART_VALUE_LABEL_FONT_SIZE, frameon=False)

    ax.set_title(title, fontsize=CHART_TITLE_FONT_SIZE, pad=CHART_TITLE_PAD)

    fig.tight_layout()
    return fig


def plot_fidelity_decay(
    depth_results: List[AggregateResult] | Sequence[List[AggregateResult]],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = CHART_DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Deprecated. Use plot_zero_error_survival_decay instead.

    This function historically plotted zero_error_fraction under the name
    "fidelity", which is a physics mislabeling.  It is kept as a compatibility
    shim only; all arguments are forwarded unchanged.
    """
    return plot_zero_error_survival_decay(
        depth_results,
        labels=labels,
        title=title,
        ax=ax,
        figsize=figsize,
    )
