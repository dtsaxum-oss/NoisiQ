"""
Aggregate error heatmap overlaid on the circuit diagram.

Renders a circuit "print-out" with gate boxes and qubit wires, with a
halo-color effect around each gate box scaled by error intensity, and
soft wire-segment halos showing idle-time T1/T2 decoherence.

Gate halos:  vertical Gaussian gradient around each gate box, driven by
             downstream Pauli-error / propagation impact. The vertical
             orientation echoes how a Pauli error at one qubit reaches
             upstream/downstream qubits at the same layer.
Wire halos:  vertical Gaussian gradient along idle wire segments, driven
             by per-segment T1+T2 decoherence from noise_config.
"""

from __future__ import annotations

import textwrap
from itertools import product
from typing import Callable, Literal, Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ...ir.circuit import Circuit, Operation
from ...ir.classical import Measurement, ConditionalOp
from ...backends.many_shot_runner import AggregateResult
from ..gate_info import GateInfoExtractor, WireSegment
from ..theme import (
    CHART_TITLE_FONT_SIZE,
    CIRCUIT_BASE_HEIGHT_PAD,
    CIRCUIT_HEIGHT_PER_QUBIT,
    CIRCUIT_MIN_WIDTH,
    CIRCUIT_WIDTH_PER_LAYER,
    FONT_FAMILY,
    GATE_EDGE_COLOR,
    GATE_EDGE_WIDTH,
    GATE_HALF_H,
    GATE_HALF_W,
    GATE_HALO_HEIGHT_IN,
    GATE_HALO_MAX_ALPHA,
    GATE_HALO_SIGMA_FRAC,
    GATE_HALO_WIDTH_IN,
    HALO_ALPHA,
    HALO_GAMMA,
    HALO_INTENSITY_FLOOR,
    HALO_PAD_X,
    HALO_PAD_Y,
    HEATMAP_COLORBAR_FRACTION,
    HEATMAP_COLORBAR_PAD,
    HEATMAP_FIGSIZE_HEIGHT_PAD,
    HEATMAP_FIGSIZE_WIDTH_PAD,
    HEATMAP_HALO_BOX_PAD,
    HEATMAP_LABEL_FONT_SIZE,
    HEATMAP_QUBIT_LABEL_X,
    HEATMAP_TICK_LABELS,
    HEATMAP_TICK_POSITIONS,
    HEATMAP_TITLE_PAD,
    HEATMAP_XLIM_LEFT,
    HEATMAP_XLIM_RIGHT_OFFSET,
    HEATMAP_YLIM_BOTTOM,
    HEATMAP_YLIM_TOP_OFFSET,
    IDLE_CBAR_ANNOTATION_COLOR,
    IDLE_CBAR_ANNOTATION_FONT_SIZE,
    IDLE_CBAR_ANNOTATION_X,
    IDLE_CBAR_ANNOTATION_Y,
    IDLE_CBAR_BBOX_Y,
    IDLE_CBAR_HEIGHT,
    IDLE_CBAR_LABEL,
    IDLE_CBAR_LABEL_COLOR,
    IDLE_CBAR_LABEL_FONT_SIZE,
    IDLE_CBAR_LABEL_PAD,
    IDLE_CBAR_TICK_FONT_SIZE,
    IDLE_CBAR_WIDTH,
    QUBIT_LABEL_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    WIRE_COLOR,
    WIRE_HALO_HEIGHT,
    WIRE_HALO_MAX_ALPHA,
    WIRE_HALO_SIGMA,
    MEASUREMENT_EDGE_COLOR,
    WIRE_LINEWIDTH,
    HOVER_WIRE_DECOHERENCE_SECTION,
    HOVER_WIRE_COMBINED_LABEL,
    HOVER_CUMULATIVE_CURSOR_SECTION,
    draw_ccz,
    draw_classical_control_wire,
    draw_cnot,
    draw_cs,
    draw_cz,
    draw_measurement,
    draw_swap,
    draw_single_gate,
    gate_color,
    get_halo_colormap,
    get_wire_halo_colormap,
    halo_color,
)


# ---------------------------------------------------------------------------
# Pauli propagation tables for downstream impact analysis
# ---------------------------------------------------------------------------

_SINGLE_QUBIT_PROPAGATION: dict[str, dict[str, str]] = {
    'H':     {'X': 'Z', 'Y': 'Y', 'Z': 'X', 'I': 'I'},
    'S':     {'X': 'Y', 'Y': 'X', 'Z': 'Z', 'I': 'I'},
    'S_DAG': {'X': 'Y', 'Y': 'X', 'Z': 'Z', 'I': 'I'},
    'X':     {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'I': 'I'},
    'Y':     {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'I': 'I'},
    'Z':     {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'I': 'I'},
    'I':     {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'I': 'I'},
    'IDLE':  {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'I': 'I'},
}

_CNOT_PROPAGATION_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    ('I', 'I'): ('I', 'I'),
    ('X', 'I'): ('X', 'X'),
    ('Y', 'I'): ('Y', 'X'),
    ('Z', 'I'): ('Z', 'I'),
    ('I', 'X'): ('I', 'X'),
    ('X', 'X'): ('X', 'I'),
    ('Y', 'X'): ('Y', 'I'),
    ('Z', 'X'): ('Z', 'X'),
    ('I', 'Y'): ('Z', 'Y'),
    ('X', 'Y'): ('Y', 'Z'),
    ('Y', 'Y'): ('X', 'Z'),
    ('Z', 'Y'): ('I', 'Y'),
    ('I', 'Z'): ('Z', 'Z'),
    ('X', 'Z'): ('Y', 'Y'),
    ('Y', 'Z'): ('X', 'Y'),
    ('Z', 'Z'): ('I', 'Z'),
}

_CZ_PROPAGATION_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    ('I', 'I'): ('I', 'I'),
    ('X', 'I'): ('X', 'Z'),
    ('Y', 'I'): ('Y', 'Z'),
    ('Z', 'I'): ('Z', 'I'),
    ('I', 'X'): ('Z', 'X'),
    ('X', 'X'): ('Y', 'Y'),
    ('Y', 'X'): ('X', 'Y'),
    ('Z', 'X'): ('I', 'X'),
    ('I', 'Y'): ('Z', 'Y'),
    ('X', 'Y'): ('Y', 'X'),
    ('Y', 'Y'): ('X', 'X'),
    ('Z', 'Y'): ('I', 'Y'),
    ('I', 'Z'): ('I', 'Z'),
    ('X', 'Z'): ('X', 'I'),
    ('Y', 'Z'): ('Y', 'I'),
    ('Z', 'Z'): ('Z', 'Z'),
}

_SWAP_PROPAGATION_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    # (P_q0, P_q1) → (P_q1, P_q0) — SWAP simply exchanges the two Pauli labels.
    ('I', 'I'): ('I', 'I'),
    ('X', 'I'): ('I', 'X'),
    ('Y', 'I'): ('I', 'Y'),
    ('Z', 'I'): ('I', 'Z'),
    ('I', 'X'): ('X', 'I'),
    ('X', 'X'): ('X', 'X'),
    ('Y', 'X'): ('X', 'Y'),
    ('Z', 'X'): ('X', 'Z'),
    ('I', 'Y'): ('Y', 'I'),
    ('X', 'Y'): ('Y', 'X'),
    ('Y', 'Y'): ('Y', 'Y'),
    ('Z', 'Y'): ('Y', 'Z'),
    ('I', 'Z'): ('Z', 'I'),
    ('X', 'Z'): ('Z', 'X'),
    ('Y', 'Z'): ('Z', 'Y'),
    ('Z', 'Z'): ('Z', 'Z'),
}


# ---------------------------------------------------------------------------
# Wire-segment computation
# ---------------------------------------------------------------------------

def _iterate_channels(ch):
    """Yield sub-channels of a CombinedChannel, or ch itself."""
    from ...noise.kraus_channels import CombinedChannel
    if ch is None:
        return
    if isinstance(ch, CombinedChannel):
        yield from ch.channels
    else:
        yield ch


def _compute_wire_segments(
    circuit: Circuit,
    noise_config: Optional[dict],
) -> list[WireSegment]:
    """Walk each qubit, find contiguous runs of IDLE ops, build WireSegments."""
    from ...ir import gates as ir_gates
    from ...noise.amplitude_damping import AmplitudeDamping
    from ...noise.t2_dephasing import Dephasing

    segments: list[WireSegment] = []

    for q in range(circuit.n_qubits):
        # Only Operation instances carry a .qubits tuple; Measurement uses .qubit
        # (singular) and ConditionalOp forwards to its inner gate.  Since
        # _compute_wire_segments only looks for IDLE operations — which are always
        # plain Operation objects — it is correct to filter to Operation here.
        ops_on_q = sorted(
            [
                (i, op)
                for i, op in enumerate(circuit.operations)
                if isinstance(op, Operation) and q in op.qubits
            ],
            key=lambda io: io[1].t,
        )

        run_idxs: list[int] = []
        for i, op in ops_on_q:
            if op.gate is ir_gates.IDLE:
                run_idxs.append(i)
            else:
                if run_idxs:
                    segments.append(
                        _segment_from_run(q, run_idxs, circuit, noise_config)
                    )
                    run_idxs = []
        if run_idxs:
            segments.append(_segment_from_run(q, run_idxs, circuit, noise_config))

    return segments


def _segment_from_run(
    qubit: int,
    run_idxs: list[int],
    circuit: Circuit,
    noise_config: Optional[dict],
) -> WireSegment:
    from ...noise.amplitude_damping import AmplitudeDamping
    from ...noise.t2_dephasing import Dephasing
    from ...noise.pauli_error import PauliError

    t_lo = circuit.operations[run_idxs[0]].t - 0.5
    t_hi = circuit.operations[run_idxs[-1]].t + 0.5

    total_t_t1 = 0.0
    total_t_t2 = 0.0
    T1_seen: Optional[float] = None
    T2_seen: Optional[float] = None
    pauli_lam_acc = 0.0   # accumulated T2-equivalent from PauliError p_z (pauli_twirl mode)
    pauli_gamma_acc = 0.0 # accumulated T1-equivalent from PauliError p_x
    duration_ns = 0.0

    for idx in run_idxs:
        op = circuit.operations[idx]
        if op.params:
            duration_ns += op.params.get("duration_ns", 0.0)
        ch = noise_config.get(idx) if noise_config else None
        for sub in _iterate_channels(ch):
            if isinstance(sub, AmplitudeDamping):
                total_t_t1 += sub.t
                T1_seen = sub.T1
            elif isinstance(sub, Dephasing):
                total_t_t2 += sub.t
                T2_seen = (
                    sub.T2 if sub.T2 is not None
                    else 1.0 / (1.0 / (2 * sub.T1) + 1.0 / sub.Tphi)
                )
            elif isinstance(sub, PauliError):
                # PauliError contributes to both T1 and T2 equivalent decoherence:
                #   γ = 4 p_x
                #   λ = 2 (p_z - p_x)
                #
                # If extra coherent-Z twirl is present, it contributes to the effective
                # phase-decay λ shown by the idle halo.
                gamma_piece = float(np.clip(4.0 * sub.p_x, 0.0, 1.0))
                lam_piece = float(np.clip(2.0 * max(0.0, sub.p_z - sub.p_x), 0.0, 1.0))

                pauli_gamma_acc = 1.0 - (1.0 - pauli_gamma_acc) * (1.0 - gamma_piece)
                pauli_lam_acc = 1.0 - (1.0 - pauli_lam_acc) * (1.0 - lam_piece)

    gamma = (1.0 - float(np.exp(-total_t_t1 / T1_seen))) if T1_seen else pauli_gamma_acc
    lam   = (1.0 - float(np.exp(-2.0 * total_t_t2 / T2_seen))) if T2_seen else pauli_lam_acc
    intensity = 1.0 - (1.0 - gamma) * (1.0 - lam)

    return WireSegment(
        qubit=qubit,
        t_lo=t_lo,
        t_hi=t_hi,
        idle_op_idxs=run_idxs,
        duration_ns=duration_ns,
        gamma=gamma,
        lam=lam,
        intensity=intensity,
    )


# ---------------------------------------------------------------------------
# Colorbar helpers
# ---------------------------------------------------------------------------

def _format_decoherence_value(value: float) -> str:
    """Format a decoherence probability for colorbar labels."""
    value = float(value)
    if value <= 0.0:
        return "0"
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{100.0 * value:.2f}%"

def _format_burden_value(value: float) -> str:
    value = float(value)
    if value <= 0.0:
        return "0"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _short_wrapped_title(label: str, *, width: int = 14) -> str:
    return textwrap.fill(label, width=width, break_long_words=False)


def _tick_positions(raw_ticks: list) -> list:
    """Normalize raw impact values to [0, 1] colorbar axis positions."""
    peak = max((float(v) for v in raw_ticks if v > 0), default=1.0)
    return [float(v) / peak for v in raw_ticks]


def _wire_visual_tick_position(raw: float, peak: float) -> float:
    raw = float(raw)
    peak = float(peak)

    if raw <= 0.0 or peak <= 0.0:
        return 0.0

    return HALO_INTENSITY_FLOOR + (
        1.0 - HALO_INTENSITY_FLOOR
    ) * (raw / peak) ** HALO_GAMMA


def _add_downstream_burden_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    impact: np.ndarray,
    heat_scale: str,
    heat_vmin: float,
    heat_vmax: float,
    show_qualitative_ticks: bool,
    burden_label: str = "Expected gate error burden / shot",
) -> None:
    cmap = get_halo_colormap()
    if heat_scale == "relative":
        _norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        _norm = mcolors.Normalize(vmin=heat_vmin, vmax=heat_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=_norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=HEATMAP_COLORBAR_FRACTION,
        pad=HEATMAP_COLORBAR_PAD,
    )

    peak = float(np.max(impact)) if len(impact) else 0.0
    positive = impact[impact > 0]
    has_zero = positive.size < impact.size

    cbar.set_label("")
    cbar.ax.set_title(
        _short_wrapped_title(burden_label, width=14),
        fontsize=HEATMAP_LABEL_FONT_SIZE,
        pad=14,
    )

    if heat_scale == "relative":
        if positive.size == 0:
            cbar.set_ticks(HEATMAP_TICK_POSITIONS)
            cbar.set_ticklabels(
                [f"{lbl}\n0" for lbl in HEATMAP_TICK_LABELS]
                if show_qualitative_ticks
                else ["0", "0", "0"],
                fontsize=HEATMAP_LABEL_FONT_SIZE,
            )
        else:
            min_nonzero = float(np.min(positive))
            peak = float(np.max(positive))

            if np.isclose(min_nonzero, peak):
                raw_ticks = [0.0, peak] if has_zero else [peak]
                labels = (
                    ["none\n0", f"all\n{_format_burden_value(peak)}"]
                    if has_zero
                    else [f"all\n{_format_burden_value(peak)}"]
                )
            else:
                mid = 0.5 * (min_nonzero + peak)

                if has_zero:
                    raw_ticks = [0.0, min_nonzero, mid, peak]
                    labels = [
                        "none\n0",
                        f"min\n{_format_burden_value(min_nonzero)}",
                        f"mid\n{_format_burden_value(mid)}",
                        f"max\n{_format_burden_value(peak)}",
                    ]
                else:
                    raw_ticks = [min_nonzero, mid, peak]
                    labels = [
                        f"min\n{_format_burden_value(min_nonzero)}",
                        f"mid\n{_format_burden_value(mid)}",
                        f"max\n{_format_burden_value(peak)}",
                    ]

            cbar.set_ticks(_tick_positions(raw_ticks))
            cbar.set_ticklabels(
                labels if show_qualitative_ticks else [
                    _format_burden_value(v) for v in raw_ticks
                ],
                fontsize=HEATMAP_LABEL_FONT_SIZE,
            )


def _add_idle_decoherence_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    segments: list[WireSegment],
    raw_vals: list[float],
) -> None:
    """Add the idle T1/T2 decoherence colorbar.

    The bar is scaled to the same relative visual intensity used for wire halos.
    Tick labels report the combined idle decoherence value:

        1 - (1 - γ_T1)(1 - λ_T2)

    The side annotation reports the observed T1 and T2 component ranges across
    the idle segments.
    """
    wire_cmap = get_wire_halo_colormap()

    # Compute tick raw-values, labels, and visual positions first so we can
    # create the ScalarMappable with vmin = the actual minimum drawn color,
    # ensuring the min tick sits at the left edge of the bar.
    raw = np.asarray(raw_vals, dtype=float)
    positive = raw[raw > 0.0]
    has_zero = positive.size < raw.size

    if positive.size == 0:
        raw_ticks: list[float] = [0.0]
        labels: list[str] = ["none\n0"]
        peak = 1.0
    else:
        min_nonzero = float(np.min(positive))
        peak = float(np.max(positive))

        if np.isclose(min_nonzero, peak):
            raw_ticks = [0.0, peak] if has_zero else [peak]
            labels = (
                ["none\n0", f"all\n{_format_decoherence_value(peak)}"]
                if has_zero
                else [f"all\n{_format_decoherence_value(peak)}"]
            )
        else:
            mid = 0.5 * (min_nonzero + peak)
            if has_zero:
                raw_ticks = [0.0, min_nonzero, mid, peak]
                labels = [
                    "none\n0",
                    f"min\n{_format_decoherence_value(min_nonzero)}",
                    f"mid\n{_format_decoherence_value(mid)}",
                    f"max\n{_format_decoherence_value(peak)}",
                ]
            else:
                raw_ticks = [min_nonzero, mid, peak]
                labels = [
                    f"min\n{_format_decoherence_value(min_nonzero)}",
                    f"mid\n{_format_decoherence_value(mid)}",
                    f"max\n{_format_decoherence_value(peak)}",
                ]

    vis_ticks = [_wire_visual_tick_position(v, peak) for v in raw_ticks]
    vis_lo = vis_ticks[0]
    vis_hi = vis_ticks[-1] if vis_ticks[-1] > vis_lo else vis_lo + 1e-6

    sm_wire = plt.cm.ScalarMappable(
        cmap=wire_cmap,
        norm=mcolors.Normalize(vmin=vis_lo, vmax=vis_hi),
    )
    sm_wire.set_array([])

    cax = inset_axes(
        ax,
        width=IDLE_CBAR_WIDTH,
        height=IDLE_CBAR_HEIGHT,
        loc="lower center",
        bbox_to_anchor=(0.0, IDLE_CBAR_BBOX_Y, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cbar_wire = fig.colorbar(
        sm_wire,
        cax=cax,
        orientation="horizontal",
    )

    cbar_wire.set_ticks(vis_ticks)
    cbar_wire.set_ticklabels(labels, fontsize=IDLE_CBAR_TICK_FONT_SIZE)

    cbar_wire.set_label(
        IDLE_CBAR_LABEL,
        fontsize=IDLE_CBAR_LABEL_FONT_SIZE,
        color=IDLE_CBAR_LABEL_COLOR,
        labelpad=IDLE_CBAR_LABEL_PAD,
    )

    positive_segments = [
        seg for seg, raw_value in zip(segments, raw_vals)
        if raw_value > 0.0
    ]

    if positive_segments:
        gamma_vals = np.asarray([seg.gamma for seg in positive_segments], dtype=float)
        lam_vals = np.asarray([seg.lam for seg in positive_segments], dtype=float)

        component_text = (
            f"T1 γ: min {_format_decoherence_value(np.min(gamma_vals))}"
            f"– max{_format_decoherence_value(np.max(gamma_vals))}\n"
            f"T2 λ: min {_format_decoherence_value(np.min(lam_vals))}"
            f"– max{_format_decoherence_value(np.max(lam_vals))}"
        )

        cbar_wire.ax.text(
            IDLE_CBAR_ANNOTATION_X,
            IDLE_CBAR_ANNOTATION_Y,
            component_text,
            transform=cbar_wire.ax.transAxes,
            ha="left",
            va="center",
            fontsize=IDLE_CBAR_ANNOTATION_FONT_SIZE,
            color=IDLE_CBAR_ANNOTATION_COLOR,
            clip_on=False,
        )

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def plot_error_heatmap(
    result: AggregateResult,
    circuit: Circuit,
    *,
    noise_config: Optional[dict] = None,
    wire_halo_metric: Optional[Callable] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple] = None,
    mcm_mode: str = "qec",
    heat_scale: Literal["relative", "absolute", "absolute_log"] = "relative",
    heat_vmin: float = 1e-4,
    heat_vmax: float = 1e-1,
    show_qualitative_ticks: bool = True,
    impact_metric: Literal["expected", "worst_case"] = "expected",
    finalize_layout: bool = True,
    purity_rho: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Draw the circuit with halo-color effects around gate boxes and
    soft wire-segment halos for idle-time decoherence.

    Parameters
    ----------
    result                : AggregateResult from ManyShotRunner.run()
    circuit               : The Circuit that was simulated
    noise_config          : Per-op noise dict from HardwareProfile.to_noise_model().
                            When supplied, wire halos are drawn for idle segments
                            (Kraus-mode noise only; Pauli-twirl dicts produced by
                            to_noise_model(circuit, representation='pauli_twirl')
                            have no T1/T2 channels to read).
    wire_halo_metric      : Optional callable (WireSegment) -> float in [0,1]
                            overriding the default combined-intensity metric.
    title                 : Optional figure title
    ax                    : Existing Axes; creates a new figure if None
    figsize               : Figure size override (auto-computed when None)
    mcm_mode              : How mid-circuit measurements affect downstream impact.
                            "qec"        — correction fires on detected error (default).
                            "worst_case" — correction never helps (maximises spread).
                            "average"    — weighted average of both branches.
                            Use plot_error_heatmap_side_by_side() for "side_by_side".
    heat_scale            : Halo intensity normalization mode.
                            "relative"     — hottest gate in this run → max intensity
                                            (default; best for single-circuit inspection).
                            "absolute"     — linear scale between heat_vmin/heat_vmax.
                            "absolute_log" — log₁₀ scale; use when comparing multiple
                                            circuits on a common scale.
    heat_vmin, heat_vmax  : Bounds used by "absolute" and "absolute_log" modes.
                            Defaults cover typical quantum error rates (1e-4 to 1e-1).
    show_qualitative_ticks: When True, colorbar uses qualitative labels
                            (trace/low/moderate/high for absolute modes,
                            none/mid/max for relative).

    Returns
    -------
    The matplotlib Figure.
    """
    if impact_metric not in ("expected", "worst_case"):
        raise ValueError(
            f"impact_metric={impact_metric!r} is not valid; choose 'expected' or 'worst_case'."
        )
    if mcm_mode not in ("qec", "worst_case", "average"):
        raise ValueError(
            f"mcm_mode={mcm_mode!r} is not valid for plot_error_heatmap. "
            "Use plot_error_heatmap_side_by_side() for side-by-side rendering, "
            "or choose 'qec', 'worst_case', or 'average'."
        )
    if not hasattr(result, 'counts_matrix') or result.counts_matrix is None:
        raise TypeError(
            "plot_error_heatmap requires an AggregateResult with a counts_matrix. "
            "If you used HardwareProfile.to_noise_model() directly, switch to "
            "to_noise_model(circuit, representation='pauli_twirl'), which produces "
            "a PauliError noise model compatible with ManyShotRunner."
        )

    n_qubits = circuit.n_qubits
    n_ops = len(circuit.operations)
    n_layers = (max(op.t for op in circuit.operations) + 1) if circuit.operations else 1

    if ax is None:
        if figsize is None:
            figsize = (
                max(CIRCUIT_MIN_WIDTH, CIRCUIT_WIDTH_PER_LAYER * n_layers)
                + HEATMAP_FIGSIZE_WIDTH_PAD,
                CIRCUIT_HEIGHT_PER_QUBIT * n_qubits
                + CIRCUIT_BASE_HEIGHT_PAD
                + HEATMAP_FIGSIZE_HEIGHT_PAD,
            )
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- Downstream propagation impact ---
    had_errors = result.counts_matrix.sum(axis=0) > 0
    impact = _compute_downstream_impact(
        circuit,
        noise_config=noise_config,
        mcm_mode=mcm_mode,
        impact_metric=impact_metric,
    )

    # When a noise_config is provided, extend had_errors to include ANY non-IDLE
    # gate that has a noise channel — including gates at the end of the circuit
    # with zero downstream spread (e.g. final H).  This ensures every noisy gate
    # shows at least HALO_INTENSITY_FLOOR in the drawing loop below, regardless
    # of sampled counts or downstream impact.
    # IDLE gates are excluded — their decoherence is shown via wire halos instead.
    if noise_config is not None:
        non_idle = np.array([
            isinstance(op, Operation) and op.gate.name.upper() not in ("I", "IDLE")
            for op in circuit.operations
        ])
        has_channel = np.array([noise_config.get(k) is not None for k in range(n_ops)])
        had_errors = had_errors | (non_idle & has_channel)

    from ..noise_metrics import normalize_heat_values
    intensities = normalize_heat_values(
        impact,
        mode=heat_scale,
        vmin=heat_vmin,
        vmax=heat_vmax,
        floor=HALO_INTENSITY_FLOOR,
        gamma=HALO_GAMMA,
    )

    # All gate and wire halos are accumulated into one composite image so that
    # interactive figures (ipympl) only serialize a single imshow instead of
    # one per halo — the main cause of Cell-7 timeouts on large circuits.
    halo_canvas = _HaloCanvas(
        x_lo=HEATMAP_XLIM_LEFT,
        x_hi=n_layers + HEATMAP_XLIM_RIGHT_OFFSET,
        y_lo=HEATMAP_YLIM_BOTTOM,
        y_hi=n_qubits + HEATMAP_YLIM_TOP_OFFSET,
    )

    # --- Qubit wires ---
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot(
            [-0.5, n_layers - 0.5], [y, y],
            linewidth=WIRE_LINEWIDTH, color=WIRE_COLOR, zorder=1,
        )

    # --- Wire halos (drawn before gates so gates sit on top) ---
    wire_raw_vals: list[float] = []

    if noise_config is not None:
        segments = _compute_wire_segments(circuit, noise_config)
        if segments:
            if wire_halo_metric is not None:
                wire_raw_vals = [wire_halo_metric(s) for s in segments]
            else:
                wire_raw_vals = [s.intensity for s in segments]

            max_val = max(wire_raw_vals) if max(wire_raw_vals) > 0 else 1.0

            for seg, raw in zip(segments, wire_raw_vals):
                y = n_qubits - 1 - seg.qubit
                vis = HALO_INTENSITY_FLOOR + (
                    1.0 - HALO_INTENSITY_FLOOR
                ) * (raw / max_val) ** HALO_GAMMA
                _draw_wire_halo(ax, seg.t_lo, seg.t_hi, y, vis, halo_canvas)
    else:
        segments = []

    # --- Gates ---
    def _preceding_meas(cbit_idx: int, before_t: int):
        candidates = [
            m for m in circuit.operations
            if isinstance(m, Measurement) and m.cbit.index == cbit_idx and m.t <= before_t
        ]
        return max(candidates, key=lambda m: m.t, default=None)

    for op_idx, op in enumerate(circuit.operations):
        x = float(op.t)
        intensity = float(intensities[op_idx])
        if intensity == 0.0 and had_errors[op_idx]:
            intensity = HALO_INTENSITY_FLOOR

        if isinstance(op, Measurement):
            y = n_qubits - 1 - op.qubit
            _draw_measurement_halo(ax, x, y, intensity, halo_canvas)
            continue

        if isinstance(op, ConditionalOp):
            inner = op.inner
            inner_name = inner.gate.name.upper()
            fill = gate_color(inner_name)
            if inner_name in ("CX", "CNOT"):
                q_ctrl, q_tgt = inner.qubits
                _draw_cnot_gate(ax, x, n_qubits-1-q_ctrl, n_qubits-1-q_tgt, fill, intensity, edge=MEASUREMENT_EDGE_COLOR, halo_canvas=halo_canvas)
            elif inner_name == "CZ":
                q1, q2 = inner.qubits
                _draw_cz_gate(ax, x, n_qubits-1-q1, n_qubits-1-q2, fill, intensity, edge=MEASUREMENT_EDGE_COLOR, halo_canvas=halo_canvas)
            elif inner_name == "SWAP":
                q1, q2 = inner.qubits
                _draw_swap_gate(ax, x, n_qubits-1-q1, n_qubits-1-q2, fill, intensity, edge=MEASUREMENT_EDGE_COLOR, halo_canvas=halo_canvas)
            elif inner_name in ("CS", "CS_DAG"):
                q_ctrl, q_tgt = inner.qubits
                _draw_cs_gate(ax, x, n_qubits-1-q_ctrl, n_qubits-1-q_tgt, fill, intensity, edge=MEASUREMENT_EDGE_COLOR, halo_canvas=halo_canvas)
            elif inner_name == "CCZ":
                q1, q2, q3 = inner.qubits
                _draw_ccz_gate(ax, x, n_qubits-1-q1, n_qubits-1-q2, n_qubits-1-q3, fill, intensity, edge=MEASUREMENT_EDGE_COLOR, halo_canvas=halo_canvas)
            else:
                (q,) = inner.qubits
                _draw_single_qubit_gate(ax, x, n_qubits-1-q, inner_name, fill, intensity, edge=MEASUREMENT_EDGE_COLOR, halo_canvas=halo_canvas)
            meas = _preceding_meas(op.condition.index, op.t)
            if meas is not None:
                draw_classical_control_wire(ax, float(meas.t), n_qubits-1-meas.qubit, x, n_qubits-1-inner.qubits[0])
            continue

        if not isinstance(op, Operation):
            continue

        name = op.gate.name.upper()
        fill = gate_color(name)

        if name in ("H", "S", "S†", "SDG", "S_DAG", "X", "Y", "Z", "T", "T†", "T_DAG", "TDG"):
            (q,) = op.qubits
            y = n_qubits - 1 - q
            _draw_single_qubit_gate(ax, x, y, name, fill, intensity, halo_canvas=halo_canvas)

        elif name in ("CX", "CNOT"):
            q_ctrl, q_tgt = op.qubits
            y_ctrl = n_qubits - 1 - q_ctrl
            y_tgt = n_qubits - 1 - q_tgt
            _draw_cnot_gate(ax, x, y_ctrl, y_tgt, fill, intensity, halo_canvas=halo_canvas)

        elif name == "CZ":
            q1, q2 = op.qubits
            y1 = n_qubits - 1 - q1
            y2 = n_qubits - 1 - q2
            _draw_cz_gate(ax, x, y1, y2, fill, intensity, halo_canvas=halo_canvas)

        elif name == "SWAP":
            q1, q2 = op.qubits
            y1 = n_qubits - 1 - q1
            y2 = n_qubits - 1 - q2
            _draw_swap_gate(ax, x, y1, y2, fill, intensity, halo_canvas=halo_canvas)

        elif name in ("CS", "CS_DAG"):
            q_ctrl, q_tgt = op.qubits
            y_ctrl = n_qubits - 1 - q_ctrl
            y_tgt = n_qubits - 1 - q_tgt
            _draw_cs_gate(ax, x, y_ctrl, y_tgt, fill, intensity, halo_canvas=halo_canvas)

        elif name == "CCZ":
            q1, q2, q3 = op.qubits
            y1 = n_qubits - 1 - q1
            y2 = n_qubits - 1 - q2
            y3 = n_qubits - 1 - q3
            _draw_ccz_gate(ax, x, y1, y2, y3, fill, intensity, halo_canvas=halo_canvas)

        elif name in ("I", "IDLE"):
            pass  # IDLE decoherence is shown via wire halos

    halo_canvas.flush(ax)

    # --- Qubit labels ---
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.text(
            HEATMAP_QUBIT_LABEL_X, y, f"q{q}",
            va="center", ha="right",
            fontsize=QUBIT_LABEL_FONT_SIZE, color=QUBIT_LABEL_COLOR,
        )

    # --- Gate colorbar (right) ---
    _add_downstream_burden_colorbar(
        fig,
        ax,
        impact=impact,
        heat_scale=heat_scale,
        heat_vmin=heat_vmin,
        heat_vmax=heat_vmax,
        show_qualitative_ticks=show_qualitative_ticks,
        burden_label=(
            "Expected gate error burden / shot"
            if impact_metric == "expected"
            else "Worst-case gate error burden / shot"
        ),
    )

    # --- T1/T2 idle decoherence colorbar (bottom) — only when IDLE segments exist ---
    ax.tick_params(axis="x", pad=4)
    if noise_config is not None and segments:
        _add_idle_decoherence_colorbar(
            fig,
            ax,
            segments=segments,
            raw_vals=wire_raw_vals,
        )

    # --- Axes formatting ---
    ax.set_xlim(HEATMAP_XLIM_LEFT, n_layers + HEATMAP_XLIM_RIGHT_OFFSET)
    ax.set_ylim(HEATMAP_YLIM_BOTTOM, n_qubits + HEATMAP_YLIM_TOP_OFFSET)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(
        [f"t={t}" for t in range(n_layers)], fontsize=HEATMAP_LABEL_FONT_SIZE
    )
    ax.set_yticks([])
    ax.set_frame_on(False)

    has_measurements = any(isinstance(op, Measurement) for op in circuit.operations)
    mode_suffix = f" [{mcm_mode}]" if has_measurements else ""
    if title:
        ax.set_title(
            title + mode_suffix,
            fontsize=CHART_TITLE_FONT_SIZE, pad=HEATMAP_TITLE_PAD,
        )
    else:
        ax.set_title(
            f"Error heatmap — {result.n_shots} shots{mode_suffix}",
            fontsize=CHART_TITLE_FONT_SIZE, pad=HEATMAP_TITLE_PAD,
        )

    if purity_rho is not None:
        from ..purity_overlay import per_qubit_purities, annotate_axes_with_purities
        purs = per_qubit_purities(purity_rho, n_qubits)
        annotate_axes_with_purities(ax, purs)

    if finalize_layout and fig is not None:
        fig.tight_layout()
        # tight_layout doesn't account for the idle-decoherence inset colorbar,
        # which is placed IDLE_CBAR_BBOX_Y axes-units below the main axes.
        # Enforce a minimum bottom margin so the colorbar stays inside the figure.
        if noise_config is not None and segments:
            fig.subplots_adjust(
                bottom=max(fig.subplotpars.bottom, 0.30)
            )
    return fig


def interactive_heatmap(
    result: AggregateResult,
    circuit: Circuit,
    *,
    noise_config: Optional[dict] = None,
    wire_halo_metric: Optional[Callable] = None,
    display_mode: Literal["panel", "annotate"] = "panel",
    title: Optional[str] = None,
    mcm_mode: str = "qec",
    impact_metric: Literal["expected", "worst_case"] = "expected",
    hover_context: Optional[dict] = None,
):
    """Render the static error heatmap with hover/click info on gates,
    qubit endcaps, and wire segments.

    Parameters
    ----------
    result           : AggregateResult from ManyShotRunner.run()
    circuit          : The circuit that was simulated
    noise_config     : Per-op noise dict; drives wire halos and hover details
    wire_halo_metric : Optional (WireSegment) -> float override
    display_mode     : "panel"    → ipywidgets.Output box next to figure
                       "annotate" → matplotlib annotation following cursor
    title            : Optional figure title
    mcm_mode         : MCM branch mode — "qec", "worst_case", or "average".

    Returns
    -------
    A handle object with .figure and mode-specific attributes.
    """
    # Compute segments once; reuse for both drawing and hit-testing.
    segments = _compute_wire_segments(circuit, noise_config) if noise_config else []

    fig = plot_error_heatmap(
        result,
        circuit,
        noise_config=noise_config,
        wire_halo_metric=wire_halo_metric,
        title=title,
        mcm_mode=mcm_mode,
        impact_metric=impact_metric,
        finalize_layout=False,
    )
    ax = fig.axes[0]

    # Extend xlim so the endcap region is reachable by the cursor.
    n_layers = (max(op.t for op in circuit.operations) + 1) if circuit.operations else 1
    ax.set_xlim(left=HEATMAP_XLIM_LEFT, right=n_layers + 0.6)

    gate_bboxes = _build_gate_bboxes(circuit)
    endcap_bboxes = _build_endcap_bboxes(circuit)
    wire_bboxes = _build_wire_segment_bboxes(segments, circuit.n_qubits)

    # Compute impact and visual intensities for hover panels using the same
    # parameters as the plot, so the displayed numbers match what is drawn.
    from ..noise_metrics import normalize_heat_values
    hover_impact = _compute_downstream_impact(
        circuit,
        noise_config=noise_config,
        mcm_mode=mcm_mode,
        impact_metric=impact_metric,
    )
    hover_intensities = normalize_heat_values(
        hover_impact,
        mode="relative",
        floor=HALO_INTENSITY_FLOOR,
        gamma=HALO_GAMMA,
    )

    if display_mode == "panel":
        return _attach_panel_hover(
            fig, ax, circuit, result, noise_config,
            gate_bboxes, endcap_bboxes, wire_bboxes,
            impact=hover_impact,
            intensities=hover_intensities,
            impact_metric=impact_metric,
            hover_context=hover_context,
        )
    elif display_mode == "annotate":
        return _attach_annotate_hover(
            fig, ax, circuit, result, noise_config,
            gate_bboxes, endcap_bboxes, wire_bboxes,
            impact=hover_impact,
            intensities=hover_intensities,
            impact_metric=impact_metric,
            hover_context=hover_context,
        )
    else:
        raise ValueError(
            f"Unknown display_mode={display_mode!r}; expected 'panel' or 'annotate'."
        )


def plot_error_heatmap_side_by_side(
    result: AggregateResult,
    circuit: Circuit,
    *,
    noise_config: Optional[dict] = None,
    wire_halo_metric: Optional[Callable] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Side-by-side heatmap: worst-case MCM (left) vs QEC-corrected (right).

    Creates a single Figure with two subplots.  The left panel shows the
    downstream impact assuming corrections never fire (maximises error spread);
    the right panel shows impact after ideal QEC corrections cancel detected
    errors.  Useful for comparing best- and worst-case MCM scenarios at a glance.

    Parameters
    ----------
    result           : AggregateResult from ManyShotRunner.run()
    circuit          : The circuit that was simulated
    noise_config     : Per-op noise dict (optional; drives wire halos)
    wire_halo_metric : Optional (WireSegment) -> float override
    title            : Optional super-title above both panels
    figsize          : Figure size override (auto-computed when None)

    Returns
    -------
    A single matplotlib Figure containing both panels.
    """
    n_qubits = circuit.n_qubits
    n_layers = (max(op.t for op in circuit.operations) + 1) if circuit.operations else 1

    if figsize is None:
        single_w = (
            max(CIRCUIT_MIN_WIDTH, CIRCUIT_WIDTH_PER_LAYER * n_layers)
            + HEATMAP_FIGSIZE_WIDTH_PAD
        )
        single_h = (
            CIRCUIT_HEIGHT_PER_QUBIT * n_qubits
            + CIRCUIT_BASE_HEIGHT_PAD
            + HEATMAP_FIGSIZE_HEIGHT_PAD
        )
        figsize = (single_w * 2 + 1.0, single_h)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    plot_error_heatmap(
        result, circuit,
        noise_config=noise_config,
        wire_halo_metric=wire_halo_metric,
        title="Worst-case MCM",
        mcm_mode="worst_case",
        ax=ax_left,
    )
    plot_error_heatmap(
        result, circuit,
        noise_config=noise_config,
        wire_halo_metric=wire_halo_metric,
        title="QEC MCM",
        mcm_mode="qec",
        ax=ax_right,
    )

    if title:
        fig.suptitle(title, fontsize=CHART_TITLE_FONT_SIZE)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pauli propagation helpers (downstream impact)
# ---------------------------------------------------------------------------

def _enumerate_initial_patterns(qubits: tuple) -> list:
    paulis = ('X', 'Y', 'Z')
    patterns = []
    if len(qubits) == 1:
        q = qubits[0]
        for p in paulis:
            patterns.append({q: p})
    else:
        q0, q1 = qubits[0], qubits[1]
        for p0 in ('I', 'X', 'Y', 'Z'):
            for p1 in ('I', 'X', 'Y', 'Z'):
                if p0 == 'I' and p1 == 'I':
                    continue
                state: dict = {}
                if p0 != 'I':
                    state[q0] = p0
                if p1 != 'I':
                    state[q1] = p1
                patterns.append(state)
    return patterns


def _propagate_pauli_through_gate(pauli_state: dict, op) -> dict | None:
    name = op.gate.name.upper()

    if name in ('T', 'T_DAG', 'TDG'):
        return None

    result = dict(pauli_state)

    if name in _SINGLE_QUBIT_PROPAGATION:
        table = _SINGLE_QUBIT_PROPAGATION[name]
        q = op.qubits[0]
        old = result.get(q, 'I')
        new = table[old]
        if new == 'I':
            result.pop(q, None)
        else:
            result[q] = new

    elif name in ('CNOT', 'CX'):
        ctrl, tgt = op.qubits[0], op.qubits[1]
        p_ctrl = result.get(ctrl, 'I')
        p_tgt = result.get(tgt, 'I')
        new_ctrl, new_tgt = _CNOT_PROPAGATION_TABLE[(p_ctrl, p_tgt)]
        for q, val in ((ctrl, new_ctrl), (tgt, new_tgt)):
            if val == 'I':
                result.pop(q, None)
            else:
                result[q] = val

    elif name == 'CZ':
        q1, q2 = op.qubits[0], op.qubits[1]
        p1 = result.get(q1, 'I')
        p2 = result.get(q2, 'I')
        new_p1, new_p2 = _CZ_PROPAGATION_TABLE[(p1, p2)]
        for q, val in ((q1, new_p1), (q2, new_p2)):
            if val == 'I':
                result.pop(q, None)
            else:
                result[q] = val

    elif name == 'SWAP':
        q1, q2 = op.qubits[0], op.qubits[1]
        p1 = result.get(q1, 'I')
        p2 = result.get(q2, 'I')
        new_p1, new_p2 = _SWAP_PROPAGATION_TABLE[(p1, p2)]
        for q, val in ((q1, new_p1), (q2, new_p2)):
            if val == 'I':
                result.pop(q, None)
            else:
                result[q] = val

    return result


# ---------------------------------------------------------------------------
# Channel-resolved expected burden helpers
# ---------------------------------------------------------------------------

_PAULI_PRODUCT_TABLE: dict[tuple[str, str], str] = {
    # Phases are intentionally discarded; the metric tracks Pauli support only.
    ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
    ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
    ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
    ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I',
}


def _state_key(state: dict[int, str]) -> tuple[tuple[int, str], ...]:
    """Hashable representation of a Pauli state dictionary."""
    return tuple(sorted((int(q), p) for q, p in state.items() if p != 'I'))


def _compose_pauli_states(left: dict[int, str], right: dict[int, str]) -> dict[int, str]:
    """Multiply two Pauli states qubitwise, ignoring global phase."""
    out = dict(left)
    for q, p_right in right.items():
        p_left = out.get(q, 'I')
        p_new = _PAULI_PRODUCT_TABLE[(p_left, p_right)]
        if p_new == 'I':
            out.pop(q, None)
        else:
            out[q] = p_new
    return out


def _merge_pauli_outcomes(
    outcomes: list[tuple[float, dict[int, str]]],
    *,
    probability_floor: float = 0.0,
) -> list[tuple[float, dict[int, str]]]:
    """Combine duplicate Pauli states after channel composition."""
    merged: dict[tuple[tuple[int, str], ...], float] = {}
    for prob, state in outcomes:
        prob = float(prob)
        if prob <= probability_floor:
            continue
        key = _state_key(state)
        merged[key] = merged.get(key, 0.0) + prob
    return [
        (prob, dict(key))
        for key, prob in merged.items()
        if prob > probability_floor
    ]


def _pauli_error_outcomes_for_targets(
    channel, op_qubits: tuple[int, ...]
) -> list[tuple[float, dict[int, str]]]:
    """Enumerate independent PauliError outcomes on every operation qubit."""
    p_i = max(0.0, 1.0 - float(channel.p_x + channel.p_y + channel.p_z))
    one_qubit = [
        ('I', p_i),
        ('X', float(channel.p_x)),
        ('Y', float(channel.p_y)),
        ('Z', float(channel.p_z)),
    ]
    outcomes: list[tuple[float, dict[int, str]]] = []
    for sampled in product(one_qubit, repeat=len(op_qubits)):
        prob = 1.0
        state: dict[int, str] = {}
        for q, (pauli_char, p_char) in zip(op_qubits, sampled):
            prob *= p_char
            if pauli_char != 'I':
                state[int(q)] = pauli_char
        outcomes.append((prob, state))
    return _merge_pauli_outcomes(outcomes)


def _correlated_pauli_outcomes_for_targets(
    channel, op_qubits: tuple[int, ...]
) -> list[tuple[float, dict[int, str]]]:
    """Enumerate joint CorrelatedPauliError outcomes on the channel target qubits."""
    if len(op_qubits) < channel.num_qubits:
        raise ValueError(
            f"CorrelatedPauliError with {channel.num_qubits} Pauli characters "
            f"cannot be mapped onto operation qubits {op_qubits!r}."
        )
    target_qubits = tuple(int(q) for q in op_qubits[:channel.num_qubits])
    total_listed = sum(float(p) for p in channel.probs.values())
    outcomes: list[tuple[float, dict[int, str]]] = [(max(0.0, 1.0 - total_listed), {})]
    for pauli_string, prob in channel.probs.items():
        state = {
            q: p_char
            for q, p_char in zip(target_qubits, pauli_string)
            if p_char != 'I'
        }
        outcomes.append((float(prob), state))
    return _merge_pauli_outcomes(outcomes)


def _kraus_channel_pauli_proxy(channel):
    """Return a Pauli-compatible visualization proxy for supported Kraus channels."""
    from ...noise.pauli_error import PauliError
    from ...noise.amplitude_damping import AmplitudeDamping
    from ...noise.t2_dephasing import Dephasing

    if isinstance(channel, AmplitudeDamping):
        p = float(channel.gamma) / 4.0
        return PauliError(p_x=p, p_y=p, p_z=p)

    if isinstance(channel, Dephasing):
        return PauliError(p_x=0.0, p_y=0.0, p_z=float(channel.lam) / 2.0)

    return None


def _channel_pauli_outcomes(
    channel,
    op_qubits: tuple[int, ...],
    *,
    probability_floor: float = 0.0,
) -> list[tuple[float, dict[int, str]]]:
    """Enumerate Pauli outcomes for a channel as (probability, Pauli-state) pairs.

    Returned outcomes include the identity outcome. Callers computing expected
    propagated burden should skip entries where the state dict is empty.
    """
    from ...noise.pauli_error import PauliError
    from ...noise.correlated_errors import CorrelatedPauliError
    from ...noise.coherent_errors import CoherentRotation, StochasticCoherentRotation
    from ...noise.kraus_channels import CombinedChannel

    if channel is None:
        return [(1.0, {})]

    if isinstance(channel, CombinedChannel):
        outcomes: list[tuple[float, dict[int, str]]] = [(1.0, {})]
        for sub in channel.channels:
            sub_outcomes = _channel_pauli_outcomes(
                sub, op_qubits, probability_floor=probability_floor,
            )
            composed: list[tuple[float, dict[int, str]]] = []
            for p_left, state_left in outcomes:
                for p_right, state_right in sub_outcomes:
                    composed.append((
                        p_left * p_right,
                        _compose_pauli_states(state_left, state_right),
                    ))
            outcomes = _merge_pauli_outcomes(composed, probability_floor=probability_floor)
        return outcomes

    if isinstance(channel, PauliError):
        return _pauli_error_outcomes_for_targets(channel, op_qubits)

    if isinstance(channel, CorrelatedPauliError):
        return _correlated_pauli_outcomes_for_targets(channel, op_qubits)

    if isinstance(channel, (CoherentRotation, StochasticCoherentRotation)):
        return _channel_pauli_outcomes(
            channel.to_pauli_error(), op_qubits, probability_floor=probability_floor,
        )

    proxy = _kraus_channel_pauli_proxy(channel)
    if proxy is not None:
        return _channel_pauli_outcomes(proxy, op_qubits, probability_floor=probability_floor)

    return [(1.0, {})]


def _downstream_spread_for_initial_state(
    ops: list,
    start_k: int,
    initial_state: dict[int, str],
    *,
    mcm_mode: str,
    circuit: Circuit,
    noise_config: Optional[dict],
) -> float:
    """Return downstream spread count for one concrete Pauli error outcome."""
    if mcm_mode == "average":
        count_qec = _downstream_walk(ops, start_k, initial_state, "qec")
        count_worst = _downstream_walk(ops, start_k, initial_state, "worst_case")
        p_wrong = _p_wrong_for_average(start_k, ops, circuit, noise_config)
        return count_qec * (1.0 - p_wrong) + count_worst * p_wrong
    return _downstream_walk(ops, start_k, initial_state, mcm_mode)


def _expected_propagated_burden_for_op(
    ops: list,
    start_k: int,
    op: Operation,
    channel,
    *,
    circuit: Circuit,
    noise_config: Optional[dict],
    mcm_mode: str,
) -> float:
    """Compute Σ_E p(E) × 1 + downstream_spread(E) for one operation."""
    total = 0.0
    for probability, initial_state in _channel_pauli_outcomes(channel, tuple(op.qubits)):
        if probability <= 0.0 or not initial_state:
            continue
        spread = _downstream_spread_for_initial_state(
            ops, start_k, initial_state,
            mcm_mode=mcm_mode, circuit=circuit, noise_config=noise_config,
        )
        total += float(probability) * (1.0 + float(spread))
    return total


# ---------------------------------------------------------------------------
# Pauli propagation helpers (downstream impact)
# ---------------------------------------------------------------------------


def _downstream_walk(
    ops: list,
    start_k: int,
    initial_state: dict,
    mcm_mode: str,
) -> float:
    """Walk ops[start_k+1:] for one initial Pauli pattern; return impacted gate count.

    For circuits with Measurement / ConditionalOp ops the walk branches based on
    mcm_mode:
      "qec"        — correction fires when anti-commuting error is detected.
      "worst_case" — correction fires only when it should NOT (maximises impact).

    ConditionalOps whose measurement hasn't been seen yet in this walk are skipped.
    """
    count = 0
    pauli_state = dict(initial_state)
    # Maps cbit.index → bool (True = anti-commutes = error detected on meas qubit)
    measurement_outcomes: dict[int, bool] = {}

    for downstream_op in ops[start_k + 1:]:
        if isinstance(downstream_op, Measurement):
            pauli_on_meas = pauli_state.get(downstream_op.qubit, 'I')
            anti_commutes = pauli_on_meas in ('X', 'Y')
            measurement_outcomes[downstream_op.cbit.index] = anti_commutes
            continue

        if isinstance(downstream_op, ConditionalOp):
            cbit_idx = downstream_op.condition.index
            if cbit_idx not in measurement_outcomes:
                continue  # measurement not yet seen in this walk
            anti_commutes = measurement_outcomes[cbit_idx]
            # actual cbit = 1 if anti_commutes (error flips Z-basis outcome), 0 otherwise.
            # The correction fires when actual_cbit == downstream_op.value.
            # worst_case inverts this so the correction fires exactly when it should not.
            correction_intended = anti_commutes == bool(downstream_op.value)
            apply_correction = correction_intended if mcm_mode == "qec" else (not correction_intended)
            if apply_correction:
                new_state = _propagate_pauli_through_gate(pauli_state, downstream_op.inner)
                if new_state is not None:
                    pauli_state = new_state
                if not pauli_state:
                    break
            continue

        # Skip any non-Operation type not already handled above (e.g. duck-typed stubs)
        if not isinstance(downstream_op, Operation):
            continue

        # Regular Operation
        if any(q in pauli_state for q in downstream_op.qubits):
            count += 1
        pauli_state = _propagate_pauli_through_gate(pauli_state, downstream_op)
        if pauli_state is None or not pauli_state:
            break

    return count


def _first_measurement_p_wrong(
    start_k: int,
    ops: list,
    noise_config: Optional[dict],
) -> float:
    """Estimate the probability of a wrong measurement at the first Measurement
    after op start_k, by summing p_x + p_y from PauliError channels on the
    measured qubit across all operations before the measurement.

    Returns 0.5 when noise_config is None (uninformative prior).
    Returns 0.0 when there are no measurements after start_k.
    """
    if noise_config is None:
        return 0.5

    from ...noise.pauli_error import PauliError

    # Find the first Measurement after start_k
    meas_qubit: Optional[int] = None
    meas_idx: Optional[int] = None
    for j, op in enumerate(ops[start_k + 1:], start=start_k + 1):
        if isinstance(op, Measurement):
            meas_qubit = op.qubit
            meas_idx = j
            break

    if meas_qubit is None:
        return 0.0  # no measurement downstream

    p_xy = 0.0
    for i in range(meas_idx):
        op = ops[i]
        if not isinstance(op, Operation):
            continue
        if meas_qubit not in op.qubits:
            continue
        for sub in _iterate_channels(noise_config.get(i)):
            if isinstance(sub, PauliError):
                p_xy += sub.p_x + sub.p_y

    return min(p_xy, 0.5)  # cap at 0.5 (maximum disorder)


def _p_wrong_for_average(
    start_k: int,
    ops: list,
    circuit: Circuit,
    noise_config: Optional[dict],
) -> float:
    """Return the probability of a wrong Z-basis measurement for the first
    Measurement after op *start_k*, used to weight average-mode MCM impact.

    Delegates to GateInfoExtractor._cumulative_decoherence so that the
    heatmap average-mode weighting and the hover-panel display share a
    single computation path (M8 integration).

    Returns 0.5 when noise_config is None (uninformative prior, matching
    the behaviour of _first_measurement_p_wrong).
    Returns 0.0 when there are no Measurements after start_k.
    """
    if noise_config is None:
        # Uninformative prior — use 0.5 (average of both branches equally).
        return 0.5

    for j, op in enumerate(ops[start_k + 1:], start=start_k + 1):
        if isinstance(op, Measurement):
            cum = GateInfoExtractor._cumulative_decoherence(j, circuit, noise_config)
            return cum.get(f"q{op.qubit}", {}).get("p_wrong_measurement", 0.0)

    return 0.0  # no Measurement downstream


def _compute_worst_case_downstream_impact(
    circuit: Circuit,
    noise_config: Optional[dict] = None,
    mcm_mode: str = "qec",
) -> np.ndarray:
    """Legacy max-spread × total-event-probability heuristic."""
    ops = circuit.operations
    n_ops = len(ops)
    impact = np.zeros(n_ops, dtype=float)

    for k in range(n_ops):
        if not isinstance(ops[k], Operation):
            continue
        best = 0.0
        for initial_state in _enumerate_initial_patterns(ops[k].qubits):
            count = _downstream_spread_for_initial_state(
                ops, k, initial_state,
                mcm_mode=mcm_mode, circuit=circuit, noise_config=noise_config,
            )
            if count > best:
                best = count
        impact[k] = best

    if noise_config is not None:
        from ..noise_metrics import channel_event_probability
        for k in range(n_ops):
            if not isinstance(ops[k], Operation):
                continue
            p_total = channel_event_probability(noise_config.get(k))
            impact[k] = impact[k] * p_total if p_total > 0 else 0.0

    return impact


def _compute_expected_downstream_impact(
    circuit: Circuit,
    noise_config: Optional[dict] = None,
    mcm_mode: str = "qec",
) -> np.ndarray:
    """Compute expected propagated Pauli burden per operation.

    For each operation k this computes:

        impact[k] = Σ_E  p(E at k) × downstream_spread(E)

    where E is a concrete Pauli/twirled error outcome from the channel attached
    to operation k.  Falls back to the worst-case geometry score when no
    noise_config is supplied (no channel probabilities to average over).
    """
    if noise_config is None:
        return _compute_worst_case_downstream_impact(circuit, noise_config=None, mcm_mode=mcm_mode)

    ops = circuit.operations
    n_ops = len(ops)
    impact = np.zeros(n_ops, dtype=float)

    for k, op in enumerate(ops):
        if not isinstance(op, Operation):
            continue
        channel = noise_config.get(k)
        if channel is None:
            continue
        impact[k] = _expected_propagated_burden_for_op(
            ops, k, op, channel,
            circuit=circuit, noise_config=noise_config, mcm_mode=mcm_mode,
        )

    return impact


def _compute_downstream_impact(
    circuit: Circuit,
    noise_config: Optional[dict] = None,
    mcm_mode: str = "qec",
    impact_metric: Literal["expected", "worst_case"] = "expected",
) -> np.ndarray:
    """Dispatch to channel-resolved expected or legacy worst-case impact.

    Parameters
    ----------
    circuit       : The simulated circuit (may contain Measurement / ConditionalOp).
    noise_config  : Per-op noise dict. Required for channel-resolved expectation.
    mcm_mode      : How to handle mid-circuit measurements.
                    "qec"        — correction fires when error is detected.
                    "worst_case" — correction fires when it should not.
                    "average"    — weighted average of both branches.
    impact_metric : "expected" computes Σ p(E) × downstream_spread(E) over the
                    actual Pauli/twirled channel outcomes.
                    "worst_case" preserves the max-spread × total-p heuristic.
    """
    if impact_metric == "expected":
        return _compute_expected_downstream_impact(circuit, noise_config=noise_config, mcm_mode=mcm_mode)
    if impact_metric == "worst_case":
        return _compute_worst_case_downstream_impact(circuit, noise_config=noise_config, mcm_mode=mcm_mode)
    raise ValueError(
        f"Unknown impact_metric={impact_metric!r}; expected 'expected' or 'worst_case'."
    )


# ---------------------------------------------------------------------------
# Halo canvas — composites all halos into one imshow to avoid ipympl overhead
# ---------------------------------------------------------------------------

class _HaloCanvas:
    """Single-image accumulator for gate and wire halos.

    Replaces one ax.imshow per halo with a single composite call, eliminating
    the ipympl serialization bottleneck on large interactive circuits.
    Paint wire halos first, then gate halos; call flush() once after all halos.
    """

    _NX: int = 500

    def __init__(self, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> None:
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._y_lo = y_lo
        self._y_hi = y_hi
        self._dx = max(x_hi - x_lo, 1e-9)
        self._dy = max(y_hi - y_lo, 1e-9)
        nx = self._NX
        ny = max(1, round(nx * self._dy / self._dx))
        self._nx = nx
        self._ny = ny
        self._img: np.ndarray = np.zeros((ny, nx, 4), dtype=np.float64)

    def _px_range(
        self, x_lo: float, x_hi: float, y_lo: float, y_hi: float
    ) -> tuple[int, int, int, int]:
        i_lo = max(0, int((x_lo - self._x_lo) / self._dx * self._nx))
        i_hi = min(self._nx, int((x_hi - self._x_lo) / self._dx * self._nx) + 1)
        j_lo = max(0, int((y_lo - self._y_lo) / self._dy * self._ny))
        j_hi = min(self._ny, int((y_hi - self._y_lo) / self._dy * self._ny) + 1)
        return i_lo, i_hi, j_lo, j_hi

    def _sub_grid(
        self, i_lo: int, i_hi: int, j_lo: int, j_hi: int
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(
            self._x_lo + (i_lo + 0.5) / self._nx * self._dx,
            self._x_lo + (i_hi - 0.5) / self._nx * self._dx,
            i_hi - i_lo,
        )
        ys = np.linspace(
            self._y_lo + (j_lo + 0.5) / self._ny * self._dy,
            self._y_lo + (j_hi - 0.5) / self._ny * self._dy,
            j_hi - j_lo,
        )
        return np.meshgrid(xs, ys)

    def _over(
        self,
        j_lo: int, j_hi: int, i_lo: int, i_hi: int,
        r: float, g: float, b: float,
        src_a: np.ndarray,
    ) -> None:
        dst = self._img[j_lo:j_hi, i_lo:i_hi]
        da = dst[:, :, 3]
        out_a = src_a + da * (1.0 - src_a)
        denom = np.maximum(out_a, 1e-9)
        for c, v in enumerate((r, g, b)):
            self._img[j_lo:j_hi, i_lo:i_hi, c] = np.where(
                out_a > 0,
                (v * src_a + dst[:, :, c] * da * (1.0 - src_a)) / denom,
                dst[:, :, c],
            )
        self._img[j_lo:j_hi, i_lo:i_hi, 3] = out_a

    def paint_gate_halo(
        self,
        cx: float, cy: float,
        extent_x: float, extent_y: float,
        sigma_x: float, sigma_y: float,
        r: float, g: float, b: float,
        peak_alpha: float,
    ) -> None:
        i_lo, i_hi, j_lo, j_hi = self._px_range(
            cx - extent_x, cx + extent_x, cy - extent_y, cy + extent_y
        )
        if i_lo >= i_hi or j_lo >= j_hi:
            return
        Xg, Yg = self._sub_grid(i_lo, i_hi, j_lo, j_hi)
        alpha = np.exp(-((Xg - cx) / sigma_x) ** 2 - ((Yg - cy) / sigma_y) ** 2) * peak_alpha
        self._over(j_lo, j_hi, i_lo, i_hi, r, g, b, alpha)

    def paint_wire_halo(
        self,
        x_lo: float, x_hi: float,
        cy: float, height: float, sigma: float,
        r: float, g: float, b: float,
        peak_alpha: float,
    ) -> None:
        i_lo, i_hi, j_lo, j_hi = self._px_range(x_lo, x_hi, cy - height, cy + height)
        if i_lo >= i_hi or j_lo >= j_hi:
            return
        _, Yg = self._sub_grid(i_lo, i_hi, j_lo, j_hi)
        alpha = np.exp(-((Yg - cy) / sigma) ** 2) * peak_alpha
        self._over(j_lo, j_hi, i_lo, i_hi, r, g, b, alpha)

    def flush(self, ax: "plt.Axes", zorder: float = 1.5) -> None:
        """Composite all accumulated halos as a single ax.imshow."""
        if not np.any(self._img[:, :, 3] > 0):
            return

        img = np.clip(self._img, 0.0, 1.0).astype(np.float32, copy=True)

        ax.imshow(
            img,
            extent=(self._x_lo, self._x_hi, self._y_lo, self._y_hi),
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            zorder=zorder,
        )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_gate_halo_gradient(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    intensity: float,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    """Vertical Gaussian halo behind a gate box.

    The halo geometry is specified in INCHES (screen units) and converted
    to data coords via the per-axis density constants. This makes the halo's
    on-screen proportions independent of how many qubits/layers the circuit
    has — without this conversion, a halo that is "circular in data coords"
    renders as a wide horizontal ellipse on screen because each x-data-unit
    occupies ~2.5× more pixels than each y-data-unit (the theme defaults).

    Parameters
    ----------
    ax          : matplotlib Axes to draw into
    x, y        : gate center in DATA coords
    w, h        : gate width/height in DATA coords. Used only to ensure the
                  halo grows to cover multi-qubit gates (CNOT, CZ) — for a
                  single-qubit gate w and h are small and don't trigger the
                  clamps below.
    intensity   : error intensity in [0, 1]; drives both color (via the halo
                  colormap) and peak opacity.
    halo_canvas : when provided, paint into the composite canvas instead of
                  adding a new ax.imshow.
    """
    if intensity <= 0:
        return

    cmap = get_halo_colormap()
    rgba = cmap(intensity)
    r, g, b = rgba[0], rgba[1], rgba[2]

    # Convert inch-based design parameters into HALF-extents in data coords.
    # `CIRCUIT_WIDTH_PER_LAYER` is inches per data-unit in x, so:
    #     half-extent_x [data] = (total inches) / (inches per data-unit) / 2
    extent_x = GATE_HALO_WIDTH_IN  / CIRCUIT_WIDTH_PER_LAYER  / 2.0
    extent_y = GATE_HALO_HEIGHT_IN / CIRCUIT_HEIGHT_PER_QUBIT / 2.0

    # For multi-qubit gates (CNOT/CZ spanning several qubits), `h` can be
    # larger than our default vertical extent. Make sure the halo always
    # comfortably covers the gate box itself.
    extent_y = max(extent_y, h / 2.0 + 0.3)
    extent_x = max(extent_x, w / 2.0 + 0.05)

    sigma_x = extent_x * GATE_HALO_SIGMA_FRAC
    sigma_y = extent_y * GATE_HALO_SIGMA_FRAC
    peak_alpha = GATE_HALO_MAX_ALPHA * intensity

    if halo_canvas is not None:
        halo_canvas.paint_gate_halo(x, y, extent_x, extent_y, sigma_x, sigma_y, r, g, b, peak_alpha)
        return

    # Build an elliptical Gaussian alpha mask. We normalize x and y by their
    # own sigma so the gradient falls off at the same rate in each direction
    # measured in screen pixels — i.e., the halo's "softness" is uniform on
    # screen, not in data coords.
    n = 120
    xs = np.linspace(-extent_x, extent_x, n)
    ys = np.linspace(-extent_y, extent_y, n)
    xg, yg = np.meshgrid(xs, ys)
    rad_sq = (xg / sigma_x) ** 2 + (yg / sigma_y) ** 2
    alpha = np.exp(-rad_sq)
    img = np.zeros((n, n, 4))
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    img[:, :, 3] = alpha * peak_alpha
    ax.imshow(
        img,
        extent=(x - extent_x, x + extent_x,
                y - extent_y, y + extent_y),
        aspect="auto",
        interpolation="bilinear",
        zorder=2,
    )


def _draw_wire_halo(
    ax: plt.Axes,
    x_lo: float,
    x_hi: float,
    y: float,
    intensity: float,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    """Soft vertical Gaussian gradient along a wire segment."""
    if intensity <= 0 or x_hi <= x_lo:
        return

    cmap = get_wire_halo_colormap()
    rgba = cmap(intensity)
    r, g, b = rgba[0], rgba[1], rgba[2]
    peak_alpha = WIRE_HALO_MAX_ALPHA * intensity

    if halo_canvas is not None:
        halo_canvas.paint_wire_halo(
            x_lo, x_hi, y, WIRE_HALO_HEIGHT, WIRE_HALO_SIGMA, r, g, b, peak_alpha
        )
        return

    n_v, n_h = 100, 2
    v_coords = np.linspace(-WIRE_HALO_HEIGHT, WIRE_HALO_HEIGHT, n_v)
    alpha_profile = np.exp(-(v_coords / WIRE_HALO_SIGMA) ** 2)
    img = np.zeros((n_v, n_h, 4))
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    img[:, :, 3] = alpha_profile[:, np.newaxis] * peak_alpha
    ax.imshow(
        img,
        extent=(x_lo, x_hi, y - WIRE_HALO_HEIGHT, y + WIRE_HALO_HEIGHT),
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )


def _draw_measurement_halo(
    ax: plt.Axes, x: float, y: float, intensity: float,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    _draw_gate_halo_gradient(ax, x, y, GATE_HALF_W * 2, GATE_HALF_H * 2, intensity, halo_canvas)
    draw_measurement(ax, x, y)


def _draw_single_qubit_gate(
    ax: plt.Axes, x: float, y: float, name: str, fill: str, intensity: float,
    edge: str | None = None,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    _draw_gate_halo_gradient(ax, x, y, GATE_HALF_W * 2, GATE_HALF_H * 2, intensity, halo_canvas)
    draw_single_gate(ax, x, y, name, fill, edge if edge is not None else GATE_EDGE_COLOR, GATE_EDGE_WIDTH)


def _draw_cnot_gate(
    ax: plt.Axes,
    x: float,
    y_ctrl: float,
    y_tgt: float,
    fill: str,
    intensity: float,
    edge: str | None = None,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    y_min = min(y_ctrl, y_tgt)
    y_max = max(y_ctrl, y_tgt)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity, halo_canvas
    )
    draw_cnot(ax, x, y_ctrl, y_tgt, fill, GATE_EDGE_WIDTH, edge=edge)


def _draw_cz_gate(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    intensity: float,
    edge: str | None = None,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity, halo_canvas
    )
    draw_cz(ax, x, y1, y2, fill, GATE_EDGE_WIDTH, edge=edge)


def _draw_swap_gate(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    intensity: float,
    edge: str | None = None,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity, halo_canvas
    )
    draw_swap(ax, x, y1, y2, fill, GATE_EDGE_WIDTH, edge=edge)


def _draw_cs_gate(
    ax: plt.Axes,
    x: float,
    y_ctrl: float,
    y_tgt: float,
    fill: str,
    intensity: float,
    edge: str | None = None,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    y_min = min(y_ctrl, y_tgt)
    y_max = max(y_ctrl, y_tgt)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity, halo_canvas
    )
    draw_cs(ax, x, y_ctrl, y_tgt, fill, edge if edge is not None else GATE_EDGE_COLOR, GATE_EDGE_WIDTH)


def _draw_ccz_gate(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    y3: float,
    fill: str,
    intensity: float,
    edge: str | None = None,
    halo_canvas: "_HaloCanvas | None" = None,
) -> None:
    y_min = min(y1, y2, y3)
    y_max = max(y1, y2, y3)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity, halo_canvas
    )
    draw_ccz(ax, x, y1, y2, y3, fill, GATE_EDGE_WIDTH, edge=edge)


# ---------------------------------------------------------------------------
# Hit-test bbox builders
# ---------------------------------------------------------------------------

def _build_gate_bboxes(circuit: Circuit) -> dict[int, tuple]:
    """op_idx → (xmin, xmax, ymin, ymax) in axes data coordinates.

    Operation ops get a standard gate box sized from GATE_HALF_W/H.
    Measurement ops also get a bbox (same size, single-qubit position) so
    that hover handlers can surface p_wrong_measurement info for them.
    ConditionalOp ops get a bbox spanning their inner gate's qubit range.
    """
    n_qubits = circuit.n_qubits
    bboxes: dict[int, tuple] = {}
    for op_idx, op in enumerate(circuit.operations):
        if isinstance(op, Measurement):
            x = float(op.t)
            y = n_qubits - 1 - op.qubit
            bboxes[op_idx] = (
                x - GATE_HALF_W, x + GATE_HALF_W,
                y - GATE_HALF_H, y + GATE_HALF_H,
            )
        elif isinstance(op, ConditionalOp):
            x = float(op.t)
            ys = [n_qubits - 1 - q for q in op.inner.qubits]
            y_lo, y_hi = min(ys), max(ys)
            bboxes[op_idx] = (
                x - GATE_HALF_W, x + GATE_HALF_W,
                y_lo - GATE_HALF_H, y_hi + GATE_HALF_H,
            )
        elif isinstance(op, Operation):
            x = float(op.t)
            if len(op.qubits) == 1:
                q = op.qubits[0]
                y = n_qubits - 1 - q
                bboxes[op_idx] = (
                    x - GATE_HALF_W, x + GATE_HALF_W,
                    y - GATE_HALF_H, y + GATE_HALF_H,
                )
            else:
                ys = [n_qubits - 1 - q for q in op.qubits]
                y_lo, y_hi = min(ys), max(ys)
                bboxes[op_idx] = (
                    x - GATE_HALF_W, x + GATE_HALF_W,
                    y_lo - GATE_HALF_H, y_hi + GATE_HALF_H,
                )
    return bboxes


def _build_endcap_bboxes(circuit: Circuit) -> dict[int, tuple]:
    """qubit → bbox for the right-end-of-wire region."""
    n_qubits = circuit.n_qubits
    n_layers = (
        max(op.t for op in circuit.operations) + 1
        if circuit.operations else 1
    )
    boxes: dict[int, tuple] = {}
    x_lo = n_layers - 0.4
    x_hi = n_layers + 0.55
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        boxes[q] = (x_lo, x_hi, y - 0.18, y + 0.18)
    return boxes


def _build_wire_segment_bboxes(
    segments: list[WireSegment],
    n_qubits: int,
) -> dict[int, tuple]:
    """segment_idx → (xmin, xmax, ymin, ymax, WireSegment)."""
    boxes: dict[int, tuple] = {}
    for i, seg in enumerate(segments):
        y = n_qubits - 1 - seg.qubit
        boxes[i] = (seg.t_lo, seg.t_hi, y - 0.18, y + 0.18, seg)
    return boxes


def _hit_test(
    x: Optional[float], y: Optional[float], bboxes: dict
) -> Optional[int]:
    """Return key whose 4-tuple bbox contains (x, y), or None."""
    if x is None or y is None:
        return None
    for key, bbox in bboxes.items():
        xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return key
    return None


def _hit_test_wire_segment(
    x: Optional[float], y: Optional[float], wire_bboxes: dict
) -> Optional[WireSegment]:
    """Return the WireSegment payload of the first matching 5-tuple bbox."""
    if x is None or y is None:
        return None
    for bbox_5 in wire_bboxes.values():
        xmin, xmax, ymin, ymax, seg = bbox_5
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return seg
    return None


# ---------------------------------------------------------------------------
# Hover mode A: ipywidgets.Output panel
# ---------------------------------------------------------------------------

def _attach_panel_hover(
    fig, ax, circuit, result, noise_config,
    gate_bboxes, endcap_bboxes, wire_bboxes,
    impact=None,
    intensities=None,
    impact_metric="expected",
    hover_context=None,
):
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib

    # ------------------------------------------------------------------
    # Backend check.
    #
    # `display_mode="panel"` puts the figure canvas inside an HBox along
    # with an ipywidgets.Output. For that to (a) display correctly and
    # (b) actually receive mouse events, `fig.canvas` must itself be an
    # ipywidget — which only happens under the `widget` (ipympl) or
    # `notebook` (nbAgg) matplotlib backends.
    #
    # Under the default `inline` backend, `fig.canvas` is a FigureCanvasAgg
    # (not a widget), HBox would raise, and no mouse events would ever
    # reach Python anyway. Catching that exception silently used to
    # leave users staring at a static PNG and an info panel that never
    # updates — which made it look like the hover code was broken.
    # Instead, fail loudly with a precise instruction.
    # ------------------------------------------------------------------
    backend = matplotlib.get_backend().lower()
    if "ipympl" not in backend and "nbagg" not in backend and "widget" not in backend:
        raise RuntimeError(
            "interactive_heatmap(display_mode='panel') requires an interactive\n"
            "matplotlib backend, but the current backend is "
            f"{matplotlib.get_backend()!r}.\n\n"
            "Add the following to the TOP of your notebook (before any pyplot\n"
            "imports or plot calls) and restart the kernel:\n\n"
            "    %pip install -q ipympl     # one-time install\n"
            "    %matplotlib widget         # every session\n\n"
            "Alternative: use display_mode='annotate' instead, which renders a\n"
            "matplotlib tooltip and works with any backend (though the\n"
            "tooltip can only be inspected on saved/displayed figures with an\n"
            "interactive backend too — for inline notebooks it's static)."
        )

    fig_h_px = int(fig.get_figheight() * fig.dpi)
    fig_w_px = int(fig.get_figwidth()  * fig.dpi)

    info_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px 12px",
            min_width="500px",
            max_width="600px",
            height=f"{fig_h_px}px",
            overflow_y="auto",
            overflow_x="hidden",
            font_family="monospace",
        ),
    )
    def _replace_panel_text(text: str) -> None:
        """Replace the side-panel contents with exactly one text block."""
        info_out.outputs = (
            {
                "output_type": "stream",
                "name": "stdout",
                "text": text.rstrip() + "\n",
            },
        )

    _replace_panel_text("Click a gate, wire segment, or qubit end-cap to inspect.")

    def on_click(event):
        if event.inaxes is not ax:
            return

        op_idx = _hit_test(event.xdata, event.ydata, gate_bboxes)
        if op_idx is not None:
            info = GateInfoExtractor.for_gate(
                op_idx, result, circuit=circuit, noise_config=noise_config,
                hover_context=hover_context,
            )
            info["op_idx"] = op_idx
            if impact is not None:
                info["gate_halo"] = {
                    "impact_metric": impact_metric,
                    "downstream_burden": float(impact[op_idx]),
                    "visual_intensity": float(intensities[op_idx]) if intensities is not None else 0.0,
                }
            _replace_panel_text(GateInfoExtractor.format_full(info))
            return

        q = _hit_test(event.xdata, event.ydata, endcap_bboxes)
        if q is not None:
            info = GateInfoExtractor.for_qubit_endcap(
                q, circuit, result, noise_config=noise_config,
                hover_context=hover_context,
            )
            _replace_panel_text(_format_endcap(info))
            return

        seg = _hit_test_wire_segment(event.xdata, event.ydata, wire_bboxes)
        if seg is not None and noise_config is not None:
            info = GateInfoExtractor.for_wire_segment(
                seg, event.xdata, circuit, result, noise_config,
                hover_context=hover_context,
            )
            _replace_panel_text(_format_wire_segment(info))

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Set a minimum canvas size so it doesn't shrink below the figure
    # dimensions, but allow it to grow to show the full colorbar area.
    fig.canvas.layout.min_width  = f"{fig_w_px}px"
    fig.canvas.layout.min_height = f"{fig_h_px}px"

    # Horizontal scroll on the container keeps both the circuit and the
    # info panel visible without either element reflowing.
    container = widgets.HBox(
        [fig.canvas, info_out],
        layout=widgets.Layout(
            display="flex",
            flex_flow="row nowrap",
            align_items="flex-start",
            overflow_x="auto",
            width="100%",
        ),
    )
    display(container)

    class _Handle:
        pass
    h = _Handle()
    h.figure = fig
    h._output = info_out
    return h


# ---------------------------------------------------------------------------
# Hover mode B: ax.annotate tooltip
# ---------------------------------------------------------------------------

def _attach_annotate_hover(
    fig, ax, circuit, result, noise_config,
    gate_bboxes, endcap_bboxes, wire_bboxes,
    impact=None,
    intensities=None,
    impact_metric="expected",
    hover_context=None,
):
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#444",
            alpha=0.95,
        ),
        fontsize=8,
        family="monospace",
        zorder=20,
        visible=False,
    )

    def on_motion(event):
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        op_idx = _hit_test(event.xdata, event.ydata, gate_bboxes)
        if op_idx is not None:
            info = GateInfoExtractor.for_gate(
                op_idx, result, circuit=circuit, noise_config=noise_config,
                hover_context=hover_context,
            )
            info["op_idx"] = op_idx
            if impact is not None:
                info["gate_halo"] = {
                    "impact_metric": impact_metric,
                    "downstream_burden": float(impact[op_idx]),
                    "visual_intensity": float(intensities[op_idx]) if intensities is not None else 0.0,
                }
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(GateInfoExtractor.format_compact(info))
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return

        q = _hit_test(event.xdata, event.ydata, endcap_bboxes)
        if q is not None:
            info = GateInfoExtractor.for_qubit_endcap(
                q, circuit, result, noise_config=noise_config,
                hover_context=hover_context,
            )
            c = info.get("cumulative", {})
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(
                f"q{q} (endcap)\n"
                f"ΣT1 {c.get('cumulative_t1_leakage', 0):.1%}"
                f"  ΣT2 {c.get('cumulative_t2_phase_decay', 0):.1%}"
            )
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return

        seg = _hit_test_wire_segment(event.xdata, event.ydata, wire_bboxes)
        if seg is not None and noise_config is not None:
            info = GateInfoExtractor.for_wire_segment(
                seg, event.xdata, circuit, result, noise_config,
                hover_context=hover_context,
            )
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(_format_wire_segment(info))
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return

        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    class _Handle:
        pass
    h = _Handle()
    h.figure = fig
    h._annotation = annot
    return h


# ---------------------------------------------------------------------------
# Formatting helpers for panel mode
# ---------------------------------------------------------------------------

def _format_endcap(info: dict) -> str:
    lines = [f"Qubit q{info['qubit']}  (end of wire)"]

    ctx = info.get("hover_context", {})
    nm = ctx.get("noise_model", {})
    run_m = ctx.get("run_metrics", {})

    if nm:
        name = nm.get("source") or (
            f"{nm['vendor']} {nm['system']}"
            if nm.get("vendor") and nm.get("system")
            else nm.get("system") or nm.get("vendor") or ""
        )
        nm_parts = [p for p in [name,
                                 f"mode={nm['mode']}" if nm.get("mode") else "",
                                 nm.get("representation", "")] if p]
        if nm_parts:
            lines.append("Noise model: " + " | ".join(nm_parts))

    c = info.get("cumulative", {})
    if c:
        t1 = c.get("cumulative_t1_leakage", 0.0)
        t2 = c.get("cumulative_t2_phase_decay", 0.0)
        bit_flip = c.get("cumulative_x_error_prob", 0.0)
        phase_flip = c.get("cumulative_z_error_prob", 0.0)
        lines += ["", "── Cumulative decoherence ──"]
        if t1 > 0 or t2 > 0:
            lines += [
                f"  T1 leakage     {t1:.2%}",
                f"  T2 phase decay {t2:.2%}",
                f"  Total T1 exposure  {c.get('total_t1_exposure_ns', 0):.1f} ns",
                f"  Total T2 exposure  {c.get('total_t2_exposure_ns', 0):.1f} ns",
            ]
        elif bit_flip > 0 or phase_flip > 0:
            lines += [
                f"  Σ bit-flip  {bit_flip:.2%}",
                f"  Σ phase-flip  {phase_flip:.2%}",
            ]

    if "per_qubit_error_rate" in info:
        lines += [
            "",
            f"Total error events: {info['total_error_events']}",
            f"Per-qubit error rate: {info['per_qubit_error_rate']:.4f}",
        ]

    if run_m:
        lines += ["", "── Run context ──"]
        for k, v in run_m.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def _format_wire_segment(info: dict) -> str:
    ps = info.get("per_segment", {})
    ctx = info.get("hover_context", {})
    nm = ctx.get("noise_model", {})
    run_m = ctx.get("run_metrics", {})

    lines = [
        f"Qubit q{info['qubit']} — idle segment",
        f"  t range:  {info['t_range'][0]:.1f} – {info['t_range'][1]:.1f}",
        f"  duration: {ps.get('duration_ns', 0):.1f} ns"
        f"  ({ps.get('idle_op_count', 0)} IDLE ops)",
    ]

    if nm:
        name = nm.get("source") or (
            f"{nm['vendor']} {nm['system']}"
            if nm.get("vendor") and nm.get("system")
            else nm.get("system") or nm.get("vendor") or ""
        )
        nm_parts = [p for p in [name,
                                 f"mode={nm['mode']}" if nm.get("mode") else "",
                                 nm.get("representation", "")] if p]
        if nm_parts:
            lines.append("Noise model: " + " | ".join(nm_parts))

    lines += [
        "",
        HOVER_WIRE_DECOHERENCE_SECTION,
        f"  T1 γ      {ps.get('gamma', 0):.3%}",
        f"  T2 λ      {ps.get('lambda', 0):.3%}",
        f"  {HOVER_WIRE_COMBINED_LABEL}:  {ps.get('intensity', 0):.3%}",
    ]

    cum = info.get("cumulative_up_to_cursor", {})
    if cum:
        lines += [
            "",
            HOVER_CUMULATIVE_CURSOR_SECTION,
            f"  T1 leakage     {cum.get('t1_leakage', 0):.2%}",
            f"  T2 phase decay {cum.get('t2_phase_decay', 0):.2%}",
            f"  T1 exposure    {cum.get('t1_exposure_ns', 0):.1f} ns",
            f"  T2 exposure    {cum.get('t2_exposure_ns', 0):.1f} ns",
        ]

    if run_m:
        lines += ["", "── Run context ──"]
        for k, v in run_m.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)
