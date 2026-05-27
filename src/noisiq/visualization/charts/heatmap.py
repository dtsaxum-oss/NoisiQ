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

from typing import Callable, Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ...ir.circuit import Circuit
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
    QUBIT_LABEL_COLOR,
    QUBIT_LABEL_FONT_SIZE,
    WIRE_COLOR,
    WIRE_HALO_HEIGHT,
    WIRE_HALO_MAX_ALPHA,
    WIRE_HALO_SIGMA,
    WIRE_LINEWIDTH,
    draw_cnot,
    draw_cz,
    draw_single_gate,
    gate_color,
    get_halo_colormap,
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
        ops_on_q = sorted(
            [(i, op) for i, op in enumerate(circuit.operations) if q in op.qubits],
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

    t_lo = circuit.operations[run_idxs[0]].t - 0.5
    t_hi = circuit.operations[run_idxs[-1]].t + 0.5

    total_t_t1 = 0.0
    total_t_t2 = 0.0
    T1_seen: Optional[float] = None
    T2_seen: Optional[float] = None
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

    gamma = (1.0 - float(np.exp(-total_t_t1 / T1_seen))) if T1_seen else 0.0
    lam = (1.0 - float(np.exp(-2.0 * total_t_t2 / T2_seen))) if T2_seen else 0.0
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
) -> plt.Figure:
    """Draw the circuit with halo-color effects around gate boxes and
    soft wire-segment halos for idle-time decoherence.

    Parameters
    ----------
    result           : AggregateResult from ManyShotRunner.run()
    circuit          : The Circuit that was simulated
    noise_config     : Per-op noise dict from HardwareProfile.to_noise_model()
                       or to_pauli_noise_model(). When supplied, wire halos
                       are drawn for idle segments (Kraus-mode noise only;
                       Pauli-twirl dicts have no T1/T2 channels to read).
    wire_halo_metric : Optional callable (WireSegment) -> float in [0,1]
                       overriding the default combined-intensity metric.
    title            : Optional figure title
    ax               : Existing Axes; creates a new figure if None
    figsize          : Figure size override (auto-computed when None)

    Returns
    -------
    The matplotlib Figure.
    """
    if not hasattr(result, 'counts_matrix') or result.counts_matrix is None:
        raise TypeError(
            "plot_error_heatmap requires an AggregateResult with a counts_matrix. "
            "If you used HardwareProfile.to_noise_model() directly, switch to "
            "to_pauli_noise_model(circuit) or to_noise_model(circuit, "
            "representation='pauli_twirl'), which produce PauliError noise models "
            "compatible with ManyShotRunner."
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
    impact = _compute_downstream_impact(circuit)

    if had_errors.any():
        max_impact = float(impact[had_errors].max())
        if max_impact > 0:
            raw = np.where(had_errors, impact / max_impact, 0.0)
        else:
            raw = had_errors.astype(float)
        intensities = np.where(
            had_errors,
            HALO_INTENSITY_FLOOR + (1.0 - HALO_INTENSITY_FLOOR) * raw ** HALO_GAMMA,
            0.0,
        )
    else:
        intensities = np.zeros(n_ops)

    # --- Qubit wires ---
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.plot(
            [-0.5, n_layers - 0.5], [y, y],
            linewidth=WIRE_LINEWIDTH, color=WIRE_COLOR, zorder=1,
        )

    # --- Wire halos (drawn before gates so gates sit on top) ---
    if noise_config is not None:
        metric = wire_halo_metric or (lambda s: s.intensity)
        segments = _compute_wire_segments(circuit, noise_config)
        for seg in segments:
            y = n_qubits - 1 - seg.qubit
            _draw_wire_halo(ax, seg.t_lo, seg.t_hi, y, metric(seg))
    else:
        segments = []

    # --- Gates ---
    for op_idx, op in enumerate(circuit.operations):
        x = float(op.t)
        name = op.gate.name.upper()
        intensity = float(intensities[op_idx])
        fill = gate_color(name)

        if name in ("H", "S", "S†", "SDG", "X", "Y", "Z", "T", "T†", "T_DAG", "TDG"):
            (q,) = op.qubits
            y = n_qubits - 1 - q
            _draw_single_qubit_gate(ax, x, y, name, fill, intensity)

        elif name in ("CX", "CNOT"):
            q_ctrl, q_tgt = op.qubits
            y_ctrl = n_qubits - 1 - q_ctrl
            y_tgt = n_qubits - 1 - q_tgt
            _draw_cnot_gate(ax, x, y_ctrl, y_tgt, fill, intensity)

        elif name == "CZ":
            q1, q2 = op.qubits
            y1 = n_qubits - 1 - q1
            y2 = n_qubits - 1 - q2
            _draw_cz_gate(ax, x, y1, y2, fill, intensity)

        elif name in ("I", "IDLE"):
            pass  # IDLE decoherence is shown via wire halos

    # --- Qubit labels ---
    for q in range(n_qubits):
        y = n_qubits - 1 - q
        ax.text(
            HEATMAP_QUBIT_LABEL_X, y, f"q{q}",
            va="center", ha="right",
            fontsize=QUBIT_LABEL_FONT_SIZE, color=QUBIT_LABEL_COLOR,
        )

    # --- Colorbar ---
    cmap = get_halo_colormap()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax,
                        fraction=HEATMAP_COLORBAR_FRACTION,
                        pad=HEATMAP_COLORBAR_PAD)
    cbar.set_label("Downstream propagation impact", fontsize=HEATMAP_LABEL_FONT_SIZE)
    cbar.set_ticks(HEATMAP_TICK_POSITIONS)
    cbar.set_ticklabels(HEATMAP_TICK_LABELS)

    # --- Axes formatting ---
    ax.set_xlim(HEATMAP_XLIM_LEFT, n_layers + HEATMAP_XLIM_RIGHT_OFFSET)
    ax.set_ylim(HEATMAP_YLIM_BOTTOM, n_qubits + HEATMAP_YLIM_TOP_OFFSET)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(
        [f"t={t}" for t in range(n_layers)], fontsize=HEATMAP_LABEL_FONT_SIZE
    )
    ax.set_yticks([])
    ax.set_frame_on(False)

    if title:
        ax.set_title(title, fontsize=CHART_TITLE_FONT_SIZE, pad=HEATMAP_TITLE_PAD)
    else:
        ax.set_title(
            f"Error heatmap — {result.n_shots} shots",
            fontsize=CHART_TITLE_FONT_SIZE, pad=HEATMAP_TITLE_PAD,
        )

    fig.tight_layout()
    return fig


def interactive_heatmap(
    result: AggregateResult,
    circuit: Circuit,
    *,
    noise_config: Optional[dict] = None,
    wire_halo_metric: Optional[Callable] = None,
    display_mode: Literal["panel", "annotate"] = "panel",
    title: Optional[str] = None,
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

    Returns
    -------
    A handle object with .figure and mode-specific attributes.
    """
    # Compute segments once; reuse for both drawing and hit-testing.
    segments = _compute_wire_segments(circuit, noise_config) if noise_config else []

    fig = plot_error_heatmap(
        result, circuit,
        noise_config=noise_config,
        wire_halo_metric=wire_halo_metric,
        title=title,
    )
    ax = fig.axes[0]

    # Extend xlim so the endcap region is reachable by the cursor.
    n_layers = (max(op.t for op in circuit.operations) + 1) if circuit.operations else 1
    ax.set_xlim(left=HEATMAP_XLIM_LEFT, right=n_layers + 0.6)

    gate_bboxes = _build_gate_bboxes(circuit)
    endcap_bboxes = _build_endcap_bboxes(circuit)
    wire_bboxes = _build_wire_segment_bboxes(segments, circuit.n_qubits)

    if display_mode == "panel":
        return _attach_panel_hover(
            fig, ax, circuit, result, noise_config,
            gate_bboxes, endcap_bboxes, wire_bboxes,
        )
    elif display_mode == "annotate":
        return _attach_annotate_hover(
            fig, ax, circuit, result, noise_config,
            gate_bboxes, endcap_bboxes, wire_bboxes,
        )
    else:
        raise ValueError(
            f"Unknown display_mode={display_mode!r}; expected 'panel' or 'annotate'."
        )


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

    return result


def _compute_downstream_impact(circuit: Circuit) -> np.ndarray:
    ops = circuit.operations
    n_ops = len(ops)
    impact = np.zeros(n_ops, dtype=np.int64)

    for k in range(n_ops):
        best = 0
        for initial_state in _enumerate_initial_patterns(ops[k].qubits):
            count = 0
            pauli_state = dict(initial_state)
            for downstream_op in ops[k + 1:]:
                if any(q in pauli_state for q in downstream_op.qubits):
                    count += 1
                pauli_state = _propagate_pauli_through_gate(pauli_state, downstream_op)
                if pauli_state is None or not pauli_state:
                    break
            if count > best:
                best = count
        impact[k] = best

    return impact


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
    ax        : matplotlib Axes to draw into
    x, y      : gate center in DATA coords
    w, h      : gate width/height in DATA coords. Used only to ensure the
                halo grows to cover multi-qubit gates (CNOT, CZ) — for a
                single-qubit gate w and h are small and don't trigger the
                clamps below.
    intensity : error intensity in [0, 1]; drives both color (via the halo
                colormap) and peak opacity.
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

    # Build an elliptical Gaussian alpha mask. We normalize x and y by their
    # own sigma so the gradient falls off at the same rate in each direction
    # measured in screen pixels — i.e., the halo's "softness" is uniform on
    # screen, not in data coords.
    n = 120
    xs = np.linspace(-extent_x, extent_x, n)
    ys = np.linspace(-extent_y, extent_y, n)
    xg, yg = np.meshgrid(xs, ys)

    sigma_x = extent_x * GATE_HALO_SIGMA_FRAC
    sigma_y = extent_y * GATE_HALO_SIGMA_FRAC
    rad_sq = (xg / sigma_x) ** 2 + (yg / sigma_y) ** 2
    alpha = np.exp(-rad_sq)

    peak_alpha = WIRE_HALO_MAX_ALPHA * intensity

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
) -> None:
    """Soft vertical Gaussian gradient along a wire segment."""
    if intensity <= 0 or x_hi <= x_lo:
        return

    cmap = get_halo_colormap()
    rgba = cmap(intensity)
    r, g, b = rgba[0], rgba[1], rgba[2]

    n_v, n_h = 100, 2
    v_coords = np.linspace(-WIRE_HALO_HEIGHT, WIRE_HALO_HEIGHT, n_v)
    alpha_profile = np.exp(-(v_coords / WIRE_HALO_SIGMA) ** 2)
    peak_alpha = WIRE_HALO_MAX_ALPHA * intensity

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


def _draw_single_qubit_gate(
    ax: plt.Axes, x: float, y: float, name: str, fill: str, intensity: float
) -> None:
    _draw_gate_halo_gradient(ax, x, y, GATE_HALF_W * 2, GATE_HALF_H * 2, intensity)
    draw_single_gate(ax, x, y, name, fill, GATE_EDGE_COLOR, GATE_EDGE_WIDTH)


def _draw_cnot_gate(
    ax: plt.Axes,
    x: float,
    y_ctrl: float,
    y_tgt: float,
    fill: str,
    intensity: float,
) -> None:
    y_min = min(y_ctrl, y_tgt)
    y_max = max(y_ctrl, y_tgt)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity
    )
    draw_cnot(ax, x, y_ctrl, y_tgt, fill, GATE_EDGE_WIDTH)


def _draw_cz_gate(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    intensity: float,
) -> None:
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    _draw_gate_halo_gradient(
        ax, x, (y_min + y_max) / 2, GATE_HALF_W * 2, y_max - y_min, intensity
    )
    draw_cz(ax, x, y1, y2, fill, GATE_EDGE_WIDTH)


# ---------------------------------------------------------------------------
# Hit-test bbox builders
# ---------------------------------------------------------------------------

def _build_gate_bboxes(circuit: Circuit) -> dict[int, tuple]:
    """op_idx → (xmin, xmax, ymin, ymax) in axes data coordinates."""
    n_qubits = circuit.n_qubits
    bboxes: dict[int, tuple] = {}
    for op_idx, op in enumerate(circuit.operations):
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

    info_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            min_width="320px",
            max_width="440px",
            height=f"{int(fig.get_figheight() * 80)}px",
            overflow="auto",
        ),
    )
    with info_out:
        print("Hover or click a gate, wire segment, or the right-end of a wire.")

    def on_motion(event):
        if event.inaxes is not ax:
            return

        op_idx = _hit_test(event.xdata, event.ydata, gate_bboxes)
        if op_idx is not None:
            info = GateInfoExtractor.for_gate(
                op_idx, result, circuit=circuit, noise_config=noise_config,
            )
            info["op_idx"] = op_idx
            with info_out:
                info_out.clear_output(wait=True)
                print(GateInfoExtractor.format_full(info))
            return

        q = _hit_test(event.xdata, event.ydata, endcap_bboxes)
        if q is not None:
            info = GateInfoExtractor.for_qubit_endcap(
                q, circuit, result, noise_config=noise_config,
            )
            with info_out:
                info_out.clear_output(wait=True)
                print(_format_endcap(info))
            return

        seg = _hit_test_wire_segment(event.xdata, event.ydata, wire_bboxes)
        if seg is not None and noise_config is not None:
            info = GateInfoExtractor.for_wire_segment(
                seg, event.xdata, circuit, result, noise_config,
            )
            with info_out:
                info_out.clear_output(wait=True)
                print(_format_wire_segment(info))

    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    # With the interactive backend check above, fig.canvas IS a widget,
    # so HBox accepts it directly. No fallback needed.
    container = widgets.HBox([fig.canvas, info_out])
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
            )
            info["op_idx"] = op_idx
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(GateInfoExtractor.format_compact(info))
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return

        q = _hit_test(event.xdata, event.ydata, endcap_bboxes)
        if q is not None:
            info = GateInfoExtractor.for_qubit_endcap(
                q, circuit, result, noise_config=noise_config,
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
            annot.xy = (event.xdata, event.ydata)
            annot.set_text(
                f"q{seg.qubit} idle  [{seg.t_lo:.1f}–{seg.t_hi:.1f}]\n"
                f"γ={seg.gamma:.2%}  λ={seg.lam:.2%}\n"
                f"combined {seg.intensity:.2%}  dur={seg.duration_ns:.0f} ns"
            )
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
    c = info.get("cumulative", {})
    if c:
        lines += [
            "",
            "── Cumulative decoherence ──",
            f"  T1 leakage     {c.get('cumulative_t1_leakage', 0):.2%}",
            f"  T2 phase decay {c.get('cumulative_t2_phase_decay', 0):.2%}",
            f"  Total T1 exposure  {c.get('total_t1_exposure_ns', 0):.1f} ns",
            f"  Total T2 exposure  {c.get('total_t2_exposure_ns', 0):.1f} ns",
        ]
    if "per_qubit_error_rate" in info:
        lines += [
            "",
            f"Total error events: {info['total_error_events']}",
            f"Per-qubit error rate: {info['per_qubit_error_rate']:.4f}",
        ]
    return "\n".join(lines)


def _format_wire_segment(info: dict) -> str:
    ps = info.get("per_segment", {})
    lines = [
        f"Qubit q{info['qubit']} — idle segment",
        f"  t range:  {info['t_range'][0]:.1f} – {info['t_range'][1]:.1f}",
        f"  duration: {ps.get('duration_ns', 0):.1f} ns"
        f"  ({ps.get('idle_op_count', 0)} IDLE ops)",
        "",
        "── Per-segment decoherence ──",
        f"  T1 γ      {ps.get('gamma', 0):.3%}",
        f"  T2 λ      {ps.get('lambda', 0):.3%}",
        f"  combined  {ps.get('intensity', 0):.3%}",
    ]
    cum = info.get("cumulative_up_to_cursor", {})
    if cum:
        lines += [
            "",
            "── Cumulative (up to cursor) ──",
            f"  T1 leakage     {cum.get('t1_leakage', 0):.2%}",
            f"  T2 phase decay {cum.get('t2_phase_decay', 0):.2%}",
            f"  T1 exposure    {cum.get('t1_exposure_ns', 0):.1f} ns",
            f"  T2 exposure    {cum.get('t2_exposure_ns', 0):.1f} ns",
        ]
    return "\n".join(lines)
