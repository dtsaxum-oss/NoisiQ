"""
Quirk-inspired visual theme constants for NoisiQ.

All visualization modules import colors, fonts, and geometry from here so
the package stays visually consistent without hardcoded values scattered
across files.

Usage:
    from noisiq.visualization.theme import gate_color, halo_color, WIRE_COLOR
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Gate categories
# ---------------------------------------------------------------------------

CLIFFORD_GATES: frozenset[str] = frozenset(
    {"H", "X", "Y", "Z", "S", "S†", "SDG", "S_DAG", "CNOT", "CX", "CZ", "I", "SWAP"}
)
T_GATES: frozenset[str] = frozenset({"T", "T†", "T_DAG", "TDG"})
# Anything not in CLIFFORD_GATES or T_GATES falls into the "other" category.


# ---------------------------------------------------------------------------
# Core palette
# ---------------------------------------------------------------------------

# Qubit wires
WIRE_COLOR: str = "#000000"
WIRE_LINEWIDTH: float = 1.5

# Classical control wires (dashed, from measurement to conditional gate)
CLASSICAL_WIRE_COLOR: str = "#555555"
CLASSICAL_WIRE_LINEWIDTH: float = 1.2
CLASSICAL_WIRE_LINESTYLE: str = "--"

# Measurement gate symbol
MEASUREMENT_FILL_COLOR: str = "#1B5E20"   # deep green
MEASUREMENT_EDGE_COLOR: str = "#0A3D00"
CBIT_LABEL_FONT_SIZE: int = 7

# Gate box fill colors
CLIFFORD_GATE_COLOR: str = "#2A2929FF"
T_GATE_COLOR: str = "#2F75B6"
OTHER_GATE_COLOR: str = "#1F4E79"

# Active T column animation color
ACTIVE_COLUMN_COLOR: str = "#51FD7170"
ACTIVE_COLUMN_PAD_X: float = 0.10
ACTIVE_COLUMN_PAD_Y: float = 0.25

# Gate box border
GATE_EDGE_COLOR: str = "#1A1A2E"
GATE_EDGE_WIDTH: float = 1.5
GATE_HIGHLIGHT_EDGE_COLOR: str = "#FF1744"
GATE_HIGHLIGHT_EDGE_WIDTH: float = 2.5

# Gate label text
GATE_LABEL_COLOR: str = "#FFFFFF"    # white on colored box
GATE_LABEL_FONT_SIZE: int = 10

# Qubit index labels (q0, q1, ...)
QUBIT_LABEL_COLOR: str = "#222222"
QUBIT_LABEL_FONT_SIZE: int = 12

# Pauli error labels (float on the wire between gates)
ERROR_COLOR: str = "#E53935"
ERROR_LABEL_FONT_SIZE: int = 10      # font size for Pauli error labels on wires
ERROR_LABEL_ALPHA: float = 1.0

# Pauli error label badge (the circled letter that rides the wire)
PAULI_ERROR_BG_COLOR: str = "#FFCEFE"   # light pink fill inside the circle badge
PAULI_ERROR_BOX_PAD: float = 0.15       # padding inside the circle boxstyle
PAULI_ERROR_LINEWIDTH: float = 1.0      # edge linewidth of the circle badge
PAULI_ERROR_X_OFFSET: float = 0.4     # rightward shift from gate center so badge clears the gate box
PAULI_ERROR_Y_OFFSET: float = 0.2     # vertical offset from the wire (0 = centered on wire)

# Identity / no-error label
NO_ERROR_COLOR: str = "#AAAAAA"

# Timestep index labels (t0, t1, … shown above gate columns)
TIMESTEP_LABEL_COLOR: str = "#888888"
TIMESTEP_LABEL_FONT_SIZE: int = 8

# ---------------------------------------------------------------------------
# Base circuit figure sizing  (shared by all circuit plot types)
#
# All circuit visualizations — static diagram, animation, heatmap, hardware
# comparison — derive their figure size from these two values plus additive
# padding for whatever extras they carry (colorbar, annotation margin, etc.).
#
#   width  = CIRCUIT_WIDTH_PER_LAYER  × n_layers  [+ per-chart pad]
#   height = CIRCUIT_HEIGHT_PER_QUBIT × n_qubits  [+ per-chart pad]
# ---------------------------------------------------------------------------

CIRCUIT_WIDTH_PER_LAYER: float = 1.1    # inches per gate-column
CIRCUIT_HEIGHT_PER_QUBIT: float = 0.44  # inches per qubit wire
CIRCUIT_MIN_WIDTH: float = 6.0          # floor so tiny circuits aren't microscopic
CIRCUIT_BASE_HEIGHT_PAD: float = 0.6    # breathing room for timestep labels / tight_layout

# ---------------------------------------------------------------------------
# Downstream aggregate burden halo palette — red sequential gradient
# ---------------------------------------------------------------------------

DOWNSTREAM_BURDEN_ZERO_COLOR: str = "#FFF5F5"  # near-white pink — zero burden
DOWNSTREAM_BURDEN_LOW_COLOR:  str = "#FCBBA1"  # light red       — low burden
DOWNSTREAM_BURDEN_MID_COLOR:  str = "#FB6A4A"  # medium red      — moderate burden
DOWNSTREAM_BURDEN_MAX_COLOR:  str = "#A50F15"  # dark red        — peak burden

_DOWNSTREAM_BURDEN_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "noisiq_downstream_burden",
    [
        DOWNSTREAM_BURDEN_ZERO_COLOR,
        DOWNSTREAM_BURDEN_LOW_COLOR,
        DOWNSTREAM_BURDEN_MID_COLOR,
        DOWNSTREAM_BURDEN_MAX_COLOR,
    ],
    N=256,
)

# Gate/qubit impact halos use the red sequential map.
_HALO_COLORMAP = _DOWNSTREAM_BURDEN_COLORMAP

HALO_ZERO_COLOR: str = DOWNSTREAM_BURDEN_ZERO_COLOR
HALO_MAX_COLOR: str = DOWNSTREAM_BURDEN_MAX_COLOR

HALO_ALPHA: float = 0.65            # transparency of the halo patch
HALO_GAMMA: float = 1.0             # power applied after range-normalization (1.0 = linear;
                                    # increase to compress high end, decrease to boost low end)
HALO_INTENSITY_FLOOR: float = 0.35 # minimum colormap intensity for any gate with ≥1 error;
                                    # keeps the lowest-error gate clearly distinct from zero.
                                    # At 0.15 the colormap is near-white (#FFF5F5) with ~11%
                                    # alpha — invisible on a white background. 0.35 maps to
                                    # DOWNSTREAM_BURDEN_LOW_COLOR territory with ~25% alpha,
                                    # which is clearly perceptible for single-qubit gates.

# Wire halo palette — blue/cyan, clearly distinct from the gate thermal palette.
# Used for idle T1/T2 decoherence stretches.
WIRE_HALO_ZERO_COLOR: str = "#0A1628"   # near-black dark navy  — negligible idle error
WIRE_HALO_MID_COLOR:  str = "#0288D1"   # medium blue           — moderate decoherence
WIRE_HALO_MAX_COLOR:  str = "#00E5FF"   # bright cyan           — maximum idle decoherence
_WIRE_HALO_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "noisiq_wire_halo",
    [WIRE_HALO_ZERO_COLOR, WIRE_HALO_MID_COLOR, WIRE_HALO_MAX_COLOR],
    N=256,
)

# Qubit error bar palette — shares the downstream burden red sequential gradient
# so gate halos and qubit error bars speak the same color language.
QUBIT_ERROR_ZERO_COLOR: str = DOWNSTREAM_BURDEN_ZERO_COLOR
QUBIT_ERROR_MID_COLOR:  str = DOWNSTREAM_BURDEN_MID_COLOR
QUBIT_ERROR_MAX_COLOR:  str = DOWNSTREAM_BURDEN_MAX_COLOR
_QUBIT_ERROR_COLORMAP = _DOWNSTREAM_BURDEN_COLORMAP

# Halo padding is split into x/y to produce a visually uniform glow at the
# non-square aspect ratio (CIRCUIT_WIDTH_PER_LAYER / CIRCUIT_HEIGHT_PER_QUBIT = 2.5).
# HALO_PAD_Y = HALO_PAD_X × (1.1 / 0.44) keeps both paddings equal in physical inches.
HALO_PAD_X: float = 0.10           # horizontal padding around gate box (data units)
HALO_PAD_Y: float = 0.25           # vertical padding around gate box (data units)

# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------

GATE_SIZE: float = 0.46           # single value — gates are square
GATE_WIDTH: float = GATE_SIZE
GATE_HEIGHT: float = GATE_SIZE
GATE_HALF_W: float = GATE_SIZE / 2
GATE_HALF_H: float = GATE_SIZE / 2
CONTROL_DOT_SIZE: float = 70.0       # scatter s= value
TARGET_CIRCLE_RADIUS: float = GATE_SIZE / 5   # ⊕ circle same width as a gate box

# ---------------------------------------------------------------------------
# Wire halo and gate halo gradient rendering
# ---------------------------------------------------------------------------

WIRE_HALO_MAX_ALPHA: float = 0.55   # peak opacity at the wire centerline
WIRE_HALO_HEIGHT: float = 0.35      # vertical extent in data coords
WIRE_HALO_SIGMA: float = 0.18       # Gaussian std-dev controlling fade rate

# Gate halo dimensions are specified in INCHES (physical screen units) rather
# than data coords. This decouples the halo's apparent proportions from the
# non-square data aspect (CIRCUIT_WIDTH_PER_LAYER / CIRCUIT_HEIGHT_PER_QUBIT
# = 1.1 / 0.44 = 2.5), which would otherwise squash a data-coord circle into
# a horizontal ellipse on screen. To get a clearly VERTICAL halo, HEIGHT_IN
# must exceed WIDTH_IN.
GATE_HALO_WIDTH_IN: float = 0.40    # halo total width on screen, inches
GATE_HALO_HEIGHT_IN: float = 0.95   # halo total height on screen, inches
GATE_HALO_MAX_ALPHA: float = 0.72   # peak opacity at the gate halo center
GATE_HALO_SIGMA_FRAC: float = 0.65  # Gaussian sigma as fraction of half-extent;
                                    # smaller = harder edge, larger = softer fade

# ---------------------------------------------------------------------------
# Font family
# ---------------------------------------------------------------------------

FONT_FAMILY: str = "sans-serif"

# ---------------------------------------------------------------------------
# Metric box  (add_metric_box)
# ---------------------------------------------------------------------------

METRIC_BOX_FONT_SIZE: float = 8.5
METRIC_BOX_FONT_FAMILY: str = "monospace"
METRIC_BOX_TEXT_COLOR: str = "#111111"
METRIC_BOX_FACE_COLOR: str = "white"
METRIC_BOX_EDGE_COLOR: str = "0.25"
METRIC_BOX_ALPHA: float = 0.88
METRIC_BOX_PAD: float = 0.35       # FancyBboxPatch round pad inside the box
METRIC_BOX_X: float = .8           # axes-transform x; >1 moves right-aligned box outside axes
METRIC_BOX_Y: float = 0.035          # axes-transform y (bottom-aligned, [0,1])
METRIC_BOX_YOFFSET_PT: float = 28.0  # points below axes bottom edge; clears x-tick labels at any figure size

# ---------------------------------------------------------------------------
# Chart styling  (used by charts/charts.py)
# ---------------------------------------------------------------------------

CHART_TITLE_FONT_SIZE: int = 11
CHART_TITLE_PAD: float = 8.0
CHART_VALUE_LABEL_FONT_SIZE: int = 9
CHART_AXIS_LABEL_FONT_SIZE: int = 10
CHART_GRID_ALPHA: float = 0.3
CHART_BAR_HEIGHT: float = 0.55
CHART_BAR_EDGE_WIDTH: float = 0.8
CHART_BAR_VALUE_X_OFFSET: float = 0.001   # gap between bar end and its value label
CHART_LINE_WIDTH: float = 2.0
CHART_MARKER_SIZE: float = 4.0
CHART_PROBABILITY_YLIM: tuple = (-0.05, 1.05)
CHART_DEFAULT_FIGSIZE: tuple = (7, 4)

# Secondary color palette for multi-curve fidelity charts
CHART_SECONDARY_COLORS: list = ["#4CAF50", "#FF9800", "#9C27B0"]
CHART_LINE_STYLES: list = ["-", "--", "-.", ":", "-"]

# ---------------------------------------------------------------------------
# Heatmap chart  (used by charts/heatmap.py)
# ---------------------------------------------------------------------------

HEATMAP_FIGSIZE_WIDTH_PAD: float = 1.5  # extra width reserved for qubit labels + colorbar
HEATMAP_FIGSIZE_HEIGHT_PAD: float = 0.8 # extra height reserved for title and tick labels
HEATMAP_QUBIT_LABEL_X: float = -0.75   # x-position of qubit labels (left of leftmost wire)
HEATMAP_XLIM_LEFT: float = -1.1
HEATMAP_XLIM_RIGHT_OFFSET: float = -0.3    # added to n_layers to get right x-limit
HEATMAP_YLIM_BOTTOM: float = -0.8
HEATMAP_YLIM_TOP_OFFSET: float = -0.2      # added to n_qubits to get top y-limit
HEATMAP_GATE_BOX_PAD: float = 0.02         # FancyBboxPatch corner-rounding pad for gate rect
HEATMAP_HALO_BOX_PAD: float = 0.04         # FancyBboxPatch corner-rounding pad for halo rect
HEATMAP_TITLE_PAD: float = 10.0            # padding above the heatmap axes title
HEATMAP_TITLE_FONT_SIZE: int = CHART_TITLE_FONT_SIZE  # per-axes title font size (mirrors CHART_TITLE_FONT_SIZE)
HEATMAP_SUPTITLE_FONT_SIZE: int = 14       # figure-level suptitle font size for multi-panel heatmap figures
HEATMAP_SUPTITLE_Y: float = 0.995         # figure-level suptitle vertical position (figure fraction)
HEATMAP_LABEL_FONT_SIZE: int = 9           # colorbar label + timestep tick labels
HEATMAP_COLORBAR_FRACTION: float = 0.03    # fraction of axes width given to colorbar
HEATMAP_COLORBAR_PAD: float = 0.02         # gap between axes and colorbar
HEATMAP_TICK_POSITIONS: list = [0.0, 0.5, 1.0]
HEATMAP_TICK_LABELS: list = ["none", "mid", "max"]

# ---------------------------------------------------------------------------
# Idle decoherence colorbar  (used by charts/heatmap.py)
# ---------------------------------------------------------------------------

# Inset geometry
IDLE_CBAR_WIDTH: str = "42%"           # inset_axes width as a % of the parent axes width
IDLE_CBAR_HEIGHT: str = "6%"           # inset_axes height as a % of the parent axes height
IDLE_CBAR_BBOX_Y: float = -0.42        # bbox_to_anchor y — how far below the main axes

# Label (colorbar title)
IDLE_CBAR_LABEL: str = "Idle decoherence"
IDLE_CBAR_LABEL_FONT_SIZE: int = HEATMAP_LABEL_FONT_SIZE
IDLE_CBAR_LABEL_COLOR: str = QUBIT_LABEL_COLOR
IDLE_CBAR_LABEL_PAD: float = 4.0       # gap in points between bar and label text

# Tick labels
IDLE_CBAR_TICK_FONT_SIZE: int = HEATMAP_LABEL_FONT_SIZE

# Side annotation (T1 γ / T2 λ range text to the right of the bar)
IDLE_CBAR_ANNOTATION_X: float = 1.04   # axes-transform x (just right of the bar)
IDLE_CBAR_ANNOTATION_Y: float = 0.5    # axes-transform y (vertically centred)
IDLE_CBAR_ANNOTATION_FONT_SIZE: int = HEATMAP_LABEL_FONT_SIZE
IDLE_CBAR_ANNOTATION_COLOR: str = QUBIT_LABEL_COLOR

# ---------------------------------------------------------------------------
# Stacked heatmap panel  (multi-row layout for build_heatmap_panel)
#
# Figure sizing:
#   width  = max(MIN_WIDTH,  WIDTH_PER_LAYER  × n_layers_max)
#   height = max(MIN_HEIGHT, HEIGHT_PER_ROW   × n_rows
#                           + HEIGHT_PER_QUBIT × n_qubits)
#
# HSPACE must be large enough for the idle decoherence colorbar that hangs
# below each row (IDLE_CBAR_BBOX_Y ≈ -0.42) plus the axes title of the
# next row.  Values below ~1.2 will cause the two to overlap.
# ---------------------------------------------------------------------------

HEATMAP_PANEL_MIN_WIDTH: float = 12.0          # floor on figure width (inches)
HEATMAP_PANEL_WIDTH_PER_LAYER: float = 0.58    # width scaling factor per gate layer
HEATMAP_PANEL_MIN_HEIGHT: float = 4.0          # floor on figure height (inches)
HEATMAP_PANEL_HEIGHT_PER_ROW: float = 2.1      # height contribution per heatmap row
HEATMAP_PANEL_HEIGHT_PER_QUBIT: float = 0.25   # additional height per qubit (shared across rows)
HEATMAP_PANEL_SUPTITLE_Y: float = 0.985        # figure-level suptitle vertical position
HEATMAP_PANEL_LEFT: float = 0.06               # left margin (figure fraction)
HEATMAP_PANEL_RIGHT: float = 0.88              # right margin — wide for purity labels + colorbar
HEATMAP_PANEL_TOP: float = 0.91                # top margin — leaves room for suptitle
HEATMAP_PANEL_BOTTOM: float = 0.08             # bottom margin
HEATMAP_PANEL_HSPACE: float = 1.4              # vertical gap between rows (fraction of mean axes height)

# ---------------------------------------------------------------------------
# Hardware comparison chart  (used by charts/hardware_comparison.py)
# ---------------------------------------------------------------------------

HARDWARE_CMP_HEATMAP_WIDTH_PAD: float = 2.0    # added to n_layers*CIRCUIT_WIDTH_PER_LAYER for figure width
HARDWARE_CMP_ASPECT_RATIO: float = 0.65         # figure height = figure width × this
HARDWARE_CMP_HEIGHT_RATIOS: list = [2.2, 1.0]  # gridspec row heights [heatmap, comparison panel]
HARDWARE_CMP_SUBPLOT_LEFT: float = 0.12
HARDWARE_CMP_SUBPLOT_RIGHT: float = 0.95
HARDWARE_CMP_SUBPLOT_TOP: float = 0.92
HARDWARE_CMP_SUBPLOT_BOTTOM: float = 0.10
HARDWARE_CMP_SUBPLOT_HSPACE: float = 1.10
HARDWARE_CMP_SUPTITLE_FONT_SIZE: int = 12      # CHART_TITLE_FONT_SIZE + 1
HARDWARE_CMP_SUPTITLE_Y: float = 0.96
HARDWARE_CMP_BAR_ALPHA: float = 0.85
HARDWARE_CMP_NOTE_X_OFFSET: float = 0.005      # x-gap between bar end and fidelity note label
HARDWARE_CMP_NOTE_FONT_SIZE: float = 7.5
HARDWARE_CMP_YTICK_FONT_SIZE: float = 8.5
HARDWARE_CMP_XLABEL_FONT_SIZE: int = 9
HARDWARE_CMP_XLIM_MAX: float = 1.35
HARDWARE_CMP_YLIM_BOTTOM_PAD: float = -0.6     # bottom y-limit for comparison bars
HARDWARE_CMP_YLIM_TOP_OFFSET: float = -0.4     # added to len(bar_labels) for top y-limit
HARDWARE_CMP_FOOTER_Y: float = -0.22           # axes-transform y for noise-params footer text
HARDWARE_CMP_FOOTER_FONT_SIZE: int = 7
HARDWARE_CMP_PANEL_TITLE_FONT_SIZE: int = 9
HARDWARE_CMP_PANEL_TITLE_PAD: float = 6.0
HARDWARE_CMP_REFLINE_LABEL_FONT_SIZE: float = 6.5
HARDWARE_CMP_REFLINE_LABEL_Y_OFFSET: float = -0.3  # subtracted from len(bar_labels)


# ---------------------------------------------------------------------------
# Per-qubit marginal purity overlay
# ---------------------------------------------------------------------------

PURITY_LABEL_FORMAT: str = "Tr(ρ_q²)={p:.3f}"
PURITY_LABEL_COLOR: str = QUBIT_LABEL_COLOR
PURITY_LABEL_FONT_SIZE: int = HEATMAP_LABEL_FONT_SIZE
PURITY_LABEL_FONT_FAMILY: str = FONT_FAMILY
PURITY_LABEL_PAD_X: float = 0.35
PURITY_LABEL_XLIM_PAD: float = 2.0
PURITY_LABEL_ZORDER: int = 8

PURITY_CAPTION_TEXT: str = (
    "Single-qubit marginal purity after tracing out all other qubits."
)
PURITY_CAPTION_FONT_SIZE: int = HEATMAP_LABEL_FONT_SIZE - 4
PURITY_CAPTION_Y_OFFSET: float = 0.5
PURITY_CAPTION_WRAP_WIDTH: int = 28


# ---------------------------------------------------------------------------
# Hover panel labels  (gate_info.py and heatmap.py formatters import these
#                      so the same wording appears in both display modes)
# ---------------------------------------------------------------------------

# Gate halo — quantity that drives halo color
HOVER_GATE_HALO_SECTION = "── Gate halo quantity ──"
HOVER_BURDEN_LABEL = "downstream burden / shot"

# Gate decoherence sections
HOVER_GATE_DECOHERENCE_SECTION = "── Gate decoherence (this op) ──"
HOVER_GATE_CUMULATIVE_SECTION = "── Cumulative on these qubits ──"

# Wire halo — quantities that drive wire halo color
HOVER_WIRE_DECOHERENCE_SECTION = "── Idle decoherence (this segment) ──"
HOVER_WIRE_COMBINED_LABEL = "combined  1-(1-γ)(1-λ)"
HOVER_CUMULATIVE_CURSOR_SECTION = "── Cumulative (up to cursor) ──"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def gate_color(gate_name: str) -> str:
    """Return the fill color for a gate box based on gate category."""
    name = gate_name.upper()
    if name in CLIFFORD_GATES:
        return CLIFFORD_GATE_COLOR
    if name in T_GATES:
        return T_GATE_COLOR
    return OTHER_GATE_COLOR


def halo_color(intensity: float) -> tuple:
    """Return red sequential downstream-burden halo color for intensity in [0, 1]."""
    intensity = float(np.clip(intensity, 0.0, 1.0))
    return _DOWNSTREAM_BURDEN_COLORMAP(intensity)


def get_halo_colormap() -> mcolors.LinearSegmentedColormap:
    """Return downstream aggregate burden colormap: near-white pink to dark red."""
    return _DOWNSTREAM_BURDEN_COLORMAP


def get_wire_halo_colormap() -> mcolors.LinearSegmentedColormap:
    """Return the wire halo colormap (cool: dark-navy → blue → bright cyan)."""
    return _WIRE_HALO_COLORMAP


def get_qubit_error_colormap() -> mcolors.LinearSegmentedColormap:
    """Return the sequential red colormap used for per-qubit error burden bars."""
    return _QUBIT_ERROR_COLORMAP


def apply_global_style() -> None:
    """
    Apply NoisiQ's global matplotlib rcParams.

    Call once at the top of a notebook or script to set fonts and
    figure defaults consistently across all visualizations.
    """
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 100,
        }
    )


# ---------------------------------------------------------------------------
# Gate drawing primitives  (shared by drawer.py and circuit_diagram.py)
# ---------------------------------------------------------------------------

def draw_single_gate(
    ax: plt.Axes,
    x: float,
    y: float,
    name: str,
    fill: str,
    edge: str,
    lw: float,
) -> None:
    """Draw a single-qubit rounded gate box with a centered label."""
    box = FancyBboxPatch(
        (x - GATE_HALF_W, y - GATE_HALF_H),
        GATE_SIZE, GATE_SIZE,
        boxstyle="round,pad=0.03",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x, y, name,
        ha="center", va="center",
        fontsize=GATE_LABEL_FONT_SIZE, color=GATE_LABEL_COLOR,
        fontweight="bold", fontfamily=FONT_FAMILY, zorder=4,
    )


def draw_cnot(
    ax: plt.Axes,
    x: float,
    y_ctrl: float,
    y_tgt: float,
    fill: str,
    lw: float,
    edge: str | None = None,
) -> None:
    """Draw a CNOT gate — filled dot on control, ⊕ symbol on target.

    *edge* tints the connecting spine; defaults to *fill*.  Pass
    ``MEASUREMENT_EDGE_COLOR`` for conditional gates.
    """
    spine = edge if edge is not None else fill
    ax.plot([x, x], [y_ctrl, y_tgt], linewidth=lw, color=spine, zorder=2)
    ax.scatter([x], [y_ctrl], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
    ax.text(
        x, y_tgt, "⊕",
        ha="center", va="center_baseline",
        fontsize=16, color=fill,
        fontweight="bold", zorder=4,
    )


def draw_cz(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    lw: float,
    edge: str | None = None,
) -> None:
    """Draw a CZ gate — filled dots on both qubits connected by a line.

    *edge* tints the connecting spine; defaults to *fill*.
    """
    spine = edge if edge is not None else fill
    ax.plot([x, x], [y1, y2], linewidth=lw, color=spine, zorder=2)
    ax.scatter([x, x], [y1, y2], s=CONTROL_DOT_SIZE, c=fill, zorder=4)


def draw_swap(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    lw: float,
    edge: str | None = None,
) -> None:
    """Draw a SWAP gate — two × marks connected by a vertical line.

    *edge* tints the connecting spine; defaults to *fill*.
    """
    spine = edge if edge is not None else fill
    ax.plot([x, x], [y1, y2], linewidth=lw, color=spine, zorder=2)
    d = GATE_SIZE / 5
    for y in (y1, y2):
        ax.plot([x - d, x + d], [y - d, y + d], linewidth=lw + 0.5, color=fill, zorder=4)
        ax.plot([x - d, x + d], [y + d, y - d], linewidth=lw + 0.5, color=fill, zorder=4)


def draw_cs(
    ax: plt.Axes,
    x: float,
    y_ctrl: float,
    y_tgt: float,
    fill: str,
    edge: str,
    lw: float,
) -> None:
    """Draw a CS gate — control dot on control qubit, S box on target."""
    ax.plot([x, x], [y_ctrl, y_tgt], linewidth=lw, color=fill, zorder=2)
    ax.scatter([x], [y_ctrl], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
    draw_single_gate(ax, x, y_tgt, "S", fill, edge, lw)


def draw_ccz(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    y3: float,
    fill: str,
    lw: float,
    edge: str | None = None,
) -> None:
    """Draw a CCZ gate — control dots on q1 and q2, Z-phase dot on q3.

    *edge* tints the connecting spine; defaults to *fill*.
    """
    spine = edge if edge is not None else fill
    y_min = min(y1, y2, y3)
    y_max = max(y1, y2, y3)
    ax.plot([x, x], [y_min, y_max], linewidth=lw, color=spine, zorder=2)
    ax.scatter([x, x], [y1, y2], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
    # Z target: open circle with a dot inside (phase-kickback convention)
    ax.scatter([x], [y3], s=CONTROL_DOT_SIZE * 1.6,
               facecolors="none", edgecolors=fill, linewidths=lw, zorder=4)
    ax.scatter([x], [y3], s=CONTROL_DOT_SIZE * 0.4, c=fill, zorder=5)


def _format_angle_compact(gate_name: str) -> str:
    """Extract and format the angle from a gate name like 'RY(0.7854)' → 'π/4'."""
    try:
        angle_str = gate_name.split("(")[1].rstrip(")")
        theta = float(angle_str)
    except (IndexError, ValueError):
        return ""
    pi = np.pi
    # Check common fractions of π
    for num, den in [(1, 8), (1, 6), (1, 4), (1, 3), (3, 8), (1, 2),
                     (2, 3), (3, 4), (5, 6), (7, 8), (1, 1), (3, 2), (2, 1)]:
        if abs(theta - num * pi / den) < 1e-4:
            label = "π" if num == 1 else f"{num}π"
            return label if den == 1 else f"{label}/{den}"
        if abs(theta + num * pi / den) < 1e-4:
            label = "π" if num == 1 else f"{num}π"
            return f"-{label}" if den == 1 else f"-{label}/{den}"
    return f"{theta:.3g}"


def draw_ry(
    ax: plt.Axes,
    x: float,
    y: float,
    gate_name: str,
    fill: str,
    edge: str,
    lw: float,
) -> None:
    """Draw an RY(θ) gate box with 'RY' label and compact angle annotation.

    Imported by circuit_diagram.py for all RY gates so the visual style
    stays consistent with theme constants.
    """
    box = FancyBboxPatch(
        (x - GATE_HALF_W, y - GATE_HALF_H),
        GATE_SIZE, GATE_SIZE,
        boxstyle="round,pad=0.03",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    angle_label = _format_angle_compact(gate_name)
    if angle_label:
        ax.text(
            x, y + 0.05, "RY",
            ha="center", va="center",
            fontsize=GATE_LABEL_FONT_SIZE, color=GATE_LABEL_COLOR,
            fontweight="bold", fontfamily=FONT_FAMILY, zorder=4,
        )
        ax.text(
            x, y - 0.10, angle_label,
            ha="center", va="center",
            fontsize=GATE_LABEL_FONT_SIZE - 3, color=GATE_LABEL_COLOR,
            fontfamily=FONT_FAMILY, zorder=4,
        )
    else:
        ax.text(
            x, y, "RY",
            ha="center", va="center",
            fontsize=GATE_LABEL_FONT_SIZE, color=GATE_LABEL_COLOR,
            fontweight="bold", fontfamily=FONT_FAMILY, zorder=4,
        )


def draw_measurement(
    ax: plt.Axes,
    x: float,
    y: float,
    cbit_label: str = "",
) -> None:
    """Draw a measurement gate box with a meter arc and needle symbol.

    The box is colored MEASUREMENT_FILL_COLOR (deep green) to distinguish
    it visually from quantum gates.  A half-circle arc with a diagonal needle
    renders the classic oscilloscope/meter icon used in circuit diagrams.
    An optional cbit_label (e.g. ``"c[0]"``) is drawn below the box.
    """
    # Gate box
    box = FancyBboxPatch(
        (x - GATE_HALF_W, y - GATE_HALF_H),
        GATE_SIZE, GATE_SIZE,
        boxstyle="round,pad=0.03",
        facecolor=MEASUREMENT_FILL_COLOR,
        edgecolor=MEASUREMENT_EDGE_COLOR,
        linewidth=GATE_EDGE_WIDTH,
        zorder=3,
    )
    ax.add_patch(box)

    # Meter arc — half-circle in the lower portion of the box
    arc_r = GATE_SIZE * 0.20
    arc_cx = x
    arc_cy = y - GATE_SIZE * 0.08
    arc = Arc(
        (arc_cx, arc_cy),
        width=arc_r * 2,
        height=arc_r * 2,
        angle=0, theta1=0, theta2=180,
        color=GATE_LABEL_COLOR,
        linewidth=1.0,
        zorder=4,
    )
    ax.add_patch(arc)

    # Needle: line from arc centre at ~60 degrees
    needle_len = arc_r * 0.90
    angle_rad = np.radians(65)
    nx = arc_cx + needle_len * np.cos(angle_rad)
    ny = arc_cy + needle_len * np.sin(angle_rad)
    ax.plot([arc_cx, nx], [arc_cy, ny],
            color=GATE_LABEL_COLOR, linewidth=1.0, zorder=5)

    # Cbit label drawn below the gate box
    if cbit_label:
        ax.text(
            x, y - GATE_HALF_H - 0.09, cbit_label,
            ha="center", va="top",
            fontsize=CBIT_LABEL_FONT_SIZE,
            color=CLASSICAL_WIRE_COLOR,
            fontfamily=FONT_FAMILY,
            zorder=4,
        )


def draw_classical_control_wire(
    ax: plt.Axes,
    x_meas: float,
    y_meas: float,
    x_gate: float,
    y_gate: float,
    y_wire: float | None = None,
) -> None:
    """Draw a dashed classical control wire from a measurement to a conditional gate.

    The wire drops from the measured qubit down to a horizontal classical-wire
    lane, runs across to the conditional gate's x-column, then rises to the
    gate qubit.  This correctly handles the common QEC pattern where the
    measured ancilla and the correction target are different qubits.

    Args:
        ax:     Axes to draw on.
        x_meas: x-column of the measurement gate.
        y_meas: y-coordinate of the measured qubit wire.
        x_gate: x-column of the conditional gate.
        y_gate: y-coordinate of the conditional gate's target qubit.
        y_wire: y-coordinate of the horizontal classical lane.  Defaults to
                ``min(y_meas, y_gate) - 0.45``.
    """
    if y_wire is None:
        y_wire = min(y_meas, y_gate) - 0.45

    kw = dict(
        color=CLASSICAL_WIRE_COLOR,
        linewidth=CLASSICAL_WIRE_LINEWIDTH,
        linestyle=CLASSICAL_WIRE_LINESTYLE,
        zorder=2,
    )
    # Drop from measurement qubit to horizontal lane
    ax.plot([x_meas, x_meas], [y_meas - GATE_HALF_H, y_wire], **kw)
    # Horizontal run to gate column
    ax.plot([x_meas, x_gate], [y_wire, y_wire], **kw)
    # Rise from lane to conditional gate qubit
    ax.plot([x_gate, x_gate], [y_wire, y_gate - GATE_HALF_H], **kw)
