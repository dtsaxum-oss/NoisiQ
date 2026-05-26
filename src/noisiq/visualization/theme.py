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
from matplotlib.patches import FancyBboxPatch
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
# Halo palette  (error-rate proxy now; downstream impact in Week 7)
# ---------------------------------------------------------------------------

HALO_ZERO_COLOR: str = "#0D47A1"    # dark blue   — zero errors at this gate
HALO_MID_COLOR: str  = "#FFD600"    # yellow      — moderate error intensity
HALO_MAX_COLOR: str  = "#FF1744"    # bright red  — maximum downstream impact

# Thermal-style colormap: dark blue → yellow → bright red.
# Three anchors give the middle range (moderate errors) a clearly distinct
# yellow band rather than blending into an indistinct purple.
_HALO_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "noisiq_halo",
    [HALO_ZERO_COLOR, HALO_MID_COLOR, HALO_MAX_COLOR],
    N=256,
)

HALO_ALPHA: float = 0.65            # transparency of the halo patch
HALO_GAMMA: float = 1.0             # power applied after range-normalization (1.0 = linear;
                                    # increase to compress high end, decrease to boost low end)
HALO_INTENSITY_FLOOR: float = 0.15 # minimum colormap intensity for any gate with ≥1 error;
                                    # keeps the lowest-error gate clearly distinct from zero

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
GATE_HALO_SIGMA_FRAC: float = 0.50  # Gaussian sigma as fraction of half-extent;
                                    # smaller = harder edge, larger = softer fade

# ---------------------------------------------------------------------------
# Font family
# ---------------------------------------------------------------------------

FONT_FAMILY: str = "sans-serif"

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
CHART_FIDELITY_YLIM_BOTTOM: float = -0.05
CHART_FIDELITY_YLIM_TOP: float = 1.05

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
HEATMAP_LABEL_FONT_SIZE: int = 9           # colorbar label + timestep tick labels
HEATMAP_COLORBAR_FRACTION: float = 0.03    # fraction of axes width given to colorbar
HEATMAP_COLORBAR_PAD: float = 0.02         # gap between axes and colorbar
HEATMAP_TICK_POSITIONS: list = [0.0, 0.5, 1.0]
HEATMAP_TICK_LABELS: list = ["none", "mid", "max"]

# ---------------------------------------------------------------------------
# Hardware comparison chart  (used by charts/hardware_comparison.py)
# ---------------------------------------------------------------------------

HARDWARE_CMP_HEATMAP_WIDTH_PAD: float = 2.0    # added to n_layers*CIRCUIT_WIDTH_PER_LAYER for figure width
HARDWARE_CMP_ASPECT_RATIO: float = 0.9         # figure height = figure width × this
HARDWARE_CMP_HEIGHT_RATIOS: list = [2.2, 1.0]  # gridspec row heights [heatmap, comparison panel]
HARDWARE_CMP_SUBPLOT_LEFT: float = 0.12
HARDWARE_CMP_SUBPLOT_RIGHT: float = 0.95
HARDWARE_CMP_SUBPLOT_TOP: float = 0.90
HARDWARE_CMP_SUBPLOT_BOTTOM: float = 0.08
HARDWARE_CMP_SUBPLOT_HSPACE: float = 0.45
HARDWARE_CMP_SUPTITLE_FONT_SIZE: int = 12      # CHART_TITLE_FONT_SIZE + 1
HARDWARE_CMP_SUPTITLE_Y: float = 0.97
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
    """
    Return an RGBA tuple for a halo given intensity in [0, 1].

    intensity=0 → light blue (zero errors)
    intensity=1 → bright red (maximum downstream impact)
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))
    return _HALO_COLORMAP(intensity)


def get_halo_colormap() -> mcolors.LinearSegmentedColormap:
    """Return the halo colormap for use in pcolormesh / colorbar."""
    return _HALO_COLORMAP


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
) -> None:
    """Draw a CNOT gate — filled dot on control, ⊕ symbol on target."""
    ax.plot([x, x], [y_ctrl, y_tgt], linewidth=lw, color=fill, zorder=2)
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
) -> None:
    """Draw a CZ gate — filled dots on both qubits connected by a line."""
    ax.plot([x, x], [y1, y2], linewidth=lw, color=fill, zorder=2)
    ax.scatter([x, x], [y1, y2], s=CONTROL_DOT_SIZE, c=fill, zorder=4)


def draw_swap(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    fill: str,
    lw: float,
) -> None:
    """Draw a SWAP gate — two × marks connected by a vertical line."""
    ax.plot([x, x], [y1, y2], linewidth=lw, color=fill, zorder=2)
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
) -> None:
    """Draw a CCZ gate — control dots on q1 and q2, Z-phase dot on q3."""
    y_min = min(y1, y2, y3)
    y_max = max(y1, y2, y3)
    ax.plot([x, x], [y_min, y_max], linewidth=lw, color=fill, zorder=2)
    ax.scatter([x, x], [y1, y2], s=CONTROL_DOT_SIZE, c=fill, zorder=4)
    # Z target: open circle with a dot inside (phase-kickback convention)
    ax.scatter([x], [y3], s=CONTROL_DOT_SIZE * 1.6,
               facecolors="none", edgecolors=fill, linewidths=lw, zorder=4)
    ax.scatter([x], [y3], s=CONTROL_DOT_SIZE * 0.4, c=fill, zorder=5)
