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
    {"H", "X", "Y", "Z", "S", "S†", "SDG", "CNOT", "CX", "CZ", "I", "SWAP"}
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
# Halo palette  (error-rate proxy now; downstream impact in Week 7)
# ---------------------------------------------------------------------------

HALO_ZERO_COLOR: str = "#00AAFF"    # light blue  — zero errors at this gate
HALO_MAX_COLOR: str = "#FF1744"     # bright red  — maximum downstream impact

# Matplotlib colormap: light blue (0) → bright red (1)
_HALO_COLORMAP = mcolors.LinearSegmentedColormap.from_list(
    "noisiq_halo",
    [HALO_ZERO_COLOR, HALO_MAX_COLOR],
    N=256,
)

HALO_ALPHA: float = 0.55            # transparency of the halo patch
HALO_PAD: float = 0.55              # extra padding around gate box for halo

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
# Font family
# ---------------------------------------------------------------------------

FONT_FAMILY: str = "sans-serif"

# ---------------------------------------------------------------------------
# Chart styling  (used by charts/charts.py)
# ---------------------------------------------------------------------------

CHART_TITLE_FONT_SIZE: int = 11
CHART_VALUE_LABEL_FONT_SIZE: int = 9
CHART_GRID_ALPHA: float = 0.3
CHART_BAR_HEIGHT: float = 0.55
CHART_BAR_EDGE_WIDTH: float = 0.8
CHART_LINE_WIDTH: float = 2.0
CHART_MARKER_SIZE: float = 4.0

# Secondary color palette for multi-curve fidelity charts
CHART_SECONDARY_COLORS: list = ["#4CAF50", "#FF9800", "#9C27B0"]
CHART_LINE_STYLES: list = ["-", "--", "-.", ":", "-"]


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
