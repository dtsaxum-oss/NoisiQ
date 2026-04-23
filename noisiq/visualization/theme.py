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
ERROR_LABEL_FONT_SIZE: int = 8       # small enough to fit between qubit lines
ERROR_LABEL_ALPHA: float = 1.0

# Identity / no-error label
NO_ERROR_COLOR: str = "#AAAAAA"

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
HALO_PAD: float = 0.05              # extra padding around gate box for halo

# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------

GATE_SIZE: float = 0.46           # single value — gates are square
GATE_WIDTH: float = GATE_SIZE
GATE_HEIGHT: float = GATE_SIZE
GATE_HALF_W: float = GATE_SIZE / 2
GATE_HALF_H: float = GATE_SIZE / 2
CONTROL_DOT_SIZE: float = 70.0       # scatter s= value
TARGET_CIRCLE_RADIUS: float = GATE_SIZE / 6   # ⊕ circle same width as a gate box

# ---------------------------------------------------------------------------
# Font family
# ---------------------------------------------------------------------------

FONT_FAMILY: str = "sans-serif"


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
