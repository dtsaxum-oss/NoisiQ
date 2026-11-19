"""Visual style shared by every NoisiQ figure.

All colors and sizes live in one :class:`Style` object instead of hundreds of module
constants. Change the defaults globally with :func:`set_style` or pass ``style=`` to
any plotting function.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = ["Style", "get_style", "set_style", "PAULI_COLORS"]

PAULI_COLORS = {"X": "#d62728", "Y": "#9467bd", "Z": "#1f77b4", "other": "#ff7f0e", "I": "#bbbbbb"}


@dataclass(frozen=True)
class Style:
    """Sizes are in data units where neighbouring qubit wires are 1 apart."""

    font: str = "DejaVu Sans"
    fontsize: float = 10.0
    gate_height: float = 0.62
    gate_min_width: float = 0.62
    char_width: float = 0.118  # extra box width per label character
    column_gap: float = 0.38
    wire_color: str = "#444444"
    wire_width: float = 1.1
    classical_color: str = "#777777"
    edge_color: str = "#222222"
    text_color: str = "#111111"
    highlight_color: str = "#fff3b0"
    done_color: str = "#f2f2f2"
    gate_colors: dict = field(
        default_factory=lambda: {
            "pauli": "#fde2e2",  # X Y Z I
            "clifford": "#dde9f9",  # H S SX ...
            "non_clifford": "#e7dcf5",  # T, rotations, custom
            "multi": "#d9f2e3",  # boxes for multi-qubit gates
            "measure": "#f4f4f4",
            "reset": "#fff2cc",
            "delay": "#ffffff",
        }
    )
    heat_cmap: str = "YlOrRd"
    deviation_cmap: str = "Reds"
    pauli_colors: dict = field(default_factory=lambda: dict(PAULI_COLORS))
    dpi: int = 110


_STYLE = Style()


def get_style() -> Style:
    """The current default style."""
    return _STYLE


def set_style(**changes) -> Style:
    """Change default style fields, e.g. ``set_style(fontsize=12, heat_cmap="viridis")``."""
    global _STYLE
    _STYLE = replace(_STYLE, **changes)
    return _STYLE
