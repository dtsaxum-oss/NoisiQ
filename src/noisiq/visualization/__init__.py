"""Visualization: circuit diagrams, error heatmaps, propagation animations, charts.

Every circuit picture is drawn by :func:`draw_circuit` from a :class:`CircuitLayout`;
change colors and sizes globally with :func:`set_style`.
"""

from .charts import (
    plot_bloch,
    plot_comparison,
    plot_counts,
    plot_density_matrix,
    plot_error_budget,
    plot_hardware_comparison,
    plot_qubit_errors,
)
from .circuit import CircuitDrawing, draw_circuit, gate_category
from .heatmap import heat_values, plot_error_heatmap
from .layout import CircuitLayout, instruction_label
from .propagation import animate_propagation, interactive_propagation, plot_propagation, save_animation
from .style import PAULI_COLORS, Style, get_style, set_style

__all__ = [
    "draw_circuit",
    "CircuitDrawing",
    "CircuitLayout",
    "instruction_label",
    "gate_category",
    "plot_error_heatmap",
    "heat_values",
    "plot_propagation",
    "animate_propagation",
    "save_animation",
    "interactive_propagation",
    "plot_error_budget",
    "plot_qubit_errors",
    "plot_counts",
    "plot_comparison",
    "plot_density_matrix",
    "plot_bloch",
    "plot_hardware_comparison",
    "Style",
    "get_style",
    "set_style",
    "PAULI_COLORS",
]
