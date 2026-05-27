"""
Visualization tools for NoisiQ.
"""

from .circuit_diagram import draw_circuit
from .drawer import draw_circuit_with_labels
from .widgets import Visualizer
from .animation import CircuitAnimator
from .export import export_gif, export_html
from .gate_info import GateInfoExtractor
from .theme import (
    apply_global_style,
    gate_color,
    get_halo_colormap,
    halo_color,
)
from . import charts
from .charts import plot_error_heatmap, plot_qubit_error_bar, plot_fidelity_decay, plot_hardware_comparison, interactive_heatmap
from .density_matrix import plot_density_matrix, plot_purity_decay, state_fidelity
from .purity_overlay import (
    per_qubit_purity,
    per_qubit_purities,
    annotate_axes_with_purities,
    add_purity_panel,
)

__all__ = [
    "draw_circuit",
    "draw_circuit_with_labels",
    "Visualizer",
    "CircuitAnimator",
    "export_gif",
    "export_html",
    "GateInfoExtractor",
    "apply_global_style",
    "gate_color",
    "get_halo_colormap",
    "halo_color",
    "charts",
    "plot_error_heatmap",
    "interactive_heatmap",
    "plot_qubit_error_bar",
    "plot_fidelity_decay",
    "plot_hardware_comparison",
    "plot_density_matrix",
    "plot_purity_decay",
    "state_fidelity",
    "per_qubit_purity",
    "per_qubit_purities",
    "annotate_axes_with_purities",
    "add_purity_panel",
]
