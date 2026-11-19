"""NoisiQ — noise-aware quantum circuit simulation and visualization.

Quick start::

    import noisiq as nq

    circuit = nq.Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
    noise = nq.NoiseModel.uniform_depolarizing(p1=0.001, p2=0.01, readout=0.01)
    result = nq.simulate(circuit, noise, shots=5000, seed=7, trace_shots=20)

    nq.plot_error_heatmap(result)          # where do errors happen?
    nq.animate_propagation(result.trace)   # how do they spread?
    nq.explain("depolarizing")             # what is the math?
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import algorithms, analysis, backends, io, ir, metrics, noise, results, suppression
from .analysis import propagate
from .backends import select_backend, simulate
from .catalog import explain
from .ir import Circuit, Gate
from .noise import NoiseModel, get_hardware, list_hardware
from .results import PropagationTrace, Result


def __getattr__(name: str):
    # Visualization pulls in matplotlib; import it lazily so `import noisiq` stays light.
    if name == "visualization":
        import importlib

        return importlib.import_module(".visualization", __name__)
    _viz = {
        "draw_circuit",
        "plot_error_heatmap",
        "plot_propagation",
        "animate_propagation",
        "save_animation",
        "plot_error_budget",
        "plot_counts",
        "plot_comparison",
        "plot_qubit_errors",
        "plot_density_matrix",
        "plot_bloch",
    }
    if name in _viz:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module 'noisiq' has no attribute {name!r}")


__all__ = [
    "__version__",
    "Circuit",
    "Gate",
    "NoiseModel",
    "Result",
    "PropagationTrace",
    "simulate",
    "select_backend",
    "propagate",
    "explain",
    "get_hardware",
    "list_hardware",
    "algorithms",
    "analysis",
    "backends",
    "io",
    "ir",
    "metrics",
    "noise",
    "results",
    "suppression",
    "visualization",
    "draw_circuit",
    "plot_error_heatmap",
    "plot_propagation",
    "animate_propagation",
    "save_animation",
    "plot_error_budget",
    "plot_counts",
    "plot_comparison",
    "plot_qubit_errors",
    "plot_density_matrix",
    "plot_bloch",
]
