"""
NoisiQ: Noise-Aware Quantum Circuit Simulation and Visualization
"""
__version__ = "0.1.0"

# Import subpackages so users can access them like `noisiq.ir` or `noisiq.results`
from . import backends, ir, noise, results, visualization

# Import the most common classes to the top-level for convenience,
# allowing users to write `noisiq.Circuit` instead of `noisiq.ir.circuit.Circuit`.
from .ir import Circuit, Gate, Operation
from .results import PauliFrameRun, SimulationResult

# Define what `from noisiq import *` will import.
__all__ = [
    "__version__",
    # Subpackages
    "ir",
    "noise",
    "backends",
    "results",
    "visualization",
    # Top-level classes
    "Circuit",
    "Gate",
    "Operation",
    "SimulationResult",
    "PauliFrameRun",
]