from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# A dictionary mapping a measured bitstring (e.g., "010") to its count.
Counts = Dict[str, int]


@dataclass(frozen=True)
class SimulationResult:
    """
    A container for the final results of a quantum circuit simulation.

    This object standardizes the output from different simulation backends.

    Attributes:
        final_state: The final quantum state of the system, which could be
                     a statevector (for pure state simulation) or a density
                     matrix (for noisy simulation).
        counts: A dictionary of measurement outcomes if the circuit was
                simulated with shots. Keys are bitstrings, values are counts.
    """

    final_state: np.ndarray
    counts: Optional[Counts] = None

    def __repr__(self) -> str:
        parts = [f"final_state=array(shape={self.final_state.shape})"]
        if self.counts is not None:
            parts.append(f"counts={self.counts!r}")
        return f"SimulationResult({', '.join(parts)})"


@dataclass(frozen=True)
class Frame:
    """
    One 'snapshot' in time for visualization / debugging.

    labels: a per-qubit label for what you want to visualize (e.g. "I", "X", "Z", "Y")
    tag: optional tag like "before", "after", "inject"
    """
    t: int
    tag: str
    labels: List[str]
    note: str = ""


@dataclass(frozen=True)
class PauliFrameRun:
    """
    The output of a Pauli-frame-based run, including step-by-step frames.
    This is primarily used for visualization and debugging.
    """
    backend: str
    seed: int
    shots: int
    frames: List[Frame]

    # optional extras (safe to ignore early on)
    stats: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None