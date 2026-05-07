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
        final_state: The final quantum state of the system, represented as a
                     numpy array. For pure state simulation, it is a 1D array
                     of complex numbers (statevector). For noisy simulation,
                     it is a 2D array of complex numbers (density matrix).
        counts: A dictionary of measurement outcomes if the circuit was
                simulated with shots. Keys are bitstrings, values are counts.
    """

    final_state: Optional[np.typing.NDArray[np.complex128]] = None
    counts: Optional[Counts] = None
    meta: Optional[Dict[str, Any]] = None

    def excited_state_probability(self, qubit: int) -> float:
        """
        Calculate the probability of measuring the given qubit in the excited state (|1>).
        """
        if self.counts is None:
            raise ValueError("Cannot calculate probability without measurement counts.")
        
        total_shots = sum(self.counts.values())
        if total_shots == 0:
            return 0.0
            
        excited_shots = sum(
            count for bitstring, count in self.counts.items()
            if len(bitstring) > qubit and bitstring[qubit] == '1'
        )
        return excited_shots / total_shots

    def __repr__(self) -> str:
        parts = []
        if self.final_state is not None:
            parts.append(f"final_state=array(shape={self.final_state.shape})")
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