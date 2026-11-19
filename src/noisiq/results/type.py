from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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
        measurements: Per-shot outcomes for each mid-circuit measurement,
                      keyed by ClassicalBit name. Each list has one entry per
                      shot (the last outcome when a cbit is measured multiple
                      times in one shot).
        acceptance_rate: Fraction of shots accepted after post-selection.
                         None when no post-selection was applied.
    """

    final_state: Optional[np.typing.NDArray[np.complex128]] = None
    counts: Optional[Counts] = None
    meta: Optional[Dict[str, Any]] = None
    measurements: Optional[Dict[str, List[int]]] = None
    acceptance_rate: Optional[float] = None

    def excited_state_probability(self, qubit: int) -> float:
        """Return P(|1⟩) for the given qubit.

        For density-matrix results (final_state is 2-D), traces out all other
        qubits and reads the [1,1] element of the reduced density matrix.
        For shot-based results (counts is set), returns the fraction of shots
        in which that qubit measured |1⟩.
        """
        if self.final_state is not None and self.final_state.ndim == 2:
            import string
            rho = self.final_state
            n_qubits = int(np.log2(rho.shape[0]))
            rho_t = rho.reshape([2] * (2 * n_qubits))
            row_chars = list(string.ascii_lowercase[:n_qubits])
            col_chars = list(string.ascii_uppercase[:n_qubits])
            for q in range(n_qubits):
                if q != qubit:
                    col_chars[q] = row_chars[q]
            einsum_str = (
                "".join(row_chars) + "".join(col_chars)
                + "->" + row_chars[qubit] + col_chars[qubit]
            )
            rho_q = np.einsum(einsum_str, rho_t)
            return float(rho_q[1, 1].real)

        if self.counts is None:
            raise ValueError(
                "excited_state_probability requires either a density matrix "
                "(final_state) or measurement counts."
            )
        total_shots = sum(self.counts.values())
        if total_shots == 0:
            return 0.0
        excited_shots = sum(
            count for bitstring, count in self.counts.items()
            if len(bitstring) > qubit and bitstring[qubit] == '1'
        )
        return excited_shots / total_shots

    def output_success_probability(
        self,
        success_condition: Callable[[str], bool],
        *,
        backend: str = "unknown",
        noise_model: str = "unknown",
        method: str = "measurement outcome filter",
    ) -> "MetricReport":
        """Return the fraction of shots whose bitstring satisfies success_condition.

        Args:
            success_condition: Callable taking a measurement bitstring and returning
                               True for successful outcomes.
            backend:           Backend name for metadata.
            noise_model:       Noise model name for metadata.
            method:            Short description of the success condition for metadata.

        Returns:
            MetricReport with kind=OUTPUT_SUCCESS_PROBABILITY and binomial uncertainty.

        Raises:
            ValueError: If counts are not available or total shots is zero.
        """
        from .metrics import MetricKind, MetricReport

        if self.counts is None:
            raise ValueError(
                "output_success_probability requires shot counts. "
                "Run the circuit with n_shots > 1 using a shot-based backend."
            )
        total = sum(self.counts.values())
        if total == 0:
            raise ValueError("counts dict is empty; cannot compute success probability")
        success = sum(
            count for bs, count in self.counts.items() if success_condition(bs)
        )
        p = success / total
        se = math.sqrt(p * (1.0 - p) / total)
        return MetricReport(
            name="Output success probability",
            kind=MetricKind.OUTPUT_SUCCESS_PROBABILITY,
            value=p,
            uncertainty=se,
            n_shots=total,
            backend=backend,
            noise_model=noise_model,
            method=method,
            definition=(
                "Fraction of shots whose measured bitstring satisfies the "
                "task-specific success condition"
            ),
            limitations=(
                "not a full quantum state fidelity; phase errors invisible to "
                "measurement are not captured",
            ),
        )

    def __repr__(self) -> str:
        parts = []
        if self.final_state is not None:
            parts.append(f"final_state=array(shape={self.final_state.shape})")
        if self.counts is not None:
            parts.append(f"counts={self.counts!r}")
        if self.measurements is not None:
            keys = list(self.measurements.keys())
            parts.append(f"measurements={{{', '.join(f'{k!r}: [...]' for k in keys)}}}")
        if self.acceptance_rate is not None:
            parts.append(f"acceptance_rate={self.acceptance_rate:.4f}")
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