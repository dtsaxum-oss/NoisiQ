"""
Many-shot simulation runner.

Runs N independent shots of a noisy circuit and accumulates error event
counts per (qubit, timestep) into an AggregateResult for downstream
analysis and heatmap visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..ir.circuit import Circuit
from ..noise.pauli_error import PauliError
from .pauli_frame import StimTableauBackend


@dataclass(frozen=True)
class AggregateResult:
    """
    Accumulated error statistics from N shots of a noisy circuit.

    counts_matrix shape: (n_qubits, n_timesteps)
    Each entry counts how many times an error occurred on that
    (qubit, operation-index) across all shots.

    zero_error_shots shape: (n_shots,)
    Boolean array; True means that shot had no error events anywhere.
    """

    counts_matrix: np.ndarray
    n_shots: int
    circuit: Circuit
    zero_error_shots: np.ndarray
    seed: Optional[int] = None

    @property
    def error_rate_matrix(self) -> np.ndarray:
        """Per-(qubit, timestep) error rate: counts / n_shots."""
        return self.counts_matrix / self.n_shots

    @property
    def zero_error_fraction(self) -> float:
        """Exact fraction of shots in which no error occurred."""
        return float(self.zero_error_shots.mean())

    @property
    def n_qubits(self) -> int:
        return self.circuit.n_qubits

    @property
    def n_timesteps(self) -> int:
        return self.counts_matrix.shape[1]


class ManyShotRunner:
    """
    Runs N-shot noisy circuit simulation and aggregates error statistics.

    Uses StimTableauBackend under the hood; each shot is independently
    seeded so results are reproducible when a top-level seed is provided.
    """

    def __init__(self, backend: Optional[StimTableauBackend] = None) -> None:
        self._backend = backend or StimTableauBackend()

    def run(
        self,
        circuit: Circuit,
        n_shots: int,
        noise_config: Optional[Dict[int, PauliError]] = None,
        seed: Optional[int] = None,
    ) -> AggregateResult:
        """
        Run n_shots independent simulations and accumulate error counts.

        Parameters
        ----------
        circuit      : Circuit to simulate.
        n_shots      : Number of independent shots (must be >= 1).
        noise_config : Mapping from operation index to PauliError noise model.
        seed         : Top-level seed for reproducibility. Each shot receives
                       a deterministic child seed derived from this value.

        Returns
        -------
        AggregateResult with counts_matrix of shape (n_qubits, n_operations).
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")

        circuit.validate()

        n_qubits = circuit.n_qubits
        n_ops = len(circuit.operations)
        counts = np.zeros((n_qubits, n_ops), dtype=np.int64)
        zero_error = np.ones(n_shots, dtype=bool)

        rng = np.random.default_rng(seed)
        shot_seeds = rng.integers(0, 2**31, size=n_shots)

        for i, shot_seed in enumerate(shot_seeds):
            result = self._backend.run_single_shot(
                circuit,
                noise_config=noise_config,
                seed=int(shot_seed),
            )
            for step in result.steps:
                for error in step.errors:
                    counts[error.qubit, error.time_step] += 1
                    zero_error[i] = False

        return AggregateResult(
            counts_matrix=counts,
            n_shots=n_shots,
            circuit=circuit,
            zero_error_shots=zero_error,
            seed=seed,
        )
