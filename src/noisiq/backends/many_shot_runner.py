"""
Many-shot simulation runner.

Runs N independent shots of a noisy circuit and accumulates error event
counts per (qubit, timestep) into an AggregateResult for downstream
analysis and heatmap visualization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from ..ir.circuit import Circuit, Operation
from ..ir.classical import Measurement, ConditionalOp
from ..noise.pauli_error import PauliError
from ..noise.correlated_errors import CorrelatedPauliError
from ..noise.kraus_channels import KrausChannel, CombinedChannel
from .pauli_frame import StimTableauBackend, PairedSimResult, PairedShotResult, _is_pauli_compatible_channel

_PAULI_IDX: Dict[str, int] = {'X': 0, 'Y': 1, 'Z': 2}
_IDX_PAULI: Dict[int, str] = {0: 'X', 1: 'Y', 2: 'Z'}


@dataclass(frozen=True)
class AggregateResult:
    """
    Accumulated error statistics from N shots of a noisy circuit.

    counts_matrix shape: (n_qubits, n_timesteps)
    Each entry counts how many times an error occurred on that
    (qubit, operation-index) across all shots.

    counts_by_pauli shape: (n_qubits, n_ops, 3) — axis 2 is [X=0, Y=1, Z=2].
    Tracks which Pauli type fired at each (qubit, op) cell across all shots.
    None when not collected (e.g., old AggregateResult instances).

    zero_error_shots shape: (n_shots,)
    Boolean array; True means that shot had no error events anywhere.
    """

    counts_matrix: np.ndarray
    n_shots: int
    circuit: Circuit
    zero_error_shots: np.ndarray
    seed: Optional[int] = None
    final_state: Optional[np.ndarray] = None
    counts_by_pauli: Optional[np.ndarray] = None
    measurements: Optional[Dict[str, List[int]]] = None
    acceptance_rate: Optional[float] = None
    final_pauli_frames: Optional[List[str]] = None
    # One n-qubit Pauli string per shot, e.g. "IXZY".
    # For non-MCM circuits: exact output-frame net error Pauli.
    # None when the underlying backend does not provide per-shot Pauli frames.
    sampled_error_products: Optional[List[str]] = None
    # One n-qubit Pauli string per shot for MCM circuits.
    # Input-frame XOR product of sampled error events — debug only.
    # Do NOT use for stabilizer/process fidelity metrics.
    branch_diverged: Optional[List[bool]] = None
    # One bool per shot for MCM circuits run via the paired ideal/noisy simulation.
    # True when the noisy classical branch diverged from the ideal branch on at least
    # one deterministic measurement (the physical outcome differed from the ideal
    # deterministic outcome, meaning the wrong correction fired).
    # None for non-MCM circuits and for MCM circuits run without the paired-sim path.

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

    @property
    def modal_pauli_matrix(self) -> np.ndarray:
        """
        Shape (n_qubits, n_ops) of strings: the most-voted Pauli at each
        (qubit, op-index) cell.  Returns 'I' where all counts are zero.
        Requires counts_by_pauli to be set; raises RuntimeError otherwise.
        """
        if self.counts_by_pauli is None:
            raise RuntimeError(
                "modal_pauli_matrix is not available: counts_by_pauli was not "
                "collected.  Re-run ManyShotRunner.run() with the current version."
            )
        cbp = self.counts_by_pauli  # (n_qubits, n_ops, 3)
        best = cbp.argmax(axis=2)   # (n_qubits, n_ops) — index of winning Pauli
        no_error = cbp.sum(axis=2) == 0  # cells where all counts are zero
        result = np.vectorize(_IDX_PAULI.get)(best).astype(object)
        result[no_error] = 'I'
        return result


    def zero_error_fraction_report(
        self,
        *,
        backend: str = "StimTableauBackend",
        noise_model: str = "unknown",
    ) -> "MetricReport":
        """Return zero_error_fraction wrapped in a MetricReport with uncertainty.

        Returns:
            MetricReport with kind=ZERO_ERROR_FRACTION and binomial standard error.
        """
        from ..results.metrics import MetricKind, MetricReport

        p = self.zero_error_fraction
        se = math.sqrt(p * (1.0 - p) / self.n_shots) if self.n_shots > 0 else 0.0
        return MetricReport(
            name="Zero-error fraction",
            kind=MetricKind.ZERO_ERROR_FRACTION,
            value=p,
            uncertainty=se,
            n_shots=self.n_shots,
            backend=backend,
            noise_model=noise_model,
            method="Monte Carlo shot fraction",
            definition=(
                "Fraction of shots in which no elementary error event was sampled"
            ),
            limitations=(
                "not a state or process fidelity; stabilizer errors that happen to "
                "preserve the target state are still counted as errors in this metric",
            ),
        )


def _validate_post_select(
    post_select: Dict[int, int],
    circuit: Circuit,
) -> None:
    """Validate post_select keys and values against circuit classical registers.

    Raises:
        ValueError: If any cbit index is not allocated by the circuit, or any
                    required value is not 0 or 1.
    """
    valid_cbit_indices = {
        reg.offset + i
        for reg in circuit.classical_registers
        for i in range(reg.size)
    }
    for cbit_index, required in post_select.items():
        if cbit_index not in valid_cbit_indices:
            raise ValueError(
                f"post_select refers to cbit index {cbit_index}, which is not "
                f"allocated by this circuit's classical registers."
            )
        if required not in (0, 1):
            raise ValueError(
                f"post_select values must be 0 or 1; got {required!r} for "
                f"cbit index {cbit_index}."
            )


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
        noise_config: Optional[Dict[int, Union[PauliError, CorrelatedPauliError, CombinedChannel]]] = None,
        seed: Optional[int] = None,
        post_select: Optional[Dict[int, int]] = None,
    ) -> AggregateResult:
        """
        Run n_shots independent simulations and accumulate error counts.

        Parameters
        ----------
        circuit      : Circuit to simulate.
        n_shots      : Number of independent shots (must be >= 1).
        noise_config : Mapping from operation index to noise channel.
        seed         : Top-level seed for reproducibility.
        post_select  : Optional mapping of {cbit_index: required_outcome}.
                       Only shots where every listed cbit equals its required
                       value are accepted.  All indices must be allocated by
                       the circuit and all values must be 0 or 1.
                       Raises ValueError when zero shots are accepted.

        Returns
        -------
        AggregateResult with counts_matrix of shape (n_qubits, n_operations).
        When post_select is given, n_shots in the result is the accepted count
        and acceptance_rate is accepted / attempted.
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")

        circuit.validate()

        _CLIFFORD_GATES = frozenset({
            'H', 'X', 'Y', 'Z', 'S', 'S_DAG',
            'CNOT', 'CX', 'CZ', 'SWAP',
            'I', 'IDLE',
        })
        for op in circuit.operations:
            if isinstance(op, ConditionalOp):
                # Inspect the inner gate — a T or RZ hidden inside a ConditionalOp
                # is still non-Clifford and cannot be run by StimTableauBackend.
                inner = op.inner
                if isinstance(inner, Operation):
                    name = inner.gate.name.upper()
                    if name not in _CLIFFORD_GATES:
                        raise NotImplementedError(
                            f"ManyShotRunner only supports Clifford circuits. "
                            f"Found non-Clifford gate inside ConditionalOp: "
                            f"{inner.gate.name}"
                        )
                continue
            if not isinstance(op, Operation):
                # Measurement — not a unitary gate; skip whitelist.
                continue
            name = op.gate.name.upper()
            if name not in _CLIFFORD_GATES:
                raise NotImplementedError(
                    f"ManyShotRunner only supports Clifford circuits. "
                    f"Found non-Clifford gate: {op.gate.name}"
                )

        if post_select is not None:
            _validate_post_select(post_select, circuit)

        # If any channel in noise_config is not Pauli-compatible (e.g. KrausChannel
        # or a CombinedChannel containing non-Pauli channels), StimTableauBackend
        # cannot handle it. Fall back to TrajectoryBackend which supports the full
        # channel set. An all-Pauli CombinedChannel stays on the Stim path.
        if noise_config is not None:
            channels = (
                list(noise_config.values())
                if isinstance(noise_config, dict)
                else [noise_config]
            )
            if any(not _is_pauli_compatible_channel(ch) for ch in channels):
                from noisiq.backends.trajectory_backend import TrajectoryBackend
                return TrajectoryBackend().run_aggregate(
                    circuit,
                    noise_model=noise_config,
                    n_shots=n_shots,
                    seed=seed,
                    post_select=post_select,
                )

        n_qubits = circuit.n_qubits
        n_ops = len(circuit.operations)
        counts = np.zeros((n_qubits, n_ops), dtype=np.int64)
        counts_by_pauli = np.zeros((n_qubits, n_ops, 3), dtype=np.int64)
        pauli_frames: List[str] = []
        sampled_products: List[str] = []

        # Per-accepted-shot lists (variable length under post-selection).
        accepted_zero_error: List[bool] = []
        accepted_measurements: Dict[str, List[int]] = {}
        accepted_branch_diverged: List[bool] = []
        accepted = 0
        attempted = n_shots

        # For MCM circuits on the Stim backend, run_paired_shot() is the source
        # of truth for every accepted shot.  It returns errors, cbits, fidelity
        # fields, and measurements all from the same shot execution, so
        # post-selection, error counts, and final_pauli_frame are always aligned.
        is_mcm_stim = isinstance(self._backend, StimTableauBackend) and any(
            isinstance(op, (Measurement, ConditionalOp))
            for op in circuit.operations
        )

        rng = np.random.default_rng(seed)
        shot_seeds = rng.integers(0, 2**31, size=n_shots)

        for shot_seed in shot_seeds:

            if is_mcm_stim:
                # ── MCM path: single paired shot as source of truth ──────────
                shot: PairedShotResult = self._backend.run_paired_shot(
                    circuit,
                    noise_config=noise_config,
                    seed=int(shot_seed),
                )
                if post_select is not None:
                    if any(shot.noisy_cbits.get(k) != v for k, v in post_select.items()):
                        continue
                accepted += 1
                for cbit_name, outcome in shot.measurements.items():
                    accepted_measurements.setdefault(cbit_name, []).append(outcome)
                shot_has_error = False
                for error in shot.errors:
                    counts[error.qubit, error.time_step] += 1
                    pauli_idx = _PAULI_IDX.get(error.pauli)
                    if pauli_idx is not None:
                        counts_by_pauli[error.qubit, error.time_step, pauli_idx] += 1
                    shot_has_error = True
                accepted_zero_error.append(not shot_has_error)
                pauli_frames.append(shot.final_pauli_frame)
                sampled_products.append(shot.sampled_error_product)
                accepted_branch_diverged.append(shot.branch_diverged)

            else:
                # ── Non-MCM path: single-shot run() as before ────────────────
                sim_result = self._backend.run(
                    circuit,
                    noise_model=noise_config,
                    n_shots=1,
                    seed=int(shot_seed),
                )
                if sim_result.meta is None or "stim_result" not in sim_result.meta:
                    raise TypeError(
                        f"Backend {type(self._backend).__name__} is incompatible with "
                        "ManyShotRunner. ManyShotRunner requires a backend that provides "
                        "step-by-step error metadata (e.g., StimTableauBackend)."
                    )
                result = sim_result.meta["stim_result"]
                if post_select is not None:
                    cbits = result.shot_cbits or {}
                    if any(cbits.get(k) != v for k, v in post_select.items()):
                        continue
                accepted += 1
                if sim_result.measurements:
                    for cbit_name, outcomes in sim_result.measurements.items():
                        accepted_measurements.setdefault(cbit_name, []).extend(outcomes)
                shot_has_error = False
                for step in result.steps:
                    for error in step.errors:
                        counts[error.qubit, error.time_step] += 1
                        pauli_idx = _PAULI_IDX.get(error.pauli)
                        if pauli_idx is not None:
                            counts_by_pauli[error.qubit, error.time_step, pauli_idx] += 1
                        shot_has_error = True
                accepted_zero_error.append(not shot_has_error)
                if result.final_pauli_frames:
                    pauli_frames.append(result.final_pauli_frames[0])
                if result.sampled_error_products:
                    sampled_products.append(result.sampled_error_products[0])

        if post_select is not None and accepted == 0:
            raise ValueError(
                f"Post-selection rejected all {attempted} shots. "
                "Check post_select conditions or increase n_shots."
            )

        acceptance_rate = accepted / attempted if post_select is not None else None
        final_n_shots = accepted if post_select is not None else attempted

        return AggregateResult(
            counts_matrix=counts,
            n_shots=final_n_shots,
            circuit=circuit,
            zero_error_shots=np.array(accepted_zero_error, dtype=bool),
            seed=seed,
            counts_by_pauli=counts_by_pauli,
            measurements=accepted_measurements if accepted_measurements else None,
            acceptance_rate=acceptance_rate,
            final_pauli_frames=pauli_frames if pauli_frames else None,
            sampled_error_products=sampled_products if sampled_products else None,
            branch_diverged=accepted_branch_diverged if accepted_branch_diverged else None,
        )
