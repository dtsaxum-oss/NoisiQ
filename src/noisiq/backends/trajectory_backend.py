"""
Monte Carlo quantum trajectory backend for Kraus and Pauli noise channels.

Classes:
    TrajectoryBackend: Runs trajectory simulation for Kraus and Pauli noise
"""

from __future__ import annotations

import string
from typing import Dict, Optional, Union

import numpy as np

from ..ir.circuit import Circuit
from ..noise.kraus_channels import KrausChannel, CombinedChannel
from ..noise.pauli_error import PauliError
from ..noise.correlated_errors import CorrelatedPauliError
from ..results import SimulationResult

# Pauli matrices for CorrelatedPauliError application
_PAULI_MATRICES: dict[str, "np.ndarray"] = {
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}



def _partial_trace(
    rho: np.ndarray, keep_qubit: int, n_qubits: int
) -> np.ndarray:
    """Compute 2x2 reduced density matrix by tracing out all qubits except keep_qubit.

    Args:
        rho:        Density matrix of shape (2^n, 2^n).
        keep_qubit: Qubit index to retain.
        n_qubits:   Total number of qubits.

    Returns:
        2x2 complex array — the reduced density matrix for keep_qubit.
    """
    rho_t = rho.reshape([2] * (2 * n_qubits))
    # Row indices use lowercase letters, col indices uppercase.
    # Setting col_char[q] = row_char[q] for traced qubits forces the
    # diagonal trace (summation over equal row/col indices).
    row_chars = list(string.ascii_lowercase[:n_qubits])
    col_chars = list(string.ascii_uppercase[:n_qubits])
    for q in range(n_qubits):
        if q != keep_qubit:
            col_chars[q] = row_chars[q]
    einsum_str = (
        "".join(row_chars)
        + "".join(col_chars)
        + "->"
        + row_chars[keep_qubit]
        + col_chars[keep_qubit]
    )
    return np.einsum(einsum_str, rho_t)


def _apply_gate_to_state(
    state: np.ndarray,
    gate_matrix: np.ndarray,
    target_qubits: list[int],
    n_qubits: int,
) -> np.ndarray:
    """Apply a matrix (unitary or Kraus operator) to target qubits of a statevector.

    Args:
        state:         Statevector of shape (2^n,).
        gate_matrix:   Matrix to apply, shape (2^k, 2^k) for k target qubits.
        target_qubits: Qubit indices the matrix acts on.
        n_qubits:      Total number of qubits.

    Returns:
        New statevector of shape (2^n,) — unnormalized if gate_matrix is not unitary.
    """
    n_targets = len(target_qubits)
    psi = state.reshape([2] * n_qubits)
    U = gate_matrix.reshape([2] * (2 * n_targets))

    # Contract input indices of U with target qubit axes of psi.
    # tensordot result axes: [out_0...out_{k-1}, non-target qubits in original order]
    psi = np.tensordot(
        U, psi,
        axes=[list(range(n_targets, 2 * n_targets)), target_qubits],
    )

    # Build permutation to restore qubit axis ordering.
    non_targets = [q for q in range(n_qubits) if q not in target_qubits]
    axis_of: dict[int, int] = {}
    for i, q in enumerate(target_qubits):
        axis_of[q] = i
    for j, q in enumerate(non_targets):
        axis_of[q] = n_targets + j

    src = [axis_of[q] for q in range(n_qubits)]
    psi = np.moveaxis(psi, src, list(range(n_qubits)))
    return psi.reshape(-1)


def _sample_and_apply_kraus(
    state: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubits: list[int],
    n_qubits: int,
    rng: np.random.Generator,
    *,
    error_set: Optional[set] = None,
) -> np.ndarray:
    """Sample one Kraus operator acting on the given qubits and apply it.

    Works for any number of target qubits (single- or multi-qubit Kraus
    channels). Sampling weight for operator K_k is ||K_k |ψ⟩||².

    Args:
        state:      Current statevector of shape (2^n,).
        kraus_ops:  Kraus operator matrices, each of shape (2^k, 2^k)
                    where k = len(qubits).
        qubits:     Qubit indices the channel acts on.
        n_qubits:   Total number of qubits in the circuit.
        rng:        NumPy random generator.

    Returns:
        Normalized statevector after applying the sampled Kraus operator.
    """
    outcomes = [
        _apply_gate_to_state(state, K, qubits, n_qubits) for K in kraus_ops
    ]
    probs = np.array([
        float(np.real(np.dot(ket.conj(), ket))) for ket in outcomes
    ])
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total == 0.0:
        return state  # unphysical channel — return state unchanged
    probs /= total

    k = int(rng.choice(len(kraus_ops), p=probs))
    if error_set is not None:
        if len(kraus_ops) == 1:
            # Single-operator channels (e.g. CoherentRotation): k is always 0, so
            # use an identity check to detect whether a non-trivial rotation fired.
            K = kraus_ops[0]
            d = K.shape[0]
            scale = K.flat[0] or 1.0
            if not np.allclose(K / scale, np.eye(d), atol=1e-9):
                error_set.update(qubits)
        elif k > 0:
            # Multi-operator channels (e.g. AmplitudeDamping): K[0] is the no-jump
            # operator by convention; k > 0 means a jump/error Kraus operator fired.
            error_set.update(qubits)
    new_state = outcomes[k]
    norm = float(np.sqrt(np.real(np.dot(new_state.conj(), new_state))))
    return new_state / norm


def _apply_kraus_to_qubit(
    state: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubit: int,
    n_qubits: int,
    rng: np.random.Generator,
    *,
    error_set: Optional[set] = None,
) -> np.ndarray:
    """Single-qubit convenience wrapper around _sample_and_apply_kraus."""
    return _sample_and_apply_kraus(state, kraus_ops, [qubit], n_qubits, rng, error_set=error_set)


def _dispatch_channel(
    state: np.ndarray,
    channel,
    op_qubits: tuple,
    n_qubits: int,
    rng: np.random.Generator,
    *,
    error_set: Optional[set] = None,
) -> np.ndarray:
    """Apply one noise channel to a statevector, dispatching on channel type.

    Handles KrausChannel (single- and multi-qubit), CorrelatedPauliError,
    PauliError, and CombinedChannel (recursively applied in order).

    Args:
        state:      Current statevector of shape (2^n,).
        channel:    Noise channel object to apply.
        op_qubits:  Qubits the parent gate acts on (used to resolve target qubits
                    for per-op channels).
        n_qubits:   Total number of qubits in the circuit.
        rng:        NumPy random generator.

    Returns:
        Updated statevector after applying the channel.
    """
    if isinstance(channel, CombinedChannel):
        for inner in channel.channels:
            state = _dispatch_channel(state, inner, op_qubits, n_qubits, rng, error_set=error_set)

    elif isinstance(channel, KrausChannel):
        # Detect qubit-count from operator shape: 2x2 → 1-qubit, 4x4 → 2-qubit, etc.
        n_chan_qubits = int(round(np.log2(channel.operators[0].shape[0])))
        if n_chan_qubits == 1:
            # Per-qubit channel: apply independently to each qubit of the operation.
            for qubit in op_qubits:
                state = _apply_kraus_to_qubit(
                    state, channel.operators, qubit, n_qubits, rng,
                    error_set=error_set,
                )
        else:
            # Multi-qubit channel: apply jointly to the first n_chan_qubits of the op.
            target_qubits = list(op_qubits)[:n_chan_qubits]
            state = _sample_and_apply_kraus(
                state, channel.operators, target_qubits, n_qubits, rng,
                error_set=error_set,
            )

    elif isinstance(channel, CorrelatedPauliError):
        pauli_str = channel.sample(rng)
        for pauli_char, qubit in zip(pauli_str, op_qubits):
            if pauli_char != 'I':
                state = _apply_gate_to_state(
                    state, _PAULI_MATRICES[pauli_char], [qubit], n_qubits
                )
                if error_set is not None:
                    error_set.add(qubit)

    elif isinstance(channel, PauliError):
        for qubit in op_qubits:
            pauli = channel.sample(rng)
            if pauli != 'I':
                state = _apply_gate_to_state(
                    state, channel.get_operator(pauli).matrix, [qubit], n_qubits
                )
                if error_set is not None:
                    error_set.add(qubit)

    return state


class TrajectoryBackend:
    """
    Monte Carlo quantum trajectory simulator for Kraus and Pauli noise.

    Each shot samples which Kraus operator (or Pauli error) applies at every
    noisy gate, evolving a pure statevector through the circuit. The approximate
    density matrix is the average of N outer products |ψ_i⟩⟨ψ_i|.

    Qubit limit: 13. The density matrix accumulator scales as 4^n in memory
    (256 MB at n=12, 1 GB at n=13). For larger circuits use a Pauli noise
    model, which BackendSelector routes to the memory-efficient Pauli-frame
    backends (StimTableauBackend or TsimBackend).
    """

    _MAX_QUBITS: int = 13

    def run(
        self,
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel, Dict[int, Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel]], None] = None,
        n_shots: int = 500,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run N trajectory shots and return the averaged density matrix.

        Args:
            circuit:     Circuit to simulate. All gates must have matrix
                         representations in the IR.
            noise_model: A single KrausChannel or PauliError applied to every
                         qubit of every gate, or a dict mapping operation index →
                         channel for per-gate control. None means noiseless
                         statevector evolution.
            n_shots:     Number of independent trajectory samples. Must be >= 1.
            seed:        Top-level RNG seed for reproducibility.

        Returns:
            SimulationResult with final_state as the density matrix of shape
            (2^n, 2^n), and meta keys n_shots, n_qubits, seed.

        Raises:
            ValueError: If n_shots < 1 or circuit has more than _MAX_QUBITS qubits.
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")
        circuit.validate()
        n = circuit.n_qubits
        if n > self._MAX_QUBITS:
            raise ValueError(
                f"TrajectoryBackend supports up to {self._MAX_QUBITS} qubits "
                f"(got {n}). The density matrix accumulator requires 4^n memory "
                f"(~1 GB at n=13). For larger circuits, switch to a Pauli noise "
                f"model — BackendSelector will automatically route those to the "
                f"memory-efficient Pauli-frame backends."
            )

        _channel_types = (KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel)
        noise_dict: Dict[int, object] = {}
        if isinstance(noise_model, _channel_types):
            noise_dict = {i: noise_model for i in range(len(circuit.operations))}
        elif isinstance(noise_model, dict):
            noise_dict = noise_model

        dim = 2**n
        rho_sum = np.zeros((dim, dim), dtype=complex)

        rng = np.random.default_rng(seed)
        shot_seeds = rng.integers(0, 2**31, size=n_shots)

        for shot_seed in shot_seeds:
            shot_rng = np.random.default_rng(int(shot_seed))
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0  # Start in |00...0⟩

            for op_idx, op in sorted(enumerate(circuit.operations),
                                     key=lambda kv: (kv[1].t, kv[0])):
                state = _apply_gate_to_state(
                    state, op.gate.matrix, list(op.qubits), n
                )
                if op_idx in noise_dict:
                    state = _dispatch_channel(
                        state, noise_dict[op_idx], op.qubits, n, shot_rng
                    )

            rho_sum += np.outer(state, state.conj())

        return SimulationResult(
            final_state=rho_sum / n_shots,
            meta={"n_shots": n_shots, "n_qubits": n, "seed": seed},
        )

    def run_aggregate(
        self,
        circuit: Circuit,
        noise_model: Union[
            KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel,
            Dict[int, Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel]],
            None,
        ] = None,
        n_shots: int = 500,
        seed: Optional[int] = None,
    ) -> "AggregateResult":
        """Run N trajectory shots and return error statistics alongside the density matrix.

        Identical physics to run(), but also tracks which Kraus / Pauli events
        fired at each (qubit, op_idx) to populate AggregateResult.counts_matrix.

        Single-operator channels (e.g. CoherentRotation) are detected via an
        identity check rather than the index-0 convention, so coherent errors
        are never silently missed. Multi-operator channels (e.g. AmplitudeDamping)
        use the standard K[0]-is-no-jump convention via k > 0.

        Args:
            circuit:     Circuit to simulate. Any gate set; no Clifford restriction.
            noise_model: A single channel applied to every qubit of every gate, or a
                         dict mapping operation index → channel. None means noiseless.
            n_shots:     Number of trajectory samples. Must be >= 1.
            seed:        Top-level RNG seed for reproducibility.

        Returns:
            AggregateResult with counts_matrix of shape (n_qubits, n_ops),
            zero_error_shots of length n_shots, and final_state density matrix
            of shape (2^n, 2^n) averaged over all shots.

        Raises:
            ValueError: If n_shots < 1 or circuit has more than _MAX_QUBITS qubits.
        """
        from ..backends.many_shot_runner import AggregateResult

        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")
        circuit.validate()
        n = circuit.n_qubits
        if n > self._MAX_QUBITS:
            raise ValueError(
                f"TrajectoryBackend supports up to {self._MAX_QUBITS} qubits "
                f"(got {n}). Switch to a Pauli noise model for larger circuits."
            )

        _channel_types = (KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel)
        noise_dict: Dict[int, object] = {}
        if isinstance(noise_model, _channel_types):
            noise_dict = {i: noise_model for i in range(len(circuit.operations))}
        elif isinstance(noise_model, dict):
            noise_dict = noise_model

        n_ops = len(circuit.operations)
        dim = 2**n
        counts = np.zeros((n, n_ops), dtype=np.int64)
        zero_error = np.ones(n_shots, dtype=bool)
        rho_sum = np.zeros((dim, dim), dtype=complex)

        # Pre-sort once outside the shot loop (run() re-sorts each shot).
        sorted_ops = sorted(enumerate(circuit.operations), key=lambda kv: (kv[1].t, kv[0]))

        rng = np.random.default_rng(seed)
        shot_seeds = rng.integers(0, 2**31, size=n_shots)

        for shot_idx, shot_seed in enumerate(shot_seeds):
            shot_rng = np.random.default_rng(int(shot_seed))
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0

            for op_idx, op in sorted_ops:
                state = _apply_gate_to_state(state, op.gate.matrix, list(op.qubits), n)

                if op_idx in noise_dict:
                    error_set: set = set()
                    state = _dispatch_channel(
                        state, noise_dict[op_idx], op.qubits, n, shot_rng,
                        error_set=error_set,
                    )
                    for q in error_set:
                        counts[q, op_idx] += 1
                    if error_set:
                        zero_error[shot_idx] = False

            rho_sum += np.outer(state, state.conj())

        return AggregateResult(
            counts_matrix=counts,
            n_shots=n_shots,
            circuit=circuit,
            zero_error_shots=zero_error,
            seed=seed,
            final_state=rho_sum / n_shots,
        )
