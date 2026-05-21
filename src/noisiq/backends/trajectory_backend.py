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
from ..noise.kraus_channels import KrausChannel
from ..noise.pauli_error import PauliError
from ..results import SimulationResult



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


def _apply_kraus_to_qubit(
    state: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubit: int,
    n_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one Kraus operator for a single qubit and apply it, then renormalize.

    Sampling weight for operator K_k is ||K_k |ψ⟩_q||².

    Args:
        state:      Current statevector.
        kraus_ops:  List of 2x2 Kraus operator matrices.
        qubit:      Qubit index the channel acts on.
        n_qubits:   Total number of qubits.
        rng:        NumPy random generator for operator selection.

    Returns:
        Normalized statevector after applying the sampled Kraus operator.
    """
    outcomes = [
        _apply_gate_to_state(state, K, [qubit], n_qubits) for K in kraus_ops
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
    new_state = outcomes[k]
    norm = float(np.sqrt(np.real(np.dot(new_state.conj(), new_state))))
    return new_state / norm


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
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]], None] = None,
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

        noise_dict: Dict[int, Union[KrausChannel, PauliError]] = {}
        if isinstance(noise_model, (KrausChannel, PauliError)):
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

            for op_idx, op in enumerate(circuit.operations):
                state = _apply_gate_to_state(
                    state, op.gate.matrix, list(op.qubits), n
                )
                if op_idx in noise_dict:
                    channel = noise_dict[op_idx]
                    if isinstance(channel, KrausChannel):
                        for qubit in op.qubits:
                            state = _apply_kraus_to_qubit(
                                state, channel.operators, qubit, n, shot_rng
                            )
                    elif isinstance(channel, PauliError):
                        for qubit in op.qubits:
                            pauli = channel.sample(shot_rng)
                            if pauli != 'I':
                                state = _apply_gate_to_state(
                                    state, channel.get_operator(pauli).matrix, [qubit], n
                                )

            rho_sum += np.outer(state, state.conj())

        return SimulationResult(
            final_state=rho_sum / n_shots,
            meta={"n_shots": n_shots, "n_qubits": n, "seed": seed},
        )
