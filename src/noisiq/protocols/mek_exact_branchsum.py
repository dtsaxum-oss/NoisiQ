"""Exact branch-summing verifier for MEK 10-to-2 H-state distillation.

The paper model has exactly ten binary resource-error sites, so the exact
verifier sums over all 2^10 resource-error patterns without any Monte Carlo
noise. This is the gold-standard check before running the Monte Carlo
trajectory simulation.

Architecture
------------
branch_statevector_mcm
    Generic exact MCM branch engine for small circuits. Recursively splits the
    statevector at every Measurement, applies ConditionalOps per branch, and
    returns a flat list of (probability-weight, cbits, final-state) triples —
    one per unique measurement-outcome sequence.

evaluate_mek_exact_branchsum
    MEK-specific wrapper: iterates over all 1024 resource-error patterns,
    calls branch_statevector_mcm for each, post-selects accepted branches, and
    accumulates a(p), u(p), and e(p) by summing over the pattern weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from noisiq.ir.circuit import Circuit, Operation
from noisiq.ir.classical import ConditionalOp, Measurement

from .mek_magic_state_distillation import (
    H_STATE_ERROR_SITES,
    find_resource_prep_ops,
    ket_H,
    mek_acceptance_probability,
    mek_output_error_probability,
    build_mek_10to2_paper_exact_circuit,
)


# ---------------------------------------------------------------------------
# Internal statevector helpers (independent of the trajectory backend)
# ---------------------------------------------------------------------------

_Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=complex)
_X_MATRIX = np.array([[0, 1], [1, 0]], dtype=complex)


def _apply_gate(state: np.ndarray, matrix: np.ndarray, qubits: List[int], n: int) -> np.ndarray:
    """Apply `matrix` to `qubits` of a 2^n statevector (tensor-contraction)."""
    k = len(qubits)
    psi = state.reshape([2] * n)
    U = matrix.reshape([2] * (2 * k))
    psi = np.tensordot(U, psi, axes=[list(range(k, 2 * k)), qubits])
    non_q = [q for q in range(n) if q not in qubits]
    src = {q: i for i, q in enumerate(qubits)}
    src.update({q: k + j for j, q in enumerate(non_q)})
    psi = np.moveaxis(psi, [src[q] for q in range(n)], list(range(n)))
    return psi.reshape(-1)


def _prob_one(state: np.ndarray, qubit: int, n: int) -> float:
    """Born probability that Z-measurement of `qubit` yields 1."""
    psi = state.reshape([2] * n)
    sl = np.take(psi, 1, axis=qubit).reshape(-1)
    return float(np.real(np.dot(sl.conj(), sl)))


def _project(state: np.ndarray, qubit: int, outcome: int, n: int) -> np.ndarray:
    """Project `state` onto Z-basis outcome (0 or 1) for `qubit`, then renormalize."""
    psi = state.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[qubit] = 1 - outcome
    psi[tuple(idx)] = 0.0
    flat = psi.reshape(-1)
    norm = float(np.sqrt(np.real(np.dot(flat.conj(), flat))))
    return flat / norm if norm > 1e-14 else flat


# ---------------------------------------------------------------------------
# BranchResult — one leaf of the measurement-outcome tree
# ---------------------------------------------------------------------------

@dataclass
class BranchResult:
    """Outcome of one measurement-branch path through a circuit.

    Attributes:
        weight:   Born-probability product of all measurements on this path.
        cbits:    Classical-bit values recorded on this path {index: 0/1}.
        state:    Normalized final statevector (length 2^n).
        accepted: True when all post_select conditions are satisfied.
    """
    weight: float
    cbits: Dict[int, int]
    state: np.ndarray
    accepted: bool = False


# ---------------------------------------------------------------------------
# Generic exact MCM branch engine
# ---------------------------------------------------------------------------

def branch_statevector_mcm(
    circuit: Circuit,
    *,
    resource_faults: Optional[Dict[str, bool]] = None,
    post_select: Optional[Dict[int, int]] = None,
) -> List[BranchResult]:
    """Exactly enumerate all measurement branches of a mid-circuit measurement circuit.

    Parameters
    ----------
    circuit:
        NoisiQ circuit. Must have circuit.n_qubits ≤ ~15 for reasonable memory.
    resource_faults:
        Optional dict {resource_site_label: True/False}. Sites mapping to True
        receive a deterministic Y error immediately after their tagged RY prep
        operation. Sites mapping to False (or absent) receive no error.
    post_select:
        Optional dict {cbit_index: required_value}. BranchResult.accepted is
        True only when all post-selection conditions are satisfied.

    Returns
    -------
    List[BranchResult]
        One entry per unique measurement-outcome sequence. Probabilities sum to 1
        over all branches (within floating-point tolerance).
    """
    n = circuit.n_qubits

    # Build fault map: op_idx → True if a Y error fires after that op
    fault_op_indices: Dict[int, bool] = {}
    if resource_faults:
        site_to_op = find_resource_prep_ops(circuit)
        for site, faulty in resource_faults.items():
            if faulty:
                fault_op_indices[site_to_op[site]] = True

    # Sort operations by time-step (mirrors trajectory backend ordering)
    sorted_ops: List[Tuple[int, object]] = sorted(
        enumerate(circuit.operations),
        key=lambda kv: (kv[1].t, kv[0]),
    )

    def _recurse(
        op_pos: int,
        state: np.ndarray,
        cbits: Dict[int, int],
        weight: float,
    ) -> List[BranchResult]:
        """Recurse through sorted_ops[op_pos:], branching at each Measurement."""
        if op_pos >= len(sorted_ops):
            accepted = True
            if post_select:
                for idx, req in post_select.items():
                    if cbits.get(idx, 0) != req:
                        accepted = False
                        break
            return [BranchResult(weight=weight, cbits=dict(cbits), state=state, accepted=accepted)]

        op_idx, op = sorted_ops[op_pos]

        # --- Measurement: split into two branches ---
        if isinstance(op, Measurement):
            p1 = _prob_one(state, op.qubit, n)
            p0 = 1.0 - p1
            results: List[BranchResult] = []
            for outcome, prob in ((0, p0), (1, p1)):
                if prob < 1e-15:
                    continue
                new_state = _project(state, op.qubit, outcome, n)
                if op.reset and outcome == 1:
                    new_state = _apply_gate(new_state, _X_MATRIX, [op.qubit], n)
                new_cbits = dict(cbits)
                new_cbits[op.cbit.index] = outcome
                results.extend(_recurse(op_pos + 1, new_state, new_cbits, weight * prob))
            return results

        # --- ConditionalOp: apply inner gate only when condition is met ---
        if isinstance(op, ConditionalOp):
            current = cbits.get(op.condition.index, 0)
            if current == op.value:
                inner = op.inner
                new_state = _apply_gate(state, inner.gate.matrix, list(inner.qubits), n)
            else:
                new_state = state
            return _recurse(op_pos + 1, new_state, cbits, weight)

        # --- Regular gate ---
        new_state = _apply_gate(state, op.gate.matrix, list(op.qubits), n)

        # Deterministic Y fault on resource-prep operations
        if op_idx in fault_op_indices:
            for q in op.qubits:
                new_state = _apply_gate(new_state, _Y_MATRIX, [q], n)

        return _recurse(op_pos + 1, new_state, cbits, weight)

    initial = np.zeros(2 ** n, dtype=complex)
    initial[0] = 1.0
    return _recurse(0, initial, {}, 1.0)


# ---------------------------------------------------------------------------
# MEK-specific helpers
# ---------------------------------------------------------------------------

def _partial_trace_one(rho: np.ndarray, keep: int, n: int) -> np.ndarray:
    """Return the 2×2 reduced density matrix for qubit `keep`.

    The density matrix rho has shape (2^n, 2^n).  After reshaping to
    [2]*2n the axes are (q0_row, q1_row, ..., q0_col, q1_col, ...).
    We want rho_keep[j, J] = sum_{i≠keep} rho[..., i, j, ..., i, J, ...].

    The einsum subscript list forces each traced-out qubit's row and col
    axes to share the same integer label (summation over equal indices),
    while the kept qubit gets two distinct labels for output.
    """
    rho_t = rho.reshape([2] * (2 * n))
    # Row subscripts: 0..n-1
    # Col subscripts: same as row for traced qubits (forces diagonal trace);
    #                 unique label `n` for the kept qubit's column.
    row_labels = list(range(n))
    col_labels = list(range(n))   # default: col[q] = row[q] → trace over q
    col_labels[keep] = n          # col[keep] = n → kept, not traced
    subscripts = row_labels + col_labels
    output = [keep, n]
    return np.einsum(rho_t, subscripts, output)


def _h_error_from_branch(branch: BranchResult, qubit: int, n: int) -> float:
    """Return 1 - ⟨H|ρ_q|H⟩ for `qubit` in the pure branch state."""
    rho_q = _partial_trace_one(
        np.outer(branch.state, branch.state.conj()), qubit, n
    )
    h = ket_H()
    return float(1.0 - np.real(h.conj() @ rho_q @ h))


# ---------------------------------------------------------------------------
# Public result type and main evaluator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MEKExactBranchsumResult:
    """Exact-branchsum statistics for the MEK circuit at one noise level p.

    Attributes:
        p:                       Input noise probability.
        acceptance:              Simulated a(p): total weight of accepted branches.
        marginal_output_error_0: Simulated e(p) for output qubit 0.
        marginal_output_error_1: Simulated e(p) for output qubit 1.
        any_output_error:        P(error on either qubit | accepted), or None.
        rejected_probability:    Total weight of rejected branches (1 - acceptance
                                 for p=0 with a correct circuit this should be 0).
        branch_diagnostics:      Per-branch summary list, populated only when
                                 evaluate_mek_exact_branchsum(...,
                                 return_branch_diagnostics=True) is called.
                                 Each entry is a dict with keys:
                                   pattern_weight, branch_weight, accepted,
                                   syndrome (tuple of accept-cbit values),
                                   scrap    (tuple of scrap-cbit values).
    """
    p: float
    acceptance: float
    marginal_output_error_0: float
    marginal_output_error_1: float
    any_output_error: Optional[float] = None
    rejected_probability: float = 0.0
    branch_diagnostics: Optional[List[Dict]] = None


def iter_resource_error_patterns() -> Iterable[Dict[str, bool]]:
    """Yield all 1024 binary Y-error patterns over the ten resource sites."""
    for bits in product((False, True), repeat=len(H_STATE_ERROR_SITES)):
        yield dict(zip(H_STATE_ERROR_SITES, bits))


def resource_pattern_probability(pattern: Dict[str, bool], p: float) -> float:
    """Independent Bernoulli weight for one ten-site resource-error pattern."""
    k = sum(1 for v in pattern.values() if v)
    n = len(H_STATE_ERROR_SITES)
    return (p ** k) * ((1.0 - p) ** (n - k))


def evaluate_mek_exact_branchsum(
    p: float,
    *,
    return_branch_diagnostics: bool = False,
) -> MEKExactBranchsumResult:
    """Exact evaluator over all 2^10 resource-error patterns.

    For each of the 1024 Y-error patterns on the ten resource sites:
      1. Assign deterministic Y errors at tagged resource-prep locations.
      2. Evolve the paper-exact MEK circuit exactly, branching at all measurements.
      3. Apply conditional feed-forward corrections per branch.
      4. Post-select branches where all acceptance cbits equal 0.
      5. Compute the marginal H-error on each output qubit.
      6. Weight by the pattern's Bernoulli probability p^k (1-p)^(10-k).

    Sums give the exact a(p), u(p), and e(p) without any Monte Carlo noise.
    Compare against mek_acceptance_probability(p) and mek_output_error_probability(p).

    Parameters
    ----------
    p : float
        Resource Y-error probability in [0, 1].
    return_branch_diagnostics : bool
        When True, populate MEKExactBranchsumResult.branch_diagnostics with a
        list of per-branch summary dicts (pattern_weight, branch_weight,
        accepted, syndrome, scrap).  Useful for pinpointing which measurement
        outcomes lead to rejected branches.

    Returns
    -------
    MEKExactBranchsumResult
    """
    from .mek_magic_state_distillation import _validate_probability
    _validate_probability(p)

    build = build_mek_10to2_paper_exact_circuit()
    circuit = build.circuit
    n = circuit.n_qubits
    post_select = build.post_select
    out_q0, out_q1 = build.output_qubits  # (1, 3)

    # Index sets for syndrome (accept) and scrap bits — used for diagnostics.
    accept_indices: Tuple[int, ...] = build.accept_cbits
    scrap_indices: Tuple[int, ...] = tuple(
        reg.offset + i
        for reg in circuit.classical_registers
        for i in range(reg.size)
        if any(kw in reg.name for kw in ("scrap", "scratch", "resource"))
    )

    total_accept_weight = 0.0
    total_reject_weight = 0.0
    weighted_error_0 = 0.0
    weighted_error_1 = 0.0
    weighted_any_error = 0.0
    diagnostics: List[Dict] = []

    for pattern in iter_resource_error_patterns():
        pat_weight = resource_pattern_probability(pattern, p)
        if pat_weight < 1e-30:
            continue

        branches = branch_statevector_mcm(
            circuit,
            resource_faults=pattern,
            post_select=post_select,
        )

        for br in branches:
            w = pat_weight * br.weight
            if br.accepted:
                total_accept_weight += w
                e0 = _h_error_from_branch(br, out_q0, n)
                e1 = _h_error_from_branch(br, out_q1, n)
                weighted_error_0 += w * e0
                weighted_error_1 += w * e1
                weighted_any_error += w * (1.0 - (1.0 - e0) * (1.0 - e1))
            else:
                total_reject_weight += w

            if return_branch_diagnostics:
                diagnostics.append({
                    "pattern_weight": pat_weight,
                    "branch_weight": br.weight,
                    "accepted": br.accepted,
                    "syndrome": tuple(br.cbits.get(idx, 0) for idx in accept_indices),
                    "scrap":    tuple(br.cbits.get(idx, 0) for idx in scrap_indices),
                })

    if total_accept_weight < 1e-20:
        return MEKExactBranchsumResult(
            p=p,
            acceptance=0.0,
            marginal_output_error_0=float("nan"),
            marginal_output_error_1=float("nan"),
            any_output_error=float("nan"),
            rejected_probability=total_reject_weight,
            branch_diagnostics=diagnostics if return_branch_diagnostics else None,
        )

    return MEKExactBranchsumResult(
        p=p,
        acceptance=total_accept_weight,
        marginal_output_error_0=weighted_error_0 / total_accept_weight,
        marginal_output_error_1=weighted_error_1 / total_accept_weight,
        any_output_error=weighted_any_error / total_accept_weight,
        rejected_probability=total_reject_weight,
        branch_diagnostics=diagnostics if return_branch_diagnostics else None,
    )


def paper_formula_reference(p: float) -> MEKExactBranchsumResult:
    """Convenience adapter exposing the paper formulas in branchsum-result shape."""
    e = mek_output_error_probability(p)
    return MEKExactBranchsumResult(
        p=p,
        acceptance=mek_acceptance_probability(p),
        marginal_output_error_0=e,
        marginal_output_error_1=e,
        any_output_error=None,
    )
