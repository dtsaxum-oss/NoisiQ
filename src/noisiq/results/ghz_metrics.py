"""
GHZ-specific density-matrix metrics.

All functions take the averaged density matrix from TrajectoryBackend and
return a MetricReport. For large Clifford/Pauli simulations use
ghz_stabilizer_state_fidelity (in this module, Phase 5) instead.

Math reference (n-qubit GHZ state |GHZ_n> = (|0...0> + |1...1>) / sqrt(2)):

    F_GHZ = <GHZ|rho|GHZ>
           = 0.5 * (rho[0,0] + rho[-1,-1]) + Re(rho[0,-1])

    subspace_population   = rho[0,0] + rho[-1,-1]
    branch_coherence_real = Re(rho[0,-1])
    branch_coherence_abs  = |rho[0,-1]|

For an ideal GHZ state: F_GHZ = 1, subspace_population = 1,
branch_coherence_real = 0.5, global_purity = 1, but each single-qubit
marginal purity = 0.5 (expected — not a failure signal).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .metrics import MetricKind, MetricReport


def _ghz_indices(n: int) -> tuple[int, int]:
    """Return (all-zeros index, all-ones index) for an n-qubit system."""
    return 0, (1 << n) - 1


def _validate_ghz_rho(rho: np.ndarray, n: int) -> np.ndarray:
    """Cast rho to complex and verify it has shape (2^n, 2^n)."""
    rho = np.asarray(rho, dtype=complex)
    d = 1 << n
    if rho.shape != (d, d):
        raise ValueError(
            f"rho must have shape ({d}, {d}) for n={n}, got {rho.shape}"
        )
    return rho


def ghz_state_fidelity(
    rho: np.ndarray,
    n: int,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Compute F = <GHZ_n|rho|GHZ_n> and return a MetricReport.

    Args:
        rho:         Full density matrix of shape (2^n, 2^n).
        n:           Number of qubits.
        backend:     Backend that produced rho.
        noise_model: Noise model used.

    Returns:
        MetricReport with kind=DENSITY_MATRIX_STATE_FIDELITY.
    """
    rho = _validate_ghz_rho(rho, n)
    i0, i1 = _ghz_indices(n)
    value = float(0.5 * (rho[i0, i0].real + rho[i1, i1].real) + rho[i0, i1].real)
    return MetricReport(
        name=f"GHZ state fidelity ({n}q)",
        kind=MetricKind.DENSITY_MATRIX_STATE_FIDELITY,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method="exact ⟨GHZ|ρ|GHZ⟩ = 0.5·(ρ[00,00] + ρ[11,11]) + Re(ρ[00,11])",
        definition="F = ⟨GHZ_n|ρ|GHZ_n⟩",
        assumptions=("ideal GHZ target state is (|0...0⟩ + |1...1⟩)/√2",),
        limitations=("requires full density matrix — not scalable to large qubit counts",),
        target=f"{n}-qubit GHZ",
    )


def phased_ghz_state_fidelity(
    rho: np.ndarray,
    n: int,
    *,
    phase_sign: int = 1,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Compute fidelity to a signed GHZ branch state.

    ``phase_sign=+1`` targets (|0...0> + |1...1>) / sqrt(2)  (GHZ+).
    ``phase_sign=-1`` targets (|0...0> - |1...1>) / sqrt(2)  (anti-GHZ / GHZ-).

    The closed-form expressions differ only in the sign of the branch-coherence term:

        F_GHZ+ = 0.5*(rho[00,00] + rho[11,11]) + Re(rho[00,11])
        F_GHZ- = 0.5*(rho[00,00] + rho[11,11]) - Re(rho[00,11])
    """
    if phase_sign not in (+1, -1):
        raise ValueError(f"phase_sign must be +1 or -1, got {phase_sign!r}")

    rho = _validate_ghz_rho(rho, n)
    i0, i1 = _ghz_indices(n)
    value = float(
        0.5 * (rho[i0, i0].real + rho[i1, i1].real)
        + float(phase_sign) * rho[i0, i1].real
    )
    target_label = "GHZ+" if phase_sign == 1 else "anti-GHZ / GHZ-"
    sign = "+" if phase_sign == 1 else "-"
    return MetricReport(
        name=f"{target_label} state fidelity ({n}q)",
        kind=MetricKind.DENSITY_MATRIX_STATE_FIDELITY,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method=(
            f"exact <{target_label}|ρ|{target_label}> = "
            f"0.5·(ρ[00,00] + ρ[11,11]) {sign} Re(ρ[00,11])"
        ),
        definition=f"F = <{target_label}_n|ρ|{target_label}_n>",
        assumptions=(f"ideal target is (|0...0> {sign} |1...1>) / sqrt(2)",),
        limitations=("requires full density matrix — not scalable to large qubit counts",),
        target=f"{n}-qubit {target_label}",
    )


def anti_ghz_state_fidelity(
    rho: np.ndarray,
    n: int,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Compute fidelity to |GHZ-> = (|0...0> - |1...1>) / sqrt(2)."""
    return phased_ghz_state_fidelity(
        rho,
        n,
        phase_sign=-1,
        backend=backend,
        noise_model=noise_model,
    )


def ghz_density_metric_bundle(
    rho: np.ndarray,
    n: int,
    *,
    phase_sign: int = 1,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> dict[str, MetricReport]:
    """Return the recommended small-system GHZ density-matrix metric bundle.

    Keys: ``target_fidelity``, ``subspace_population``,
    ``branch_coherence_real``, ``branch_coherence_abs``.

    Global and per-qubit purity are intentionally excluded — they live in
    the visualization helpers and should be reported alongside this bundle.
    """
    return {
        "target_fidelity": phased_ghz_state_fidelity(
            rho, n, phase_sign=phase_sign, backend=backend, noise_model=noise_model
        ),
        "subspace_population": ghz_subspace_population(
            rho, n, backend=backend, noise_model=noise_model
        ),
        "branch_coherence_real": ghz_branch_coherence_real(
            rho, n, backend=backend, noise_model=noise_model
        ),
        "branch_coherence_abs": ghz_branch_coherence_abs(
            rho, n, backend=backend, noise_model=noise_model
        ),
    }


def ghz_subspace_population(
    rho: np.ndarray,
    n: int,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Compute P(|0...0⟩) + P(|1...1⟩) and return a MetricReport.

    Args:
        rho:         Full density matrix of shape (2^n, 2^n).
        n:           Number of qubits.
        backend:     Backend that produced rho.
        noise_model: Noise model used.

    Returns:
        MetricReport with kind=GHZ_SUBSPACE_POPULATION.
    """
    rho = _validate_ghz_rho(rho, n)
    i0, i1 = _ghz_indices(n)
    value = float(rho[i0, i0].real + rho[i1, i1].real)
    return MetricReport(
        name=f"GHZ subspace population ({n}q)",
        kind=MetricKind.GHZ_SUBSPACE_POPULATION,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method="ρ[00...0, 00...0] + ρ[11...1, 11...1]",
        definition="Total population in the two GHZ branches |0...0⟩ and |1...1⟩",
        limitations=("requires full density matrix",),
        target=f"{n}-qubit GHZ",
    )


def ghz_branch_coherence_real(
    rho: np.ndarray,
    n: int,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Compute Re(rho[00...0, 11...1]) and return a MetricReport.

    A classical mixture of |0...0⟩ and |1...1⟩ has zero coherence even at
    full subspace population — this term distinguishes true GHZ entanglement
    from a classical mixture.

    Args:
        rho:         Full density matrix of shape (2^n, 2^n).
        n:           Number of qubits.
        backend:     Backend that produced rho.
        noise_model: Noise model used.

    Returns:
        MetricReport with kind=GHZ_BRANCH_COHERENCE_REAL.
    """
    rho = _validate_ghz_rho(rho, n)
    i0, i1 = _ghz_indices(n)
    value = float(rho[i0, i1].real)
    return MetricReport(
        name=f"GHZ branch coherence Re (real part) ({n}q)",
        kind=MetricKind.GHZ_BRANCH_COHERENCE_REAL,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method="Re(ρ[00...0, 11...1])",
        definition="Real part of the off-diagonal coherence between the two GHZ branches",
        limitations=("requires full density matrix",),
        target=f"{n}-qubit GHZ",
    )


def ghz_branch_coherence_abs(
    rho: np.ndarray,
    n: int,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Compute |rho[00...0, 11...1]| and return a MetricReport.

    Args:
        rho:         Full density matrix of shape (2^n, 2^n).
        n:           Number of qubits.
        backend:     Backend that produced rho.
        noise_model: Noise model used.

    Returns:
        MetricReport with kind=GHZ_BRANCH_COHERENCE_ABS.
    """
    rho = _validate_ghz_rho(rho, n)
    i0, i1 = _ghz_indices(n)
    value = float(abs(rho[i0, i1]))
    return MetricReport(
        name=f"GHZ branch coherence |C| (absolute) ({n}q)",
        kind=MetricKind.GHZ_BRANCH_COHERENCE_ABS,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method="|ρ[00...0, 11...1]|",
        definition="Absolute off-diagonal coherence between the two GHZ branches",
        limitations=("requires full density matrix",),
        target=f"{n}-qubit GHZ",
    )


def ghz_stabilizer_state_fidelity(
    final_frames: List[str],
    n: int,
    *,
    backend: str = "StimTableauBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Estimate GHZ state fidelity via stabilizer membership (scalable).

    Automatically builds the n-qubit GHZ stabilizer generators and delegates
    to stabilizer_state_fidelity. Scales to large n where density-matrix
    methods are impractical.

    GHZ stabilizer generators for n qubits:
      X⊗n           (all-X generator)
      Z_i Z_{i+1}   for i = 0 .. n-2   (consecutive ZZ pairs)

    Args:
        final_frames: Per-shot output-frame Pauli strings from
                      AggregateResult.final_pauli_frames.
        n:            Number of qubits.
        backend:      Backend name for metadata.
        noise_model:  Noise model name for metadata.

    Returns:
        MetricReport with kind=GHZ_STABILIZER_STATE_FIDELITY.
    """
    from .stabilizer_metrics import stabilizer_state_fidelity as _sf

    generators: List[str] = ['X' * n]
    for i in range(n - 1):
        gen = ['I'] * n
        gen[i] = 'Z'
        gen[i + 1] = 'Z'
        generators.append(''.join(gen))

    base = _sf(
        final_frames,
        generators,
        backend=backend,
        noise_model=noise_model,
        target=f"{n}-qubit GHZ",
    )

    return MetricReport(
        name=f"GHZ stabilizer state fidelity ({n}q)",
        kind=MetricKind.GHZ_STABILIZER_STATE_FIDELITY,
        value=base.value,
        uncertainty=base.uncertainty,
        n_shots=base.n_shots,
        backend=base.backend,
        noise_model=base.noise_model,
        method=base.method,
        definition=(
            f"Fraction of shots where the output-frame net error Pauli is "
            f"in the {n}-qubit GHZ stabilizer group"
        ),
        assumptions=base.assumptions,
        limitations=base.limitations,
        target=f"{n}-qubit GHZ",
    )
