"""
Density matrix visualization and fidelity metrics for trajectory simulation results.

Provides two views of non-Pauli simulation results:
  1. plot_density_matrix  — Re(ρ) and Im(ρ) as side-by-side color heatmaps.
     Makes decoherence (off-diagonal decay) and population shifts visible.
  2. plot_purity_decay    — Tr(ρ²) vs time.  Noise-model-agnostic quality
     metric: 1 = pure state, 1/2^n = maximally mixed.

Metric utilities:
  3. density_matrix_state_fidelity — Compare a noisy density matrix to an ideal
     pure state or density matrix; returns a MetricReport.
  4. global_purity  — Tr(ρ²) as a MetricReport.
  5. state_fidelity — Backwards-compatible float alias for density_matrix_state_fidelity.

Functions:
    density_matrix_state_fidelity : Exact state fidelity returning MetricReport
    global_purity                 : Tr(ρ²) returning MetricReport
    state_fidelity                : Float alias (backwards compat)
    plot_density_matrix           : Side-by-side real/imag heatmaps of ρ
    plot_purity_decay             : Tr(ρ²) vs time from a list of SimulationResults
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..results import SimulationResult
from ..results.metrics import MetricKind, MetricReport


# ==============================================================================
# Internal helpers
# ==============================================================================


def _basis_labels(n_qubits: int) -> list[str]:
    """Return computational basis labels for an n-qubit system.

    Example: n_qubits=2 → ['|00⟩', '|01⟩', '|10⟩', '|11⟩']

    Args:
        n_qubits: Number of qubits.

    Returns:
        List of 2^n label strings in standard binary order.
    """
    return [f"|{i:0{n_qubits}b}⟩" for i in range(2**n_qubits)]


def _purity(rho: np.ndarray) -> float:
    """Compute Tr(ρ²) for a density matrix.

    Args:
        rho: Complex density matrix of shape (d, d).

    Returns:
        Real-valued purity in (0, 1].
    """
    return float(np.real(np.trace(rho @ rho)))


def _matrix_sqrt(rho: np.ndarray) -> np.ndarray:
    """Compute the matrix square root of a positive-semidefinite Hermitian matrix.

    Uses eigendecomposition: sqrt(ρ) = V diag(sqrt(λ)) V†.
    Negative eigenvalues from numerical noise are clipped to zero.

    Args:
        rho: Hermitian positive-semidefinite matrix.

    Returns:
        Matrix square root of the same shape.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T


# ==============================================================================
# Public: metrics
# ==============================================================================


def global_purity(
    rho: np.ndarray,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Return Tr(ρ²) as a MetricReport.

    Args:
        rho:         Complex density matrix of shape (d, d).
        backend:     Name of the backend that produced rho.
        noise_model: Name of the noise model used.

    Returns:
        MetricReport with kind=GLOBAL_PURITY, value=Tr(ρ²), uncertainty=None.
    """
    value = _purity(np.asarray(rho, dtype=complex))
    return MetricReport(
        name="Global purity",
        kind=MetricKind.GLOBAL_PURITY,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method="exact Tr(ρ²)",
        definition="Tr(ρ²): 1 for a pure state, 1/d for a maximally mixed state",
        limitations=("requires full density matrix — not scalable to large qubit counts",),
    )


def density_matrix_state_fidelity(
    ideal: np.ndarray,
    noisy: np.ndarray,
    *,
    backend: str = "TrajectoryBackend",
    noise_model: str = "unknown",
    target: Optional[str] = None,
) -> MetricReport:
    """Compute exact state fidelity and return a MetricReport.

    Two calling conventions:

    1. Pure-state reference: pass ideal as a 1-D statevector |ψ⟩.
       Computes F = ⟨ψ|ρ|ψ⟩.

    2. Mixed-state reference: pass ideal as a 2-D density matrix σ.
       Computes the Uhlmann fidelity F = [Tr(√(√σ ρ √σ))]².

    Args:
        ideal:       Ideal state — 1-D statevector or 2-D density matrix.
        noisy:       Noisy density matrix (e.g. SimulationResult.final_state).
        backend:     Name of the backend that produced noisy.
        noise_model: Name of the noise model used.
        target:      Human-readable description of the target state (e.g. "3-qubit GHZ").

    Returns:
        MetricReport with kind=DENSITY_MATRIX_STATE_FIDELITY.
    """
    value = state_fidelity(ideal, noisy)
    ideal_arr = np.asarray(ideal)
    if ideal_arr.ndim == 1:
        method = "exact ⟨ψ|ρ|ψ⟩"
        definition = "F = ⟨ψ_ideal|ρ_noisy|ψ_ideal⟩"
    else:
        method = "Uhlmann fidelity [Tr(√(√σ ρ √σ))]²"
        definition = "F(σ,ρ) = [Tr(√(√σ ρ √σ))]²"
    return MetricReport(
        name="Density-matrix state fidelity",
        kind=MetricKind.DENSITY_MATRIX_STATE_FIDELITY,
        value=value,
        uncertainty=None,
        n_shots=None,
        backend=backend,
        noise_model=noise_model,
        method=method,
        definition=definition,
        assumptions=("ideal target state is known exactly",),
        limitations=("requires full density matrix — not scalable to large qubit counts",),
        target=target,
    )


def state_fidelity(
    ideal: np.ndarray,
    noisy: np.ndarray,
) -> float:
    """Compute quantum state fidelity and return a bare float.

    Backwards-compatible alias. Prefer density_matrix_state_fidelity() for
    new code — it returns a MetricReport with full metadata.

    Two calling conventions:

    1. Pure-state reference: ideal is a 1-D statevector |ψ⟩.  F = ⟨ψ|ρ|ψ⟩.
    2. Mixed-state reference: ideal is a 2-D density matrix σ.
       F = [Tr(√(√σ ρ √σ))]².

    Args:
        ideal: 1-D statevector or 2-D density matrix.
        noisy: Noisy density matrix of shape (2^n, 2^n).

    Returns:
        Fidelity F in [0, 1].

    Raises:
        ValueError: If array shapes are incompatible.
    """
    ideal = np.asarray(ideal, dtype=complex)
    noisy = np.asarray(noisy, dtype=complex)

    if noisy.ndim != 2 or noisy.shape[0] != noisy.shape[1]:
        raise ValueError(
            f"noisy must be a square 2-D density matrix, got shape {noisy.shape}"
        )

    if ideal.ndim == 1:
        if ideal.shape[0] != noisy.shape[0]:
            raise ValueError(
                f"ideal statevector length {ideal.shape[0]} does not match "
                f"density matrix dimension {noisy.shape[0]}"
            )
        return float(np.real(ideal.conj() @ noisy @ ideal))

    if ideal.shape != noisy.shape:
        raise ValueError(
            f"ideal shape {ideal.shape} does not match noisy shape {noisy.shape}"
        )
    sqrt_ideal = _matrix_sqrt(ideal)
    M = sqrt_ideal @ noisy @ sqrt_ideal
    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return float(np.sum(np.sqrt(eigenvalues)) ** 2)


def plot_density_matrix(
    rho: np.ndarray,
    title: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    """Plot the real and imaginary parts of a density matrix as side-by-side heatmaps.

    Layout:
      Left Axes  — Re(ρ), diverging colormap centered at zero.
      Right Axes — Im(ρ), same colormap and scale.
    Both share a single colorbar.  Tick labels use computational basis strings
    (e.g. |00⟩, |01⟩, …) derived from the matrix dimension.  The default
    title includes purity Tr(ρ²).

    Args:
        rho:   Complex density matrix of shape (2^n, 2^n).
        title: Optional figure suptitle.  If None, displays "Density Matrix
               (purity = {Tr(ρ²):.4f})".
        fig:   Existing Figure to draw into, or None to create a new one.

    Returns:
        matplotlib Figure containing two Axes.

    Raises:
        ValueError: If rho is not square or its dimension is not a power of 2.
    """
    if hasattr(rho, "final_state"):
        rho = rho.final_state
    d = rho.shape[0]
    if rho.ndim != 2 or rho.shape[1] != d:
        raise ValueError(f"rho must be square, got shape {rho.shape}")
    if d == 0 or (d & (d - 1)) != 0:
        raise ValueError(f"rho dimension must be a power of 2, got {d}")

    n_qubits = int(np.log2(d))
    labels = _basis_labels(n_qubits)
    purity = _purity(rho)

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    else:
        axes = fig.subplots(1, 2)

    vmax = max(np.abs(rho.real).max(), np.abs(rho.imag).max())
    if vmax == 0.0:
        vmax = 1.0

    for ax, data, part_label in zip(
        axes, [rho.real, rho.imag], ["Re(ρ)", "Im(ρ)"]
    ):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(d))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(6, 9 - n_qubits))
        ax.set_yticks(range(d))
        ax.set_yticklabels(labels, fontsize=max(6, 9 - n_qubits))
        ax.set_title(part_label)

    fig.colorbar(im, ax=axes.tolist(), label="Amplitude", shrink=0.8)

    suptitle = title if title is not None else f"Density Matrix  (purity = {purity:.4f})"
    fig.suptitle(suptitle)
    return fig


def plot_purity_decay(
    results: list[SimulationResult],
    t_values: np.ndarray,
    label: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot Tr(ρ²) (purity) vs time from a list of SimulationResults.

    Purity is a noise-model-agnostic scalar quality metric:
      - 1.0  → perfectly pure state (no decoherence)
      - 1/d  → maximally mixed state (d = 2^n_qubits)

    Works for any KrausChannel or CombinedChannel — does not require
    discrete error events the way AggregateResult fidelity estimates do.

    Args:
        results:  List of SimulationResult from TrajectoryBackend, one per
                  entry in t_values. Each result.final_state must be a 2-D
                  density matrix.
        t_values: 1D array of time values in seconds.
        label:    Optional legend label for the purity curve.
        title:    Optional Axes title.
        ax:       Existing Axes to draw on, or None to create a new Figure.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If len(results) != len(t_values).
    """
    if len(results) != len(t_values):
        raise ValueError(
            f"results and t_values must have the same length, "
            f"got {len(results)} and {len(t_values)}"
        )

    t_values = np.asarray(t_values)
    d = results[0].final_state.shape[0]
    min_purity = 1.0 / d
    purities = [_purity(r.final_state) for r in results]
    t_us = t_values * 1e6  # display in µs

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    ax.plot(t_us, purities, marker="o", linewidth=1.8, markersize=5, label=label)
    ax.axhline(
        min_purity,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"maximally mixed (1/{d})",
    )
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Purity  Tr(ρ²)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    if title:
        ax.set_title(title)
    if label:
        ax.legend()

    fig.tight_layout()
    return fig
