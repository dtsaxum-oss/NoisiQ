"""
Density matrix visualization and fidelity metrics for trajectory simulation results.

Provides two views of non-Pauli simulation results:
  1. plot_density_matrix  — Re(ρ) and Im(ρ) as side-by-side color heatmaps.
     Makes decoherence (off-diagonal decay) and population shifts visible.
  2. plot_purity_decay    — Tr(ρ²) vs time.  Noise-model-agnostic quality
     metric: 1 = pure state, 1/2^n = maximally mixed.

And a fidelity utility:
  3. state_fidelity       — Compare a noisy density matrix to an ideal pure
     state or density matrix.  Used for hardware comparison experiments.

Functions:
    state_fidelity      : Quantum state fidelity between ideal and noisy states
    plot_density_matrix : Side-by-side real/imag heatmaps of ρ
    plot_purity_decay   : Tr(ρ²) vs time from a list of SimulationResults
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from ..results import SimulationResult


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
# Public: fidelity
# ==============================================================================


def state_fidelity(
    ideal: np.ndarray,
    noisy: np.ndarray,
) -> float:
    """Compute quantum state fidelity between an ideal and a noisy state.

    Two calling conventions are supported:

    1. Pure-state reference (fastest, most common):
       Pass ideal as a 1-D statevector |ψ⟩.  Computes F = ⟨ψ|ρ|ψ⟩.

    2. Mixed-state reference:
       Pass ideal as a 2-D density matrix σ.  Computes the general
       Uhlmann fidelity F = [Tr(√(√σ ρ √σ))]².

    In both cases noisy must be a 2-D density matrix (e.g. from
    SimulationResult.final_state returned by TrajectoryBackend).

    Args:
        ideal: Ideal state — either a 1-D complex statevector of length 2^n
               or a 2-D complex density matrix of shape (2^n, 2^n).
        noisy: Noisy density matrix of shape (2^n, 2^n), e.g.
               SimulationResult.final_state.

    Returns:
        Fidelity F in [0, 1].  F = 1 means perfect match; F = 0 means
        orthogonal states.

    Raises:
        ValueError: If array shapes are incompatible.

    Example:
        ghz = np.zeros(8)
        ghz[0] = ghz[7] = 1.0 / np.sqrt(2)          # |000⟩ + |111⟩
        F = state_fidelity(ghz, result.final_state)
    """
    ideal = np.asarray(ideal, dtype=complex)
    noisy = np.asarray(noisy, dtype=complex)

    if noisy.ndim != 2 or noisy.shape[0] != noisy.shape[1]:
        raise ValueError(
            f"noisy must be a square 2-D density matrix, got shape {noisy.shape}"
        )

    if ideal.ndim == 1:
        # Pure-state shortcut: F = ⟨ψ|ρ|ψ⟩
        if ideal.shape[0] != noisy.shape[0]:
            raise ValueError(
                f"ideal statevector length {ideal.shape[0]} does not match "
                f"density matrix dimension {noisy.shape[0]}"
            )
        return float(np.real(ideal.conj() @ noisy @ ideal))

    # General case: F = [Tr(sqrt(sqrt(σ) ρ sqrt(σ)))]²
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
    (e.g. |00⟩, |01⟩, …) derived from the matrix dimension.

    Steps (to implement):
        1. Compute n_qubits = log2(rho.shape[0]); generate basis labels.
        2. vmax = max(|Re(ρ)|, |Im(ρ)|) for a symmetric colormap range.
        3. Plot Re(ρ) with ax.imshow(..., cmap='RdBu_r', vmin=-vmax, vmax=vmax).
        4. Plot Im(ρ) identically on the second Axes.
        5. Add a shared colorbar; annotate purity Tr(ρ²) in the title.

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
    labels: Optional[list[str]] = None,
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
        labels:   Optional legend labels (one per curve).
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

    curve_label = labels[0] if labels else None
    ax.plot(t_us, purities, marker="o", linewidth=1.8, markersize=5, label=curve_label)
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
    if labels:
        ax.legend()

    fig.tight_layout()
    return fig
