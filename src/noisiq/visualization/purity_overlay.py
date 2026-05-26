"""
Per-qubit purity overlay helpers for NoisiQ visualizations.

A noisy density matrix ρ has a single scalar purity Tr(ρ²) ∈ [1/d, 1] that
summarises how much the global state has decohered. For circuit-level visuals
we want the *per-qubit* version: for each qubit q, trace out the rest of the
system to get ρ_q (a 2×2 matrix), then compute Tr(ρ_q²) ∈ [0.5, 1.0].

Per-qubit purity is the right quantity for the right-hand annotations on
both the Quirk-style heatmap and the error-propagation GIF: it gives a
single number per wire that drops as decoherence builds along that qubit's
trajectory through the circuit.

Functions
---------
per_qubit_purity     : Tr(ρ_q²) for one specific qubit
per_qubit_purities   : list of Tr(ρ_q²) values for all qubits in order
annotate_axes_with_purities : matplotlib helper — draws labels at right edge
"""

from __future__ import annotations

import string
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _partial_trace_keep_one(rho: np.ndarray, keep_qubit: int, n_qubits: int) -> np.ndarray:
    """Trace out every qubit except keep_qubit; return a 2×2 reduced ρ_q.

    Args:
        rho:        Full density matrix of shape (2^n, 2^n).
        keep_qubit: The single qubit whose reduced state we want.
        n_qubits:   Total qubit count (so we know 2^n).

    Returns:
        2×2 complex array — the reduced density matrix of the kept qubit.

    Notes:
        Logic mirrors the private `_partial_trace` in
        backends/trajectory_backend.py — duplicated here so visualization has
        no dependency on a backend internal. When MSD lands and that helper
        is promoted to a public function, this can call into it instead.
    """
    # Reshape (2^n, 2^n) → 2n axes each of size 2; row-axes first, col-axes second.
    rho_t = rho.reshape([2] * (2 * n_qubits))

    # Build an einsum string that contracts (sums) every (row,col) pair where
    # the qubit index is NOT keep_qubit. The result is a 2×2 matrix indexed by
    # the row/col indices of keep_qubit.
    row_chars = list(string.ascii_lowercase[:n_qubits])
    col_chars = list(string.ascii_uppercase[:n_qubits])
    for q in range(n_qubits):
        if q != keep_qubit:
            col_chars[q] = row_chars[q]   # equal indices → diagonal sum

    einsum_str = (
        "".join(row_chars)
        + "".join(col_chars)
        + "->"
        + row_chars[keep_qubit]
        + col_chars[keep_qubit]
    )
    return np.einsum(einsum_str, rho_t)


def per_qubit_purity(rho: np.ndarray, n_qubits: int, qubit: int) -> float:
    """Return Tr(ρ_q²) for a single qubit q after tracing out the rest.

    Args:
        rho:      Full density matrix of shape (2^n, 2^n).
        n_qubits: Total qubit count.
        qubit:    Index of the qubit to evaluate.

    Returns:
        Real-valued purity in [0.5, 1.0].
        1.0  → that qubit's reduced state is pure (no decoherence on this qubit)
        0.5  → that qubit's reduced state is maximally mixed
    """
    rho_q = _partial_trace_keep_one(rho, keep_qubit=qubit, n_qubits=n_qubits)
    return float(np.real(np.trace(rho_q @ rho_q)))


def per_qubit_purities(rho: np.ndarray, n_qubits: int) -> List[float]:
    """Return [Tr(ρ_0²), Tr(ρ_1²), …, Tr(ρ_{n-1}²)] in qubit-index order."""
    return [per_qubit_purity(rho, n_qubits, q) for q in range(n_qubits)]


# ---------------------------------------------------------------------------
# Matplotlib overlay
# ---------------------------------------------------------------------------

def annotate_axes_with_purities(
    ax: plt.Axes,
    purities: List[float],
    *,
    x_pos: Optional[float] = None,
    color: str = "#1A237E",
    fontsize: int = 9,
    fmt: str = "Tr(ρ²)={p:.3f}",
    pad_x: float = 0.35,
) -> None:
    """Draw a 'Tr(ρ²)=...' label at the right edge of each qubit wire.

    Designed to be called AFTER `plot_error_heatmap` (which draws qubit wires
    from x=-0.5 to x=n_layers-0.5 with qubit index q at y = n_qubits-1-q),
    and at the end of an animation (last frame only).

    Args:
        ax:       The Axes already containing the circuit diagram.
        purities: List of per-qubit purities, in qubit-index order. Length
                  must equal the qubit count of the drawn circuit.
        x_pos:    X-coordinate to place the labels. If None, places labels
                  `pad_x` data units past the right edge of the visible
                  x-axis range.
        color:    Text color.
        fontsize: Text size.
        fmt:      Format string with a `{p}` placeholder for the float.
        pad_x:    When x_pos is None, distance past x-axis right edge.
    """
    n_qubits = len(purities)
    if x_pos is None:
        x_pos = ax.get_xlim()[1] + pad_x

    for q in range(n_qubits):
        # Match the y-coordinate convention used by plot_error_heatmap:
        # q=0 is the TOP wire (y = n_qubits-1), q=n_qubits-1 is the BOTTOM (y=0).
        y = n_qubits - 1 - q
        ax.text(
            x_pos, y,
            fmt.format(p=purities[q]),
            ha="left", va="center",
            color=color, fontsize=fontsize,
            fontfamily="monospace",
        )

    # Make sure the labels are not cut off
    ax.set_xlim(right=x_pos + 2.0)


# ---------------------------------------------------------------------------
# Convenience: one-shot wrapper around an existing heatmap call
# ---------------------------------------------------------------------------

def add_purity_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    rho: np.ndarray,
    n_qubits: int,
    **kwargs,
) -> List[float]:
    """Compute per-qubit purities from ρ and overlay them onto the heatmap axes.

    Convenience wrapper combining `per_qubit_purities` + `annotate_axes_with_purities`.
    Returns the purity list so callers can also print/log it.

    Example:
        fig = plot_error_heatmap(result, circuit, ...)
        purs = add_purity_panel(fig, fig.axes[0], rho_kraus, circuit.n_qubits)
        print('per-qubit purities:', purs)
    """
    purs = per_qubit_purities(rho, n_qubits)
    annotate_axes_with_purities(ax, purs, **kwargs)
    return purs
