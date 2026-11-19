"""Summary charts for simulation results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import bloch_vector, hellinger_fidelity
from ..results.result import Result
from .style import Style, get_style

__all__ = [
    "plot_error_budget",
    "plot_qubit_errors",
    "plot_counts",
    "plot_comparison",
    "plot_density_matrix",
    "plot_bloch",
    "plot_hardware_comparison",
]


def _ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=get_style().dpi)
    return ax.get_figure(), ax


def plot_error_budget(
    result: Result,
    *,
    metric: str = "rate",
    top: int = 12,
    include_idle: bool = True,
    ax: plt.Axes | None = None,
    style: Style | None = None,
):
    """Horizontal bar chart of the largest error sources with their share in percent.

    See :meth:`Result.error_budget` and ``nq.explain("error_burden")``.
    """
    style = style or get_style()
    rows = result.error_budget(metric=metric, include_idle=include_idle)
    shown = rows[:top]
    rest = rows[top:]
    if rest:
        shown = shown + [
            {
                "label": f"{len(rest)} others",
                "value": sum(r["value"] for r in rest),
                "share": sum(r["share"] for r in rest),
            }
        ]
    fig, ax = _ax(ax, (7, 0.38 * max(len(shown), 3) + 1))
    labels = [r["label"] for r in shown][::-1]
    vals = [r["value"] for r in shown][::-1]
    shares = [r["share"] for r in shown][::-1]
    cmap = plt.get_cmap(style.heat_cmap)
    vmax = max(vals) if vals else 1
    bars = ax.barh(labels, vals, color=[cmap(0.25 + 0.75 * v / vmax) for v in vals], edgecolor="#333333", lw=0.6)
    for b, sh in zip(bars, shares):
        ax.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            f"  {100 * sh:.1f}%",
            va="center",
            fontsize=style.fontsize * 0.8,
        )
    ax.set_xlabel("fault probability per shot" if metric == "rate" else "expected corrupted outputs per shot")
    ax.set_title(f"Error budget — {result.circuit.name or 'circuit'} ({result.noise_name})", fontsize=style.fontsize)
    ax.margins(x=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_qubit_errors(result: Result, *, ax: plt.Axes | None = None, style: Style | None = None):
    """Stacked bars of X / Y / Z / other fault rates per qubit (gates + idle)."""
    style = style or get_style()
    rates = result.qubit_error_rates()
    fig, ax = _ax(ax, (max(4, 0.45 * result.n_qubits + 2), 3.2))
    x = np.arange(result.n_qubits)
    bottom = np.zeros(result.n_qubits)
    for k, name in enumerate(("X", "Y", "Z", "other")):
        if rates[:, k].any():
            ax.bar(x, rates[:, k], bottom=bottom, color=style.pauli_colors[name], label=name, edgecolor="white", lw=0.5)
            bottom += rates[:, k]
    ax.set_xticks(x, [f"q{q}" for q in x])
    ax.set_ylabel("faults per shot")
    ax.set_title("Faults per qubit", fontsize=style.fontsize)
    ax.legend(frameon=False, fontsize=style.fontsize * 0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_counts(
    result: Result,
    ideal: Union[Result, Mapping[str, float]] | None = None,
    *,
    top: int = 16,
    ax: plt.Axes | None = None,
    style: Style | None = None,
):
    """Outcome probabilities, optionally against an ideal (noiseless) distribution."""
    style = style or get_style()
    p = result.probabilities()
    q = None
    if ideal is not None:
        q = (
            ideal.probabilities()
            if isinstance(ideal, Result)
            else {k: v / sum(ideal.values()) for k, v in ideal.items()}
        )
    keys = sorted(set(p) | set(q or {}), key=lambda k: -(max(p.get(k, 0), (q or {}).get(k, 0))))[:top]
    keys = sorted(keys)
    fig, ax = _ax(ax, (max(4, 0.45 * len(keys) + 2), 3.4))
    x = np.arange(len(keys))
    ax.bar(x, [p.get(k, 0) for k in keys], color="#4c78a8", label="noisy" if q else "simulated")
    if q is not None:
        ax.scatter(x, [q.get(k, 0) for k in keys], marker="_", s=400, color="#d62728", lw=2.5, zorder=3, label="ideal")
        ax.set_title(f"Outcomes (Hellinger fidelity to ideal {hellinger_fidelity(p, q):.3f})", fontsize=style.fontsize)
        ax.legend(frameon=False, fontsize=style.fontsize * 0.8)
    else:
        ax.set_title("Outcomes", fontsize=style.fontsize)
    ax.set_xticks(x, keys, rotation=60 if len(keys) > 8 or (keys and len(keys[0]) > 4) else 0, family="monospace")
    ax.set_ylabel("probability")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _metric(result: Result, metric: str, ideal):
    if metric == "fault_free":
        e = result.fault_free_fraction
        return e.value, e.stderr
    if metric == "output_error":
        e = result.output_error_rate()
        return e.value, e.stderr
    if metric == "fidelity":
        if result.fidelity is None:
            raise ValueError(f"{result.backend} result has no fidelity; use backend='statevector'.")
        return result.fidelity.value, result.fidelity.stderr
    if metric == "hellinger":
        if ideal is None:
            raise ValueError("metric='hellinger' needs ideal=...")
        return result.compare(ideal)["hellinger_fidelity"], 0.0
    raise ValueError("metric must be 'fault_free', 'output_error', 'fidelity' or 'hellinger'.")


def plot_comparison(
    results: Mapping[str, Result],
    metric: str = "fault_free",
    *,
    ideal=None,
    ax: plt.Axes | None = None,
    style: Style | None = None,
):
    """Compare several runs (e.g. different noise models, with/without DD) on one metric."""
    style = style or get_style()
    labels = list(results)
    vals = [_metric(results[k], metric, ideal) for k in labels]
    fig, ax = _ax(ax, (max(4, 1.1 * len(labels) + 1.5), 3.4))
    x = np.arange(len(labels))
    ax.bar(x, [v for v, _ in vals], yerr=[e for _, e in vals], color="#72b7b2", edgecolor="#333333", capsize=4)
    for xi, (v, _) in zip(x, vals):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=style.fontsize * 0.8)
    ax.set_xticks(x, labels, rotation=20 if len(labels) > 4 else 0)
    names = {
        "fault_free": "fault-free fraction",
        "output_error": "output error rate",
        "fidelity": "fidelity to ideal",
        "hellinger": "Hellinger fidelity to ideal",
    }
    ax.set_ylabel(names[metric])
    ax.set_title(f"Comparison: {names[metric]}", fontsize=style.fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_density_matrix(state: Union[Result, np.ndarray], *, max_qubits: int = 6, style: Style | None = None):
    """Real and imaginary parts of a density matrix as heatmaps."""
    style = style or get_style()
    rho = state.density_matrix if isinstance(state, Result) else np.asarray(state)
    if rho is None:
        raise ValueError(
            "This result has no density matrix (use the statevector backend with n ≤ 8 or density_matrix)."
        )
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    n = int(round(np.log2(rho.shape[0])))
    if n > max_qubits:
        raise ValueError(f"{n} qubits is too many to display (max_qubits={max_qubits}).")
    labels = [format(i, f"0{n}b") for i in range(2**n)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=style.dpi)
    lim = float(np.abs(rho).max()) or 1.0
    for ax, part, name in zip(axes, (rho.real, rho.imag), ("Re ρ", "Im ρ")):
        im = ax.imshow(part, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(name)
        if n <= 4:
            ax.set_xticks(range(2**n), labels, rotation=90, family="monospace", fontsize=7)
            ax.set_yticks(range(2**n), labels, family="monospace", fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.03)
    return fig


def plot_bloch(state: Union[Result, np.ndarray], qubits: Sequence[int] | None = None, *, style: Style | None = None):
    """Bloch vectors of single-qubit reduced states (length < 1 ⇒ mixed / entangled)."""
    style = style or get_style()
    states = state.qubit_states if isinstance(state, Result) else np.asarray(state)
    if states is None:
        raise ValueError("This result has no reduced states (use the statevector or density_matrix backend).")
    if states.ndim == 2:
        states = states[None]
    qubits = list(range(len(states))) if qubits is None else list(qubits)
    cols = min(len(qubits), 4)
    rows = int(np.ceil(len(qubits) / cols))
    fig = plt.figure(figsize=(3 * cols, 3 * rows), dpi=style.dpi)
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    for i, q in enumerate(qubits):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#cccccc", lw=0.4)
        bx, by, bz = bloch_vector(states[q])
        ax.quiver(0, 0, 0, bx, by, bz, color="#d62728", lw=2, arrow_length_ratio=0.12)
        r = np.sqrt(bx**2 + by**2 + bz**2)
        ax.set_title(f"q{q}  |r|={r:.2f}", fontsize=style.fontsize * 0.9)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        for a, b, c, t in ((1.2, 0, 0, "x"), (0, 1.2, 0, "y"), (0, 0, 1.2, "|0⟩"), (0, 0, -1.3, "|1⟩")):
            ax.text(a, b, c, t, fontsize=7, color="#666666")
    fig.tight_layout()
    return fig


def plot_hardware_comparison(
    circuit,
    profiles: Sequence[str] | None = None,
    *,
    metric: str = "auto",
    shots: int = 2000,
    seed: int | None = 0,
    representation: str = "pauli",
    ax: plt.Axes | None = None,
    style: Style | None = None,
):
    """Simulate ``circuit`` under several hardware profiles and compare one metric.

    ``metric="auto"`` uses the output error rate for Clifford circuits (frame backend) and
    the fidelity to ideal otherwise.
    """
    from ..backends import simulate
    from ..noise.hardware import get_hardware, list_hardware

    names = list(profiles) if profiles is not None else list_hardware()
    results = {}
    for name in names:
        prof = get_hardware(name)
        nm = prof.noise_model(representation=representation)
        results[f"{prof.vendor}\n{prof.system}"] = simulate(circuit, nm, shots, seed=seed)
    if metric == "auto":
        metric = "output_error" if all(r.measurement_errors is not None for r in results.values()) else "fidelity"
    fig = plot_comparison(results, metric, ax=ax, style=style)
    fig.axes[0].set_title(
        f"{circuit.name or 'circuit'} on different hardware", fontsize=(style or get_style()).fontsize
    )
    return fig
