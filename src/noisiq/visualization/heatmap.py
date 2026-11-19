"""Error heatmap: the circuit diagram with a glow around every gate showing how much
error the *simulation* attributed to it.

Halo values come from :class:`~noisiq.Result` fault statistics, so they change with the
noise model (unlike the old structure-only halos):

* ``metric="rate"`` — probability that a fault fired at that gate on that qubit
  (``nq.explain("fault_rate")``).
* ``metric="burden"`` — rate weighted by how many output bits the fault corrupts after
  propagating through the rest of the circuit (``nq.explain("error_burden")``; Clifford
  circuits only).

Idle (decoherence) faults are drawn as colored bands along the wires.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from ..results.result import Result
from .circuit import draw_circuit
from .style import Style, get_style

__all__ = ["plot_error_heatmap", "heat_values"]


def heat_values(result: Result, metric: str = "rate") -> tuple[np.ndarray, np.ndarray]:
    """Per-(instruction, qubit) and per-(layer, qubit) values for the heatmap."""
    if metric == "rate":
        return result.fault_rates(), result.idle_fault_rates()
    if metric == "burden":
        from ..analysis.propagation import fault_spread

        w_inst, w_idle = fault_spread(result.circuit)
        inst = (result.fault_rates(by_pauli=True)[:, :, :3] * w_inst).sum(axis=2)
        idle = (result.idle_fault_rates(by_pauli=True)[:, :, :3] * w_idle).sum(axis=2)
        return inst, idle
    raise ValueError("metric must be 'rate' or 'burden'.")


def _glow(ax, x, y, width, height, color, strength, layers=6, z=2.0):
    for i in range(layers, 0, -1):
        pad = 0.06 + 0.07 * i
        alpha = strength * (1.0 - i / (layers + 1)) * 0.55
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2 - pad, y - height / 2 - pad),
                width + 2 * pad,
                height + 2 * pad,
                boxstyle=f"round,pad=0,rounding_size={0.1 + pad}",
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=z,
            )
        )


def plot_error_heatmap(
    result: Result,
    *,
    metric: str = "rate",
    ax: plt.Axes | None = None,
    scale: str = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
    show_idle: bool = True,
    annotate: bool = False,
    colorbar: bool = True,
    title: str | None = None,
    style: Style | None = None,
):
    """Draw the circuit with halos sized and colored by simulated error.

    Args:
        result: A :class:`~noisiq.Result` from the ``frame`` or ``statevector`` backend.
        metric: ``"rate"`` or ``"burden"`` (see module docs).
        ax: Axes to draw on.
        scale: ``"linear"`` or ``"log"`` color scale.
        vmin, vmax: Color range (use the same values to compare several heatmaps).
        show_idle: Draw idle-decoherence bands on the wires.
        annotate: Write the value (in %) under each gate.
        colorbar: Add a colorbar.
        title: Figure title (defaults to circuit name + noise model).

    Returns:
        The :class:`~noisiq.visualization.CircuitDrawing` (``.fig``, ``.ax``, ``.layout``).
    """
    style = style or get_style()
    inst_v, idle_v = heat_values(result, metric)
    circuit = result.circuit
    if title is None:
        what = "fault rate" if metric == "rate" else "error burden"
        title = f"{circuit.name or 'circuit'} — {result.noise_name} ({what})"
    drawing = draw_circuit(circuit, ax, title=title, style=style)
    ax = drawing.ax
    layout = drawing.layout

    values = [inst_v[inst_v > 0]]
    if show_idle and idle_v.size:
        values.append(idle_v[idle_v > 0])
    allv = np.concatenate(values) if values else np.array([])
    hi = vmax if vmax is not None else (float(allv.max()) if allv.size else 1.0)
    if scale == "log":
        lo = vmin if vmin is not None else (max(float(allv.min()), hi * 1e-3) if allv.size else hi * 1e-3)
        norm = mcolors.LogNorm(vmin=lo, vmax=max(hi, lo * 1.0001))
    elif scale == "linear":
        norm = mcolors.Normalize(vmin=vmin if vmin is not None else 0.0, vmax=hi if hi > 0 else 1.0)
    else:
        raise ValueError("scale must be 'linear' or 'log'.")
    cmap = plt.get_cmap(style.heat_cmap)

    def strength(v: float) -> float:
        return float(np.clip(norm(v), 0.05, 1.0)) if v > 0 else 0.0

    if show_idle:
        for t in range(min(idle_v.shape[0], len(layout.layer_columns))):
            a, b = layout.layer_span(t)
            for q in range(circuit.n_qubits):
                v = idle_v[t, q]
                if v > 0:
                    y = layout.qubit_y(q)
                    s = strength(v)
                    ax.fill_between(
                        [a, b],
                        y - 0.09 - 0.12 * s,
                        y + 0.09 + 0.12 * s,
                        color=cmap(norm(v)),
                        alpha=0.25 + 0.55 * s,
                        lw=0,
                        zorder=0.8,
                    )

    for idx, p in layout.placed.items():
        for q in circuit[idx].qubits:
            v = inst_v[idx, q] if idx < inst_v.shape[0] else 0.0
            if v <= 0:
                continue
            y = layout.qubit_y(q)
            _glow(ax, p.x, y, p.width, style.gate_height, cmap(norm(v)), strength(v))
            if annotate:
                ax.text(
                    p.x,
                    y - style.gate_height / 2 - 0.05,
                    f"{100 * v:.2g}%",
                    ha="center",
                    va="top",
                    fontsize=style.fontsize * 0.6,
                    zorder=4,
                    color="#333333",
                )

    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        label = "fault probability (per gate & qubit)" if metric == "rate" else "expected corrupted outputs"
        cb = drawing.fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(label, fontsize=style.fontsize * 0.8)
    return drawing
