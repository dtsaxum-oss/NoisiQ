"""Step-through views and animations of error propagation.

Works with any :class:`~noisiq.results.PropagationTrace`:

* ``kind="pauli"`` (Clifford circuits — from ``nq.propagate`` or the frame backend with
  ``trace_shots``): colored X/Y/Z badges show the exact error frame after each layer, and
  colored wire segments show where the error has travelled.
* ``kind="deviation"`` (statevector backend): wire segments are colored by how far each
  qubit's state is from ideal (``1 − fidelity``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, RegularPolygon

from ..results.result import Result
from ..results.trace import PropagationTrace
from .circuit import draw_circuit
from .layout import CircuitLayout
from .style import Style, get_style

__all__ = ["plot_propagation", "animate_propagation", "save_animation", "interactive_propagation"]


def _as_trace(obj: Union[PropagationTrace, Result]) -> PropagationTrace:
    if isinstance(obj, PropagationTrace):
        return obj
    if isinstance(obj, Result):
        if obj.trace is None:
            raise ValueError("This result has no trace; run nq.simulate(..., trace_shots=10).")
        return obj.trace
    raise TypeError(f"Expected a PropagationTrace or Result, got {type(obj).__name__}.")


def _pick_shot(trace: PropagationTrace, shot: int | None) -> int:
    if shot is not None:
        if not 0 <= shot < trace.n_shots:
            raise ValueError(f"shot must be in [0, {trace.n_shots}).")
        return shot
    # Prefer a shot where an error actually flipped a measurement (the instructive case),
    # then any shot with a fault, then shot 0.
    for candidate in (trace.first_faulty_shot(harmful=True), trace.first_faulty_shot()):
        if candidate is not None:
            return candidate
    return 0


def plot_propagation(
    trace: Union[PropagationTrace, Result],
    step: int | None = None,
    *,
    shot: int | None = None,
    ax: plt.Axes | None = None,
    style: Style | None = None,
    layout: CircuitLayout | None = None,
    title: str | None = None,
):
    """Show the error state after ``step`` layers (default: the end of the circuit).

    Args:
        trace: A trace (or a Result that carries one).
        step: Number of completed layers (0 … n_layers).
        shot: Which recorded shot to show (default: the first one where an error flipped a
            measurement, else the first one with any fault).
    """
    trace = _as_trace(trace)
    style = style or get_style()
    shot = _pick_shot(trace, shot)
    n_layers = trace.n_layers
    step = n_layers if step is None else int(np.clip(step, 0, n_layers))
    circuit = trace.circuit
    layout = layout or CircuitLayout.from_circuit(circuit, style)
    if title is None:
        title = f"{circuit.name or 'circuit'} — layer {step}/{n_layers}"
        if trace.kind == "pauli":
            title += f"   error frame: {trace.pauli_string(step, shot)}"
    drawing = draw_circuit(
        circuit,
        ax,
        style=style,
        layout=layout,
        title=title,
        completed_layers=step,
        highlight_layer=step - 1 if step > 0 else None,
    )
    ax = drawing.ax
    colors = style.pauli_colors
    lw = 5.0

    # trail: where the error lived in each completed layer
    for t in range(1, step + 1):
        a = layout.boundary_x(t - 1)
        b = layout.boundary_x(t)
        for q in range(circuit.n_qubits):
            y = layout.qubit_y(q)
            if trace.kind == "pauli":
                ch = _char(trace, t, q, shot)
                if ch != "I":
                    ax.plot([a, b], [y, y], color=colors[ch], lw=lw, alpha=0.45, solid_capstyle="butt", zorder=1.5)
            else:
                v = float(trace.deviation[t, q, shot])
                if v > 1e-4:
                    cmap = plt.get_cmap(style.deviation_cmap)
                    ax.plot(
                        [a, b],
                        [y, y],
                        color=cmap(0.3 + 0.6 * min(1.0, v)),
                        lw=lw,
                        alpha=0.35 + 0.6 * min(1.0, v),
                        solid_capstyle="butt",
                        zorder=1.5,
                    )

    # fault events so far
    for e in trace.events_for(shot):
        visible = e.layer < step or (e.kind == "injected" and e.index is None and step == 0 and e.layer == 0)
        if not visible:
            continue
        color = colors.get(e.pauli, colors["other"])
        if e.index is not None and e.index in layout.placed:
            # directly above the gate (or the measurement, for readout faults)
            x = layout.placed[e.index].x
            y = layout.qubit_y(e.qubit) + style.gate_height / 2 + 0.17
        else:
            # idle fault: on the wire, in the middle of that layer's time window
            if e.layer < len(layout.layer_columns):
                a, b = layout.layer_span(e.layer)
                x = (a + b) / 2
            else:
                x = layout.x_max - 0.3
            y = layout.qubit_y(e.qubit)
        ax.add_patch(RegularPolygon((x, y), 5, radius=0.12, color=color, ec="white", lw=0.8, zorder=7))

    # badges at the current boundary
    xb = layout.boundary_x(step)
    for q in range(circuit.n_qubits):
        y = layout.qubit_y(q)
        if trace.kind == "pauli":
            ch = _char(trace, step, q, shot)
            if ch == "I":
                continue
            ax.add_patch(Circle((xb, y), 0.2, color=colors[ch], zorder=6))
            ax.text(
                xb,
                y,
                ch,
                color="white",
                ha="center",
                va="center",
                fontsize=style.fontsize * 0.85,
                fontweight="bold",
                zorder=6.1,
            )
        else:
            v = float(trace.deviation[step, q, shot]) if step > 0 else 0.0
            if v > 1e-3:
                ax.text(
                    xb,
                    y + 0.3,
                    f"{100 * v:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=style.fontsize * 0.65,
                    color="#aa0000",
                    zorder=6,
                )

    # measurements whose recorded bit was flipped (shown once that measurement has happened)
    if trace.measurement_flips is not None:
        flips = trace.measurement_flips[shot]
        from ..ir.instructions import Measure

        for idx, p in layout.placed.items():
            inst = circuit[idx]
            if isinstance(inst, Measure) and inst.layer < step and inst.clbit < len(flips) and flips[inst.clbit]:
                y = layout.qubit_y(inst.qubit)
                ax.add_patch(Circle((p.x, y), 0.42, fill=False, ec="#d62728", lw=2.2, zorder=6))
    return drawing


def _char(trace: PropagationTrace, step: int, q: int, shot: int) -> str:
    x = int(trace.frames_x[step, q, shot])
    z = int(trace.frames_z[step, q, shot])
    return {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}[(x, z)]


def animate_propagation(
    trace: Union[PropagationTrace, Result],
    *,
    shot: int | None = None,
    interval: int = 700,
    style: Style | None = None,
    repeat: bool = True,
) -> FuncAnimation:
    """Animate the error state layer by layer.

    In Jupyter, display with ``from IPython.display import HTML; HTML(anim.to_jshtml())``
    or save with :func:`save_animation`.
    """
    trace = _as_trace(trace)
    style = style or get_style()
    shot = _pick_shot(trace, shot)
    layout = CircuitLayout.from_circuit(trace.circuit, style)
    first = plot_propagation(trace, 0, shot=shot, style=style, layout=layout)
    fig, ax = first.fig, first.ax

    def update(step):
        ax.clear()
        plot_propagation(trace, step, shot=shot, ax=ax, style=style, layout=layout)
        return []

    anim = FuncAnimation(fig, update, frames=range(trace.n_layers + 1), interval=interval, repeat=repeat)
    plt.close(fig)
    return anim


def save_animation(anim: FuncAnimation, path: Union[str, Path], fps: int = 2, dpi: int | None = None) -> Path:
    """Save an animation as ``.gif`` (Pillow) or ``.html`` (self-contained)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    elif path.suffix.lower() in (".html", ".htm"):
        anim._interval = int(1000 / fps)
        path.write_text(anim.to_jshtml(fps=fps), encoding="utf-8")
    else:
        raise ValueError("Use a .gif or .html file name.")
    return path


def interactive_propagation(
    trace: Union[PropagationTrace, Result], *, shot: int | None = None, style: Style | None = None
):
    """Jupyter slider to step through the trace (needs ``pip install noisiq[widgets]``)."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover
        raise ImportError("interactive_propagation needs ipywidgets: pip install 'noisiq[widgets]'") from exc
    trace = _as_trace(trace)
    style = style or get_style()
    layout = CircuitLayout.from_circuit(trace.circuit, style)
    shots = list(range(trace.n_shots))
    out = widgets.Output()
    step_w = widgets.IntSlider(value=0, min=0, max=trace.n_layers, description="layer")
    shot_w = widgets.Dropdown(options=shots, value=_pick_shot(trace, shot), description="shot")

    def render(*_):
        with out:
            out.clear_output(wait=True)
            d = plot_propagation(trace, step_w.value, shot=shot_w.value, style=style, layout=layout)
            plt.show(d.fig)

    step_w.observe(render, names="value")
    shot_w.observe(render, names="value")
    render()
    box = widgets.VBox([widgets.HBox([step_w, shot_w]), out])
    display(box)
    return box
