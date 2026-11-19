"""Draw any circuit with matplotlib.

:func:`draw_circuit` is the only circuit renderer in NoisiQ; heatmaps and
propagation animations call it and then add overlays using the returned
:class:`CircuitDrawing` (which exposes the :class:`~noisiq.visualization.CircuitLayout`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch

from ..ir.circuit import Circuit
from ..ir.instructions import Delay, Measure, Reset
from .layout import CircuitLayout, instruction_label
from .style import Style, get_style

__all__ = ["draw_circuit", "CircuitDrawing", "gate_category"]


def gate_category(inst) -> str:
    """Color category of an instruction (keys of ``Style.gate_colors``)."""
    if isinstance(inst, Measure):
        return "measure"
    if isinstance(inst, Reset):
        return "reset"
    if isinstance(inst, Delay):
        return "delay"
    g = inst.gate
    if g.num_qubits > 1 and not g.num_controls:
        return "multi"
    base = g.base if (g.num_controls and g.base is not None) else g
    if base.is_pauli:
        return "pauli"
    if base.is_clifford:
        return "clifford"
    return "non_clifford"


@dataclass
class CircuitDrawing:
    """Handle returned by :func:`draw_circuit`."""

    fig: plt.Figure
    ax: plt.Axes
    layout: CircuitLayout

    def gate_center(self, index: int, qubit: int | None = None) -> tuple[float, float]:
        """Data coordinates of instruction ``index`` (at ``qubit``'s wire if given)."""
        p = self.layout.placed[index]
        y = self.layout.qubit_y(qubit) if qubit is not None else (p.y_top + p.y_bottom) / 2
        return p.x, y

    def _repr_html_(self):  # pragma: no cover - notebook display
        return None


def draw_circuit(
    circuit: Circuit,
    ax: plt.Axes | None = None,
    *,
    title: str | None = None,
    style: Style | None = None,
    layout: CircuitLayout | None = None,
    highlight_layer: int | None = None,
    completed_layers: int | None = None,
    gate_facecolor: Callable[[int], str | None] | None = None,
    qubit_labels: list[str] | None = None,
    show_layer_numbers: bool = True,
    scale: float = 1.0,
) -> CircuitDrawing:
    """Draw ``circuit``.

    Args:
        circuit: Any circuit.
        ax: Axes to draw on (a new figure is created otherwise).
        title: Axes title (defaults to the circuit name).
        style: Visual style (default: :func:`~noisiq.visualization.get_style`).
        layout: Pre-computed layout.
        highlight_layer: Shade this layer's columns.
        completed_layers: Grey out the first ``completed_layers`` layers.
        gate_facecolor: ``f(instruction_index) -> color`` overriding box colors.
        qubit_labels: Wire labels (default ``q0``, ``q1`` …).
        show_layer_numbers: Print layer indices above the columns.
        scale: Figure size multiplier when a new figure is created.

    Returns:
        :class:`CircuitDrawing` with ``fig``, ``ax`` and ``layout``.
    """
    style = style or get_style()
    layout = layout or CircuitLayout.from_circuit(circuit, style)
    n = circuit.n_qubits
    if ax is None:
        w = max(4.0, 0.75 * (layout.x_max - layout.x_min)) * scale
        h = max(1.6, 0.62 * (layout.y_max - layout.y_min)) * scale
        fig, ax = plt.subplots(figsize=(min(w, 40), min(h, 30)), dpi=style.dpi)
    else:
        fig = ax.get_figure()
    fs = style.fontsize
    ax.set_xlim(layout.x_min - 0.3, layout.x_max + 0.1)
    ax.set_ylim(layout.y_min, layout.y_max + (0.3 if show_layer_numbers else 0))
    ax.set_aspect("equal")
    ax.axis("off")

    # progress shading
    if completed_layers and layout.layer_columns:
        x_done = layout.boundary_x(completed_layers)
        ax.axvspan(layout.x_min + 0.3, x_done, color=style.done_color, zorder=0)
    if highlight_layer is not None and 0 <= highlight_layer < len(layout.layer_columns):
        a, b = layout.layer_span(highlight_layer)
        ax.axvspan(a, b, color=style.highlight_color, zorder=0.1)

    # wires
    labels = qubit_labels or [f"q{q}" for q in range(n)]
    for q in range(n):
        y = layout.qubit_y(q)
        ax.plot([layout.x_min + 0.3, layout.x_max], [y, y], color=style.wire_color, lw=style.wire_width, zorder=1)
        ax.text(
            layout.x_min + 0.2,
            y,
            labels[q],
            ha="right",
            va="center",
            fontsize=fs,
            family=style.font,
            color=style.text_color,
        )
    cy = layout.classical_y
    if cy is not None:
        for dy in (0.04, -0.04):
            ax.plot(
                [layout.x_min + 0.3, layout.x_max], [cy + dy, cy + dy], color=style.classical_color, lw=0.8, zorder=1
            )
        ax.text(
            layout.x_min + 0.2,
            cy,
            f"c ({max(circuit.n_clbits, 1)})",
            ha="right",
            va="center",
            fontsize=fs * 0.85,
            color=style.classical_color,
            family=style.font,
        )
    if show_layer_numbers:
        for t in range(len(layout.layer_columns)):
            a, b = layout.layer_span(t)
            ax.text(
                (a + b) / 2,
                layout.y_max - 0.05,
                str(t),
                ha="center",
                va="bottom",
                fontsize=fs * 0.7,
                color="#888888",
                family=style.font,
            )
    for x, qs in layout.barriers:
        ys = [layout.qubit_y(q) for q in qs]
        ax.plot([x, x], [min(ys) - 0.4, max(ys) + 0.4], ls="--", color="#999999", lw=0.9, zorder=1.2)

    for idx, p in layout.placed.items():
        inst = circuit[idx]
        face = (gate_facecolor(idx) if gate_facecolor else None) or style.gate_colors[gate_category(inst)]
        _draw_instruction(ax, inst, p, layout, style, face)

    if title is None:
        title = circuit.name
    if title:
        ax.set_title(title, fontsize=fs * 1.15, family=style.font)
    return CircuitDrawing(fig, ax, layout)


# ---------------------------------------------------------------------------- glyphs
def _box(ax, x, y_top, y_bottom, width, label, face, style, *, dashed=False, z=3):
    h = style.gate_height
    y0 = y_bottom - h / 2
    height = (y_top - y_bottom) + h
    patch = FancyBboxPatch(
        (x - width / 2, y0),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=face,
        edgecolor=style.edge_color,
        lw=1.0,
        zorder=z,
        ls="--" if dashed else "-",
    )
    ax.add_patch(patch)
    if label:
        size = style.fontsize * (0.95 if len(label) <= 3 else 0.8)
        ax.text(
            x,
            (y_top + y_bottom) / 2,
            label,
            ha="center",
            va="center",
            fontsize=size,
            family=style.font,
            color=style.text_color,
            zorder=z + 0.1,
        )


def _dot(ax, x, y, style, z=3):
    ax.add_patch(Circle((x, y), 0.09, color=style.edge_color, zorder=z))


def _oplus(ax, x, y, style, z=3):
    r = 0.2
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=style.edge_color, lw=1.1, zorder=z))
    ax.plot([x - r, x + r], [y, y], color=style.edge_color, lw=1.1, zorder=z + 0.1)
    ax.plot([x, x], [y - r, y + r], color=style.edge_color, lw=1.1, zorder=z + 0.1)


def _cross(ax, x, y, style, z=3):
    d = 0.13
    ax.plot([x - d, x + d], [y - d, y + d], color=style.edge_color, lw=1.4, zorder=z)
    ax.plot([x - d, x + d], [y + d, y - d], color=style.edge_color, lw=1.4, zorder=z)


def _classical_link(ax, x, y_from, layout, style, text):
    cy = layout.classical_y
    if cy is None:
        return
    for dx in (-0.025, 0.025):
        ax.plot([x + dx, x + dx], [y_from, cy + 0.08], color=style.classical_color, lw=0.8, zorder=2)
    ax.add_patch(Circle((x, cy), 0.06, color=style.classical_color, zorder=2.5))
    ax.text(
        x + 0.08,
        cy - 0.12,
        text,
        ha="left",
        va="top",
        fontsize=style.fontsize * 0.65,
        color=style.classical_color,
        family=style.font,
    )


def _draw_instruction(ax, inst, p, layout, style, face) -> None:
    x = p.x
    if isinstance(inst, Measure):
        y = layout.qubit_y(inst.qubit)
        _box(ax, x, y, y, p.width, "", face, style)
        ax.add_patch(Arc((x, y - 0.12), 0.38, 0.32, theta1=0, theta2=180, color=style.edge_color, lw=1, zorder=3.2))
        ax.plot([x, x + 0.14], [y - 0.12, y + 0.12], color=style.edge_color, lw=1, zorder=3.2)
        if inst.basis != "Z":
            ax.text(
                x - p.width / 2 + 0.06,
                y + 0.2,
                inst.basis,
                fontsize=style.fontsize * 0.6,
                ha="left",
                va="top",
                zorder=3.3,
                family=style.font,
            )
        if inst.reset:
            ax.text(x + p.width / 2 + 0.02, y + 0.22, "↺", fontsize=style.fontsize * 0.7, zorder=3.3)
        _classical_link(ax, x, y - style.gate_height / 2, layout, style, f"c{inst.clbit}")
        return
    if isinstance(inst, (Reset, Delay)):
        y = layout.qubit_y(inst.qubit)
        _box(ax, x, y, y, p.width, instruction_label(inst), face, style, dashed=isinstance(inst, Delay))
        return

    g = inst.gate
    ys = [layout.qubit_y(q) for q in inst.qubits]
    top, bottom = max(ys), min(ys)
    if inst.condition is not None:
        bits = inst.condition.clbits
        reg = next((r.name for r in layout.circuit.registers if r.clbits == bits), None)
        if reg is not None:
            txt = f"{reg}={inst.condition.value}"
        elif len(bits) == 1:
            txt = f"c{bits[0]}={inst.condition.value}"
        else:
            txt = f"c{list(bits)}={inst.condition.value}"
        _classical_link(ax, x, bottom - style.gate_height / 2, layout, style, txt)
    if len(inst.qubits) > 1:
        ax.plot([x, x], [bottom, top], color=style.edge_color, lw=1.2, zorder=2.5)
    if g.num_controls and g.base is not None:
        ctrl = inst.qubits[: g.num_controls]
        targ = inst.qubits[g.num_controls :]
        for q in ctrl:
            _dot(ax, x, layout.qubit_y(q), style)
        tys = [layout.qubit_y(q) for q in targ]
        if g.base.name == "X" and len(targ) == 1:
            _oplus(ax, x, tys[0], style)
        elif g.base.name == "Z" and len(targ) == 1:
            _dot(ax, x, tys[0], style)
        elif g.base.name == "SWAP" and len(targ) == 2:
            for ty in tys:
                _cross(ax, x, ty, style)
        else:
            _box(ax, x, max(tys), min(tys), p.width, instruction_label(inst), face, style)
        return
    if g.name == "SWAP":
        for y in ys:
            _cross(ax, x, y, style)
        return
    if len(inst.qubits) == 1:
        _box(ax, x, top, bottom, p.width, instruction_label(inst), face, style)
        return
    _box(ax, x, top, bottom, p.width, instruction_label(inst), face, style)
    for i, y in enumerate(ys):
        ax.text(
            x - p.width / 2 + 0.04,
            y,
            str(i),
            fontsize=style.fontsize * 0.55,
            ha="left",
            va="center",
            color="#555555",
            zorder=3.3,
        )
