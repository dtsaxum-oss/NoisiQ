"""Geometry of a circuit diagram, independent of any plotting library.

:class:`CircuitLayout` turns any :class:`~noisiq.Circuit` into coordinates: where each
wire, column and instruction sits and how wide every gate box is. All NoisiQ figures
(plain diagrams, heatmaps, propagation animations) are drawn from the same layout, and
:meth:`CircuitLayout.to_dict` exports it as plain data for other front-ends (e.g. a
future web renderer).

Rules
-----
* Each circuit layer becomes one or more *columns*. Instructions in the same layer whose
  vertical extents overlap (e.g. ``CX(0, 2)`` and ``H(1)``) go into separate columns.
* Box widths grow with the label length (``RZ(π/4)`` is wider than ``H``).
* Qubit ``q`` sits at ``y = -q``; the classical wire (if any) below the last qubit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.circuit import Circuit
from ..ir.instructions import Barrier, Delay, Measure, Operation, Reset
from .style import Style, get_style

__all__ = ["CircuitLayout", "PlacedInstruction", "instruction_label"]


def _fmt_seconds(t: float) -> str:
    for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if t >= scale or unit == "ns":
            return f"{t / scale:.3g}{unit}"
    return f"{t:.3g}s"  # pragma: no cover


def _airy(label: str) -> str:
    """Stack parametric labels: ``P(π/8)`` becomes ``P`` over ``π/8`` (two lines).

    Keeps boxes narrow and avoids the angle text crowding the gate name.
    """
    if label.endswith(")") and "(" in label:
        name, _, args = label.partition("(")
        return f"{name}\n{args[:-1]}"
    return label


def label_width_chars(label: str) -> float:
    """Effective width of a (possibly stacked) label, in characters."""
    lines = label.split("\n")
    if len(lines) == 1:
        return float(len(label))
    return max(len(lines[0]), 0.8 * max(len(x) for x in lines[1:]))


def instruction_label(inst) -> str:
    """Short label drawn inside an instruction's box."""
    if isinstance(inst, Operation):
        g = inst.gate
        if g.base is not None and g.num_controls:
            return _airy(g.base.display_label if not g.params else g.display_label.lstrip("C"))
        return _airy(g.display_label)
    if isinstance(inst, Measure):
        return "M" if inst.basis == "Z" else f"M{inst.basis}"
    if isinstance(inst, Reset):
        return "|0⟩"
    if isinstance(inst, Delay):
        return f"wait {_fmt_seconds(inst.duration)}"
    return ""


@dataclass(frozen=True)
class PlacedInstruction:
    index: int
    layer: int
    column: int
    x: float  # center
    width: float
    y_top: float  # topmost involved qubit (largest y)
    y_bottom: float  # lowest involved qubit (smallest y)
    to_classical: bool


@dataclass
class CircuitLayout:
    """Coordinates for drawing ``circuit``; build with :meth:`from_circuit`."""

    circuit: Circuit
    style: Style
    placed: dict[int, PlacedInstruction] = field(default_factory=dict)
    column_x: list[float] = field(default_factory=list)
    column_width: list[float] = field(default_factory=list)
    layer_columns: list[list[int]] = field(default_factory=list)
    barriers: list[tuple[float, tuple[int, ...]]] = field(default_factory=list)
    x_min: float = 0.0
    x_max: float = 0.0

    @classmethod
    def from_circuit(cls, circuit: Circuit, style: Style | None = None) -> CircuitLayout:
        style = style or get_style()
        lay = cls(circuit, style)
        cl_y = lay.classical_y
        x = 0.0
        barrier_layers: dict[int, list[tuple[int, ...]]] = {}
        for inst in circuit:
            if isinstance(inst, Barrier):
                barrier_layers.setdefault(inst.layer, []).append(inst.qubits)
        for t, insts in enumerate(circuit.layers()):
            if t in barrier_layers:
                for qs in barrier_layers.pop(t):
                    lay.barriers.append((x - style.column_gap / 2, qs))
            columns: list[list[tuple[float, float]]] = []
            assign: list[tuple[int, int, float, float, float, bool]] = []
            for idx, inst in sorted(insts, key=lambda kv: min(kv[1].qubits)):
                ys = [-q for q in inst.qubits]
                to_cl = isinstance(inst, Measure) or (isinstance(inst, Operation) and inst.condition is not None)
                top, bottom = max(ys), min(ys)
                lo = cl_y if (to_cl and cl_y is not None) else bottom
                span = (lo - 0.45, top + 0.45)
                col = 0
                while col < len(columns) and any(not (span[1] <= a or span[0] >= b) for a, b in columns[col]):
                    col += 1
                if col == len(columns):
                    columns.append([])
                columns[col].append(span)
                label = instruction_label(inst)
                width = max(style.gate_min_width, 0.42 + style.char_width * label_width_chars(label))
                if (
                    isinstance(inst, Operation)
                    and inst.gate.num_controls
                    and inst.gate.base is not None
                    and inst.gate.base.name in ("X", "Z", "SWAP")
                ):
                    width = style.gate_min_width * 0.8
                if isinstance(inst, Operation) and inst.gate.name == "SWAP":
                    width = style.gate_min_width * 0.8
                assign.append((idx, col, width, top, bottom, to_cl))
            n_cols = max(len(columns), 1)
            widths = [style.gate_min_width] * n_cols
            for _, col, w, *_ in assign:
                widths[col] = max(widths[col], w)
            cols_here = []
            for c in range(n_cols):
                center = x + widths[c] / 2
                lay.column_x.append(center)
                lay.column_width.append(widths[c])
                cols_here.append(len(lay.column_x) - 1)
                x += widths[c] + style.column_gap
            lay.layer_columns.append(cols_here)
            for idx, col, w, top, bottom, to_cl in assign:
                ci = cols_here[col]
                lay.placed[idx] = PlacedInstruction(idx, t, ci, lay.column_x[ci], w, top, bottom, to_cl)
        for qs_list in barrier_layers.values():  # trailing barriers
            for qs in qs_list:
                lay.barriers.append((x - style.column_gap / 2, qs))
        lay.x_min = -0.9
        lay.x_max = max(x - style.column_gap + 0.5, 0.6)
        return lay

    # ------------------------------------------------------------------ queries
    @property
    def n_qubits(self) -> int:
        return self.circuit.n_qubits

    @property
    def classical_y(self) -> float | None:
        if self.circuit.n_clbits == 0 and not self.circuit.has_measurements:
            return None
        return -self.circuit.n_qubits - 0.1

    def qubit_y(self, q: int) -> float:
        return -float(q)

    @property
    def y_min(self) -> float:
        cy = self.classical_y
        return (cy if cy is not None else -(self.n_qubits - 1)) - 0.8

    @property
    def y_max(self) -> float:
        return 0.8

    def layer_span(self, layer: int) -> tuple[float, float]:
        """``(x_left, x_right)`` of a layer's columns."""
        if not 0 <= layer < len(self.layer_columns):
            raise IndexError(f"layer {layer} out of range")
        cols = self.layer_columns[layer]
        left = self.column_x[cols[0]] - self.column_width[cols[0]] / 2 - self.style.column_gap / 2
        right = self.column_x[cols[-1]] + self.column_width[cols[-1]] / 2 + self.style.column_gap / 2
        return left, right

    def boundary_x(self, step: int) -> float:
        """x position *after* ``step`` layers (``step=0``: start of the circuit)."""
        if step <= 0 or not self.layer_columns:
            return self.x_min + 0.45
        return self.layer_span(min(step, len(self.layer_columns)) - 1)[1]

    def to_dict(self) -> dict:
        """Plain-data export of the layout (for non-matplotlib renderers)."""
        out = {
            "n_qubits": self.n_qubits,
            "classical_y": self.classical_y,
            "x_range": [self.x_min, self.x_max],
            "instructions": [],
            "barriers": [],
        }
        for idx, p in sorted(self.placed.items()):
            inst = self.circuit[idx]
            out["instructions"].append(
                {
                    "index": idx,
                    "layer": p.layer,
                    "kind": type(inst).__name__,
                    "label": instruction_label(inst),
                    "qubits": list(inst.qubits),
                    "x": p.x,
                    "width": p.width,
                    "y_top": p.y_top,
                    "y_bottom": p.y_bottom,
                }
            )
        out["barriers"] = [{"x": x, "qubits": list(qs)} for x, qs in self.barriers]
        return out
