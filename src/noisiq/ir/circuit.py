"""The :class:`Circuit` container.

A circuit is an ordered list of :mod:`instructions <noisiq.ir.instructions>`, each
placed in a *layer* (time step). Instructions are scheduled "as soon as possible"
unless an explicit ``layer=`` is given; two instructions may never share a qubit in
the same layer. Simulators execute instructions in ``(layer, insertion order)``.

Example:
    >>> import noisiq as nq
    >>> c = nq.Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
    >>> c.depth
    4
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from typing import Union

import numpy as np

from . import gates as G
from .gates import Gate
from .instructions import (
    Barrier,
    ClassicalRegister,
    Condition,
    Delay,
    Instruction,
    Measure,
    Operation,
    Reset,
)

__all__ = ["Circuit"]

QubitArg = Union[int, Iterable[int]]
ConditionArg = Union[None, Condition, int, ClassicalRegister, tuple]


def _as_qubit_list(q: QubitArg) -> list[int]:
    if isinstance(q, (int, np.integer)):
        return [int(q)]
    return [int(x) for x in q]


class Circuit:
    """A quantum circuit on ``n_qubits`` qubits and ``n_clbits`` classical bits.

    Args:
        n_qubits: Number of qubits (>= 1).
        n_clbits: Number of classical bits to pre-allocate. More are allocated
            automatically by :meth:`measure`.
        name: Optional name used in plot titles.

    All builder methods return ``self`` so calls can be chained. Single-qubit
    builders accept an int or an iterable of qubits (``c.h(range(3))``).
    """

    def __init__(self, n_qubits: int, n_clbits: int = 0, name: str | None = None):
        if not isinstance(n_qubits, (int, np.integer)) or n_qubits < 1:
            raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}.")
        if n_clbits < 0:
            raise ValueError("n_clbits must be >= 0.")
        self.n_qubits: int = int(n_qubits)
        self.n_clbits: int = int(n_clbits)
        self.name = name
        self.registers: list[ClassicalRegister] = []
        self._instructions: list[Instruction] = []
        self._frontier = [0] * self.n_qubits  # first free layer per qubit
        self._clbit_ready: dict[int, int] = {}  # clbit -> first layer after its last write
        self._occupied: set[tuple[int, int]] = set()  # (layer, qubit)

    # ================================================================ core
    @property
    def instructions(self) -> list[Instruction]:
        """All instructions in insertion order (read-only copy)."""
        return list(self._instructions)

    @property
    def operations(self) -> list[Operation]:
        """Only the gate operations."""
        return [i for i in self._instructions if isinstance(i, Operation)]

    def ordered(self) -> list[tuple[int, Instruction]]:
        """``(index, instruction)`` pairs in execution order ``(layer, index)``."""
        return sorted(enumerate(self._instructions), key=lambda kv: (kv[1].layer, kv[0]))

    def __len__(self) -> int:
        return len(self._instructions)

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions)

    def __getitem__(self, i: int) -> Instruction:
        return self._instructions[i]

    @property
    def num_qubits(self) -> int:
        return self.n_qubits

    @property
    def depth(self) -> int:
        """Number of layers (``max layer + 1``), ignoring barriers."""
        layers = [i.layer for i in self._instructions if not isinstance(i, Barrier)]
        return max(layers) + 1 if layers else 0

    def layers(self) -> list[list[tuple[int, Instruction]]]:
        """Instructions grouped by layer: ``layers()[t]`` = ``[(index, inst), ...]``."""
        out: list[list[tuple[int, Instruction]]] = [[] for _ in range(self.depth)]
        for idx, inst in self.ordered():
            if not isinstance(inst, Barrier):
                out[inst.layer].append((idx, inst))
        return out

    def count_ops(self) -> dict[str, int]:
        """Count instructions by name, e.g. ``{"H": 1, "CX": 2, "MEASURE": 3}``."""
        return dict(Counter(i.name for i in self._instructions if not isinstance(i, Barrier)))

    @property
    def has_measurements(self) -> bool:
        return any(isinstance(i, Measure) for i in self._instructions)

    @property
    def has_conditionals(self) -> bool:
        return any(isinstance(i, Operation) and i.condition is not None for i in self._instructions)

    def is_clifford(self) -> bool:
        """True if every gate is a Clifford gate (measurements/resets allowed)."""
        return all(op.gate.is_clifford for op in self.operations)

    # =========================================================== validation
    def _check_qubits(self, qubits: Sequence[int], what: str) -> tuple[int, ...]:
        qs = tuple(int(q) for q in qubits)
        for q in qs:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"{what}: qubit {q} is out of range for a {self.n_qubits}-qubit circuit.")
        if len(set(qs)) != len(qs):
            raise ValueError(f"{what}: qubits {list(qs)} contain duplicates.")
        return qs

    def _check_clbit(self, c: int, what: str) -> int:
        c = int(c)
        if not 0 <= c < self.n_clbits:
            raise ValueError(f"{what}: classical bit {c} does not exist (circuit has {self.n_clbits}).")
        return c

    def _place(self, qubits: tuple[int, ...], layer: int | None, min_layer: int, what: str) -> int:
        if layer is None:
            layer = max([self._frontier[q] for q in qubits] + [min_layer])
        else:
            layer = int(layer)
            if layer < 0:
                raise ValueError(f"{what}: layer must be >= 0.")
            clash = [q for q in qubits if (layer, q) in self._occupied]
            if clash:
                raise ValueError(f"{what}: qubit(s) {clash} already busy in layer {layer}.")
        for q in qubits:
            self._occupied.add((layer, q))
            self._frontier[q] = max(self._frontier[q], layer + 1)
        return layer

    def _normalize_condition(self, condition: ConditionArg) -> Condition | None:
        if condition is None or isinstance(condition, Condition):
            cond = condition
        elif isinstance(condition, (int, np.integer)):
            cond = Condition((int(condition),), 1)
        elif isinstance(condition, ClassicalRegister):
            cond = Condition(condition.clbits, 1)
        elif isinstance(condition, tuple) and len(condition) == 2:
            bits, value = condition
            if isinstance(bits, ClassicalRegister):
                bits = bits.clbits
            elif isinstance(bits, (int, np.integer)):
                bits = (int(bits),)
            cond = Condition(tuple(bits), int(value))
        else:
            raise TypeError(f"Cannot interpret condition {condition!r}.")
        if cond is not None:
            for c in cond.clbits:
                self._check_clbit(c, "condition")
        return cond

    # ============================================================ builders
    def append(
        self,
        gate: Gate,
        qubits: QubitArg,
        *,
        layer: int | None = None,
        condition: ConditionArg = None,
        tag: str | None = None,
    ) -> Circuit:
        """Append ``gate`` on ``qubits``.

        Args:
            gate: Any :class:`Gate` (library or custom).
            qubits: Target qubits, in the gate's matrix order.
            layer: Explicit layer; default schedules as early as possible.
            condition: Classical condition: a clbit index, ``(clbits, value)``,
                a :class:`ClassicalRegister`, or a :class:`Condition`.
            tag: Free-form label (usable in noise rules).
        """
        if not isinstance(gate, Gate):
            raise TypeError(f"Expected a Gate, got {type(gate).__name__}.")
        qs = self._check_qubits(_as_qubit_list(qubits), f"Gate '{gate.name}'")
        if len(qs) != gate.num_qubits:
            raise ValueError(f"Gate '{gate.name}' acts on {gate.num_qubits} qubit(s) but got {len(qs)}: {list(qs)}.")
        cond = self._normalize_condition(condition)
        min_layer = max((self._clbit_ready.get(c, 0) for c in cond.clbits), default=0) if cond else 0
        lay = self._place(qs, layer, min_layer, f"Gate '{gate.name}'")
        self._instructions.append(Operation(gate, qs, lay, cond, tag))
        return self

    def _single(self, gate: Gate, qubits: QubitArg, layer, condition) -> Circuit:
        for q in _as_qubit_list(qubits):
            self.append(gate, [q], layer=layer, condition=condition)
        return self

    # --- single-qubit fixed gates
    def i(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        """Identity gate (occupies a layer; picks up gate noise if modelled)."""
        return self._single(G.I, q, layer, condition)

    def x(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.X, q, layer, condition)

    def y(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.Y, q, layer, condition)

    def z(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.Z, q, layer, condition)

    def h(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.H, q, layer, condition)

    def s(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.S, q, layer, condition)

    def sdg(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.SDG, q, layer, condition)

    def t(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        """T gate (π/8 gate)."""
        return self._single(G.T, q, layer, condition)

    def tdg(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.TDG, q, layer, condition)

    def sx(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.SX, q, layer, condition)

    def sxdg(self, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.SXDG, q, layer, condition)

    # --- single-qubit parametric gates
    def rx(self, theta: float, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.rx(theta), q, layer, condition)

    def ry(self, theta: float, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.ry(theta), q, layer, condition)

    def rz(self, theta: float, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.rz(theta), q, layer, condition)

    def p(self, theta: float, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.p(theta), q, layer, condition)

    def u(self, theta: float, phi: float, lam: float, q: QubitArg, *, layer=None, condition=None) -> Circuit:
        return self._single(G.u(theta, phi, lam), q, layer, condition)

    # --- multi-qubit gates
    def cx(self, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CX, [control, target], layer=layer, condition=condition)

    cnot = cx

    def cy(self, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CY, [control, target], layer=layer, condition=condition)

    def cz(self, q0: int, q1: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CZ, [q0, q1], layer=layer, condition=condition)

    def ch(self, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CH, [control, target], layer=layer, condition=condition)

    def swap(self, q0: int, q1: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.SWAP, [q0, q1], layer=layer, condition=condition)

    def iswap(self, q0: int, q1: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.ISWAP, [q0, q1], layer=layer, condition=condition)

    def cp(self, theta: float, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.cp(theta), [control, target], layer=layer, condition=condition)

    def crx(self, theta: float, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.crx(theta), [control, target], layer=layer, condition=condition)

    def cry(self, theta: float, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.cry(theta), [control, target], layer=layer, condition=condition)

    def crz(self, theta: float, control: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.crz(theta), [control, target], layer=layer, condition=condition)

    def ccx(self, c0: int, c1: int, target: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CCX, [c0, c1, target], layer=layer, condition=condition)

    toffoli = ccx

    def ccz(self, q0: int, q1: int, q2: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CCZ, [q0, q1, q2], layer=layer, condition=condition)

    def cswap(self, control: int, q0: int, q1: int, *, layer=None, condition=None) -> Circuit:
        return self.append(G.CSWAP, [control, q0, q1], layer=layer, condition=condition)

    def mcx(self, controls: Sequence[int], target: int, *, layer=None, condition=None) -> Circuit:
        """Multi-controlled X (any number of controls)."""
        controls = list(controls)
        gate = G.X if not controls else G.controlled(G.X, len(controls))
        return self.append(gate, controls + [target], layer=layer, condition=condition)

    def mcz(self, controls: Sequence[int], target: int, *, layer=None, condition=None) -> Circuit:
        """Multi-controlled Z (any number of controls)."""
        controls = list(controls)
        gate = G.Z if not controls else G.controlled(G.Z, len(controls))
        return self.append(gate, controls + [target], layer=layer, condition=condition)

    def unitary(self, matrix: np.ndarray, qubits: QubitArg, name: str = "U", *, layer=None, condition=None) -> Circuit:
        """Apply an arbitrary unitary matrix (big-endian in ``qubits`` order)."""
        return self.append(G.unitary(matrix, name), qubits, layer=layer, condition=condition)

    # ============================================================ classical
    def add_register(self, name: str, size: int) -> ClassicalRegister:
        """Allocate ``size`` new classical bits as a named register."""
        if size < 1:
            raise ValueError("Register size must be >= 1.")
        if any(r.name == name for r in self.registers):
            raise ValueError(f"Register '{name}' already exists.")
        reg = ClassicalRegister(name, self.n_clbits, size)
        self.n_clbits += size
        self.registers.append(reg)
        return reg

    def _new_clbit(self) -> int:
        self.n_clbits += 1
        return self.n_clbits - 1

    def measure(
        self,
        qubit: QubitArg,
        clbit: Union[int, Sequence[int]] | None = None,
        *,
        basis: str = "Z",
        reset: bool = False,
        layer: int | None = None,
    ) -> Circuit:
        """Measure qubit(s) in ``basis`` ("X", "Y" or "Z").

        If ``clbit`` is omitted a fresh classical bit is allocated per qubit.
        With ``reset=True`` the qubit is reset to |0⟩ after the measurement.
        """
        qs = _as_qubit_list(qubit)
        if clbit is None:
            cs = [self._new_clbit() for _ in qs]
        else:
            cs = _as_qubit_list(clbit)
            if len(cs) != len(qs):
                raise ValueError("measure: number of qubits and clbits differ.")
        for q, c in zip(qs, cs):
            (q,) = self._check_qubits([q], "measure")
            c = self._check_clbit(c, "measure")
            lay = self._place((q,), layer, 0, "measure")
            self._instructions.append(Measure(q, c, lay, basis, reset))
            self._clbit_ready[c] = max(self._clbit_ready.get(c, 0), lay + 1)
        return self

    def measure_all(self, basis: str = "Z", *, barrier: bool = True) -> Circuit:
        """Measure every qubit into fresh classical bits (clbit k ↔ qubit k when the circuit had none)."""
        if barrier:
            self.barrier()
        return self.measure(range(self.n_qubits), basis=basis)

    def reset(self, qubit: QubitArg, *, layer: int | None = None) -> Circuit:
        """Reset qubit(s) to |0⟩."""
        for q in _as_qubit_list(qubit):
            (q,) = self._check_qubits([q], "reset")
            lay = self._place((q,), layer, 0, "reset")
            self._instructions.append(Reset(q, lay))
        return self

    def delay(self, qubit: QubitArg, duration: float, *, layer: int | None = None) -> Circuit:
        """Let qubit(s) idle for ``duration`` seconds (one layer)."""
        for q in _as_qubit_list(qubit):
            (q,) = self._check_qubits([q], "delay")
            lay = self._place((q,), layer, 0, "delay")
            self._instructions.append(Delay(q, float(duration), lay))
        return self

    def barrier(self, *qubits: QubitArg) -> Circuit:
        """Align the given qubits (default: all) so later gates start together."""
        qs: list[int] = []
        for q in qubits:
            qs.extend(_as_qubit_list(q))
        qs = list(self._check_qubits(qs or range(self.n_qubits), "barrier"))
        lay = max(self._frontier[q] for q in qs)
        for q in qs:
            self._frontier[q] = lay
        self._instructions.append(Barrier(tuple(qs), lay))
        return self

    def c_if(self, clbits: Union[int, Sequence[int], ClassicalRegister], value: int = 1) -> _Conditioned:
        """Condition the next gate on classical bits.

        Example:
            ``c.c_if(0).x(2)`` applies X to qubit 2 if clbit 0 is 1;
            ``c.c_if([0, 1], 0b10).z(2)`` fires when clbit 0 = 0 and clbit 1 = 1.
        """
        if isinstance(clbits, ClassicalRegister):
            bits = clbits.clbits
        else:
            bits = tuple(_as_qubit_list(clbits))
        cond = Condition(bits, value)
        self._normalize_condition(cond)
        return _Conditioned(self, cond)

    # ============================================================ transforms
    def copy(self, name: str | None = None) -> Circuit:
        """Deep copy (instructions are immutable, so this is cheap)."""
        new = Circuit(self.n_qubits, self.n_clbits, name if name is not None else self.name)
        new.registers = list(self.registers)
        new._instructions = list(self._instructions)
        new._frontier = list(self._frontier)
        new._clbit_ready = dict(self._clbit_ready)
        new._occupied = set(self._occupied)
        return new

    def _append_instruction(self, inst: Instruction, qmap, cmap, keep_layers: bool, offset: int = 0) -> None:
        layer = inst.layer + offset if keep_layers else None
        if isinstance(inst, Operation):
            cond = None
            if inst.condition is not None:
                cond = Condition(tuple(cmap[c] for c in inst.condition.clbits), inst.condition.value)
            self.append(inst.gate, [qmap[q] for q in inst.qubits], layer=layer, condition=cond, tag=inst.tag)
        elif isinstance(inst, Measure):
            self.measure(qmap[inst.qubit], cmap[inst.clbit], basis=inst.basis, reset=inst.reset, layer=layer)
        elif isinstance(inst, Reset):
            self.reset(qmap[inst.qubit], layer=layer)
        elif isinstance(inst, Delay):
            self.delay(qmap[inst.qubit], inst.duration, layer=layer)
        elif isinstance(inst, Barrier):
            self.barrier([qmap[q] for q in inst.qubits])

    def compose(
        self,
        other: Circuit,
        qubits: Sequence[int] | None = None,
        clbits: Sequence[int] | None = None,
    ) -> Circuit:
        """Append ``other`` (in place). ``qubits[k]`` is where other's qubit k goes."""
        qmap = list(qubits) if qubits is not None else list(range(other.n_qubits))
        if len(qmap) != other.n_qubits:
            raise ValueError("compose: qubit map must have one entry per qubit of the other circuit.")
        if clbits is None:
            while self.n_clbits < other.n_clbits:
                self._new_clbit()
            cmap = list(range(other.n_clbits))
        else:
            cmap = list(clbits)
        for _, inst in other.ordered():
            self._append_instruction(inst, qmap, cmap, keep_layers=False)
        return self

    def __add__(self, other: Circuit) -> Circuit:
        if other.n_qubits != self.n_qubits:
            raise ValueError("Can only add circuits with the same number of qubits.")
        return self.copy().compose(other)

    def inverse(self) -> Circuit:
        """Return the inverse circuit ``U†`` (gates reversed and inverted).

        Raises:
            ValueError: If the circuit contains measurements, resets or conditionals.
        """
        new = Circuit(self.n_qubits, self.n_clbits, f"{self.name}_inv" if self.name else None)
        for _, inst in reversed(self.ordered()):
            if isinstance(inst, Operation):
                if inst.condition is not None:
                    raise ValueError("Cannot invert a circuit with classically-conditioned gates.")
                new.append(inst.gate.inverse(), inst.qubits, tag=inst.tag)
            elif isinstance(inst, (Measure, Reset)):
                raise ValueError("Cannot invert a circuit containing measurements or resets.")
            elif isinstance(inst, Delay):
                new.delay(inst.qubit, inst.duration)
            elif isinstance(inst, Barrier):
                new.barrier(list(inst.qubits))
        return new

    def repeat(self, n: int) -> Circuit:
        """Return the circuit repeated ``n`` times."""
        out = Circuit(self.n_qubits, 0, self.name)
        for _ in range(n):
            out.compose(self, clbits=None)
        return out

    def without_measurements(self) -> Circuit:
        """Copy with all measurements and barriers removed (gates keep their order)."""
        new = Circuit(self.n_qubits, self.n_clbits, self.name)
        for _, inst in self.ordered():
            if not isinstance(inst, (Measure, Barrier)):
                new._append_instruction(inst, list(range(self.n_qubits)), list(range(self.n_clbits)), False)
        return new

    # ================================================================ misc
    def draw(self, **kwargs):
        """Draw the circuit with matplotlib (see :func:`noisiq.visualization.draw_circuit`)."""
        from ..visualization import draw_circuit

        return draw_circuit(self, **kwargs)

    def to_qasm(self) -> str:
        """Export to OpenQASM 2.0 (see :func:`noisiq.io.to_qasm`)."""
        from ..io.qasm import to_qasm

        return to_qasm(self)

    @classmethod
    def from_qasm(cls, source: str) -> Circuit:
        """Parse OpenQASM 2.0 source (see :func:`noisiq.io.from_qasm`)."""
        from ..io.qasm import from_qasm

        return from_qasm(source)

    def __repr__(self) -> str:
        name = f"{self.name!r}, " if self.name else ""
        return (
            f"Circuit({name}n_qubits={self.n_qubits}, n_clbits={self.n_clbits}, "
            f"instructions={len(self)}, depth={self.depth})"
        )

    def __str__(self) -> str:
        lines = [repr(self)]
        for idx, inst in self.ordered():
            lines.append(f"  [{idx:>3}] {inst!r}")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return (
            self.n_qubits == other.n_qubits
            and self.n_clbits == other.n_clbits
            and self._instructions == other._instructions
        )

    __hash__ = None  # mutable


class _Conditioned:
    """Proxy returned by :meth:`Circuit.c_if`; forwards one gate call with a condition."""

    _GATE_METHODS = {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "p",
        "u",
        "cx",
        "cnot",
        "cy",
        "cz",
        "ch",
        "swap",
        "iswap",
        "cp",
        "crx",
        "cry",
        "crz",
        "ccx",
        "toffoli",
        "ccz",
        "cswap",
        "mcx",
        "mcz",
        "unitary",
        "append",
    }

    def __init__(self, circuit: Circuit, condition: Condition):
        self._circuit = circuit
        self._condition = condition

    def __getattr__(self, name: str):
        if name not in self._GATE_METHODS:
            raise AttributeError(f"c_if(...) can only wrap gate methods, not '{name}'.")
        method = getattr(self._circuit, name)

        def call(*args, **kwargs):
            kwargs["condition"] = self._condition
            return method(*args, **kwargs)

        return call
