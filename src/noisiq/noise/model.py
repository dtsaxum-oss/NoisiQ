"""Rule-based noise models.

A :class:`NoiseModel` is a set of *rules* ("depolarize every CX with p=1%",
"T1/T2 decay on every idle qubit", "2% readout error on qubit 3"). Rules are resolved
against a circuit only when it is simulated, so editing the circuit — adding gates,
inserting dynamical decoupling, re-scheduling — can never shift noise onto the wrong
gate (a problem with the old ``{op_index: channel}`` dictionaries).

Timing
------
Time-dependent noise (thermal relaxation, idling, quasi-static detuning) needs gate
durations. Every instruction gets a duration from :class:`Durations`; a layer lasts as
long as its slowest instruction; a qubit that is not busy for the whole layer *idles*
for the remainder and receives the idle rules for that time.

Example:
    >>> nm = NoiseModel()
    >>> nm.add_gate_error(depolarizing(1e-3), num_qubits=1)
    >>> nm.add_gate_error(depolarizing(1e-2, 2), gates="CX")
    >>> nm.add_thermal_relaxation(t1=100e-6, t2=80e-6)
    >>> nm.add_readout_error(0.02)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Union

import numpy as np

from ..ir.gates import Gate
from ..ir.instructions import Delay, Instruction, Measure, Operation, Reset
from .channels import (
    Channel,
    PauliChannel,
    ReadoutError,
    ThermalRelaxation,
    bit_flip,
    depolarizing,
    phase_flip,
)

__all__ = ["NoiseModel", "Durations", "Fault", "ChannelSpec"]

ChannelSpec = Union[Channel, Callable[[float], Channel | None]]
"""A channel, or a function ``duration -> channel`` for time-dependent noise."""

_ALIASES = {"CNOT": "CX", "TOFFOLI": "CCX", "FREDKIN": "CSWAP", "S_DAG": "SDG", "T_DAG": "TDG", "ID": "I"}


def _norm_gate_name(name: str) -> str:
    key = name.upper()
    return _ALIASES.get(key, key)


@dataclass(frozen=True)
class Durations:
    """Instruction durations in seconds.

    Args:
        single_qubit: Duration of 1-qubit gates.
        two_qubit: Duration of 2-qubit gates.
        multi_qubit: Duration of gates on 3+ qubits.
        measure: Duration of a measurement.
        reset: Duration of a reset.
        per_gate: Overrides by gate name, e.g. ``{"RZ": 0.0}`` for virtual Z gates.
    """

    single_qubit: float = 50e-9
    two_qubit: float = 300e-9
    multi_qubit: float = 600e-9
    measure: float = 1e-6
    reset: float = 1e-6
    per_gate: Mapping[str, float] = field(default_factory=dict)

    def of(self, inst: Instruction) -> float:
        """Duration of one instruction."""
        if isinstance(inst, Operation):
            name = _norm_gate_name(inst.gate.name)
            for k, v in self.per_gate.items():
                if _norm_gate_name(k) == name:
                    return float(v)
            n = inst.gate.num_qubits
            return self.single_qubit if n == 1 else self.two_qubit if n == 2 else self.multi_qubit
        if isinstance(inst, Measure):
            return self.measure + (self.reset if inst.reset else 0.0)
        if isinstance(inst, Reset):
            return self.reset
        if isinstance(inst, Delay):
            return inst.duration
        return 0.0


@dataclass(frozen=True)
class Fault:
    """A resolved noise location: apply ``channel`` to ``qubits``."""

    channel: Channel
    qubits: tuple[int, ...]


@dataclass
class _GateRule:
    spec: ChannelSpec
    names: frozenset[str] | None
    gate_objs: tuple[Gate, ...]
    qubits: frozenset[int] | None
    num_qubits: int | None
    where: Callable[[Operation], bool] | None

    def matches(self, op: Operation) -> bool:
        if self.num_qubits is not None and op.gate.num_qubits != self.num_qubits:
            return False
        if self.names is not None or self.gate_objs:
            name_ok = self.names is not None and _norm_gate_name(op.gate.name) in self.names
            obj_ok = any(op.gate == g for g in self.gate_objs)
            if not (name_ok or obj_ok):
                return False
        if self.qubits is not None and not set(op.qubits) <= self.qubits:
            return False
        if self.where is not None and not self.where(op):
            return False
        return True


@dataclass
class _QubitRule:
    spec: object
    qubits: frozenset[int] | None

    def applies(self, q: int) -> bool:
        return self.qubits is None or q in self.qubits


def _qubit_set(qubits) -> frozenset[int] | None:
    if qubits is None:
        return None
    if isinstance(qubits, (int, np.integer)):
        return frozenset([int(qubits)])
    return frozenset(int(q) for q in qubits)


def _realize(spec: ChannelSpec, duration: float) -> Channel | None:
    if isinstance(spec, Channel):
        return spec
    ch = spec(duration)
    if ch is not None and not isinstance(ch, Channel):
        raise TypeError("A channel factory must return a Channel or None.")
    return ch


class NoiseModel:
    """A collection of noise rules; see the module docstring.

    Args:
        name: Shown in plots and result summaries.
        durations: Instruction durations used by time-dependent rules.
    """

    def __init__(self, name: str | None = None, durations: Durations | None = None):
        self.name = name or "custom noise"
        self.durations = durations or Durations()
        self._gate_rules: list[_GateRule] = []
        self._idle_rules: list[_QubitRule] = []
        self._readout_rules: list[_QubitRule] = []
        self._reset_rules: list[_QubitRule] = []
        self._explicit: dict[int, list[tuple[ChannelSpec, tuple[int, ...] | None]]] = {}
        self._quasistatic: list[_QubitRule] = []
        self.description: list[str] = []

    # ------------------------------------------------------------------ rules
    def add_gate_error(
        self,
        channel: ChannelSpec,
        gates: Union[None, str, Gate, Iterable[Union[str, Gate]]] = None,
        qubits: Union[None, int, Iterable[int]] = None,
        *,
        num_qubits: int | None = None,
        where: Callable[[Operation], bool] | None = None,
    ) -> NoiseModel:
        """Apply ``channel`` right after every matching gate.

        Args:
            channel: A :class:`Channel`, or ``f(duration) -> Channel`` (receives the
                gate's duration). A 1-qubit channel on a k-qubit gate is applied to
                each qubit independently; a k-qubit channel is applied jointly.
            gates: Gate names (``"CX"``, case-insensitive) and/or :class:`Gate`
                objects. ``None`` matches every gate.
            qubits: Only gates whose qubits are all in this set.
            num_qubits: Only gates of this arity (e.g. ``1`` or ``2``).
            where: Extra predicate on the :class:`~noisiq.ir.Operation`
                (e.g. ``lambda op: op.tag == "faulty"``).
        """
        names: set[str] | None = None
        objs: list[Gate] = []
        if gates is not None:
            items = [gates] if isinstance(gates, (str, Gate)) else list(gates)
            names = set()
            for g in items:
                if isinstance(g, Gate):
                    objs.append(g)
                else:
                    names.add(_norm_gate_name(str(g)))
        self._gate_rules.append(
            _GateRule(
                channel,
                frozenset(names) if names is not None else None,
                tuple(objs),
                _qubit_set(qubits),
                num_qubits,
                where,
            )
        )
        target = (
            "all gates"
            if gates is None
            else (
                gates
                if isinstance(gates, str)
                else ", ".join(
                    g.name if isinstance(g, Gate) else str(g) for g in ([gates] if isinstance(gates, Gate) else gates)
                )
            )
        )
        if num_qubits is not None:
            target += f" ({num_qubits}-qubit)"
        if qubits is not None:
            target += f" on qubits {sorted(_qubit_set(qubits))}"
        if where is not None:
            target += " (filtered)"
        self.description.append(f"gate error {_spec_repr(channel)} after {target}")
        return self

    def add_idle_error(self, channel: ChannelSpec, qubits=None) -> NoiseModel:
        """Apply ``channel`` to qubits while they idle (receives the idle duration)."""
        self._idle_rules.append(_QubitRule(channel, _qubit_set(qubits)))
        self.description.append(f"idle error {_spec_repr(channel)}")
        return self

    def add_thermal_relaxation(
        self,
        t1: Union[float, Mapping[int, float]],
        t2: Union[float, Mapping[int, float]],
        qubits=None,
        *,
        on_gates: bool = True,
        on_idle: bool = True,
    ) -> NoiseModel:
        """T1/T2 decoherence during gates and/or idle periods.

        ``t1``/``t2`` may be numbers or per-qubit dicts. Uses
        :class:`~noisiq.noise.ThermalRelaxation` with the instruction duration.
        """
        qset = _qubit_set(qubits)
        targets: list[int | None]
        if isinstance(t1, Mapping) or isinstance(t2, Mapping):
            keys = set(t1) if isinstance(t1, Mapping) else set(t2)
            targets = sorted(keys if qset is None else keys & qset)
        elif qset is not None:
            targets = sorted(qset)
        else:
            targets = [None]

        def make(q: int | None) -> Callable[[float], Channel | None]:
            a = t1[q] if isinstance(t1, Mapping) else t1
            b = t2[q] if isinstance(t2, Mapping) else t2

            def f(duration: float) -> Channel | None:
                return ThermalRelaxation(a, b, duration) if duration > 0 else None

            return f

        for q in targets:
            qs = None if q is None else [q]
            if on_gates:
                self._gate_rules.append(_GateRule(make(q), None, (), _qubit_set(qs), 1, None))
                # multi-qubit gates: relax each qubit independently
                self._gate_rules.append(_GateRule(make(q), None, (), None, None, _MultiQubitPicker(q)))
            if on_idle:
                self._idle_rules.append(_QubitRule(make(q), _qubit_set(qs)))
        self.description.append(f"thermal relaxation T1={_fmt_t(t1)}, T2={_fmt_t(t2)}")
        return self

    def add_readout_error(self, p01: Union[float, ReadoutError], p10: float | None = None, qubits=None) -> NoiseModel:
        """Classical bit-flip on measurement outcomes (``p01``: 0→1, ``p10``: 1→0)."""
        err = p01 if isinstance(p01, ReadoutError) else ReadoutError(p01, p10)
        self._readout_rules.append(_QubitRule(err, _qubit_set(qubits)))
        self.description.append(f"readout error {err!r}")
        return self

    def add_reset_error(self, p: float, qubits=None) -> NoiseModel:
        """After a reset, the qubit is left in |1⟩ with probability ``p``."""
        self._reset_rules.append(_QubitRule(bit_flip(p), _qubit_set(qubits)))
        self.description.append(f"reset error p={p}")
        return self

    def add_quasistatic_detuning(self, sigma: float, qubits=None) -> NoiseModel:
        """Shot-to-shot frequency drift (low-frequency / 1/f noise).

        At the start of every shot each qubit draws a detuning ``δ ~ Normal(0, σ)``
        (σ in rad/s). During any interval ``τ`` the qubit accumulates ``RZ(δ τ)``.
        Because ``δ`` is constant within a shot, echo pulses (dynamical decoupling)
        refocus it — exactly the effect DD is designed to cancel. Only the
        statevector backend simulates this correlation exactly; Pauli twirling
        replaces it by independent phase flips.
        """
        if sigma < 0:
            raise ValueError("sigma must be >= 0.")
        self._quasistatic.append(_QubitRule(float(sigma), _qubit_set(qubits)))
        self.description.append(f"quasi-static detuning σ={sigma:.3g} rad/s")
        return self

    def add_error_after(
        self, instruction_index: int, channel: ChannelSpec, qubits: Sequence[int] | None = None
    ) -> NoiseModel:
        """Apply ``channel`` after the instruction at ``instruction_index``.

        Useful for teaching ("inject an X error after gate 3"). Note that indices
        refer to a specific circuit; prefer gate/qubit rules for reusable models.
        """
        self._explicit.setdefault(int(instruction_index), []).append(
            (channel, tuple(qubits) if qubits is not None else None)
        )
        self.description.append(f"explicit {_spec_repr(channel)} after instruction {instruction_index}")
        return self

    # ------------------------------------------------------------ resolution
    def gate_faults(self, index: int, op: Operation, duration: float) -> list[Fault]:
        """Faults applied after gate ``op`` (instruction ``index``)."""
        out: list[Fault] = []
        for rule in self._gate_rules:
            if isinstance(rule.where, _MultiQubitPicker):
                if op.gate.num_qubits < 2:
                    continue
                for q in op.qubits:
                    if rule.where.qubit is None or rule.where.qubit == q:
                        ch = _realize(rule.spec, duration)
                        if ch is not None:
                            out.append(Fault(ch, (q,)))
                continue
            if rule.matches(op):
                ch = _realize(rule.spec, duration)
                if ch is not None:
                    out.extend(_expand(ch, op.qubits, f"gate '{op.gate.name}'"))
        for spec, qs in self._explicit.get(index, []):
            ch = _realize(spec, duration)
            if ch is not None:
                out.extend(_expand(ch, qs if qs is not None else op.qubits, f"instruction {index}"))
        return out

    def explicit_faults(self, index: int, inst: Instruction, duration: float) -> list[Fault]:
        """Explicit faults for non-gate instructions (e.g. after a measurement)."""
        if isinstance(inst, Operation):
            return []
        out = []
        for spec, qs in self._explicit.get(index, []):
            ch = _realize(spec, duration)
            if ch is not None:
                out.extend(_expand(ch, qs if qs is not None else inst.qubits, f"instruction {index}"))
        return out

    def idle_faults(self, qubit: int, duration: float) -> list[Fault]:
        """Faults for ``qubit`` idling for ``duration`` seconds."""
        if duration <= 0:
            return []
        out = []
        for rule in self._idle_rules:
            if rule.applies(qubit):
                ch = _realize(rule.spec, duration)
                if ch is not None:
                    out.extend(_expand(ch, (qubit,), "idle"))
        return out

    def readout_error(self, qubit: int) -> ReadoutError | None:
        """Combined readout error for ``qubit`` (independent flips compose)."""
        p01, p10 = 0.0, 0.0
        found = False
        for rule in self._readout_rules:
            if rule.applies(qubit):
                e: ReadoutError = rule.spec
                p01 = p01 + e.p01 - 2 * p01 * e.p01
                p10 = p10 + e.p10 - 2 * p10 * e.p10
                found = True
        return ReadoutError(p01, p10) if found else None

    def reset_faults(self, qubit: int) -> list[Fault]:
        return [Fault(r.spec, (qubit,)) for r in self._reset_rules if r.applies(qubit)]

    def detuning_sigma(self, qubit: int) -> float:
        """Total quasi-static detuning σ (rad/s) for ``qubit``."""
        return float(np.sqrt(sum(r.spec**2 for r in self._quasistatic if r.applies(qubit))))

    @property
    def has_quasistatic(self) -> bool:
        return bool(self._quasistatic)

    @property
    def is_empty(self) -> bool:
        return not (
            self._gate_rules
            or self._idle_rules
            or self._readout_rules
            or self._reset_rules
            or self._explicit
            or self._quasistatic
        )

    # -------------------------------------------------------------- transforms
    def twirled(self) -> NoiseModel:
        """Return a copy where every channel is replaced by its Pauli twirl.

        The result runs on the fast stabilizer (frame) backend. Coherent errors lose
        their quadratic accumulation and quasi-static detuning becomes independent
        phase flips with ``p = (1 - exp(-σ²τ²/2)) / 2`` per interval τ.
        """
        new = NoiseModel(f"{self.name} (Pauli-twirled)", self.durations)
        new._gate_rules = [replace(r, spec=_twirl_spec(r.spec)) for r in self._gate_rules]
        new._idle_rules = [_QubitRule(_twirl_spec(r.spec), r.qubits) for r in self._idle_rules]
        new._readout_rules = list(self._readout_rules)
        new._reset_rules = [_QubitRule(r.spec.pauli_twirl(), r.qubits) for r in self._reset_rules]
        new._explicit = {k: [(_twirl_spec(s), q) for s, q in v] for k, v in self._explicit.items()}
        for r in self._quasistatic:
            sigma = r.spec

            def dephase(duration: float, sigma=sigma) -> Channel | None:
                p = 0.5 * (1.0 - np.exp(-0.5 * (sigma * duration) ** 2))
                return phase_flip(p) if p > 0 else None

            new._idle_rules.append(_QubitRule(dephase, r.qubits))
            if r.qubits is None:
                new._gate_rules.append(_GateRule(dephase, None, (), None, 1, None))
                new._gate_rules.append(_GateRule(dephase, None, (), None, None, _MultiQubitPicker(None)))
            else:
                for q in sorted(r.qubits):
                    new._gate_rules.append(_GateRule(dephase, None, (), frozenset([q]), 1, None))
                    new._gate_rules.append(_GateRule(dephase, None, (), None, None, _MultiQubitPicker(q)))
        new.description = list(self.description) + ["(all channels Pauli-twirled)"]
        return new

    def copy(self) -> NoiseModel:
        new = NoiseModel(self.name, self.durations)
        new._gate_rules = list(self._gate_rules)
        new._idle_rules = list(self._idle_rules)
        new._readout_rules = list(self._readout_rules)
        new._reset_rules = list(self._reset_rules)
        new._explicit = {k: list(v) for k, v in self._explicit.items()}
        new._quasistatic = list(self._quasistatic)
        new.description = list(self.description)
        return new

    # ----------------------------------------------------------- constructors
    @staticmethod
    def depolarizing_from_infidelity(r: float, num_qubits: int = 1) -> PauliChannel:
        """Depolarizing channel whose *average gate infidelity* is ``r``.

        Uses ``p = r (d + 1) / d`` with ``d = 2**num_qubits`` (see ``nq.explain("depolarizing")``).
        """
        d = 2**num_qubits
        return depolarizing(min(1.0, r * (d + 1) / d), num_qubits)

    @classmethod
    def uniform_depolarizing(
        cls, p1: float, p2: float | None = None, readout: float = 0.0, name: str | None = None
    ) -> NoiseModel:
        """Depolarizing noise ``p1`` after 1-qubit gates, ``p2`` (default ``10·p1``)
        after 2-qubit gates (as a 2-qubit channel) and symmetric readout error."""
        p2 = min(1.0, 10 * p1) if p2 is None else p2
        nm = cls(name or f"depolarizing p1={p1:g}, p2={p2:g}")
        if p1 > 0:
            nm.add_gate_error(depolarizing(p1), num_qubits=1)
        if p2 > 0:
            nm.add_gate_error(depolarizing(p2, 2), num_qubits=2)
            nm.add_gate_error(depolarizing(p2), where=lambda op: op.gate.num_qubits > 2)
        if readout > 0:
            nm.add_readout_error(readout)
        return nm

    @classmethod
    def from_hardware(cls, profile, **kwargs) -> NoiseModel:
        """Build from a :class:`~noisiq.noise.HardwareProfile` or registered name."""
        from .hardware import get_hardware

        if isinstance(profile, str):
            profile = get_hardware(profile)
        return profile.noise_model(**kwargs)

    def __repr__(self) -> str:
        body = "\n  ".join(self.description) or "(no noise)"
        return f"NoiseModel({self.name!r})\n  {body}"


class _MultiQubitPicker:
    """Marker 'where' used by thermal relaxation to relax each qubit of multi-qubit gates."""

    def __init__(self, qubit: int | None):
        self.qubit = qubit

    def __call__(self, op: Operation) -> bool:  # pragma: no cover - never used as predicate
        return op.gate.num_qubits >= 2


def _expand(ch: Channel, qubits: Sequence[int], where: str) -> list[Fault]:
    qubits = tuple(qubits)
    if ch.num_qubits == len(qubits):
        return [Fault(ch, qubits)]
    if ch.num_qubits == 1:
        return [Fault(ch, (q,)) for q in qubits]
    raise ValueError(
        f"A {ch.num_qubits}-qubit channel cannot be applied to {where} on {len(qubits)} qubit(s). "
        "Use a 1-qubit channel (applied per qubit) or a channel matching the gate size."
    )


def _twirl_spec(spec: ChannelSpec) -> ChannelSpec:
    if isinstance(spec, Channel):
        return spec.pauli_twirl()

    def f(duration: float) -> Channel | None:
        ch = spec(duration)
        return None if ch is None else ch.pauli_twirl()

    return f


def _spec_repr(spec) -> str:
    return repr(spec) if isinstance(spec, Channel) else "f(duration)"


def _fmt_t(v) -> str:
    if isinstance(v, Mapping):
        return "per-qubit"
    return f"{v * 1e6:.4g} µs"
