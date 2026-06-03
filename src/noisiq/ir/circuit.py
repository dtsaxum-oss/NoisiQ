from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from . import gates
from .classical import ClassicalBit, ClassicalRegister, Measurement, ConditionalOp

Qubits = Tuple[int, ...]

# Discriminated union of all operation types that may appear in circuit.operations.
# All loops over circuit.operations MUST isinstance-check before accessing
# Operation-specific attributes (op.gate, op.qubits).
CircuitOp = Union["Operation", Measurement, ConditionalOp]


def _get_qubits(op: Any) -> Tuple[int, ...]:
    """Return the qubit indices touched by *any* CircuitOp variant.

    - ``Operation``   → ``op.qubits`` (tuple field)
    - ``ConditionalOp`` → ``op.qubits`` (property → inner.qubits)
    - ``Measurement`` → ``(op.qubit,)`` (singular int field)
    - unknown         → empty tuple (safe fallback for future types)
    """
    if isinstance(op, Operation):
        return op.qubits
    if isinstance(op, Measurement):
        return (op.qubit,)
    if isinstance(op, ConditionalOp):
        return op.qubits
    # Fallback for any future union member or test stubs
    if hasattr(op, "qubits"):
        return tuple(op.qubits)
    if hasattr(op, "qubit"):
        return (op.qubit,)
    return ()

## Represents a single operation/gate in a quantum circuit, with its name, target qubits, time, and optional parameters and metadata.
@dataclass(frozen=True)
class Operation:
    """
    One gate/operation in a circuit.

    Example:
        Operation(gate=gates.H, qubits=(0,), t=0)
    """
    gate: gates.Gate
    qubits: Qubits
    t: int
    params: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    ## Removes parameters and meta from printing if they're None, to reduce clutter
    def __repr__(self) -> str:
        parts = [f"gate={self.gate.name!r}", f"qubits={self.qubits!r}", f"t={self.t!r}"]
        if self.params is not None:
            parts.append(f"params={self.params!r}")
        if self.meta is not None:
            parts.append(f"meta={self.meta!r}")
        return f"Operation({', '.join(parts)})"

## A quantum circuit, consisting of a number of qubits and a list of operations based on the operation class above.
@dataclass
class Circuit:
    """
    A quantum circuit described as:
      - number of qubits
      - a list/tuple of operations
    """
    n_qubits: int
    # CircuitOp = Union[Operation, Measurement, ConditionalOp].
    # Do not assume op.gate exists — use isinstance(op, Operation) first.
    operations: List["CircuitOp"] = field(default_factory=list)
    classical_registers: List[ClassicalRegister] = field(default_factory=list)
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def add_gate(
        self,
        gate: gates.Gate,
        qubits: Qubits | List[int],
        t: Optional[int] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Adds a gate operation to the circuit.

        Args:
            gate:   The gate to add (e.g., gates.H, gates.IDLE).
            qubits: The qubit(s) the gate acts on.
            t:      Optional explicit time step (layer index).  When omitted,
                    the gate is auto-scheduled into the earliest available slot
                    where none of its qubits conflict with an existing operation.
                    When provided, the gate is placed at exactly that t — but
                    a ValueError is raised if any qubit is already occupied at
                    that t by another gate.
            params: Optional per-op parameters. For IDLE: {"duration_ns": float}.
            meta:   Optional per-op metadata (free-form).

        Raises:
            ValueError: If the number of qubits provided does not match the
                        gate's definition, if any qubit index is out of bounds,
                        or if any qubit is already in use at the requested t.
        """
        if isinstance(qubits, list):
            qubits = tuple(qubits)

        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"Gate '{gate.name}' acts on {gate.num_qubits} qubit(s), "
                f"but was applied to {len(qubits)}."
            )

        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(
                    f"Qubit index {q} is out of bounds for a circuit with "
                    f"{self.n_qubits} qubits."
                )

        if t is None:
            # Greedy scheduling: earliest slot where all required qubits are free.
            # _get_qubits() handles all CircuitOp variants (Operation, Measurement,
            # ConditionalOp) so Measurement ops on the same qubit are respected.
            qubit_last_t: Dict[int, int] = {}
            for op in self.operations:
                for q in _get_qubits(op):
                    if op.t > qubit_last_t.get(q, -1):
                        qubit_last_t[q] = op.t
            t = max((qubit_last_t.get(q, -1) for q in qubits), default=-1) + 1

        # Qubit-collision check: no two gates at the same t may share a qubit.
        new_qubits = set(qubits)
        for existing in self.operations:
            if existing.t == t:
                conflict = new_qubits & set(_get_qubits(existing))
                if conflict:
                    existing_desc = (
                        existing.gate.name
                        if isinstance(existing, Operation)
                        else type(existing).__name__
                    )
                    raise ValueError(
                        f"Cannot place '{gate.name}' on qubit(s) {sorted(qubits)} at t={t}: "
                        f"qubit(s) {sorted(conflict)} already used by "
                        f"'{existing_desc}' on {list(_get_qubits(existing))} at t={t}."
                    )

        op = Operation(gate=gate, qubits=qubits, t=t, params=params, meta=meta)
        self.operations.append(op)
        return self

    # ------------------------------------------------------------------
    # Fluent builder methods — each returns self for chaining, e.g.:
    #   Circuit(3).h(0).cnot(0, 1).cnot(1, 2)
    # Single-qubit gates accept an optional t= to pin the layer.
    # ------------------------------------------------------------------

    def h(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.H, (qubit,), t=t)

    def x(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.X, (qubit,), t=t)

    def y(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.Y, (qubit,), t=t)

    def z(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.Z, (qubit,), t=t)

    def s(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.S, (qubit,), t=t)

    def s_dag(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.S_DAG, (qubit,), t=t)

    def tgate(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        """T gate — named tgate to avoid collision with the t= layer parameter."""
        return self.add_gate(gates.T, (qubit,), t=t)

    def t_dag(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.T_DAG, (qubit,), t=t)

    def identity(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.I, (qubit,), t=t)

    def idle(self, qubit: int, duration_ns: float, t: Optional[int] = None) -> "Circuit":
        """Insert an IDLE gate lasting duration_ns on qubit.

        IDLE represents a qubit sitting idle while something else runs on
        another qubit. Only T1/T2 decoherence accrues — no gate error.
        duration_ns is required because it drives the noise model.
        """
        return self.add_gate(
            gates.IDLE,
            (qubit,),
            t=t,
            params={"duration_ns": float(duration_ns)},
        )

    def cnot(self, control: int, target: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CNOT, (control, target), t=t)

    def cx(self, control: int, target: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CNOT, (control, target), t=t)

    def cz(self, q1: int, q2: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CZ, (q1, q2), t=t)

    def swap(self, q1: int, q2: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.SWAP, (q1, q2), t=t)

    def cs(self, control: int, target: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CS, (control, target), t=t)

    def cs_dag(self, control: int, target: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CS_DAG, (control, target), t=t)

    def ccz(self, q1: int, q2: int, q3: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CCZ, (q1, q2, q3), t=t)

    def p(self, qubit: int, theta: float, t: Optional[int] = None) -> "Circuit":
        """Phase gate P(θ) = [[1, 0], [0, e^{iθ}]]."""
        return self.add_gate(gates.phase_gate(theta), (qubit,), t=t)

    def rz(self, qubit: int, theta: float, t: Optional[int] = None) -> "Circuit":
        """Rotation-Z gate RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]."""
        return self.add_gate(gates.rz_gate(theta), (qubit,), t=t)

    def ry(
        self,
        qubit: int,
        theta: float,
        t: Optional[int] = None,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Circuit":
        """Rotation-Y gate RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]].

        Accepts an optional `meta` dict for tagging resource sites (e.g. MEK |H>
        preparation sites) without affecting circuit execution.
        """
        return self.add_gate(gates.ry_gate(theta), (qubit,), t=t, meta=meta)

    ## Removes name and metadata from printing if they're None, to reduce clutter
    def __repr__(self) -> str:
        parts = [f"n_qubits={self.n_qubits!r}", f"operations={self.operations!r}"]
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        if self.metadata is not None:
            parts.append(f"metadata={self.metadata!r}")
        return f"Circuit({', '.join(parts)})"

    ## Validates the circuit for basic sanity checks, like non-negative time and valid qubit indices.
    def validate(self) -> None:
        """Basic safety checks so we catch mistakes early."""
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be > 0")

        valid_cbit_indices = {
            reg.offset + i
            for reg in self.classical_registers
            for i in range(reg.size)
        }

        for op in self.operations:
            if isinstance(op, Operation):
                if op.t < 0:
                    raise ValueError(f"Operation time t must be >= 0: {op}")
                for q in op.qubits:
                    if not (0 <= q < self.n_qubits):
                        raise ValueError(f"Qubit index out of range: {q} for {op}")
            elif isinstance(op, Measurement):
                if op.t < 0:
                    raise ValueError(f"Measurement time t must be >= 0: {op}")
                if not (0 <= op.qubit < self.n_qubits):
                    raise ValueError(
                        f"Measurement qubit index out of range: {op.qubit} for {op}"
                    )
                if op.cbit.index not in valid_cbit_indices:
                    raise ValueError(
                        f"Measurement.cbit '{op.cbit.name}' (index {op.cbit.index}) "
                        f"is not allocated by this circuit."
                    )
            elif isinstance(op, ConditionalOp):
                if op.t < 0:
                    raise ValueError(f"ConditionalOp time t must be >= 0: {op}")
                for q in op.qubits:
                    if not (0 <= q < self.n_qubits):
                        raise ValueError(
                            f"Qubit index out of range: {q} for {op}"
                        )
                if op.condition.index not in valid_cbit_indices:
                    raise ValueError(
                        f"ConditionalOp.condition '{op.condition.name}' "
                        f"(index {op.condition.index}) is not allocated by this circuit."
                    )

    # ------------------------------------------------------------------
    # Classical-register management
    # ------------------------------------------------------------------

    def add_classical_register(self, name: str, size: int) -> ClassicalRegister:
        """Allocate a named classical register on this circuit.

        Returns the new ``ClassicalRegister`` so callers can index into it::

            syndrome = circuit.add_classical_register("syndrome", 2)
            circuit.measure(0, syndrome[0])

        Args:
            name: Register name (must be unique within this circuit).
            size: Number of bits (must be >= 1).

        Raises:
            ValueError: If *name* is already used by another register, or
                        if *size* is less than 1.
        """
        if size < 1:
            raise ValueError(
                f"Classical register size must be >= 1; got {size}."
            )
        for existing in self.classical_registers:
            if existing.name == name:
                raise ValueError(
                    f"Classical register '{name}' already exists on this circuit."
                )
        offset = sum(r.size for r in self.classical_registers)
        reg = ClassicalRegister(name=name, size=size, offset=offset)
        self.classical_registers.append(reg)
        return reg

    def measure(
        self,
        qubit: int,
        cbit: ClassicalBit,
        *,
        reset: bool = False,
        t: Optional[int] = None,
    ) -> "Circuit":
        """Append a Z-basis measurement of *qubit* writing the outcome into *cbit*.

        Scheduling follows the same greedy rule as ``add_gate``: when *t* is
        omitted the measurement is placed at ``max_t_on_qubit + 1``.

        Args:
            qubit: Qubit to measure (must be in range).
            cbit:  Destination classical bit (must be allocated via
                   ``add_classical_register``).
            reset: When ``True``, reset the qubit to |0⟩ after measurement.
            t:     Optional explicit time-step.  When provided, a ValueError is
                   raised if *qubit* is already occupied at that time-step.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If *qubit* is out of bounds, *cbit* is not allocated
                        by this circuit, or an explicit *t* collides with an
                        existing operation on *qubit*.
        """
        if not (0 <= qubit < self.n_qubits):
            raise ValueError(
                f"Qubit index {qubit} is out of bounds for a circuit with "
                f"{self.n_qubits} qubits."
            )

        if self.classical_registers:
            valid_cbit_indices = {
                reg.offset + i
                for reg in self.classical_registers
                for i in range(reg.size)
            }
            if cbit.index not in valid_cbit_indices:
                raise ValueError(
                    f"ClassicalBit at index {cbit.index} ('{cbit.name}') is not "
                    f"allocated by this circuit. Use add_classical_register() first."
                )

        if t is None:
            last_t = -1
            for op in self.operations:
                if qubit in _get_qubits(op) and op.t > last_t:
                    last_t = op.t
            t = last_t + 1
        else:
            # Explicit t: collision check mirrors add_gate().
            for existing in self.operations:
                if existing.t == t and qubit in set(_get_qubits(existing)):
                    existing_desc = (
                        existing.gate.name
                        if isinstance(existing, Operation)
                        else type(existing).__name__
                    )
                    raise ValueError(
                        f"Cannot place Measurement on qubit {qubit} at t={t}: "
                        f"qubit {qubit} already used by '{existing_desc}' at t={t}."
                    )

        self.operations.append(Measurement(qubit=qubit, cbit=cbit, t=t, reset=reset))
        return self

    def c_if(
        self,
        cbit: ClassicalBit,
        value: int = 1,
    ) -> "_ConditionalBuilder":
        """Return a builder that wraps the next gate in a ``ConditionalOp``.

        Usage::

            circuit.c_if(syndrome[0], value=1).x(target)
            circuit.c_if(syndrome[1]).cnot(0, 1)

        Args:
            cbit:  Classical bit to check.
            value: Required value (0 or 1) for the gate to fire.
        """
        return _ConditionalBuilder(self, cbit, value)


class _ConditionalBuilder:
    """Fluent helper returned by ``Circuit.c_if()``.

    Each method wraps the corresponding gate in a ``ConditionalOp`` and
    appends it to the parent circuit, returning the circuit for chaining::

        circuit.c_if(syndrome[0]).x(target)   # X fires only when bit=1
        circuit.c_if(anc, 0).z(0)             # Z fires only when bit=0

    Scheduling mirrors ``Circuit.add_gate``: the wrapped gate lands at
    ``max_t_across_target_qubits + 1`` when no explicit ``t`` is given.
    """

    def __init__(
        self,
        circuit: Circuit,
        condition: ClassicalBit,
        value: int = 1,
    ) -> None:
        self._circuit = circuit
        self._condition = condition
        self._value = value

    def _add(
        self,
        gate: gates.Gate,
        qubits: Qubits,
        t: Optional[int] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Circuit:
        """Build ``ConditionalOp(Operation(...))`` and append to circuit."""
        if isinstance(qubits, list):
            qubits = tuple(qubits)

        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"Gate '{gate.name}' acts on {gate.num_qubits} qubit(s), "
                f"but was applied to {len(qubits)}."
            )
        for q in qubits:
            if not 0 <= q < self._circuit.n_qubits:
                raise ValueError(
                    f"Qubit index {q} is out of bounds for a circuit with "
                    f"{self._circuit.n_qubits} qubits."
                )

        # Condition cbit must be allocated by the circuit.
        if self._circuit.classical_registers:
            valid_cbit_indices = {
                reg.offset + i
                for reg in self._circuit.classical_registers
                for i in range(reg.size)
            }
            if self._condition.index not in valid_cbit_indices:
                raise ValueError(
                    f"Condition ClassicalBit at index {self._condition.index} "
                    f"('{self._condition.name}') is not allocated by this circuit. "
                    f"Use add_classical_register() first."
                )

        if t is None:
            qubit_last_t: Dict[int, int] = {}
            for op in self._circuit.operations:
                for q in _get_qubits(op):
                    if op.t > qubit_last_t.get(q, -1):
                        qubit_last_t[q] = op.t
            latest_qubit_t = max((qubit_last_t.get(q, -1) for q in qubits), default=-1)
            # Ensure the conditional fires after the measurement that sets the condition bit.
            latest_cbit_write_t = -1
            for op in self._circuit.operations:
                if isinstance(op, Measurement) and op.cbit.index == self._condition.index:
                    if op.t > latest_cbit_write_t:
                        latest_cbit_write_t = op.t
            t = max(latest_qubit_t, latest_cbit_write_t) + 1
        else:
            # Explicit t: collision check mirrors add_gate().
            new_qubits = set(qubits)
            for existing in self._circuit.operations:
                if existing.t == t:
                    conflict = new_qubits & set(_get_qubits(existing))
                    if conflict:
                        existing_desc = (
                            existing.gate.name
                            if isinstance(existing, Operation)
                            else type(existing).__name__
                        )
                        raise ValueError(
                            f"Cannot place conditional '{gate.name}' on qubit(s) "
                            f"{sorted(qubits)} at t={t}: qubit(s) {sorted(conflict)} "
                            f"already used by '{existing_desc}' at t={t}."
                        )

        inner = Operation(gate=gate, qubits=qubits, t=t, params=params, meta=meta)
        cond = ConditionalOp(inner=inner, condition=self._condition, value=self._value)
        self._circuit.operations.append(cond)
        return self._circuit

    # Mirrors the single-qubit fluent methods on Circuit.
    def h(self, qubit: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.H, (qubit,), t)

    def x(self, qubit: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.X, (qubit,), t)

    def y(self, qubit: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.Y, (qubit,), t)

    def z(self, qubit: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.Z, (qubit,), t)

    def s(self, qubit: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.S, (qubit,), t)

    def cnot(self, control: int, target: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.CNOT, (control, target), t)

    def cx(self, control: int, target: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.CNOT, (control, target), t)

    def cz(self, q1: int, q2: int, t: Optional[int] = None) -> Circuit:
        return self._add(gates.CZ, (q1, q2), t)

