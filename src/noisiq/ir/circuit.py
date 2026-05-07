from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import gates


Qubits = Tuple[int, ...]

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
    operations: List[Operation] = field(default_factory=list)
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def add_gate(self, gate: gates.Gate, qubits: Qubits | List[int], t: Optional[int] = None):
        """
        Adds a gate operation to the circuit.

        Args:
            gate:   The gate to add (e.g., gates.H).
            qubits: The qubit(s) the gate acts on.
            t:      Optional explicit time step (layer index).  When omitted,
                    the gate is auto-scheduled into the earliest available slot
                    where none of its qubits conflict with an existing operation.
                    When provided, the gate is placed at exactly that t — but
                    a ValueError is raised if any qubit is already occupied at
                    that t by another gate.

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
            qubit_last_t: Dict[int, int] = {}
            for op in self.operations:
                for q in op.qubits:
                    if op.t > qubit_last_t.get(q, -1):
                        qubit_last_t[q] = op.t
            t = max((qubit_last_t.get(q, -1) for q in qubits), default=-1) + 1

        # Qubit-collision check: no two gates at the same t may share a qubit.
        new_qubits = set(qubits)
        for existing in self.operations:
            if existing.t == t:
                conflict = new_qubits & set(existing.qubits)
                if conflict:
                    raise ValueError(
                        f"Cannot place '{gate.name}' on qubit(s) {sorted(qubits)} at t={t}: "
                        f"qubit(s) {sorted(conflict)} already used by "
                        f"'{existing.gate.name}' on {list(existing.qubits)} at t={t}."
                    )

        op = Operation(gate=gate, qubits=qubits, t=t)
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

    def tgate(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        """T gate — named tgate to avoid collision with the t= layer parameter."""
        return self.add_gate(gates.T, (qubit,), t=t)

    def identity(self, qubit: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.I, (qubit,), t=t)

    def cnot(self, control: int, target: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CNOT, (control, target), t=t)

    def cx(self, control: int, target: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CNOT, (control, target), t=t)

    def cz(self, q1: int, q2: int, t: Optional[int] = None) -> "Circuit":
        return self.add_gate(gates.CZ, (q1, q2), t=t)

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

        for op in self.operations:
            if op.t < 0:
                raise ValueError(f"Operation time t must be >= 0: {op}")

            for q in op.qubits:
                if not (0 <= q < self.n_qubits):
                    raise ValueError(f"Qubit index out of range: {q} for {op}")