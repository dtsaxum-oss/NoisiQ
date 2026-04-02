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

    def add_gate(self, gate: gates.Gate, qubits: Qubits | List[int]):
        """
        Adds a gate operation to the circuit.

        Args:
            gate: The gate to add (e.g., gates.H).
            qubits: The qubit(s) the gate acts on.

        Raises:
            ValueError: If the number of qubits provided does not match the
                        gate's definition or if any qubit index is out of bounds.
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

        # For now, we assume each gate is one time step.
        # TODO: Add support for gates with different durations and parallel operations.
        t = len(self.operations)
        op = Operation(gate=gate, qubits=qubits, t=t)
        self.operations.append(op)

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