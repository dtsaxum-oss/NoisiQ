from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


Qubits = Tuple[int, ...]

## Represents a single operation/gate in a quantum circuit, with its name, target qubits, time, and optional parameters and metadata.
@dataclass(frozen=True)
class Operation:
    """
    One gate/operation in a circuit.

    Example:
        Operation(name="cz", qubits=(1,2), t=3)
    """
    name: str
    qubits: Qubits
    t: int
    params: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    ## Removes parameters and meta from printing if they're None, to reduce clutter
    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"qubits={self.qubits!r}", f"t={self.t!r}"]
        if self.params is not None:
            parts.append(f"params={self.params!r}")
        if self.meta is not None:
            parts.append(f"meta={self.meta!r}")
        return f"Operation({', '.join(parts)})"

## A quantum circuit, consisting of a number of qubits and a list of operations based on the operation class above.
@dataclass(frozen=True)
class Circuit:
    """
    A quantum circuit described as:
      - number of qubits
      - a list/tuple of operations
    """
    n_qubits: int
    ops: Tuple[Operation, ...] = field(default_factory=tuple)
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    ## Removes name and metadata from printing if they're None, to reduce clutter
    def __repr__(self) -> str:
        parts = [f"n_qubits={self.n_qubits!r}", f"ops={self.ops!r}"]
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

        for op in self.ops:
            if op.t < 0:
                raise ValueError(f"Operation time t must be >= 0: {op}")

            for q in op.qubits:
                if not (0 <= q < self.n_qubits):
                    raise ValueError(f"Qubit index out of range: {q} for {op}")
