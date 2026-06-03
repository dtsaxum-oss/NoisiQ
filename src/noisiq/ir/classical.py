"""
Classical IR primitives — measurement, classical bits, conditional operations.

These types complete the CircuitOp union:
    CircuitOp = Union[Operation, Measurement, ConditionalOp]

Every variant exposes `.t` (time-step) and a qubit accessor so that all loops
over circuit.operations work uniformly:
  - Operation   → .t, .qubits (tuple)
  - Measurement → .t, .qubit (int, singular), .qubits property returning (qubit,)
  - ConditionalOp → .t property (→ inner.t), .qubits property (→ inner.qubits)

Default-zero cbit convention (mirrors OpenQASM classical register initialisation):
  All simulation backends read unmeasured cbits as 0.  A ConditionalOp with
  value=0 therefore fires before its cbit has been written; value=1 does not.
  This is enforced by using `shot_cbits.get(cbit_index, 0)` at every evaluation
  site rather than a bare `.get()` whose None default would silently suppress
  all conditionals on unwritten bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class ClassicalBit:
    """A single classical bit holding one measurement outcome.

    Attributes:
        name:  Human-readable identifier, e.g. ``"syndrome_0"`` or ``"c[0]"``.
        index: Global index within the circuit's classical address space.
    """
    name: str
    index: int


@dataclass(frozen=True)
class ClassicalRegister:
    """A named contiguous block of classical bits.

    Attributes:
        name:   Register name, e.g. ``"syndrome"``.
        size:   Number of bits in the register.
        offset: Starting index in the circuit's global classical address space.
    """
    name: str
    size: int
    offset: int = 0

    def __getitem__(self, i: int) -> ClassicalBit:
        """Return the ClassicalBit at local index *i*.

        Raises:
            IndexError: If *i* is outside ``[0, size)``.
        """
        if not 0 <= i < self.size:
            raise IndexError(
                f"ClassicalRegister '{self.name}' has size {self.size}; "
                f"index {i} is out of range."
            )
        return ClassicalBit(name=f"{self.name}[{i}]", index=self.offset + i)


@dataclass(frozen=True)
class Measurement:
    """Mid-circuit Z-basis measurement of a single qubit.

    Collapses the qubit state and writes the outcome (0 or 1) into a
    classical bit.  Sits in ``circuit.operations`` at a specific time-step
    ``t``, just like an ``Operation``.

    Attributes:
        qubit: Target qubit to measure.
        cbit:  Classical bit that receives the outcome.
        t:     Scheduling time-step (mirrors ``Operation.t``).
        basis: Measurement basis; only ``"Z"`` is supported in M5.
        reset: When ``True``, reset the qubit to |0⟩ after measurement
               (measure-and-reset, MR in Stim).
    """
    qubit: int
    cbit: ClassicalBit
    t: int
    basis: str = "Z"
    reset: bool = False

    def __post_init__(self) -> None:
        if self.basis != "Z":
            raise ValueError(
                f"Measurement basis must be 'Z'; got {self.basis!r}. "
                "Non-Z measurements are not yet supported."
            )
        if not isinstance(self.qubit, int) or self.qubit < 0:
            raise ValueError(
                f"Measurement.qubit must be a non-negative integer; got {self.qubit!r}."
            )

    @property
    def qubits(self) -> Tuple[int, ...]:
        """Single-element tuple ``(self.qubit,)`` for uniform CircuitOp iteration."""
        return (self.qubit,)


@dataclass(frozen=True)
class ConditionalOp:
    """Wrap an Operation so it only fires when a classical bit equals a value.

    Used for correction gates in distillation / QEC protocols::

        circuit.c_if(syndrome[0], value=1).x(target)

    Attributes:
        inner:      The underlying ``Operation`` that conditionally executes.
        condition:  Which classical bit to check.
        value:      Required value (0 or 1) for ``inner`` to run.

    Properties:
        t:      Forwards to ``inner.t``.
        qubits: Forwards to ``inner.qubits``.
    """
    inner: Any          # Operation; typed Any to avoid circular import
    condition: ClassicalBit
    value: int = 1

    def __post_init__(self) -> None:
        if self.value not in (0, 1):
            raise ValueError(
                f"ConditionalOp.value must be 0 or 1; got {self.value!r}."
            )

    @property
    def t(self) -> int:
        """Time-step of the wrapped inner operation."""
        return self.inner.t  # type: ignore[union-attr]

    @property
    def qubits(self) -> Tuple[int, ...]:
        """Qubit indices of the wrapped inner operation."""
        return self.inner.qubits  # type: ignore[union-attr]
