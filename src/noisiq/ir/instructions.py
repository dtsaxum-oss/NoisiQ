"""Circuit instructions.

Every instruction is an immutable dataclass with a ``layer`` (time step) and a
``qubits`` tuple, so any code can iterate a circuit uniformly:

* :class:`Operation` — a unitary gate, optionally classically conditioned.
* :class:`Measure` — single-qubit measurement in the X, Y or Z basis.
* :class:`Reset` — reset a qubit to |0⟩.
* :class:`Delay` — the qubit idles for a given duration (seconds).
* :class:`Barrier` — scheduling fence; no physical effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

from .gates import Gate

__all__ = [
    "Condition",
    "Operation",
    "Measure",
    "Reset",
    "Delay",
    "Barrier",
    "Instruction",
    "ClassicalRegister",
]


@dataclass(frozen=True)
class Condition:
    """Classical condition: fire only if the listed bits encode ``value``.

    ``clbits[0]`` is the least-significant bit of ``value`` (OpenQASM convention).
    Bits that have never been written read as 0.

    Example:
        ``Condition((3,), 1)`` fires when clbit 3 is 1;
        ``Condition((0, 1), 0b10)`` fires when clbit 0 is 0 and clbit 1 is 1.
    """

    clbits: tuple[int, ...]
    value: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "clbits", tuple(int(c) for c in self.clbits))
        if not self.clbits:
            raise ValueError("A Condition needs at least one classical bit.")
        if len(set(self.clbits)) != len(self.clbits):
            raise ValueError("Condition clbits must be distinct.")
        if not 0 <= self.value < 2 ** len(self.clbits):
            raise ValueError(f"Condition value {self.value} does not fit in {len(self.clbits)} bit(s).")

    def is_met(self, bits: Sequence[int]) -> bool:
        """Evaluate against a full classical-bit vector ``bits``."""
        v = 0
        for i, c in enumerate(self.clbits):
            v |= (int(bits[c]) & 1) << i
        return v == self.value

    def __str__(self) -> str:
        if len(self.clbits) == 1:
            return f"c{self.clbits[0]}=={self.value}"
        return f"c{list(self.clbits)}=={self.value}"


@dataclass(frozen=True)
class Operation:
    """A gate applied to ``qubits`` at ``layer``, optionally conditioned."""

    gate: Gate
    qubits: tuple[int, ...]
    layer: int
    condition: Condition | None = None
    tag: str | None = None

    @property
    def name(self) -> str:
        return self.gate.name

    def __repr__(self) -> str:
        extra = f", if {self.condition}" if self.condition else ""
        tag = f", tag={self.tag!r}" if self.tag else ""
        return f"Operation({self.gate.display_label} {list(self.qubits)} @{self.layer}{extra}{tag})"


@dataclass(frozen=True)
class Measure:
    """Measure ``qubit`` in ``basis`` (``"X"``, ``"Y"`` or ``"Z"``) into ``clbit``.

    The post-measurement state is the measured eigenstate. With ``reset=True`` the
    qubit is then returned to |0⟩ (measure-and-reset).
    """

    qubit: int
    clbit: int
    layer: int
    basis: str = "Z"
    reset: bool = False

    def __post_init__(self) -> None:
        b = str(self.basis).upper()
        if b not in ("X", "Y", "Z"):
            raise ValueError(f"Measurement basis must be 'X', 'Y' or 'Z', got {self.basis!r}.")
        object.__setattr__(self, "basis", b)

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.qubit,)

    @property
    def name(self) -> str:
        return "MEASURE"

    def __repr__(self) -> str:
        r = "+reset" if self.reset else ""
        return f"Measure({self.basis}{r} q{self.qubit}->c{self.clbit} @{self.layer})"


@dataclass(frozen=True)
class Reset:
    """Reset ``qubit`` to |0⟩."""

    qubit: int
    layer: int

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.qubit,)

    @property
    def name(self) -> str:
        return "RESET"


@dataclass(frozen=True)
class Delay:
    """``qubit`` idles for ``duration`` seconds (picks up idle noise only)."""

    qubit: int
    duration: float
    layer: int

    def __post_init__(self) -> None:
        if not self.duration >= 0:
            raise ValueError(f"Delay duration must be >= 0, got {self.duration}.")

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.qubit,)

    @property
    def name(self) -> str:
        return "DELAY"


@dataclass(frozen=True)
class Barrier:
    """Scheduling fence: later instructions on these qubits start at ``layer`` or later."""

    qubits: tuple[int, ...]
    layer: int

    @property
    def name(self) -> str:
        return "BARRIER"


Instruction = Union[Operation, Measure, Reset, Delay, Barrier]


@dataclass(frozen=True)
class ClassicalRegister:
    """A named, contiguous slice of a circuit's classical bits.

    Indexing returns global clbit indices: ``reg[0]`` → ``offset``.
    """

    name: str
    offset: int
    size: int

    def __getitem__(self, i: int) -> int:
        if not -self.size <= i < self.size:
            raise IndexError(f"Register '{self.name}' has size {self.size}; index {i} out of range.")
        return self.offset + (i % self.size)

    def __iter__(self):
        return iter(range(self.offset, self.offset + self.size))

    def __len__(self) -> int:
        return self.size

    @property
    def clbits(self) -> tuple[int, ...]:
        return tuple(self)
