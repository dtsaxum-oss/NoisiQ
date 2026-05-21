"""
Abstract base class for quantum channels described by Kraus operators.

Classes:
    KrausChannel: Base class for any CPTP channel defined by Kraus operators
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class KrausChannel(ABC):
    """
    Base class for completely positive trace-preserving (CPTP) channels.

    Subclasses compute Kraus operators K_k and pass them to super().__init__().
    The base class stores them, validates trace preservation automatically, and
    provides a public validate() hook for explicit re-checking.

    The Kraus form guarantees complete positivity automatically; the only
    constraint is the trace-preservation sum rule:

        Σ_k  K_k† K_k  =  I
    """

    def __init__(self, operators: list[np.ndarray]) -> None:
        self.operators = operators
        self._validate_operators()

    @abstractmethod
    def describe(self) -> dict:
        """Return a dict summary of the channel type and parameters."""

    def _validate_operators(self) -> None:
        if not self.operators:
            raise ValueError("Kraus operators list cannot be empty.")
        shape = self.operators[0].shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Kraus operators must be square matrices.")
        for K in self.operators:
            if K.shape != shape:
                raise ValueError("All Kraus operators must have the same shape.")
        total = sum(K.conj().T @ K for K in self.operators)
        if not np.allclose(total, np.eye(shape[0]), atol=1e-10):
            raise ValueError(
                f"{type(self).__name__}: Kraus operators violate trace "
                f"preservation (Σ K†K ≠ I). Got:\n{total}"
            )

    def validate(self) -> None:
        """Explicitly re-validate trace preservation. Useful after modifying operators."""
        self._validate_operators()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()})"
