"""
Noise models for quantum circuit simulation.
Classes:
    PauliError: Pauli error channel (X, Y, Z errors)
"""

from .pauli_error import (
    PauliError,
    bit_flip_error,
    depolarizing_error,
    dephasing_error,
)

__all__ = [
    "PauliError",
    "depolarizing_error",
    "dephasing_error",
    "bit_flip_error",
]
