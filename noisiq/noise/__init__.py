"""
Noise models for quantum circuit simulation.
"""

from .pauli_error import (
    PauliError,
    bit_flip_error,
    depolarizing_error,
    dephasing_error,
)
from .pauli_channels import (
    BitFlipChannel,
    DephaseChannel,
    DepolarizingChannel,
    PauliChannel,
    PhaseFlipChannel,
)

__all__ = [
    # Low-level Pauli error dataclass + factory functions
    "PauliError",
    "depolarizing_error",
    "dephasing_error",
    "bit_flip_error",
    # Named single-parameter channel classes
    "PauliChannel",
    "DepolarizingChannel",
    "DephaseChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
]
