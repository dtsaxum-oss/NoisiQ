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
from .kraus_channels import KrausChannel
from .amplitude_damping import AmplitudeDamping
from .t2_dephasing import Dephasing
from .hardware_noise import (
    GHZResult,
    GateTimes,
    HardwareProfile,
    get_hardware,
    list_hardware,
    register_hardware,
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
    # Kraus channel base class and non-Pauli channels
    "KrausChannel",
    "AmplitudeDamping",
    "Dephasing",
    # Hardware profiles and registry
    "GHZResult",
    "GateTimes",
    "HardwareProfile",
    "get_hardware",
    "list_hardware",
    "register_hardware",
]
