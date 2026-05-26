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
from .kraus_channels import KrausChannel, CombinedChannel
from .amplitude_damping import AmplitudeDamping
from .t2_dephasing import Dephasing
from .coherent_errors import CoherentRotation
from .correlated_errors import CorrelatedPauliError
from .hardware_noise import (
    GHZResult,
    GateTimes,
    HardwareProfile,
    get_hardware,
    list_hardware,
    register_hardware,
)
from .idle_fill import fill_idle_with_identities, idle_kraus, idle_pauli_twirl

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
    "CombinedChannel",
    "AmplitudeDamping",
    "Dephasing",
    # Coherent and correlated error channels
    "CoherentRotation",
    "CorrelatedPauliError",
    # Hardware profiles and registry
    "GHZResult",
    "GateTimes",
    "HardwareProfile",
    "get_hardware",
    "list_hardware",
    "register_hardware",
    # Idle-slot filler and standalone idle channel builders
    "fill_idle_with_identities",
    "idle_kraus",
    "idle_pauli_twirl",
]
