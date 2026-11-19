"""Noise channels, rule-based noise models and hardware profiles.

Run ``noisiq.explain("<name>")`` for the math behind any channel.
"""

from .channels import (
    AmplitudeDamping,
    Channel,
    CoherentRotation,
    ComposedChannel,
    KrausChannel,
    PauliChannel,
    PhaseDamping,
    ReadoutError,
    ThermalRelaxation,
    UnitaryChannel,
    bit_flip,
    bit_phase_flip,
    depolarizing,
    pauli_channel,
    pauli_matrix,
    phase_flip,
)
from .hardware import GHZResult, HardwareProfile, get_hardware, list_hardware, register_hardware
from .model import Durations, Fault, NoiseModel

__all__ = [
    "Channel",
    "PauliChannel",
    "KrausChannel",
    "UnitaryChannel",
    "AmplitudeDamping",
    "PhaseDamping",
    "ThermalRelaxation",
    "CoherentRotation",
    "ComposedChannel",
    "ReadoutError",
    "pauli_channel",
    "depolarizing",
    "bit_flip",
    "phase_flip",
    "bit_phase_flip",
    "pauli_matrix",
    "NoiseModel",
    "Durations",
    "Fault",
    "HardwareProfile",
    "GHZResult",
    "get_hardware",
    "list_hardware",
    "register_hardware",
]
