"""
Named single-parameter Pauli noise channels.

Each class wraps PauliError with a fixed error structure and a single
probability parameter p, providing a clean interface for common channel types.

Classes:
    PauliChannel:       Abstract base for all single-parameter Pauli channels
    DepolarizingChannel: Equal X/Y/Z error — models generic gate noise
    DephaseChannel:     Z-only error — models pure phase randomization
    BitFlipChannel:     X-only error — models classical bit-flip noise
    PhaseFlipChannel:   Z-only error — synonym for DephaseChannel, distinct name
                        for pedagogical clarity
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .pauli_error import PauliError, bit_flip_error, depolarizing_error, dephasing_error


class PauliChannel(ABC):
    """
    Base class for single-parameter Pauli noise channels.

    All subclasses accept a single probability p in [0, 1] and expose
    a consistent interface for converting to the underlying PauliError
    and describing the channel.
    """

    def __init__(self, p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"{type(self).__name__} probability must be in [0, 1], got {p}"
            )
        self.p = p

    @abstractmethod
    def to_pauli_error(self) -> PauliError:
        """Return the underlying PauliError representation of this channel."""

    @abstractmethod
    def describe(self) -> dict:
        """Return a dict summary of the channel type and parameters."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class DepolarizingChannel(PauliChannel):
    """
    Depolarizing channel: equal probability of X, Y, or Z error.

    Models generic unstructured gate noise. Each error type occurs with
    probability p/3, so total error probability is p.

        ρ → (1-p)ρ + (p/3)(XρX† + YρY† + ZρZ†)
    """

    def to_pauli_error(self) -> PauliError:
        return depolarizing_error(self.p)

    def describe(self) -> dict:
        return {
            "channel": "depolarizing",
            "p": self.p,
            "p_x": self.p / 3,
            "p_y": self.p / 3,
            "p_z": self.p / 3,
        }


class DephaseChannel(PauliChannel):
    """
    Dephasing channel: Z-only errors model pure phase randomization.

    Destroys off-diagonal coherence in the Z basis without causing bit flips.
    Relevant for T2 dephasing noise in most qubit modalities.

        ρ → (1-p)ρ + p ZρZ†
    """

    def to_pauli_error(self) -> PauliError:
        return dephasing_error(self.p)

    def describe(self) -> dict:
        return {
            "channel": "dephasing",
            "p": self.p,
            "p_x": 0.0,
            "p_y": 0.0,
            "p_z": self.p,
        }


class BitFlipChannel(PauliChannel):
    """
    Bit-flip channel: X-only errors flip the qubit state.

    Models classical bit-flip noise or readout errors. Does not affect phase.

        ρ → (1-p)ρ + p XρX†
    """

    def to_pauli_error(self) -> PauliError:
        return bit_flip_error(self.p)

    def describe(self) -> dict:
        return {
            "channel": "bit_flip",
            "p": self.p,
            "p_x": self.p,
            "p_y": 0.0,
            "p_z": 0.0,
        }


class PhaseFlipChannel(PauliChannel):
    """
    Phase-flip channel: Z-only errors flip the qubit phase.

    Physically identical to DephaseChannel; provided as a separate class
    because the two names appear in different pedagogical contexts.
    "Phase flip" is common in error-correction literature; "dephasing" in
    open-systems literature.

        ρ → (1-p)ρ + p ZρZ†
    """

    def to_pauli_error(self) -> PauliError:
        return dephasing_error(self.p)

    def describe(self) -> dict:
        return {
            "channel": "phase_flip",
            "p": self.p,
            "p_x": 0.0,
            "p_y": 0.0,
            "p_z": self.p,
        }
