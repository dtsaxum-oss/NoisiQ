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


class CombinedChannel:
    """
    Multiple noise channels applied sequentially to the same gate.

    Wraps a list of heterogeneous channels (KrausChannel subclasses,
    PauliError, CorrelatedPauliError) into a single object so that
    to_noise_model() can compose decoherence + depolarizing + coherent ZZ
    + spectator errors while preserving the dict[op_idx → single channel]
    contract that backends and tests already expect.

    The channels are applied in list order (index 0 first). Because each
    inner channel may act on a different number of qubits (e.g. single-qubit
    Dephasing followed by two-qubit CoherentRotation), no Kraus Cartesian
    product is taken — the backend dispatches each inner channel individually
    using its own type-specific path.

    Args:
        channels: Non-empty list of channel objects. Accepted types are any
                  KrausChannel subclass, PauliError, or CorrelatedPauliError.

    Raises:
        ValueError: If channels is empty.

    Example:
        from noisiq.noise import AmplitudeDamping, CoherentRotation
        from noisiq.noise.kraus_channels import CombinedChannel

        combined = CombinedChannel([
            AmplitudeDamping(T1=300e-6, t=200e-9),
            CoherentRotation(axis='Z', epsilon=0.05),
        ])
    """

    def __init__(self, channels: list) -> None:
        if not channels:
            raise ValueError("CombinedChannel requires at least one channel.")
        self.channels = list(channels)

    def describe(self) -> dict:
        """Return a summary dict listing each inner channel."""
        return {
            "type": "CombinedChannel",
            "n_channels": len(self.channels),
            "channels": [
                c.describe() if hasattr(c, "describe") else repr(c)
                for c in self.channels
            ],
        }

    def __repr__(self) -> str:
        return f"CombinedChannel({len(self.channels)} channels: {[type(c).__name__ for c in self.channels]})"
