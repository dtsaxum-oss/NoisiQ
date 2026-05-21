"""
CombinedChannel: stacks a Kraus (T1/T2) channel with an optional Pauli error layer.

In real hardware both noise sources are always present simultaneously:
  1. T1/T2 decoherence accumulates during the gate's physical duration.
  2. Gate imperfections (calibration error, crosstalk) add Pauli-type noise on top.

This channel applies them in sequence per qubit per gate and is used with
TrajectoryBackend.  Pass it wherever a KrausChannel is accepted.

Classes:
    CombinedChannel: dataclass wrapping KrausChannel + optional PauliChannel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .kraus_channels import KrausChannel
from .pauli_channels import PauliChannel
from .pauli_error import PauliError


@dataclass
class CombinedChannel:
    """
    Stacks a T1/T2 Kraus channel with an optional Pauli gate-error layer.

    Applied per qubit per gate inside TrajectoryBackend:
        1. Sample and apply one Kraus operator (T1/T2 decoherence).
        2. Sample and apply one Pauli operator (gate imperfection) — if set.

    Because the two error sources are independent their effects compound:
    combined fidelity loss > either channel alone, matching real hardware
    where both are always active.

    Args:
        kraus_channel: KrausChannel for decoherence (AmplitudeDamping or T2Dephasing).
        pauli_channel: Optional PauliChannel for gate imperfection noise.
                       Defaults to None (Kraus-only, no Pauli layer).

    Example:
        noise = CombinedChannel(
            kraus_channel=AmplitudeDamping(T1=50e-6, t=40e-9),
            pauli_channel=DepolarizingChannel(p=0.001),
        )
        result = TrajectoryBackend().run(circuit, noise_model=noise, n_shots=1000)

    Note — TrajectoryBackend extension (Week 7):
        TrajectoryBackend.run() must be updated to check
        isinstance(noise_model, CombinedChannel) and call both
        _apply_kraus_to_qubit() and _apply_pauli_to_qubit() per gate.
        See trajectory_backend.py for the injection point.
    """

    kraus_channel: KrausChannel
    pauli_channel: Optional[PauliChannel] = field(default=None)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the Kraus channel operators (Σ K†K = I).

        The Pauli channel validates its probability bounds on construction,
        so no separate call is needed here.

        Raises:
            ValueError: If kraus_channel fails trace-preservation check.
        """
        self.kraus_channel.validate()

    # ------------------------------------------------------------------
    # Pauli layer access
    # ------------------------------------------------------------------

    def pauli_error(self) -> Optional[PauliError]:
        """Return the underlying PauliError for the Pauli layer, or None.

        Returns:
            PauliError with p_x, p_y, p_z set, or None if no Pauli layer.
        """
        if self.pauli_channel is None:
            return None
        return self.pauli_channel.to_pauli_error()

    def sample_pauli(self, rng: np.random.Generator) -> Optional[str]:
        """Sample one Pauli character from the gate-error layer.

        Args:
            rng: NumPy random generator.

        Returns:
            One of 'I', 'X', 'Y', 'Z' if a Pauli layer is set; None otherwise.
        """
        err = self.pauli_error()
        if err is None:
            return None
        return err.sample(rng)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return a dict summary of both noise layers.

        Returns:
            Dict with key 'kraus' always present; key 'pauli' present only
            when a Pauli layer is attached.
        """
        d: dict = {"kraus": self.kraus_channel.describe()}
        if self.pauli_channel is not None:
            d["pauli"] = self.pauli_channel.describe()
        return d

    def __repr__(self) -> str:
        pauli_str = repr(self.pauli_channel) if self.pauli_channel else "None"
        return (
            f"CombinedChannel(\n"
            f"  kraus={self.kraus_channel!r},\n"
            f"  pauli={pauli_str}\n"
            f")"
        )
