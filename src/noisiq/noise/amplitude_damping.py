"""
T1 amplitude damping channel for energy relaxation noise.

Classes:
    AmplitudeDamping: T1 channel via Kraus operators parameterized by T1 and gate time t
"""

from __future__ import annotations

import numpy as np

from .kraus_channels import KrausChannel


class AmplitudeDamping(KrausChannel):
    """
    T1 amplitude damping channel modelling spontaneous emission.

    A qubit in |1⟩ decays to |0⟩ at rate 1/T1. After gate time t the
    decay probability is γ = 1 - exp(-t / T1). The channel acts as:

        ρ → K0 ρ K0†  +  K1 ρ K1†

    with Kraus operators:

        K0 = [[1,          0        ],     (no decay)
              [0,  sqrt(1 - γ)]]

        K1 = [[0,  sqrt(γ)],               (decay |1⟩ → |0⟩)
              [0,     0   ]]

    Args:
        T1: Energy relaxation time in seconds. Must be > 0.
        t:  Gate duration (exposure time) in seconds. Must be >= 0.

    Raises:
        ValueError: If T1 <= 0 or t < 0.
    """

    def __init__(self, T1: float, t: float) -> None:
        if T1 <= 0:
            raise ValueError(f"T1 must be > 0, got {T1}")
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}")
        self.T1 = T1
        self.t = t
        self._gamma = 1.0 - np.exp(-t / T1)
        K0 = np.array([[1.0, 0.0],
                        [0.0, np.sqrt(1.0 - self._gamma)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(self._gamma)],
                        [0.0, 0.0]], dtype=complex)
        super().__init__([K0, K1])

    @property
    def gamma(self) -> float:
        """Decay probability γ = 1 - exp(-t / T1)."""
        return float(self._gamma)

    def describe(self) -> dict:
        return {
            "channel": "amplitude_damping",
            "T1": self.T1,
            "t": self.t,
            "gamma": self._gamma,
        }

    def __repr__(self) -> str:
        return (
            f"AmplitudeDamping(T1={self.T1}, t={self.t}, "
            f"gamma={self._gamma:.6f})"
        )
