"""
T2 dephasing (phase damping) channel for phase coherence decay.

Classes:
    Dephasing: Phase damping channel parameterized by gate time t and either
               T2 directly, or T1 + Tphi via the relation 1/T2 = 1/(2·T1) + 1/Tphi
"""

from __future__ import annotations

import numpy as np

from .kraus_channels import KrausChannel


class Dephasing(KrausChannel):
    """
    T2 dephasing channel modelling transverse coherence decay.

    Off-diagonal density matrix elements decay as exp(-t / T2), where
    the total dephasing rate satisfies 1/T2 = 1/(2·T1) + 1/Tphi.

    The phase damping channel is:

        ρ → K0 ρ K0†  +  K1 ρ K1†

    with Kraus operators:

        K0 = [[1,           0        ],
              [0,  sqrt(1 - λ)]]

        K1 = [[0,  0],
              [0,  sqrt(λ)]]

        where λ = 1 - exp(-2t / T2)

    Off-diagonal decay: ρ_01 → sqrt(1 - λ) · ρ_01 = exp(-t / T2) · ρ_01

    Args:
        t:    Gate duration in seconds. Must be >= 0.
        T2:   Total transverse coherence time in seconds. Must be > 0.
              Provide either T2, or both T1 and Tphi.
        T1:   Energy relaxation time in seconds. Must be > 0.
              Used with Tphi to compute T2 via 1/T2 = 1/(2·T1) + 1/Tphi.
        Tphi: Pure dephasing time in seconds. Must be > 0.

    Raises:
        ValueError: If t < 0, T2 <= 0, T1 <= 0, Tphi <= 0, or neither
                    T2 nor (T1, Tphi) are provided.
    """

    def __init__(
        self,
        t: float,
        T2: float | None = None,
        T1: float | None = None,
        Tphi: float | None = None,
    ) -> None:
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}")

        if T2 is not None:
            if T2 <= 0:
                raise ValueError(f"T2 must be > 0, got {T2}")
            rate = 1.0 / T2
            self.T2 = T2
            self.T1 = None
            self.Tphi = None
        elif T1 is not None and Tphi is not None:
            if T1 <= 0:
                raise ValueError(f"T1 must be > 0, got {T1}")
            if Tphi <= 0:
                raise ValueError(f"Tphi must be > 0, got {Tphi}")
            rate = 1.0 / (2.0 * T1) + 1.0 / Tphi
            self.T2 = None
            self.T1 = T1
            self.Tphi = Tphi
        else:
            raise ValueError("Must provide either T2 or both T1 and Tphi.")

        self.t = t
        self._lambda = 1.0 - np.exp(-2.0 * t * rate)
        lam = self._lambda
        K0 = np.array([[1.0, 0.0],
                        [0.0, np.sqrt(1.0 - lam)]], dtype=complex)
        K1 = np.array([[0.0, 0.0],
                        [0.0, np.sqrt(lam)]], dtype=complex)
        super().__init__([K0, K1])

    @property
    def lam(self) -> float:
        """Phase damping parameter λ = 1 - exp(-2t / T2)."""
        return float(self._lambda)

    def describe(self) -> dict:
        d: dict = {
            "channel": "dephasing",
            "t": self.t,
            "lambda": self._lambda,
        }
        if self.T2 is not None:
            d["T2"] = self.T2
        else:
            d["T1"] = self.T1
            d["Tphi"] = self.Tphi
        return d

    def __repr__(self) -> str:
        if self.T2 is not None:
            params = f"T2={self.T2}, t={self.t}"
        else:
            params = f"T1={self.T1}, Tphi={self.Tphi}, t={self.t}"
        return f"Dephasing({params}, lambda={self._lambda:.6f})"
