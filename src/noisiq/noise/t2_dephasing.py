import numpy as np
from typing import Optional
from .kraus_channels import KrausChannel

class Dephasing(KrausChannel):
    """
    T2 Dephasing Channel.
    Models phase randomization over time.
    """
    def __init__(self, t: float, T2: Optional[float] = None, T1: Optional[float] = None, Tphi: Optional[float] = None):
        if t < 0:
            raise ValueError("Time t must be non-negative.")
            
        if T2 is not None:
            if T2 <= 0:
                raise ValueError("T2 must be greater than 0.")
            rate = 1.0 / T2
        elif T1 is not None and Tphi is not None:
            if T1 <= 0 or Tphi <= 0:
                raise ValueError("T1 and Tphi must be greater than 0.")
            rate = 1.0 / (2 * T1) + 1.0 / Tphi
        else:
            raise ValueError("Must provide either T2 or both T1 and Tphi.")
            
        self.t = t
        
        # p is the probability of a phase error
        # The off-diagonal elements decay as exp(-t/T2)
        # For a dephasing channel: p = (1 - exp(-t/T2)) / 2
        p = 0.5 * (1.0 - np.exp(-t * rate))
        
        K0 = np.sqrt(1 - p) * np.array([[1, 0], [0, 1]], dtype=complex)
        K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
        
        super().__init__([K0, K1])
