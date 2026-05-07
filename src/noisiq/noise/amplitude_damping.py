import numpy as np
from .kraus_channels import KrausChannel

class AmplitudeDamping(KrausChannel):
    """
    T1 Amplitude Damping Channel.
    Models a qubit decaying to the |0> state over time.
    """
    def __init__(self, T1: float, t: float):
        if T1 <= 0:
            raise ValueError("T1 must be greater than 0.")
        if t < 0:
            raise ValueError("Time t must be non-negative.")
            
        self.T1 = T1
        self.t = t
        
        gamma = 1.0 - np.exp(-t / T1)
        
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        
        super().__init__([K0, K1])
