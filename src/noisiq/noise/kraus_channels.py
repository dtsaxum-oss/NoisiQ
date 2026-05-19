import numpy as np
from typing import List
from abc import ABC

class KrausChannel(ABC):
    """
    Abstract base class for non-Pauli noise channels represented by Kraus operators.
    """
    def __init__(self, operators: List[np.ndarray]):
        self.operators = operators
        self._validate_operators()

    def _validate_operators(self):
        """
        Validates that the Kraus operators satisfy trace preservation:
        sum_k K_k^\\dagger K_k = I
        """
        if not self.operators:
            raise ValueError("Kraus operators list cannot be empty.")
            
        shape = self.operators[0].shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Kraus operators must be square matrices.")
            
        identity = np.eye(shape[0], dtype=complex)
        sum_k = np.zeros_like(identity, dtype=complex)
        
        for k in self.operators:
            if k.shape != shape:
                raise ValueError("All Kraus operators must have the same shape.")
            sum_k += k.conj().T @ k
            
        if not np.allclose(sum_k, identity, atol=1e-8):
            raise ValueError("Kraus operators are not trace-preserving (sum K^\\dagger K != I).")
