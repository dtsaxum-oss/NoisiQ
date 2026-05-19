"""
Classes:
    PauliError: Configurable Pauli channel with X, Y, Z error probabilities
    
Functions:
    depolarizing_error: Create uniform Pauli noise
    dephasing_error: Create pure phase-flip noise
    bit_flip_error: Create pure bit-flip noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..ir import gates


@dataclass
class PauliError:
    """
    A Pauli error channel with probabilities for X, Y, Z errors.
    
    Represents a quantum channel of the form:
        ρ → (1-p_X-p_Y-p_Z)ρ + p_X XρX† + p_Y YρY† + p_Z ZρZ†
    
    Where:
        - p_X: Probability of bit-flip (X) error
        - p_Y: Probability of bit-phase-flip (Y) error  
        - p_Z: Probability of phase-flip (Z) error
        - Identity occurs with probability 1 - (p_X + p_Y + p_Z)
    
    The probabilities must satisfy:
        - 0 ≤ p_X, p_Y, p_Z ≤ 1  (each probability is valid)
        - p_X + p_Y + p_Z ≤ 1     (total error probability is physical)
    
    Attributes:
        p_x: Probability of X error (0 ≤ p_x ≤ 1)
        p_y: Probability of Y error (0 ≤ p_y ≤ 1)
        p_z: Probability of Z error (0 ≤ p_z ≤ 1)
        
    Raises:
        ValueError: If probabilities are negative or sum to > 1
    """
    
    p_x: float
    p_y: float
    p_z: float
    
    def __post_init__(self):
        """
        Validate that probabilities are physical.

        Raises:
            ValueError: If any probability is negative
            ValueError: If sum of probabilities exceeds 1
        """
        if self.p_x < 0:
            raise ValueError(f"p_x must be non-negative, got {self.p_x}")
        if self.p_y < 0:
            raise ValueError(f"p_y must be non-negative, got {self.p_y}")
        if self.p_z < 0:
            raise ValueError(f"p_z must be non-negative, got {self.p_z}")
    
        if self.p_x > 1:
            raise ValueError(f"p_x must be ≤ 1, got {self.p_x}")
        if self.p_y > 1:
            raise ValueError(f"p_y must be ≤ 1, got {self.p_y}")
        if self.p_z > 1:
            raise ValueError(f"p_z must be ≤ 1, got {self.p_z}")
        
        total = self.p_x + self.p_y + self.p_z
        if total > 1:
            raise ValueError(
                f"Sum of error probabilities must be ≤ 1, got {total:.6f} "
                f"(p_x={self.p_x}, p_y={self.p_y}, p_z={self.p_z})"
            )
    
    def sample(self, rng: np.random.Generator) -> str:
        """ 
        Args:
            rng: NumPy random number generator
                 Use np.random.default_rng(seed=...)
                
        Returns:
            'I', 'X', 'Y', 'Z' based on the configured probabilities
        """
        r = rng.random()
        
        # Use cumulative probabilities for stable sampling, avoids floating-point comparison issues
        if r < self.p_x:
            return 'X'
        elif r < self.p_x + self.p_y:
            return 'Y'
        elif r < self.p_x + self.p_y + self.p_z:
            return 'Z'
        else:
            return 'I'
    
    def get_operator(self, pauli_character: str) -> gates.Gate:
        """
        Provides the unitary matrix to be applied when error occurs.
        
        Args:
            pauli_character: 'I', 'X', 'Y', 'Z'
            
        Returns:
            Corresponding Gate object from gates module
            
        Raises:
            ValueError: If pauli_character is not 'I', 'X', 'Y', 'Z'
        """
        # Map string to gate object using existing gate definitions
        pauli_map = {
            'I': gates.I,
            'X': gates.X,
            'Y': gates.Y,
            'Z': gates.Z,
        }
        
        if pauli_character not in pauli_map:
            raise ValueError(
                f"Invalid Pauli character '{pauli_character}'. "
                f"Must be one of {list(pauli_map.keys())}"
            )
        
        return pauli_map[pauli_character]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'p_x': self.p_x,
            'p_y': self.p_y,
            'p_z': self.p_z,
        }
    
    def __repr__(self) -> str:
        """
        Concise representation showing non-zero probabilities.
        """
        # Collect non-zero terms for cleaner output
        terms = []
        if self.p_x != 0:
            terms.append(f"p_x={self.p_x}")
        if self.p_y != 0:
            terms.append(f"p_y={self.p_y}")
        if self.p_z != 0:
            terms.append(f"p_z={self.p_z}")
        
        if not terms:
            terms.append("p_x=0.0, p_y=0.0, p_z=0.0")  # Explicitly show if all are zero
        
        return f"PauliError({', '.join(terms)})"

def depolarizing_error(p: float) -> PauliError:
    """
    Create a depolarizing error channel.
    
    Args:
        p: Total depolarizing probability (0 ≤ p ≤ 1)
           p=0 means no error, p=1 means complete depolarization
        
    Returns:
        PauliError with equal X, Y, Z probabilities of p/3 each
        
    Raises:
        ValueError: If p is not in the range [0, 1]
    """
    if not 0 <= p <= 1:
        raise ValueError(
            f"Depolarizing probability must be in [0, 1], got {p}"
        )

    p_each = p / 3.0
    return PauliError(p_x=p_each, p_y=p_each, p_z=p_each)


def dephasing_error(p: float) -> PauliError:
    """
    Create a pure dephasing (Z) error channel.

    Args:
        p: Phase-flip probability (0 ≤ p ≤ 1)
           p=0 means no dephasing, p=1 means deterministic phase flip
        
    Returns:
        PauliError with only Z errors (p_x=0, p_y=0, p_z=p)
        
    Raises:
        ValueError: If p is not in the range [0, 1]
    """
    if not 0 <= p <= 1:
        raise ValueError(
            f"Dephasing probability must be in [0, 1], got {p}"
        )
    
    return PauliError(p_x=0.0, p_y=0.0, p_z=p)


def bit_flip_error(p: float) -> PauliError:
    """
    Args:
        p: Bit-flip probability (0 ≤ p ≤ 1)
           p=0 means no bit flips, p=1 means deterministic bit flip
        
    Returns:
        PauliError with only X errors (p_x=p, p_y=0, p_z=0)
        
    Raises:
        ValueError: If p is not in the range [0, 1]
    """
    if not 0 <= p <= 1:
        raise ValueError(
            f"Bit-flip probability must be in [0, 1], got {p}"
        )
    
    return PauliError(p_x=p, p_y=0.0, p_z=0.0)
