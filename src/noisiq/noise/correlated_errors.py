"""
Correlated multi-qubit Pauli error channels.

A CorrelatedPauliError is a stochastic joint Pauli channel acting on two or
more qubits simultaneously. It models noise sources where a single physical
event corrupts multiple qubits at once:

  - ZZ crosstalk: always-on residual coupling between superconducting qubits
    causes joint ZZ phase errors during idle and gate time.
  - Spectator errors: a two-qubit gate on (q0, q1) causes a stray Z error on
    a neighboring spectator qubit q2, because the control pulse partially
    addresses q2 (ion-trap beam crosstalk) or dispersively couples to it
    (superconducting ZZ coupling).
  - Pauli-twirled multi-qubit coherent errors: CoherentRotation.to_pauli_error()
    produces a CorrelatedPauliError when the axis has length ≥ 2.

The difference from independent PauliErrors applied per qubit is that here a
single event hits multiple qubits *together* — the joint error probability is
primary, not a product of marginals. This explains the fidelity gap between
independent noise models (predict F ≈ 0.7) and hardware (F ≈ 0.5): independent
models undercount correlated events.

Classes:
    CorrelatedPauliError: A stochastic multi-qubit Pauli channel.

Maps to STIM's CORRELATED_ERROR directive for Clifford-backend routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CorrelatedPauliError:
    """
    A stochastic multi-qubit Pauli channel.

    Stores a mapping from Pauli strings (e.g. 'ZZ', 'IX', 'XY') to their
    joint probability. Identity probability is 1 − Σp; missing strings default
    to probability 0. All strings must act on the same number of qubits.

    Args:
        probs: Dict mapping Pauli string → probability. Keys must be
               strings of I/X/Y/Z characters, all the same length (≥ 2).
               Values must be non-negative and sum to ≤ 1.

    Raises:
        ValueError: If probs is empty, strings have inconsistent lengths,
                    any string has length < 2, any character is outside IXYZ,
                    any probability is negative, or total probability exceeds 1.

    Example:
        # ZZ crosstalk on a 2q gate at 0.5% rate
        err = CorrelatedPauliError({'ZZ': 0.005})

        # Mixed correlated channel from a Pauli-Lindblad model
        err = CorrelatedPauliError({
            'IX': 0.001, 'XI': 0.001, 'XX': 0.0005,
            'IZ': 0.002, 'ZI': 0.002, 'ZZ': 0.001,
        })
    """

    probs: Dict[str, float] = field(default_factory=dict)
    num_qubits: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.probs:
            raise ValueError("CorrelatedPauliError: probs dict cannot be empty.")

        lengths = {len(s) for s in self.probs}
        if len(lengths) != 1:
            raise ValueError(
                f"All Pauli strings must have the same length; got lengths {lengths}."
            )
        n = lengths.pop()
        if n < 2:
            raise ValueError(
                f"CorrelatedPauliError requires strings of length ≥ 2 (got {n}). "
                f"Use PauliError for single-qubit channels."
            )
        object.__setattr__(self, 'num_qubits', n)

        for s, p in self.probs.items():
            if any(ch not in 'IXYZ' for ch in s):
                raise ValueError(
                    f"Invalid Pauli string {s!r}: every character must be one of IXYZ."
                )
            if p < 0:
                raise ValueError(
                    f"Probability for {s!r} must be non-negative, got {p}."
                )

        total = sum(self.probs.values())
        if total > 1.0 + 1e-12:
            raise ValueError(
                f"Sum of probabilities must be ≤ 1, got {total:.6f}."
            )

    def sample(self, rng: np.random.Generator) -> str:
        """Sample one Pauli string outcome for this shot.

        Returns the identity string 'II...I' with probability
        (1 − Σ listed probabilities).

        Args:
            rng: NumPy random generator.

        Returns:
            A Pauli string of length num_qubits, e.g. 'ZZ', 'IX', or 'II'.
        """
        strings = list(self.probs.keys())
        weights = list(self.probs.values())
        identity = 'I' * self.num_qubits
        p_identity = max(0.0, 1.0 - sum(weights))
        strings.append(identity)
        weights.append(p_identity)
        idx = int(rng.choice(len(strings), p=np.array(weights) / sum(weights)))
        return strings[idx]

    def to_stim_correlated_error_directives(
        self, qubits: List[int]
    ) -> List[Tuple[str, float, List[Tuple[str, int]]]]:
        """Render this channel as STIM CORRELATED_ERROR instruction data.

        Each non-identity Pauli in probs becomes one directive entry of the form
        ('CORRELATED_ERROR', probability, [(pauli_char, qubit_index), ...]).

        Args:
            qubits: Physical qubit indices the channel acts on, in the same
                    order as the Pauli string characters.

        Returns:
            List of (instruction_name, probability, targets) tuples.
        """
        directives = []
        for pauli_string, prob in self.probs.items():
            if prob == 0.0:
                continue
            targets = [
                (p, q)
                for p, q in zip(pauli_string, qubits)
                if p != 'I'
            ]
            if targets:
                directives.append(('CORRELATED_ERROR', prob, targets))
        return directives

    def __repr__(self) -> str:
        terms = ", ".join(f"{s!r}: {p:.4g}" for s, p in self.probs.items())
        return f"CorrelatedPauliError({{{terms}}})"
