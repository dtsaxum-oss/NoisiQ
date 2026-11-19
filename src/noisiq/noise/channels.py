"""Quantum noise channels.

Every channel is a completely-positive trace-preserving (CPTP) map on ``num_qubits``
qubits and exposes the same interface, so backends never need ``isinstance`` ladders:

* :meth:`Channel.kraus` — Kraus operators ``{K_k}`` with ``Σ K_k† K_k = I``.
* :meth:`Channel.pauli_twirl` — the exact Pauli-twirled channel (a :class:`PauliChannel`),
  used to run non-Pauli noise on the fast stabilizer backend.
* :attr:`Channel.is_pauli` — True for Pauli channels (natively stabilizer-simulable).
* :attr:`Channel.error_probability` — probability that the twirled channel applies a
  non-identity Pauli; used for visual summaries.

The mathematics of each channel is documented in :mod:`noisiq.catalog` and printed by
``noisiq.explain("<channel name>")``.

Multi-qubit Kraus matrices are big-endian in the order of the qubits they are
applied to (same convention as gates).
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property

import numpy as np

__all__ = [
    "Channel",
    "PauliChannel",
    "KrausChannel",
    "UnitaryChannel",
    "AmplitudeDamping",
    "PhaseDamping",
    "ThermalRelaxation",
    "CoherentRotation",
    "ComposedChannel",
    "ReadoutError",
    "pauli_channel",
    "depolarizing",
    "bit_flip",
    "phase_flip",
    "bit_phase_flip",
    "pauli_matrix",
]

_PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(label: str) -> np.ndarray:
    """Matrix of a Pauli string such as ``"XZ"`` (big-endian: first char = first qubit)."""
    if not label or any(ch not in _PAULI_1Q for ch in label):
        raise ValueError(f"Invalid Pauli string {label!r}; use characters I, X, Y, Z.")
    m = np.array([[1.0 + 0j]])
    for ch in label:
        m = np.kron(m, _PAULI_1Q[ch])
    return m


def _all_paulis(n: int) -> list[str]:
    return ["".join(t) for t in itertools.product("IXYZ", repeat=n)]


def _check_prob(p: float, name: str = "p") -> float:
    p = float(p)
    if not (0.0 <= p <= 1.0) or np.isnan(p):
        raise ValueError(f"{name} must be in [0, 1], got {p}.")
    return p


class Channel(ABC):
    """Abstract base class for all noise channels."""

    #: key into :mod:`noisiq.catalog` with the math for this channel
    catalog_key: str = "kraus"

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits the channel acts on."""

    @abstractmethod
    def kraus(self) -> list[np.ndarray]:
        """Kraus operators of the channel."""

    @property
    def is_pauli(self) -> bool:
        """True if this is a stochastic Pauli channel."""
        return False

    @property
    def is_unitary(self) -> bool:
        """True if the channel is a single unitary (a coherent error)."""
        return len(self.kraus()) == 1

    def pauli_twirl(self) -> PauliChannel:
        """Return the exact Pauli twirl of this channel.

        ``p_P = Σ_k |Tr(P K_k)|² / d²`` — the diagonal of the channel's χ-matrix in
        the Pauli basis. Twirling preserves the process fidelity ``p_I``.
        """
        n = self.num_qubits
        d = 2**n
        ks = self.kraus()
        probs = {}
        for label in _all_paulis(n):
            P = pauli_matrix(label)
            probs[label] = float(sum(abs(np.trace(P @ K)) ** 2 for K in ks) / d**2)
        probs = {k: v for k, v in probs.items() if set(k) != {"I"} and v > 1e-15}
        return PauliChannel(probs, num_qubits=n, name=f"twirl({self.name})")

    @property
    def error_probability(self) -> float:
        """Probability of a non-identity Pauli after twirling (``1 - process fidelity``)."""
        return self.pauli_twirl().error_probability

    @property
    def name(self) -> str:
        return type(self).__name__

    def superoperator_apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply the channel to a ``num_qubits``-qubit density matrix."""
        return sum(K @ rho @ K.conj().T for K in self.kraus())

    def describe(self) -> dict:
        """Dictionary summary (type, parameters, twirled error probability)."""
        return {
            "channel": self.name,
            "num_qubits": self.num_qubits,
            "error_probability": self.error_probability,
            **self._params(),
        }

    def _params(self) -> dict:
        return {}

    def then(self, other: Channel) -> ComposedChannel:
        """Apply ``self`` then ``other`` (same number of qubits)."""
        return ComposedChannel([self, other])

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={_fmt(v)}" for k, v in self._params().items())
        return f"{self.name}({params})"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return repr(v)


def _validate_kraus(ops: Sequence[np.ndarray], atol: float = 1e-8) -> list[np.ndarray]:
    ops = [np.array(K, dtype=complex) for K in ops]
    if not ops:
        raise ValueError("At least one Kraus operator is required.")
    d = ops[0].shape[0]
    for K in ops:
        if K.shape != (d, d):
            raise ValueError("Kraus operators must all be square and the same size.")
    n = int(round(np.log2(d)))
    if 2**n != d or n < 1:
        raise ValueError(f"Kraus operator dimension {d} is not 2**n.")
    total = sum(K.conj().T @ K for K in ops)
    if not np.allclose(total, np.eye(d), atol=atol):
        raise ValueError("Kraus operators are not trace preserving (Σ K†K ≠ I).")
    return ops


# =============================================================================
# Pauli channels
# =============================================================================


class PauliChannel(Channel):
    """Stochastic Pauli channel ``ρ → p_I ρ + Σ_P p_P P ρ P``.

    Args:
        probs: Mapping from Pauli string (length ``num_qubits``, characters IXYZ)
            to probability. The identity gets the remaining probability.
        num_qubits: Inferred from the keys when omitted.
        name: Display name.

    Example:
        >>> PauliChannel({"X": 0.01, "Z": 0.02})          # 1-qubit
        >>> PauliChannel({"ZZ": 0.005, "XX": 0.001})      # correlated 2-qubit
    """

    catalog_key = "pauli"

    def __init__(self, probs: Mapping[str, float], num_qubits: int | None = None, name: str | None = None):
        lengths = {len(k) for k in probs}
        if num_qubits is None:
            if len(lengths) != 1:
                raise ValueError("Cannot infer num_qubits: give Pauli strings of equal length or pass num_qubits.")
            num_qubits = lengths.pop()
        elif lengths and lengths != {num_qubits}:
            raise ValueError(f"All Pauli strings must have length {num_qubits}.")
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1.")
        clean: dict[str, float] = {}
        for label, prob in probs.items():
            label = label.upper()
            pauli_matrix(label)  # validates characters
            prob = _check_prob(prob, f"probability of {label}")
            if set(label) == {"I"} or prob == 0.0:
                continue
            clean[label] = clean.get(label, 0.0) + prob
        total = sum(clean.values())
        if total > 1 + 1e-12:
            raise ValueError(f"Pauli probabilities sum to {total:.6g} > 1.")
        self._n = int(num_qubits)
        self._probs = clean
        self._name = name or ("PauliChannel" if self._n == 1 else f"PauliChannel{self._n}")

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def name(self) -> str:
        return self._name

    @property
    def probabilities(self) -> dict[str, float]:
        """Non-identity Pauli probabilities."""
        return dict(self._probs)

    @property
    def p_identity(self) -> float:
        return max(0.0, 1.0 - sum(self._probs.values()))

    @property
    def is_pauli(self) -> bool:
        return True

    @property
    def is_unitary(self) -> bool:
        return not self._probs or (len(self._probs) == 1 and self.p_identity < 1e-15)

    @property
    def error_probability(self) -> float:
        return float(sum(self._probs.values()))

    def pauli_twirl(self) -> PauliChannel:
        return self

    def kraus(self) -> list[np.ndarray]:
        ops = []
        if self.p_identity > 0:
            ops.append(np.sqrt(self.p_identity) * np.eye(2**self._n, dtype=complex))
        for label, prob in self._probs.items():
            ops.append(np.sqrt(prob) * pauli_matrix(label))
        if not ops:  # pathological: all probability on identity rounding
            ops.append(np.eye(2**self._n, dtype=complex))
        return ops

    @cached_property
    def _table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(probabilities, x-bits, z-bits) arrays over the non-identity Paulis."""
        labels = list(self._probs)
        p = np.array([self._probs[k] for k in labels], dtype=float)
        x = np.array([[ch in "XY" for ch in k] for k in labels], dtype=np.uint8).reshape(len(labels), self._n)
        z = np.array([[ch in "ZY" for ch in k] for k in labels], dtype=np.uint8).reshape(len(labels), self._n)
        return p, x, z

    def sample_bits(self, rng: np.random.Generator, shots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``shots`` independent Paulis.

        Returns:
            ``(hit, x, z)``: ``hit`` is a bool array (shots,), ``x``/``z`` are uint8
            arrays (shots, num_qubits). Identity samples have all-zero bits.
        """
        p, xb, zb = self._table
        x = np.zeros((shots, self._n), dtype=np.uint8)
        z = np.zeros((shots, self._n), dtype=np.uint8)
        total = p.sum()
        if total <= 0:
            return np.zeros(shots, bool), x, z
        hit = rng.random(shots) < total
        idx = np.flatnonzero(hit)
        if idx.size:
            choice = rng.choice(len(p), size=idx.size, p=p / total)
            x[idx] = xb[choice]
            z[idx] = zb[choice]
        return hit, x, z

    def sample(self, rng: np.random.Generator) -> str:
        """Sample one Pauli string (``"I...I"`` for no error)."""
        r = rng.random()
        acc = 0.0
        for label, prob in self._probs.items():
            acc += prob
            if r < acc:
                return label
        return "I" * self._n

    def _params(self) -> dict:
        return {"probabilities": {k: round(v, 12) for k, v in self._probs.items()}}

    def __repr__(self) -> str:
        summary = getattr(self, "_summary", None)
        if summary:
            return summary
        terms = ", ".join(f"{k}: {v:.4g}" for k, v in self._probs.items()) or "noiseless"
        return f"{self._name}({{{terms}}})"


def pauli_channel(p_x: float = 0.0, p_y: float = 0.0, p_z: float = 0.0) -> PauliChannel:
    """Single-qubit Pauli channel with independent X, Y, Z probabilities."""
    return PauliChannel({"X": p_x, "Y": p_y, "Z": p_z}, num_qubits=1, name="PauliChannel")


def depolarizing(p: float, num_qubits: int = 1) -> PauliChannel:
    """Depolarizing channel: with probability ``p`` apply a uniformly random
    non-identity Pauli (``4**n - 1`` choices).

    Note:
        ``p`` is the *total Pauli error probability* (process infidelity). The
        average gate infidelity is ``r = p · d / (d + 1)`` with ``d = 2**n``; use
        :meth:`noisiq.noise.NoiseModel.depolarizing_from_infidelity` to convert.
        Qiskit's ``depolarizing_error(λ)`` corresponds to ``p = λ (4**n - 1) / 4**n``.
    """
    p = _check_prob(p)
    labels = [s for s in _all_paulis(num_qubits) if set(s) != {"I"}]
    ch = PauliChannel({s: p / len(labels) for s in labels}, num_qubits=num_qubits, name="Depolarizing")
    ch.catalog_key = "depolarizing"
    ch._summary = f"Depolarizing(p={p:.4g}" + (f", num_qubits={num_qubits})" if num_qubits > 1 else ")")
    return ch


def bit_flip(p: float) -> PauliChannel:
    """``ρ → (1-p) ρ + p XρX``."""
    ch = PauliChannel({"X": _check_prob(p)}, num_qubits=1, name="BitFlip")
    ch.catalog_key = "bit_flip"
    return ch


def phase_flip(p: float) -> PauliChannel:
    """``ρ → (1-p) ρ + p ZρZ`` (Pauli dephasing)."""
    ch = PauliChannel({"Z": _check_prob(p)}, num_qubits=1, name="PhaseFlip")
    ch.catalog_key = "phase_flip"
    return ch


def bit_phase_flip(p: float) -> PauliChannel:
    """``ρ → (1-p) ρ + p YρY``."""
    ch = PauliChannel({"Y": _check_prob(p)}, num_qubits=1, name="BitPhaseFlip")
    ch.catalog_key = "pauli"
    return ch


# =============================================================================
# General Kraus channels
# =============================================================================


class KrausChannel(Channel):
    """Arbitrary CPTP channel given by Kraus operators.

    Args:
        operators: Square matrices of size ``2**n`` with ``Σ K†K = I``.
        name: Display name.
    """

    catalog_key = "kraus"

    def __init__(self, operators: Sequence[np.ndarray], name: str = "KrausChannel"):
        self._ops = _validate_kraus(operators)
        self._name = name

    @property
    def num_qubits(self) -> int:
        return int(round(np.log2(self._ops[0].shape[0])))

    @property
    def name(self) -> str:
        return self._name

    def kraus(self) -> list[np.ndarray]:
        return [K.copy() for K in self._ops]

    def _params(self) -> dict:
        return {"n_operators": len(self._ops)}


class UnitaryChannel(Channel):
    """A coherent error: always apply the unitary ``U``."""

    catalog_key = "coherent"

    def __init__(self, matrix: np.ndarray, name: str = "UnitaryError"):
        from ..ir.gates import is_unitary

        m = np.array(matrix, dtype=complex)
        if not is_unitary(m):
            raise ValueError("UnitaryChannel matrix must be unitary.")
        _validate_kraus([m])
        self._U = m
        self._name = name

    @property
    def num_qubits(self) -> int:
        return int(round(np.log2(self._U.shape[0])))

    @property
    def name(self) -> str:
        return self._name

    @property
    def matrix(self) -> np.ndarray:
        return self._U.copy()

    def kraus(self) -> list[np.ndarray]:
        return [self._U.copy()]


class CoherentRotation(UnitaryChannel):
    """Systematic over/under-rotation ``U = exp(-i θ P / 2)`` about a Pauli axis.

    Args:
        axis: Pauli string, e.g. ``"Z"`` (1 qubit) or ``"ZZ"`` (2-qubit crosstalk).
        angle: Rotation angle θ in radians (the miscalibration).

    The Pauli twirl is ``p_P = sin²(θ/2)``. Unlike that stochastic approximation,
    coherent errors add in *amplitude*: ``N`` repetitions give an error probability
    ``sin²(Nθ/2) ≈ N² θ²/4`` instead of ``N θ²/4``.
    """

    catalog_key = "coherent"

    def __init__(self, axis: str, angle: float):
        axis = axis.upper()
        P = pauli_matrix(axis)
        if set(axis) == {"I"}:
            raise ValueError("Rotation axis must contain a non-identity Pauli.")
        self.axis = axis
        self.angle = float(angle)
        U = np.cos(angle / 2) * np.eye(P.shape[0]) - 1j * np.sin(angle / 2) * P
        super().__init__(U, name="CoherentRotation")

    def pauli_twirl(self) -> PauliChannel:
        p = float(np.sin(self.angle / 2) ** 2)
        return PauliChannel({self.axis: p}, num_qubits=len(self.axis), name=f"twirl({self.name})")

    def _params(self) -> dict:
        return {"axis": self.axis, "angle": self.angle}


class AmplitudeDamping(Channel):
    """Energy relaxation (T1): |1⟩ decays to |0⟩ with probability γ.

    ``K0 = [[1, 0], [0, √(1-γ)]]``, ``K1 = [[0, √γ], [0, 0]]``.
    Use :meth:`from_t1` to set γ from a relaxation time: ``γ = 1 - e^{-t/T1}``.
    """

    catalog_key = "amplitude_damping"

    def __init__(self, gamma: float):
        self.gamma = _check_prob(gamma, "gamma")

    @classmethod
    def from_t1(cls, t1: float, duration: float) -> AmplitudeDamping:
        if t1 <= 0 or duration < 0:
            raise ValueError("Need t1 > 0 and duration >= 0.")
        return cls(1.0 - np.exp(-duration / t1))

    @property
    def num_qubits(self) -> int:
        return 1

    def kraus(self) -> list[np.ndarray]:
        g = self.gamma
        return [
            np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex),
            np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex),
        ]

    def _params(self) -> dict:
        return {"gamma": self.gamma}


class PhaseDamping(Channel):
    """Pure dephasing: coherences shrink by ``√(1-λ)``, populations unchanged.

    ``K0 = [[1, 0], [0, √(1-λ)]]``, ``K1 = [[0, 0], [0, √λ]]``.
    Use :meth:`from_tphi`: coherence factor ``e^{-t/Tφ}`` ⇒ ``λ = 1 - e^{-2t/Tφ}``.
    Equivalent to a phase flip with ``p = (1 - √(1-λ)) / 2``.
    """

    catalog_key = "phase_damping"

    def __init__(self, lam: float):
        self.lam = _check_prob(lam, "lam")

    @classmethod
    def from_tphi(cls, tphi: float, duration: float) -> PhaseDamping:
        if tphi <= 0 or duration < 0:
            raise ValueError("Need tphi > 0 and duration >= 0.")
        return cls(1.0 - np.exp(-2.0 * duration / tphi))

    @property
    def num_qubits(self) -> int:
        return 1

    def kraus(self) -> list[np.ndarray]:
        lam = self.lam
        return [
            np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex),
        ]

    def _params(self) -> dict:
        return {"lam": self.lam}


class ThermalRelaxation(Channel):
    """Combined T1/T2 decoherence of an idle or gated qubit (zero temperature).

    After time ``t``: ``ρ11 → ρ11 e^{-t/T1}`` and ``ρ01 → ρ01 e^{-t/T2}``.
    Implemented as amplitude damping (γ = 1 - e^{-t/T1}) followed by pure dephasing
    with ``1/Tφ = 1/T2 - 1/(2 T1)``, so T1 is **not** double counted.

    Args:
        t1: Relaxation time (s), > 0.
        t2: Coherence time (s), ``0 < T2 <= 2 T1``.
        duration: Exposure time (s), >= 0.
    """

    catalog_key = "thermal_relaxation"

    def __init__(self, t1: float, t2: float, duration: float):
        if t1 <= 0 or t2 <= 0:
            raise ValueError("T1 and T2 must be positive.")
        if t2 > 2 * t1 * (1 + 1e-12):
            raise ValueError(f"Unphysical: T2 ({t2}) must be <= 2*T1 ({2 * t1}).")
        if duration < 0:
            raise ValueError("duration must be >= 0.")
        self.t1, self.t2, self.duration = float(t1), float(t2), float(duration)

    @property
    def num_qubits(self) -> int:
        return 1

    @property
    def gamma(self) -> float:
        return float(1.0 - np.exp(-self.duration / self.t1))

    @property
    def lam(self) -> float:
        rate_phi = 1.0 / self.t2 - 1.0 / (2.0 * self.t1)
        rate_phi = max(rate_phi, 0.0)
        return float(1.0 - np.exp(-2.0 * self.duration * rate_phi))

    def kraus(self) -> list[np.ndarray]:
        ad = AmplitudeDamping(self.gamma).kraus()
        pd = PhaseDamping(self.lam).kraus()
        ops = [B @ A for A in ad for B in pd]
        return [K for K in ops if np.linalg.norm(K) > 1e-15]

    def _params(self) -> dict:
        return {"t1": self.t1, "t2": self.t2, "duration": self.duration}


class ComposedChannel(Channel):
    """Apply several channels on the same qubits in sequence (first one first)."""

    catalog_key = "kraus"

    def __init__(self, channels: Sequence[Channel]):
        chans: list[Channel] = []
        for c in channels:
            if not isinstance(c, Channel):
                raise TypeError(f"Expected a Channel, got {type(c).__name__}.")
            chans.extend(c.channels if isinstance(c, ComposedChannel) else [c])
        if not chans:
            raise ValueError("ComposedChannel needs at least one channel.")
        n = {c.num_qubits for c in chans}
        if len(n) != 1:
            raise ValueError("All composed channels must act on the same number of qubits.")
        self.channels = chans

    @property
    def num_qubits(self) -> int:
        return self.channels[0].num_qubits

    @property
    def is_pauli(self) -> bool:
        return all(c.is_pauli for c in self.channels)

    def kraus(self) -> list[np.ndarray]:
        ops = [np.eye(2**self.num_qubits, dtype=complex)]
        for c in self.channels:
            ops = [K @ A for A in ops for K in c.kraus()]
        return [K for K in ops if np.linalg.norm(K) > 1e-15]

    def pauli_twirl(self) -> PauliChannel:
        if self.is_pauli:
            return _compose_pauli([c.pauli_twirl() for c in self.channels])
        return super().pauli_twirl()

    @property
    def name(self) -> str:
        return " ∘ ".join(c.name for c in reversed(self.channels))

    def _params(self) -> dict:
        return {"channels": [repr(c) for c in self.channels]}

    def __repr__(self) -> str:
        return f"ComposedChannel([{', '.join(repr(c) for c in self.channels)}])"


def _pauli_product(a: str, b: str) -> str:
    table = {("I", q): q for q in "IXYZ"} | {(q, "I"): q for q in "IXYZ"}
    table |= {(q, q): "I" for q in "XYZ"}
    table |= {("X", "Y"): "Z", ("Y", "X"): "Z", ("Y", "Z"): "X", ("Z", "Y"): "X", ("Z", "X"): "Y", ("X", "Z"): "Y"}
    return "".join(table[(x, y)] for x, y in zip(a, b))


def _compose_pauli(chs: Sequence[PauliChannel]) -> PauliChannel:
    n = chs[0].num_qubits
    dist = {"I" * n: 1.0}
    for ch in chs:
        step = {"I" * n: ch.p_identity, **ch.probabilities}
        new: dict[str, float] = {}
        for a, pa in dist.items():
            for b, pb in step.items():
                k = _pauli_product(a, b)
                new[k] = new.get(k, 0.0) + pa * pb
        dist = new
    return PauliChannel(dist, num_qubits=n, name="ComposedPauli")


# =============================================================================
# Classical readout error
# =============================================================================


class ReadoutError:
    """Classical measurement-assignment error.

    Args:
        p01: Probability of reporting 1 when the outcome was 0.
        p10: Probability of reporting 0 when the outcome was 1 (defaults to ``p01``).

    Applied to the recorded classical bit only; the post-measurement quantum state is
    unaffected. Feed-forward (``c_if``) sees the *reported* (possibly wrong) bit.
    """

    catalog_key = "readout"

    def __init__(self, p01: float, p10: float | None = None):
        self.p01 = _check_prob(p01, "p01")
        self.p10 = self.p01 if p10 is None else _check_prob(p10, "p10")

    @property
    def confusion_matrix(self) -> np.ndarray:
        """``M[reported, true]``."""
        return np.array([[1 - self.p01, self.p10], [self.p01, 1 - self.p10]])

    @property
    def error_probability(self) -> float:
        return 0.5 * (self.p01 + self.p10)

    def __repr__(self) -> str:
        return f"ReadoutError(p01={self.p01:.4g}, p10={self.p10:.4g})"
