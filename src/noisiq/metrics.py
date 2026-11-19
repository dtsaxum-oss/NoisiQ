"""State and distribution metrics. The math for each lives in :mod:`noisiq.catalog`
(``noisiq.explain("state_fidelity")`` etc.)."""

from __future__ import annotations

import math
import string
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "state_fidelity",
    "purity",
    "reduced_density_matrix",
    "bloch_vector",
    "trace_distance",
    "hellinger_fidelity",
    "total_variation_distance",
    "ghz_fidelity",
    "binomial_stderr",
    "to_density_matrix",
]


def to_density_matrix(state: np.ndarray) -> np.ndarray:
    """Return ``|ψ⟩⟨ψ|`` for a statevector, or the input if already a matrix."""
    s = np.asarray(state, dtype=complex)
    if s.ndim == 1:
        return np.outer(s, s.conj())
    if s.ndim == 2 and s.shape[0] == s.shape[1]:
        return s
    raise ValueError(f"Expected a statevector or square density matrix, got shape {s.shape}.")


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((m + m.conj().T) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Uhlmann fidelity ``F(ρ, σ) = (Tr √(√ρ σ √ρ))²``; statevectors are accepted."""
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    if a.ndim == 1 and b.ndim == 1:
        return float(abs(np.vdot(a, b)) ** 2)
    if a.ndim == 1:
        return float(np.real(a.conj() @ b @ a))
    if b.ndim == 1:
        return float(np.real(b.conj() @ a @ b))
    sa = _sqrtm_psd(a)
    inner = _sqrtm_psd(sa @ b @ sa)
    return float(np.clip(np.real(np.trace(inner)) ** 2, 0.0, 1.0))


def purity(rho: np.ndarray) -> float:
    """``Tr(ρ²)``."""
    rho = to_density_matrix(rho)
    return float(np.real(np.einsum("ij,ji->", rho, rho)))


def reduced_density_matrix(state: np.ndarray, keep: Sequence[int], n_qubits: int | None = None) -> np.ndarray:
    """Partial trace keeping qubits ``keep`` (big-endian, qubit 0 most significant)."""
    s = np.asarray(state, dtype=complex)
    dim = s.shape[0]
    n = n_qubits if n_qubits is not None else int(round(math.log2(dim)))
    keep = list(keep)
    if s.ndim == 1:
        psi = s.reshape([2] * n)
        others = [q for q in range(n) if q not in keep]
        psi = np.transpose(psi, keep + others).reshape(2 ** len(keep), -1)
        return psi @ psi.conj().T
    rho = s.reshape([2] * (2 * n))
    letters = string.ascii_letters
    row = list(letters[:n])
    col = list(letters[n : 2 * n])
    for q in range(n):
        if q not in keep:
            col[q] = row[q]
    out = "".join(row[q] for q in keep) + "".join(col[q] for q in keep)
    r = np.einsum("".join(row) + "".join(col) + "->" + out, rho)
    k = 2 ** len(keep)
    return r.reshape(k, k)


def bloch_vector(rho: np.ndarray) -> tuple[float, float, float]:
    """Bloch vector ``(⟨X⟩, ⟨Y⟩, ⟨Z⟩)`` of a single-qubit state."""
    rho = to_density_matrix(rho)
    if rho.shape != (2, 2):
        raise ValueError("bloch_vector needs a single-qubit state.")
    return (float(2 * rho[0, 1].real), float(-2 * rho[0, 1].imag), float((rho[0, 0] - rho[1, 1]).real))


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``½ ‖ρ − σ‖₁``."""
    d = to_density_matrix(a) - to_density_matrix(b)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh((d + d.conj().T) / 2))))


def _normalize(counts: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Empty distribution.")
    return {k: v / total for k, v in counts.items()}


def hellinger_fidelity(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """``(Σ √(p_x q_x))²`` between two count/probability dictionaries."""
    p, q = _normalize(p), _normalize(q)
    return float(sum(math.sqrt(p[k] * q.get(k, 0.0)) for k in p) ** 2)


def total_variation_distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """``½ Σ |p_x − q_x|`` between two count/probability dictionaries."""
    p, q = _normalize(p), _normalize(q)
    return float(0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q)))


def ghz_fidelity(rho: np.ndarray) -> float:
    """Fidelity with ``(|0…0⟩ + |1…1⟩)/√2``."""
    rho = to_density_matrix(rho)
    d = rho.shape[0]
    return float(0.5 * (rho[0, 0].real + rho[d - 1, d - 1].real) + rho[0, d - 1].real)


def binomial_stderr(p: float, n: int) -> float:
    """Standard error ``√(p(1−p)/n)`` of a proportion."""
    return 0.0 if n <= 0 else math.sqrt(max(p * (1 - p), 0.0) / n)
