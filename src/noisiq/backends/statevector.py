"""Monte-Carlo trajectory (statevector) backend: any gate, any noise channel.

Each shot evolves a pure state ``|ψ⟩``. A noise channel with Kraus operators
``{K_k}`` is applied by choosing ``k`` with probability ``‖K_k ψ‖²`` and renormalizing —
averaging over shots reproduces the exact density-matrix evolution.

Shots are simulated **together**: the state is an array of shape ``(batch, 2ⁿ)`` and every
step is a vectorized numpy operation, with shots processed in chunks so memory stays
bounded (``batch · 2ⁿ ≲ 4M`` amplitudes).

Alongside the noisy state an *ideal* state is evolved (no noise; measurement outcomes
post-selected to match the noisy shot) so the backend can report fidelity to ideal and
per-qubit deviations for animations. Per-qubit reduced states are always accumulated;
the full ``4ⁿ`` density matrix only when requested and small (default ``n ≤ 8``).
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..ir.circuit import Circuit
from ..ir.gates import SDG, H, S
from ..noise.channels import Channel, PauliChannel, UnitaryChannel, pauli_matrix
from ..noise.model import NoiseModel
from ..results.result import Estimate, Result
from ..results.trace import FaultEvent, PropagationTrace
from .frame import IncompatibleCircuitError, _counts_from_memory
from .program import DetuneStep, FaultStep, GateStep, LayerEnd, MeasureStep, Program, ResetStep, compile_program

__all__ = ["StatevectorBackend", "apply_matrix", "MAX_QUBITS"]

MAX_QUBITS = 26
_CHUNK_AMPLITUDES = 1 << 22
_X = pauli_matrix("X")
_PAULI_MATS = {ch: pauli_matrix(ch) for ch in "XYZ"}
_BASIS_IN = {"Z": [], "X": [H.matrix], "Y": [SDG.matrix, H.matrix]}
_BASIS_OUT = {"Z": [], "X": [H.matrix], "Y": [H.matrix, S.matrix]}


# ============================================================================ kernels
def apply_matrix(state: np.ndarray, matrix: np.ndarray, qubits, n: int) -> np.ndarray:
    """Apply a ``2^k × 2^k`` matrix to ``qubits`` of an ``n``-qubit statevector (big-endian)."""
    return _apply(state[None, :], matrix, qubits, n)[0]


def _apply(psi: np.ndarray, m: np.ndarray, qubits, n: int) -> np.ndarray:
    """Batched gate application; ``psi`` has shape (B, 2**n)."""
    qubits = list(qubits)
    k = len(qubits)
    B = psi.shape[0]
    if k == 1:
        q = qubits[0]
        a = psi.reshape(B, 2**q, 2, -1)
        out = np.empty_like(a, dtype=complex)
        a0, a1 = a[:, :, 0], a[:, :, 1]
        if m[0, 1] == 0 and m[1, 0] == 0:
            np.multiply(a0, m[0, 0], out=out[:, :, 0])
            np.multiply(a1, m[1, 1], out=out[:, :, 1])
        else:
            out[:, :, 0] = m[0, 0] * a0 + m[0, 1] * a1
            out[:, :, 1] = m[1, 0] * a0 + m[1, 1] * a1
        return out.reshape(B, -1)
    t = psi.reshape([B] + [2] * n)
    u = m.reshape([2] * (2 * k))
    t = np.tensordot(u, t, axes=(list(range(k, 2 * k)), [q + 1 for q in qubits]))
    # axes now: out_0..out_{k-1}, B, remaining qubits in order
    t = np.moveaxis(t, list(range(k)) + [k], [q + 1 for q in qubits] + [0])
    return t.reshape(B, -1)


def _apply_masked(psi: np.ndarray, m: np.ndarray, qubits, n: int, mask: np.ndarray) -> np.ndarray:
    if mask.all():
        return _apply(psi, m, qubits, n)
    if mask.any():
        psi = psi.copy()
        psi[mask] = _apply(psi[mask], m, qubits, n)
    return psi


def _normalize(psi: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(psi, axis=1)
    norms[norms < 1e-300] = 1.0
    return psi / norms[:, None]


def _rho_1q(psi: np.ndarray, q: int) -> np.ndarray:
    """(B, 2, 2) reduced density matrices of qubit q."""
    B = psi.shape[0]
    a = psi.reshape(B, 2**q, 2, -1)
    return np.einsum("bxiy,bxjy->bij", a, a.conj())


def _rho_all_1q(psi: np.ndarray, n: int) -> np.ndarray:
    """(B, n, 2, 2)."""
    return np.stack([_rho_1q(psi, q) for q in range(n)], axis=1)


def _prob_one(psi: np.ndarray, q: int) -> np.ndarray:
    B = psi.shape[0]
    a = psi.reshape(B, 2**q, 2, -1)[:, :, 1]
    return np.real(np.einsum("bxy,bxy->b", a, a.conj()))


def _project(psi: np.ndarray, q: int, outcome: np.ndarray) -> np.ndarray:
    B = psi.shape[0]
    a = psi.reshape(B, 2**q, 2, -1).copy()
    zero_one = outcome.astype(bool)
    a[zero_one, :, 0] = 0
    a[~zero_one, :, 1] = 0
    return _normalize(a.reshape(B, -1))


def _fid_1q(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fidelity between batches of 2x2 density matrices (last two axes)."""
    tr_ab = np.real(np.einsum("...ij,...ji->...", a, b))
    det = np.real(np.linalg.det(a)) * np.real(np.linalg.det(b))
    return np.clip(tr_ab + 2 * np.sqrt(np.clip(det, 0, None)), 0.0, 1.0)


# ============================================================================ backend
class StatevectorBackend:
    """Universal trajectory simulator (any gates, Kraus/coherent noise, feed-forward)."""

    name = "statevector"

    def run(
        self,
        circuit: Circuit,
        noise: NoiseModel | None = None,
        shots: int = 1024,
        seed: int | None = None,
        *,
        trace_shots: int = 0,
        density_matrix: Union[bool, str] = "auto",
        program: Program | None = None,
    ) -> Result:
        """Simulate ``circuit`` (see :func:`noisiq.simulate`).

        Args:
            density_matrix: Accumulate the averaged full density matrix
                (``"auto"``: only for ``n ≤ 8``).
        """
        if shots < 1:
            raise ValueError("shots must be >= 1.")
        n = circuit.n_qubits
        if n > MAX_QUBITS:
            raise IncompatibleCircuitError(
                f"The statevector backend is limited to {MAX_QUBITS} qubits (got {n}). "
                "For larger circuits use Clifford gates with Pauli (or twirled) noise on backend='frame'."
            )
        program = program or compile_program(circuit, noise)
        keep_dm = (n <= 8) if density_matrix == "auto" else bool(density_matrix)
        rng = np.random.default_rng(seed)
        T = min(trace_shots, shots)
        acc = _Accumulator(program, shots, T, keep_dm)
        chunk = max(1, min(shots, _CHUNK_AMPLITUDES // 2**n))
        start = 0
        while start < shots:
            b = min(chunk, shots - start)
            _Chunk(program, acc, rng, start, b).run()
            start += b

        trace = None
        if T:
            trace = PropagationTrace(
                circuit=circuit,
                kind="deviation",
                n_shots=T,
                events=acc.events,
                deviation=acc.deviation,
                description=noise.name if noise else "noiseless",
            )
        fid = acc.fid
        return Result(
            circuit=circuit,
            backend=self.name,
            shots=shots,
            seed=seed,
            counts=_counts_from_memory(acc.memory),
            memory=acc.memory,
            implicit_measurement=program.implicit_measurement,
            noise_name=noise.name if noise else "noiseless",
            instruction_faults=acc.inst_f,
            idle_faults=acc.idle_f[: program.n_layers],
            readout_faults=acc.ro_f,
            fault_free=acc.fault_free,
            fidelity=Estimate(float(fid.mean()), float(fid.std(ddof=1) / np.sqrt(shots)) if shots > 1 else 0.0),
            qubit_states=acc.red_sum / shots,
            density_matrix=None if acc.dm_sum is None else acc.dm_sum / shots,
            trace=trace,
        )


class _Accumulator:
    def __init__(self, program: Program, shots: int, T: int, keep_dm: bool):
        n = program.n_qubits
        self.inst_f = np.zeros((program.n_instructions, n, 4))
        self.idle_f = np.zeros((max(program.n_layers, 1), n, 4))
        self.ro_f = np.zeros(n)
        self.fault_free = np.ones(shots, bool)
        self.memory = np.zeros((shots, program.n_clbits), np.uint8)
        self.fid = np.zeros(shots)
        self.red_sum = np.zeros((n, 2, 2), complex)
        self.dm_sum = np.zeros((2**n, 2**n), complex) if keep_dm else None
        self.events: list[FaultEvent] = []
        self.T = T
        self.deviation = np.zeros((program.n_layers + 1, n, T)) if T else None


class _Chunk:
    """Simulates shots ``[start, start + B)`` together."""

    def __init__(self, program: Program, acc: _Accumulator, rng, start: int, B: int):
        self.p, self.acc, self.rng, self.start, self.B = program, acc, rng, start, B
        n = program.n_qubits
        self.n = n
        self.psi = np.zeros((B, 2**n), complex)
        self.psi[:, 0] = 1
        self.ideal = self.psi.copy()
        self.ideal_ok = np.ones(B, bool)
        self.bits = np.zeros((B, program.n_clbits), np.uint8)
        self.fired: dict[int, np.ndarray] = {}
        self.faulted = np.zeros(B, bool)
        sigma = np.asarray(program.detuning_sigma, float)
        self.delta = rng.normal(0.0, 1.0, (B, n)) * sigma if sigma.any() else None
        self.T_local = max(0, min(acc.T - start, B))

    # ------------------------------------------------------------------ record
    def record(self, index, layer, q, col, weights: np.ndarray, kind: str, pauli: str) -> None:
        """weights: (B,) per-shot fault weight (0 = no fault)."""
        tot = float(weights.sum())
        if tot == 0:
            return
        tgt = self.acc.idle_f[layer, q] if kind == "idle" else self.acc.inst_f[index, q]
        tgt[col] += tot
        self.faulted |= weights > 0
        for b in np.flatnonzero(weights[: self.T_local] > 0):
            self.acc.events.append(FaultEvent(self.start + int(b), layer, index, q, pauli, kind))

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        p = self.p
        for pos, step in enumerate(p.steps):
            if pos == p.snapshot_position:
                self.snapshot()
            if isinstance(step, GateStep):
                self.gate(step)
            elif isinstance(step, FaultStep):
                self.fault(step)
            elif isinstance(step, MeasureStep):
                self.measure(step)
            elif isinstance(step, ResetStep):
                self.reset(step.qubit)
            elif isinstance(step, DetuneStep):
                self.detune(step)
            elif isinstance(step, LayerEnd) and self.T_local:
                self.trace_layer(step.layer)
        if p.snapshot_position >= len(p.steps):
            self.snapshot()
        sl = slice(self.start, self.start + self.B)
        self.acc.memory[sl] = self.bits
        self.acc.fault_free[sl] = ~self.faulted

    def gate(self, s: GateStep) -> None:
        m = s.gate.matrix
        if s.condition is None:
            self.psi = _apply(self.psi, m, s.qubits, self.n)
            self.ideal = _apply(self.ideal, m, s.qubits, self.n)
            return
        fire = self._cond(s.condition)
        self.fired[s.index] = fire
        self.psi = _apply_masked(self.psi, m, s.qubits, self.n, fire)
        self.ideal = _apply_masked(self.ideal, m, s.qubits, self.n, fire)

    def _cond(self, cond) -> np.ndarray:
        v = np.zeros(self.B, np.int64)
        for i, c in enumerate(cond.clbits):
            v |= self.bits[:, c].astype(np.int64) << i
        return v == cond.value

    def fault(self, s: FaultStep) -> None:
        ch: Channel = s.channel
        mask = self.fired.get(s.index) if s.condition is not None else None
        n = self.n
        if isinstance(ch, PauliChannel):
            _, x, z = ch.sample_bits(self.rng, self.B)
            if mask is not None:
                x = x & mask[:, None]
                z = z & mask[:, None]
            for j, q in enumerate(s.qubits):
                xs, zs = x[:, j].astype(bool), z[:, j].astype(bool)
                if xs.any():
                    self.psi = _apply_masked(self.psi, _PAULI_MATS["X"], (q,), n, xs)
                if zs.any():
                    self.psi = _apply_masked(self.psi, _PAULI_MATS["Z"], (q,), n, zs)
                for col, sel in ((0, xs & ~zs), (1, xs & zs), (2, ~xs & zs)):
                    self.record(s.index, s.layer, q, col, sel.astype(float), s.kind, "XYZ"[col])
            return
        if isinstance(ch, UnitaryChannel):
            sel = np.ones(self.B, bool) if mask is None else mask
            before = self.psi
            self.psi = _apply_masked(self.psi, ch.matrix, s.qubits, n, sel)
            w = 1.0 - np.abs(np.einsum("bi,bi->b", before.conj(), self.psi)) ** 2
            w[w < 1e-12] = 0
            for q in s.qubits:
                self.record(s.index, s.layer, q, 3, w, s.kind, "other")
            return
        ops = _kraus_cache(ch)
        if len(s.qubits) == 1:
            rho = _rho_1q(self.psi, s.qubits[0])
            probs = np.stack([np.real(np.einsum("ij,bji->b", K.conj().T @ K, rho)) for K in ops], axis=1)
        else:
            outs = [_apply(self.psi, K, s.qubits, n) for K in ops]
            probs = np.stack([np.real(np.einsum("bi,bi->b", o.conj(), o)) for o in outs], axis=1)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum(axis=1, keepdims=True)
        choice = (self.rng.random(self.B)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
        choice = np.minimum(choice, len(ops) - 1)
        trivial = int(np.argmax([abs(np.trace(K)) for K in ops]))
        if mask is not None:
            choice = np.where(mask, trivial, choice)  # gate did not fire → no fault
            # (the trivial Kraus operator is still applied below only where mask is True)
        new = self.psi.copy()
        for k, K in enumerate(ops):
            sel = choice == k
            if mask is not None:
                sel &= mask
            if sel.any():
                new[sel] = _apply(self.psi[sel], K, s.qubits, n)
        self.psi = _normalize(new)
        jumped = (choice != trivial).astype(float)
        for q in s.qubits:
            self.record(s.index, s.layer, q, 3, jumped, s.kind, "other")

    def measure(self, s: MeasureStep) -> None:
        q, n = s.qubit, self.n
        for m in _BASIS_IN[s.basis]:
            self.psi = _apply(self.psi, m, (q,), n)
            self.ideal = _apply(self.ideal, m, (q,), n)
        p1 = _prob_one(self.psi, q)
        out = (self.rng.random(self.B) < p1).astype(np.uint8)
        self.psi = _project(self.psi, q, out)
        pi1 = _prob_one(self.ideal, q)
        p_ideal_out = np.where(out == 1, pi1, 1 - pi1)
        self.ideal_ok &= p_ideal_out > 1e-12
        self.ideal = _project(self.ideal, q, out)
        if s.reset:
            flip = out.astype(bool)
            self.psi = _apply_masked(self.psi, _X, (q,), n, flip)
            self.ideal = _apply_masked(self.ideal, _X, (q,), n, flip)
        else:
            for m in _BASIS_OUT[s.basis]:
                self.psi = _apply(self.psi, m, (q,), n)
                self.ideal = _apply(self.ideal, m, (q,), n)
        if s.readout is not None:
            prob = np.where(out == 1, s.readout.p10, s.readout.p01)
            flip = self.rng.random(self.B) < prob
            out = out ^ flip.astype(np.uint8)
            self.acc.ro_f[q] += flip.sum()
            if s.index >= 0:
                self.record(s.index, s.layer, q, 3, flip.astype(float), "readout", "other")
            else:
                self.faulted |= flip
        self.bits[:, s.clbit] = out

    def reset(self, q: int) -> None:
        n = self.n
        out = (self.rng.random(self.B) < _prob_one(self.psi, q)).astype(np.uint8)
        self.psi = _apply_masked(_project(self.psi, q, out), _X, (q,), n, out.astype(bool))
        pi1 = _prob_one(self.ideal, q)
        oi = np.where(out == 1, pi1 > 1e-12, ~(1 - pi1 > 1e-12)).astype(np.uint8)
        self.ideal = _apply_masked(_project(self.ideal, q, oi), _X, (q,), n, oi.astype(bool))

    def detune(self, s: DetuneStep) -> None:
        ang = self.delta[:, s.qubit] * s.duration  # (B,)
        if not np.any(ang):
            return
        B, q = self.B, s.qubit
        before = self.psi
        a = self.psi.reshape(B, 2**q, 2, -1).copy()
        a[:, :, 0] *= np.exp(-0.5j * ang)[:, None, None]
        a[:, :, 1] *= np.exp(0.5j * ang)[:, None, None]
        self.psi = a.reshape(B, -1)
        w = 1.0 - np.abs(np.einsum("bi,bi->b", before.conj(), self.psi)) ** 2
        w[w < 1e-12] = 0
        tot = float(w.sum())
        if tot:
            self.acc.idle_f[s.layer, q, 3] += tot

    def snapshot(self) -> None:
        sl = slice(self.start, self.start + self.B)
        f = np.abs(np.einsum("bi,bi->b", self.ideal.conj(), self.psi)) ** 2
        self.acc.fid[sl] = np.where(self.ideal_ok, f, 0.0)
        self.acc.red_sum += _rho_all_1q(self.psi, self.n).sum(axis=0)
        if self.acc.dm_sum is not None:
            self.acc.dm_sum += self.psi.T @ self.psi.conj()

    def trace_layer(self, layer: int) -> None:
        k = self.T_local
        a = _rho_all_1q(self.psi[:k], self.n)
        b = _rho_all_1q(self.ideal[:k], self.n)
        dev = 1.0 - _fid_1q(a, b)  # (k, n)
        dev[~self.ideal_ok[:k]] = 1.0
        self.acc.deviation[layer + 1, :, self.start : self.start + k] = dev.T


_KRAUS_CACHE: dict[int, tuple[Channel, list[np.ndarray]]] = {}


def _kraus_cache(ch: Channel) -> list[np.ndarray]:
    """Kraus operators, computed once per channel object."""
    hit = _KRAUS_CACHE.get(id(ch))
    if hit is None or hit[0] is not ch:
        if len(_KRAUS_CACHE) > 4096:
            _KRAUS_CACHE.clear()
        hit = (ch, ch.kraus())
        _KRAUS_CACHE[id(ch)] = hit
    return hit[1]
