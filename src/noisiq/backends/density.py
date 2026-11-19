"""Exact density-matrix backend (small circuits; the reference for testing).

The state is a classical-quantum state: a dictionary mapping each classical-bit
record to an unnormalized density matrix. Measurements split branches, readout errors
split the *recorded* bit, and conditional gates act per branch — so mid-circuit
measurement and feed-forward are exact, with no sampling noise. Counts are drawn
from the exact outcome distribution, which is also returned in
``Result.probabilities_exact``.
"""

from __future__ import annotations

import numpy as np

from ..ir.circuit import Circuit
from ..ir.gates import SDG, H, S
from ..metrics import state_fidelity
from ..noise.channels import pauli_matrix
from ..noise.model import NoiseModel
from ..results.result import Estimate, Result
from .frame import IncompatibleCircuitError, _counts_from_memory
from .program import DetuneStep, FaultStep, GateStep, MeasureStep, Program, ResetStep, compile_program
from .statevector import apply_matrix

__all__ = ["DensityMatrixBackend"]

MAX_QUBITS = 10
_P0 = np.array([[1, 0], [0, 0]], complex)
_P1 = np.array([[0, 0], [0, 1]], complex)
_X = pauli_matrix("X")
_BASIS_IN = {"Z": [], "X": [H], "Y": [SDG, H]}
_BASIS_OUT = {"Z": [], "X": [H], "Y": [H, S]}


def _apply_left(rho: np.ndarray, m: np.ndarray, qubits, n: int) -> np.ndarray:
    """``M ρ`` on the given qubits (ρ is 2ⁿ×2ⁿ)."""
    d = 2**n
    qubits = list(qubits)
    k = len(qubits)
    t = rho.reshape([2] * n + [d])
    u = m.reshape([2] * (2 * k))
    t = np.tensordot(u, t, axes=(list(range(k, 2 * k)), qubits))
    t = np.moveaxis(t, list(range(k)), qubits)
    return t.reshape(d, d)


def _conj_apply(rho: np.ndarray, m: np.ndarray, qubits, n: int) -> np.ndarray:
    """``M ρ M†``."""
    left = _apply_left(rho, m, qubits, n)
    return _apply_left(left.conj().T, m, qubits, n).conj().T


def _kraus(rho, ops, qubits, n):
    return sum(_conj_apply(rho, K, qubits, n) for K in ops)


class DensityMatrixBackend:
    """Exact simulation by density matrices (``n ≤ 10``)."""

    name = "density_matrix"

    def run(
        self,
        circuit: Circuit,
        noise: NoiseModel | None = None,
        shots: int = 1024,
        seed: int | None = None,
        *,
        program: Program | None = None,
        **_ignored,
    ) -> Result:
        n = circuit.n_qubits
        if n > MAX_QUBITS:
            raise IncompatibleCircuitError(f"density_matrix backend is limited to {MAX_QUBITS} qubits (got {n}).")
        program = program or compile_program(circuit, noise)
        if program.has_detuning:
            raise IncompatibleCircuitError(
                "Quasi-static detuning is correlated across a shot and cannot be represented exactly by "
                "a density matrix; use backend='statevector'."
            )
        nc = program.n_clbits
        rho0 = np.zeros((2**n, 2**n), complex)
        rho0[0, 0] = 1
        branches: dict[tuple[int, ...], np.ndarray] = {(0,) * nc: rho0}
        snapshot = None
        ideal = self._ideal_state(program, n)

        for pos, step in enumerate(program.steps):
            if pos == program.snapshot_position:
                snapshot = sum(branches.values())
            if isinstance(step, GateStep):
                for key, rho in branches.items():
                    if step.condition is None or step.condition.is_met(key):
                        branches[key] = _conj_apply(rho, step.gate.matrix, step.qubits, n)
            elif isinstance(step, FaultStep):
                ops = step.channel.kraus()
                for key, rho in branches.items():
                    if step.condition is None or step.condition.is_met(key):
                        branches[key] = _kraus(rho, ops, step.qubits, n)
            elif isinstance(step, MeasureStep):
                branches = self._measure(branches, step, n)
            elif isinstance(step, ResetStep):
                for key, rho in branches.items():
                    a = _conj_apply(rho, _P0, (step.qubit,), n)
                    b = _conj_apply(_conj_apply(rho, _P1, (step.qubit,), n), _X, (step.qubit,), n)
                    branches[key] = a + b
            elif isinstance(step, DetuneStep):  # pragma: no cover - rejected above
                raise IncompatibleCircuitError("detuning not supported")
        if snapshot is None:
            snapshot = sum(branches.values())

        probs = {"".join(map(str, k)): float(np.real(np.trace(r))) for k, r in branches.items()}
        probs = {k: v for k, v in sorted(probs.items()) if v > 1e-14}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        rng = np.random.default_rng(seed)
        keys = list(probs)
        draws = rng.choice(len(keys), size=shots, p=np.array([probs[k] for k in keys]))
        memory = np.array([[int(ch) for ch in keys[i]] for i in draws], dtype=np.uint8).reshape(shots, nc)
        from ..metrics import reduced_density_matrix

        fid = None
        if ideal is not None:
            fid = Estimate(state_fidelity(ideal, snapshot), 0.0)
        return Result(
            circuit=circuit,
            backend=self.name,
            shots=shots,
            seed=seed,
            counts=_counts_from_memory(memory),
            memory=memory,
            implicit_measurement=program.implicit_measurement,
            noise_name=noise.name if noise else "noiseless",
            probabilities_exact=probs,
            fidelity=fid,
            density_matrix=snapshot,
            qubit_states=np.array([reduced_density_matrix(snapshot, [q], n) for q in range(n)]),
        )

    @staticmethod
    def _ideal_state(program: Program, n: int) -> np.ndarray | None:
        """Noiseless statevector at the snapshot, if the pre-snapshot part is unitary."""
        psi = np.zeros(2**n, complex)
        psi[0] = 1
        for step in program.steps[: program.snapshot_position]:
            if isinstance(step, (MeasureStep, ResetStep)):
                return None
            if isinstance(step, GateStep):
                if step.condition is not None:
                    return None
                psi = apply_matrix(psi, step.gate.matrix, step.qubits, n)
        return psi

    @staticmethod
    def _measure(branches, step: MeasureStep, n):
        q = step.qubit
        new: dict[tuple[int, ...], np.ndarray] = {}
        ro = step.readout
        for key, rho in branches.items():
            for g in _BASIS_IN[step.basis]:
                rho = _conj_apply(rho, g.matrix, (q,), n)
            for outcome, proj in ((0, _P0), (1, _P1)):
                r = _conj_apply(rho, proj, (q,), n)
                if np.real(np.trace(r)) < 1e-15:
                    continue
                if step.reset:
                    if outcome:
                        r = _conj_apply(r, _X, (q,), n)
                else:
                    for g in _BASIS_OUT[step.basis]:
                        r = _conj_apply(r, g.matrix, (q,), n)
                if ro is None:
                    splits = [(outcome, 1.0)]
                else:
                    p_flip = ro.p10 if outcome else ro.p01
                    splits = [(outcome, 1 - p_flip), (1 - outcome, p_flip)]
                for reported, w in splits:
                    if w <= 0:
                        continue
                    k = list(key)
                    k[step.clbit] = reported
                    k = tuple(k)
                    new[k] = new.get(k, 0) + w * r
        return new
