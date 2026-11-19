"""Vectorized Pauli-frame simulator for Clifford circuits with Pauli noise.

How it works
------------
One noiseless *reference* shot is simulated with :class:`stim.TableauSimulator`.
Every other shot is described by a *Pauli frame*: the Pauli operator that maps the
reference state to that shot's state. Frames for all shots are stored as bit arrays
``x, z`` of shape ``(n_qubits, shots)`` and updated together with one small binary
matrix multiplication per gate (the gate's symplectic matrix, derived from its Stim
tableau — so *any* Clifford gate works, including user-defined ones).

Two frames are tracked per shot:

* ``E`` — the **error frame**: difference between the noisy shot and an ideal shot with
  the same measurement randomness. This is what NoisiQ visualizes.
* ``R`` — the **randomization frame** ("stabilizer randomization", as in Stim): random
  Z (X/Y) flips injected after preparations/measurements so that measurements with
  random outcomes are sampled correctly.

A measurement outcome is ``reference ⊕ component(E ⊕ R)``; its *error* is
``component(E)``. Classically-controlled Pauli gates are supported by toggling the
frame wherever a shot's condition differs from the reference.

Cost: ``O(gates × shots)`` bit operations — hundreds of qubits and ~10⁵ shots are fine.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import stim

from ..ir.circuit import Circuit
from ..noise.channels import Channel, PauliChannel, phase_flip
from ..noise.model import NoiseModel
from ..results.result import Result
from ..results.trace import FaultEvent, PropagationTrace
from .program import (
    DetuneStep,
    FaultStep,
    GateStep,
    LayerEnd,
    MeasureStep,
    Program,
    ResetStep,
    compile_program,
)

__all__ = ["FrameBackend", "IncompatibleCircuitError"]

_MEASURE_OP = {"Z": "M", "X": "MX", "Y": "MY"}


class IncompatibleCircuitError(ValueError):
    """Raised when a circuit/noise combination cannot run on the chosen backend."""


def frame_incompatibilities(program: Program, approximate: bool = False) -> list[str]:
    """Reasons ``program`` cannot run on the frame backend (empty list = OK)."""
    problems = []
    bad_gates = sorted(
        {s.gate.display_label for s in program.steps if isinstance(s, GateStep) and not s.gate.is_clifford}
    )
    if bad_gates:
        problems.append(f"non-Clifford gates: {', '.join(bad_gates[:6])}")
    bad_cond = sorted(
        {s.gate.display_label for s in program.steps if isinstance(s, GateStep) and s.condition and not s.gate.is_pauli}
    )
    if bad_cond:
        problems.append(f"classically-controlled non-Pauli gates: {', '.join(bad_cond[:6])}")
    if not approximate:
        bad_noise = sorted({ch.name for ch in program.channels() if not ch.is_pauli})
        if bad_noise:
            problems.append(f"non-Pauli noise: {', '.join(bad_noise[:6])} (use noise.twirled() or approximate=True)")
        if program.has_detuning:
            problems.append("quasi-static detuning (use noise.twirled() or approximate=True)")
    return problems


def _pauli(ch: Channel, approximate: bool) -> PauliChannel:
    if isinstance(ch, PauliChannel):
        return ch
    if ch.is_pauli or approximate:
        return ch.pauli_twirl()
    raise IncompatibleCircuitError(f"Channel {ch!r} is not a Pauli channel.")


def _cond_mask(cond, bits: np.ndarray) -> np.ndarray:
    v = np.zeros(bits.shape[1], dtype=np.int64)
    for i, c in enumerate(cond.clbits):
        v |= bits[c].astype(np.int64) << i
    return v == cond.value


def _apply_symplectic(x: np.ndarray, z: np.ndarray, m: np.ndarray, qubits: tuple[int, ...]) -> None:
    q = list(qubits)
    k = len(q)
    bits = np.concatenate([x[q], z[q]], axis=0)
    new = (m @ bits) & 1
    x[q] = new[:k]
    z[q] = new[k:]


class _Run:
    """Mutable state of one frame simulation."""

    def __init__(
        self,
        program: Program,
        shots: int,
        rng: np.random.Generator,
        *,
        randomize: bool,
        approximate: bool,
        apply_noise: bool,
        trace_shots: int,
        record: bool,
        inject: Callable[[int, _Run], None] | None = None,
    ):
        self.p = program
        self.S = shots
        self.rng = rng
        self.randomize = randomize
        self.approximate = approximate
        self.apply_noise = apply_noise
        self.T = min(trace_shots, shots)
        self.record = record
        self.inject = inject
        n, nc = program.n_qubits, program.n_clbits
        u8 = np.uint8
        self.Ex = np.zeros((n, shots), u8)
        self.Ez = np.zeros((n, shots), u8)
        self.Rx = np.zeros((n, shots), u8)
        self.Rz = self._rand((n, shots))
        self.ref = stim.TableauSimulator(seed=int(rng.integers(2**63 - 1)))
        self.ref.set_num_qubits(n)
        self.ref_bits = np.zeros(nc, u8)
        self.rec = np.zeros((nc, shots), u8)  # recorded flip vs reference (incl. readout)
        self.rnd = np.zeros((nc, shots), u8)  # flip of the ideal (randomization-only) run
        self.fire: dict[int, np.ndarray] = {}
        if record:
            self.inst_f = np.zeros((program.n_instructions, n, 4))
            self.idle_f = np.zeros((max(program.n_layers, 1), n, 4))
            self.ro_f = np.zeros(n)
            self.faulted = np.zeros(shots, bool)
        self.events: list[FaultEvent] = []
        self.frames_x: list[np.ndarray] = []
        self.frames_z: list[np.ndarray] = []
        self.final: tuple[np.ndarray, np.ndarray] | None = None

    def _rand(self, shape) -> np.ndarray:
        if not self.randomize:
            return np.zeros(shape, np.uint8)
        return self.rng.integers(0, 2, shape, dtype=np.uint8)

    # ------------------------------------------------------------------
    def snapshot(self) -> None:
        if self.T:
            self.frames_x.append(self.Ex[:, : self.T].copy())
            self.frames_z.append(self.Ez[:, : self.T].copy())

    def tally(self, step: FaultStep | None, index, layer, qubits, x, z, kind: str) -> None:
        """x, z: (shots, k) sampled bits."""
        if not self.record:
            return
        for j, q in enumerate(qubits):
            xj, zj = x[:, j].astype(bool), z[:, j].astype(bool)
            counts = (np.count_nonzero(xj & ~zj), np.count_nonzero(xj & zj), np.count_nonzero(~xj & zj))
            target = self.idle_f[layer, q] if kind == "idle" else self.inst_f[index, q]
            target[:3] += counts
            if self.T:
                for s in np.flatnonzero((xj | zj)[: self.T]):
                    p = "XYZ"[[(1, 0), (1, 1), (0, 1)].index((int(xj[s]), int(zj[s])))]
                    self.events.append(FaultEvent(int(s), layer, index, q, p, kind))
        self.faulted |= x.any(axis=1) | z.any(axis=1)

    def run(self) -> None:
        self.snapshot()
        for pos, step in enumerate(self.p.steps):
            if pos == self.p.snapshot_position:
                self.final = (self.Ex.copy(), self.Ez.copy())
            if self.inject is not None:
                self.inject(pos, self)
            if isinstance(step, GateStep):
                self.gate(step)
            elif isinstance(step, FaultStep):
                if self.apply_noise:
                    self.fault(step)
            elif isinstance(step, MeasureStep):
                self.measure(step)
            elif isinstance(step, ResetStep):
                self.reset(step.qubit)
            elif isinstance(step, DetuneStep):
                if self.apply_noise:
                    self.detune(step)
            elif isinstance(step, LayerEnd):
                self.snapshot()
        if self.final is None:
            self.final = (self.Ex.copy(), self.Ez.copy())

    def gate(self, s: GateStep) -> None:
        g = s.gate
        if s.condition is None:
            self.ref.do_tableau(g.tableau, list(s.qubits))
            m = g.symplectic
            if not g.is_pauli:
                _apply_symplectic(self.Ex, self.Ez, m, s.qubits)
                _apply_symplectic(self.Rx, self.Rz, m, s.qubits)
            return
        fire_ref = s.condition.is_met(self.ref_bits)
        if fire_ref:
            self.ref.do_tableau(g.tableau, list(s.qubits))
        ref_col = self.ref_bits[:, None]
        fire_act = _cond_mask(s.condition, ref_col ^ self.rec)
        fire_ideal = _cond_mask(s.condition, ref_col ^ self.rnd)
        self.fire[s.index] = fire_act
        tog_r = (fire_ideal != fire_ref).astype(np.uint8)
        tog_e = (fire_act != fire_ideal).astype(np.uint8)
        px, pz = g.pauli_bits
        for j, q in enumerate(s.qubits):
            if px[j]:
                self.Rx[q] ^= tog_r
                self.Ex[q] ^= tog_e
            if pz[j]:
                self.Rz[q] ^= tog_r
                self.Ez[q] ^= tog_e

    def fault(self, s: FaultStep) -> None:
        ch = _pauli(s.channel, self.approximate)
        hit, x, z = ch.sample_bits(self.rng, self.S)
        if s.condition is not None and s.index in self.fire:
            mask = self.fire[s.index]
            x &= mask[:, None].astype(np.uint8)
            z &= mask[:, None].astype(np.uint8)
        for j, q in enumerate(s.qubits):
            self.Ex[q] ^= x[:, j]
            self.Ez[q] ^= z[:, j]
        self.tally(s, s.index, s.layer, s.qubits, x, z, s.kind)

    def detune(self, s: DetuneStep) -> None:
        sigma = self.p.detuning_sigma[s.qubit]
        p = 0.5 * (1.0 - np.exp(-0.5 * (sigma * s.duration) ** 2))
        if p <= 0:
            return
        _, x, z = phase_flip(p).sample_bits(self.rng, self.S)
        self.Ez[s.qubit] ^= z[:, 0]
        self.tally(None, None, s.layer, (s.qubit,), x, z, "idle")

    def measure(self, s: MeasureStep) -> None:
        q, b = s.qubit, s.basis
        self.ref.do(stim.Circuit(f"{_MEASURE_OP[b]} {q}"))
        ref_bit = int(self.ref.current_measurement_record()[-1])
        if b == "Z":
            ce, cr = self.Ex[q].copy(), self.Rx[q].copy()
        elif b == "X":
            ce, cr = self.Ez[q].copy(), self.Rz[q].copy()
        else:
            ce, cr = self.Ex[q] ^ self.Ez[q], self.Rx[q] ^ self.Rz[q]
        full = ce ^ cr
        rec = full
        if s.readout is not None and self.apply_noise:
            true_bit = ref_bit ^ full
            prob = np.where(true_bit == 1, s.readout.p10, s.readout.p01)
            flip = (self.rng.random(self.S) < prob).astype(np.uint8)
            rec = full ^ flip
            if self.record:
                n_flip = int(flip.sum())
                self.ro_f[q] += n_flip
                if s.index >= 0:
                    self.inst_f[s.index, q, 3] += n_flip
                self.faulted |= flip.astype(bool)
                if self.T:
                    for sh in np.flatnonzero(flip[: self.T]):
                        self.events.append(
                            FaultEvent(int(sh), s.layer, s.index if s.index >= 0 else None, q, "other", "readout")
                        )
        c = s.clbit
        self.ref_bits[c] = ref_bit
        self.rec[c] = rec
        self.rnd[c] = cr
        r = self._rand(self.S)
        if b == "Z":
            self.Ez[q] = 0
            self.Rz[q] = r
        elif b == "X":
            self.Ex[q] = 0
            self.Rx[q] = r
        else:
            self.Ex[q], self.Ez[q] = ce, 0
            self.Rx[q], self.Rz[q] = cr ^ r, r
        if s.reset:
            self.ref.do(stim.Circuit(f"R {q}"))
            self._clear(q)

    def reset(self, q: int) -> None:
        self.ref.do(stim.Circuit(f"R {q}"))
        self._clear(q)

    def _clear(self, q: int) -> None:
        self.Ex[q] = 0
        self.Ez[q] = 0
        self.Rx[q] = 0
        self.Rz[q] = self._rand(self.S)


class FrameBackend:
    """Fast stabilizer backend: Clifford gates + Pauli noise, any size.

    Supports X/Y/Z-basis measurements, resets, readout errors and classically
    controlled Pauli corrections. Non-Pauli noise must be twirled first
    (``noise.twirled()``) or pass ``approximate=True`` to twirl automatically.
    """

    name = "frame"

    def run(
        self,
        circuit: Circuit,
        noise: NoiseModel | None = None,
        shots: int = 1024,
        seed: int | None = None,
        *,
        approximate: bool = False,
        trace_shots: int = 0,
        program: Program | None = None,
    ) -> Result:
        """Simulate ``circuit`` for ``shots`` shots (see :func:`noisiq.simulate`)."""
        if shots < 1:
            raise ValueError("shots must be >= 1.")
        program = program or compile_program(circuit, noise)
        problems = frame_incompatibilities(program, approximate)
        if problems:
            raise IncompatibleCircuitError(
                "The frame (stabilizer) backend cannot run this circuit: "
                + "; ".join(problems)
                + ". Use backend='statevector' for small circuits."
            )
        rng = np.random.default_rng(seed)
        run = _Run(
            program,
            shots,
            rng,
            randomize=True,
            approximate=approximate,
            apply_noise=True,
            trace_shots=trace_shots,
            record=True,
        )
        run.run()

        actual = (run.ref_bits[:, None] ^ run.rec).T.astype(np.uint8)  # (shots, n_clbits)
        counts = _counts_from_memory(actual)
        trace = None
        if run.T:
            trace = PropagationTrace(
                circuit=circuit,
                kind="pauli",
                n_shots=run.T,
                events=run.events,
                frames_x=np.array(run.frames_x, dtype=np.uint8),
                frames_z=np.array(run.frames_z, dtype=np.uint8),
                measurement_flips=((run.rec ^ run.rnd).T[: run.T]).astype(bool),
                description=noise.name if noise else "noiseless",
            )
        return Result(
            circuit=circuit,
            backend=self.name,
            shots=shots,
            seed=seed,
            counts=counts,
            memory=actual,
            implicit_measurement=program.implicit_measurement,
            noise_name=(noise.name + (" (twirled)" if approximate else "")) if noise else "noiseless",
            instruction_faults=run.inst_f,
            idle_faults=run.idle_f[: program.n_layers],
            readout_faults=run.ro_f,
            fault_free=~run.faulted,
            measurement_errors=(run.rec ^ run.rnd).T.astype(bool),
            final_errors=(run.final[0].astype(bool), run.final[1].astype(bool)),
            trace=trace,
            metadata={"approximate": approximate},
        )


def _counts_from_memory(mem: np.ndarray) -> dict[str, int]:
    if mem.shape[1] == 0:
        return {"": mem.shape[0]}
    rows, cnt = np.unique(mem, axis=0, return_counts=True)
    return {"".join(map(str, r)): int(c) for r, c in zip(rows, cnt)}
