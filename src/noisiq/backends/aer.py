"""Optional Qiskit Aer backend (``pip install noisiq[qiskit]``).

Translates the compiled program instruction-by-instruction: gates become
``UnitaryGate`` objects and every noise location becomes an explicit ``Kraus``
instruction, so the physics is identical to NoiseQ's own backends — useful for
cross-validation. Qiskit is little-endian, so qubit lists are reversed when
appending matrices that NoisiQ stores big-endian.
"""

from __future__ import annotations

import numpy as np

from ..ir.circuit import Circuit
from ..noise.model import NoiseModel
from ..results.result import Result
from .frame import IncompatibleCircuitError
from .program import DetuneStep, FaultStep, GateStep, MeasureStep, Program, ResetStep, compile_program

__all__ = ["AerBackend"]


def _require_qiskit():
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("The Aer backend needs qiskit and qiskit-aer: pip install 'noisiq[qiskit]'") from exc


class AerBackend:
    """Run NoisiQ circuits on Qiskit Aer (density-matrix method for ≤ 12 qubits)."""

    name = "aer"

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
        _require_qiskit()
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import Kraus
        from qiskit_aer import AerSimulator

        program = program or compile_program(circuit, noise)
        if program.has_detuning:
            raise IncompatibleCircuitError("Aer cannot represent quasi-static detuning; use backend='statevector'.")
        n, nc = program.n_qubits, program.n_clbits
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(max(nc, 1), "c")
        qc = QuantumCircuit(qr, cr)

        def rev(qs):
            return [qr[q] for q in reversed(qs)]

        def cond_ctx(cond):
            if cond is None:
                return None
            if len(cond.clbits) != 1:
                raise IncompatibleCircuitError("The Aer backend supports single-bit conditions only.")
            return qc.if_test((cr[cond.clbits[0]], bool(cond.value)))

        for step in program.steps:
            if isinstance(step, GateStep):
                ctx = cond_ctx(step.condition)
                # copies: Aer misbehaves with read-only numpy arrays
                gate = UnitaryGate(np.array(step.gate.matrix, copy=True))
                if ctx is None:
                    qc.append(gate, rev(step.qubits))
                else:
                    with ctx:
                        qc.append(gate, rev(step.qubits))
            elif isinstance(step, FaultStep):
                instr = Kraus([np.array(K, copy=True) for K in step.channel.kraus()])
                ctx = cond_ctx(step.condition)
                if ctx is None:
                    qc.append(instr, rev(step.qubits))
                else:
                    with ctx:
                        qc.append(instr, rev(step.qubits))
            elif isinstance(step, MeasureStep):
                q = qr[step.qubit]
                if step.basis == "X":
                    qc.h(q)
                elif step.basis == "Y":
                    qc.sdg(q)
                    qc.h(q)
                qc.measure(q, cr[step.clbit])
                if step.reset:
                    qc.reset(q)
                elif step.basis == "X":
                    qc.h(q)
                elif step.basis == "Y":
                    qc.h(q)
                    qc.s(q)
                # readout errors are applied classically after sampling (see below)
            elif isinstance(step, ResetStep):
                qc.reset(qr[step.qubit])
            elif isinstance(step, DetuneStep):  # pragma: no cover
                raise IncompatibleCircuitError("detuning")

        method = "density_matrix" if n <= 12 else "statevector"
        sim = AerSimulator(method=method, seed_simulator=seed)
        res = sim.run(qc, shots=shots, memory=True).result()
        mem = res.get_memory()
        memory = np.array([[int(b) for b in reversed(m.replace(" ", ""))][:nc] for m in mem], dtype=np.uint8).reshape(
            shots, nc
        )
        rng = np.random.default_rng(seed)
        # Readout errors (classical) are applied in post-processing, per measurement step.
        ro = {}
        for step in program.steps:
            if isinstance(step, MeasureStep) and step.readout is not None:
                ro[step.clbit] = step.readout
        if ro and any(isinstance(s, GateStep) and s.condition for s in program.steps):
            raise IncompatibleCircuitError("Aer backend: readout errors combined with feed-forward are not supported.")
        for c, err in ro.items():
            true = memory[:, c].copy()
            p = np.where(true == 1, err.p10, err.p01)
            memory[:, c] = true ^ (rng.random(shots) < p)
        from .frame import _counts_from_memory

        return Result(
            circuit=circuit,
            backend=self.name,
            shots=shots,
            seed=seed,
            counts=_counts_from_memory(memory),
            memory=memory,
            implicit_measurement=program.implicit_measurement,
            noise_name=noise.name if noise else "noiseless",
            metadata={"aer_method": method},
        )
