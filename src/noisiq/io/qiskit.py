"""Adapters between NoisiQ and Qiskit circuits (``pip install noisiq[qiskit]``).

Qiskit orders qubits little-endian inside matrices; NoisiQ is big-endian. Gates
without a library equivalent are converted through their matrix with the qubit
list reversed, which makes the two conventions agree exactly.
"""

from __future__ import annotations

from ..ir import gates as G
from ..ir.circuit import Circuit
from ..ir.instructions import Barrier, Condition, Delay, Measure, Operation, Reset

__all__ = ["from_qiskit", "to_qiskit"]

_FROM = {
    "id": "I",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdg": "SDG",
    "t": "T",
    "tdg": "TDG",
    "sx": "SX",
    "sxdg": "SXDG",
    "cx": "CX",
    "cy": "CY",
    "cz": "CZ",
    "ch": "CH",
    "swap": "SWAP",
    "iswap": "ISWAP",
    "ccx": "CCX",
    "ccz": "CCZ",
    "cswap": "CSWAP",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "p": "P",
    "u1": "P",
    "u": "U",
    "u3": "U",
    "cp": "CP",
    "cu1": "CP",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
}
_TO = {v: k for k, v in _FROM.items() if k not in ("u1", "u3", "cu1")}


def _require():
    try:
        import qiskit  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Qiskit adapters need qiskit: pip install 'noisiq[qiskit]'") from exc


def from_qiskit(qc) -> Circuit:
    """Convert a ``qiskit.QuantumCircuit`` into a NoisiQ :class:`Circuit`.

    Supports standard gates, arbitrary unitary gates (via their matrix), measurement,
    reset, barrier, delay (with a time unit) and legacy ``c_if`` conditions.
    """
    _require()
    from qiskit.quantum_info import Operator

    out = Circuit(qc.num_qubits, qc.num_clbits, name=qc.name)
    qidx = {q: i for i, q in enumerate(qc.qubits)}
    cidx = {c: i for i, c in enumerate(qc.clbits)}
    for item in qc.data:
        op = item.operation
        qs = [qidx[q] for q in item.qubits]
        name = op.name
        cond = None
        legacy = getattr(op, "_condition", None)
        if legacy is not None:
            target, value = legacy
            bits = [cidx[target]] if target in cidx else [cidx[b] for b in target]
            cond = Condition(tuple(bits), int(value))
        if name == "measure":
            out.measure(qs[0], cidx[item.clbits[0]])
        elif name == "reset":
            out.reset(qs[0])
        elif name == "barrier":
            out.barrier(qs)
        elif name == "delay":
            unit = getattr(op, "unit", "dt")
            scale = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}.get(unit)
            if scale is None:
                raise ValueError("Delays in 'dt' units cannot be converted to seconds.")
            for q in qs:
                out.delay(q, float(op.duration) * scale)
        elif name in ("if_else", "while_loop", "for_loop", "switch_case"):
            raise ValueError(f"Control-flow instruction '{name}' is not supported; use c_if-style conditions.")
        elif name in _FROM:
            params = [float(p) for p in op.params]
            out.append(G.gate_from_name(_FROM[name], params), qs, condition=cond)
        else:
            try:
                m = Operator(op).data
            except Exception as exc:
                raise ValueError(f"Cannot convert Qiskit instruction '{name}'.") from exc
            out.append(G.unitary(m, name=name.upper()), list(reversed(qs)), condition=cond)
    return out


def to_qiskit(circuit: Circuit):
    """Convert a NoisiQ :class:`Circuit` into a ``qiskit.QuantumCircuit``."""
    _require()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    qr = QuantumRegister(circuit.n_qubits, "q")
    regs = [qr]
    cr = None
    if circuit.n_clbits:
        cr = ClassicalRegister(circuit.n_clbits, "c")
        regs.append(cr)
    qc = QuantumCircuit(*regs, name=circuit.name or "noisiq")

    def emit(inst: Operation):
        g = inst.gate
        qs = [qr[q] for q in inst.qubits]
        if g.name in _TO and hasattr(qc, _TO[g.name]):
            getattr(qc, _TO[g.name])(*g.params, *qs)
        else:
            qc.append(UnitaryGate(g.matrix.copy(), label=g.name), list(reversed(qs)))

    for _, inst in circuit.ordered():
        if isinstance(inst, Operation):
            if inst.condition is None:
                emit(inst)
            else:
                if len(inst.condition.clbits) != 1:
                    raise ValueError("to_qiskit supports single-bit conditions only.")
                with qc.if_test((cr[inst.condition.clbits[0]], bool(inst.condition.value))):
                    emit(inst)
        elif isinstance(inst, Measure):
            q = qr[inst.qubit]
            if inst.basis == "X":
                qc.h(q)
            elif inst.basis == "Y":
                qc.sdg(q)
                qc.h(q)
            qc.measure(q, cr[inst.clbit])
            if inst.reset:
                qc.reset(q)
            elif inst.basis == "X":
                qc.h(q)
            elif inst.basis == "Y":
                qc.h(q)
                qc.s(q)
        elif isinstance(inst, Reset):
            qc.reset(qr[inst.qubit])
        elif isinstance(inst, Barrier):
            qc.barrier(*[qr[q] for q in inst.qubits])
        elif isinstance(inst, Delay):
            qc.delay(inst.duration, qr[inst.qubit], unit="s")
    return qc
