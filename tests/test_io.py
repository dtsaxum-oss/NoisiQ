import numpy as np
import pytest

import noisiq as nq
from noisiq.io import QASMExportError, QASMParseError, from_qasm, to_qasm
from tests.conftest import random_circuit, tvd

QASM = """
OPENQASM 2.0;
include "qelib1.inc";
// a comment
gate bell a, b { h a; cx a, b; }
gate myrot(theta) q { rz(theta/2) q; rx(-theta) q; }
qreg q[3];
qreg anc[1];
creg c[3];
creg flag[1];
bell q[0], q[1];
myrot(pi/3) q[2];
u3(0.1, -pi/2, 2*pi) anc[0];
u2(0, pi) q[2];
cu1(pi/4) q[0], anc[0];
barrier q;
measure anc[0] -> flag[0];
if (flag == 1) x q[2];
reset anc[0];
measure q -> c;
"""


def test_parse_rich_program():
    c = from_qasm(QASM)
    assert c.n_qubits == 4 and c.n_clbits == 4
    assert [r.name for r in c.registers] == ["c", "flag"]
    names = [i.name for i in c]
    assert names.count("MEASURE") == 4 and "RESET" in names and "BARRIER" in names
    cond = [i for i in c if getattr(i, "condition", None)]
    assert cond[0].condition.clbits == (3,) and cond[0].condition.value == 1
    rz = [i for i in c if i.name == "RZ"][0]
    assert rz.gate.params[0] == pytest.approx(np.pi / 6)


@pytest.mark.parametrize(
    "bad",
    [
        "qreg q[1]; foo q[0];",
        "qreg q[1]; h q[3];",
        "qreg q[1]; rx q[0];",
        "creg c[1];",
        "qreg q[1]; qreg q[2];",
        "qreg q[2]; cx q[0], q[0];",
        "qreg q[1]; opaque g q;",
        "qreg q[1]; rx(pi + foo) q[0];",
        "qreg q[2]; creg c[1]; measure q -> c;",
    ],
)
def test_parse_errors(bad):
    with pytest.raises(QASMParseError):
        from_qasm(bad)


@pytest.mark.parametrize("seed", range(5))
def test_round_trip_random_circuits(seed):
    rng = np.random.default_rng(seed)
    c = random_circuit(3, 20, rng, clifford=True, mid_measure=True, conditionals=True, resets=True, bases=True)
    # custom (random Clifford) gates have no QASM name: replace by exporting only library gates
    c2 = nq.Circuit(c.n_qubits, c.n_clbits)
    for _, inst in c.ordered():
        if inst.name in ("RC1", "RC2"):
            continue
        c2._append_instruction(inst, list(range(c.n_qubits)), list(range(c.n_clbits)), False)
    back = from_qasm(to_qasm(c2))
    p1 = nq.simulate(c2, shots=1, backend="density_matrix").probabilities()
    p2 = nq.simulate(back, shots=1, backend="density_matrix").probabilities()
    assert tvd(p1, p2) < 1e-9


def test_round_trip_parametric_and_named_registers():
    c = nq.Circuit(3).rx(0.123456789, 0).u(1, 2, 3, 1).cp(0.7, 0, 2).crx(0.4, 1, 2).cry(0.2, 2, 0).ccz(0, 1, 2)
    s = c.add_register("syn", 2)
    c.measure(0, s[0]).measure(1, s[1])
    c.c_if(s, 2).z(2)
    text = to_qasm(c)
    assert "creg syn[2];" in text and "if(syn==2)" in text
    back = from_qasm(text)
    p1 = nq.simulate(c, shots=1, backend="density_matrix").probabilities()
    p2 = nq.simulate(back, shots=1, backend="density_matrix").probabilities()
    assert tvd(p1, p2) < 1e-9


def test_export_errors():
    with pytest.raises(QASMExportError):
        to_qasm(nq.Circuit(1).unitary(np.eye(2), [0], name="CUSTOM"))
    with pytest.raises(QASMExportError):
        to_qasm(nq.Circuit(1).delay(0, 1e-6))
    c = nq.Circuit(2).measure(0).measure(1)
    c.c_if([0, 1], 3).x(0)
    with pytest.raises(QASMExportError):
        to_qasm(c)


def test_circuit_convenience_methods():
    c = nq.Circuit(2).h(0).cx(0, 1)
    assert nq.Circuit.from_qasm(c.to_qasm()) == c


# ----------------------------------------------------------------------------- qiskit


@pytest.mark.qiskit
@pytest.mark.parametrize("seed", range(4))
def test_qiskit_round_trip_preserves_unitary(seed):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(seed)
    c = random_circuit(3, 15, rng, clifford=False)
    qc = nq.io.to_qiskit(c)
    back = nq.io.from_qiskit(qc)
    u1 = Operator(qc).data
    u2 = Operator(nq.io.to_qiskit(back)).data
    assert np.allclose(u1, u2)
    p1 = nq.simulate(c, shots=1, backend="density_matrix").probabilities()
    p2 = nq.simulate(back, shots=1, backend="density_matrix").probabilities()
    assert tvd(p1, p2) < 1e-9


@pytest.mark.qiskit
def test_from_qiskit_library_circuit():
    pytest.importorskip("qiskit")
    from qiskit.circuit.library import QFT

    qc = QFT(3).decompose()
    c = nq.io.from_qiskit(qc)
    from qiskit.quantum_info import Statevector

    sv = Statevector(qc).probabilities_dict()
    ours = nq.simulate(c, shots=1, backend="density_matrix").probabilities()
    theirs = {k[::-1]: v for k, v in sv.items()}
    assert tvd(ours, theirs) < 1e-9


@pytest.mark.qiskit
def test_qiskit_measure_reset_condition():
    pytest.importorskip("qiskit")
    c = nq.algorithms.teleportation()
    qc = nq.io.to_qiskit(c)
    assert qc.num_clbits == 3
    c2 = nq.Circuit(2).h(0).measure(0).reset(0).barrier().delay(1, 1e-6)
    back = nq.io.from_qiskit(nq.io.to_qiskit(c2))
    assert back.count_ops() == c2.count_ops()
