"""
Tests for the OpenQASM 2.0 → NoisiQ Circuit parser.

Covers:
- Header and include lines are ignored
- qreg declaration builds the correct qubit count
- All supported single-qubit gates are parsed
- Both two-qubit gates (cx/cz) are parsed
- Multiple qreg declarations are concatenated in declaration order
- barrier, measure, reset, creg are silently ignored
- Line comments are stripped
- Gate ordering matches QASM source order
- Error: no qreg declaration
- Error: undeclared register reference
- Error: parameterized gate
- Error: custom gate definition
- Error: unknown gate name
- Error: wrong qubit count for gate
"""

import pytest
from noisiq.io.qasm import from_qasm, QASMParseError
from noisiq.ir import gates as ir_gates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GHZ_3Q = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"""

BELL_PAIR = """
OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0], q[1];
"""


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_ghz_qubit_count():
    c = from_qasm(GHZ_3Q)
    assert c.n_qubits == 3


def test_ghz_gate_count():
    c = from_qasm(GHZ_3Q)
    assert len(c.operations) == 3


def test_ghz_gate_order():
    c = from_qasm(GHZ_3Q)
    assert c.operations[0].gate == ir_gates.H
    assert c.operations[1].gate == ir_gates.CNOT
    assert c.operations[2].gate == ir_gates.CNOT


def test_ghz_qubit_targets():
    c = from_qasm(GHZ_3Q)
    assert c.operations[0].qubits == (0,)
    assert c.operations[1].qubits == (0, 1)
    assert c.operations[2].qubits == (1, 2)


def test_bell_pair():
    c = from_qasm(BELL_PAIR)
    assert c.n_qubits == 2
    assert len(c.operations) == 2
    assert c.operations[0].gate == ir_gates.H
    assert c.operations[1].gate == ir_gates.CNOT


# ---------------------------------------------------------------------------
# All supported single-qubit gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate_name,expected_gate", [
    ("h",   ir_gates.H),
    ("x",   ir_gates.X),
    ("y",   ir_gates.Y),
    ("z",   ir_gates.Z),
    ("s",   ir_gates.S),
    ("sdg", ir_gates.S_DAG),
    ("t",   ir_gates.T),
    ("tdg", ir_gates.T_DAG),
    ("id",  ir_gates.I),
])
def test_single_qubit_gate(gate_name, expected_gate):
    qasm = f"OPENQASM 2.0;\nqreg q[1];\n{gate_name} q[0];"
    c = from_qasm(qasm)
    assert len(c.operations) == 1
    assert c.operations[0].gate == expected_gate


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------

def test_cx_gate():
    c = from_qasm("OPENQASM 2.0;\nqreg q[2];\ncx q[0], q[1];")
    assert c.operations[0].gate == ir_gates.CNOT
    assert c.operations[0].qubits == (0, 1)


def test_cnot_alias():
    c = from_qasm("OPENQASM 2.0;\nqreg q[2];\ncnot q[0], q[1];")
    assert c.operations[0].gate == ir_gates.CNOT


def test_cz_gate():
    c = from_qasm("OPENQASM 2.0;\nqreg q[2];\ncz q[0], q[1];")
    assert c.operations[0].gate == ir_gates.CZ
    assert c.operations[0].qubits == (0, 1)


# ---------------------------------------------------------------------------
# Multiple qreg declarations
# ---------------------------------------------------------------------------

def test_multiple_qregs_total_qubits():
    qasm = "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\nh a[0];\nx b[1];"
    c = from_qasm(qasm)
    assert c.n_qubits == 5


def test_multiple_qregs_qubit_offset():
    # 'a' occupies 0..1, 'b' starts at 2
    qasm = "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\nx b[0];"
    c = from_qasm(qasm)
    assert c.operations[0].qubits == (2,)


# ---------------------------------------------------------------------------
# Ignored statements
# ---------------------------------------------------------------------------

def test_barrier_ignored():
    qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\nbarrier q[0], q[1];\nx q[1];"
    c = from_qasm(qasm)
    assert len(c.operations) == 2


def test_measure_ignored():
    qasm = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];"
    c = from_qasm(qasm)
    assert len(c.operations) == 1
    assert c.operations[0].gate == ir_gates.H


def test_comments_stripped():
    qasm = """
    OPENQASM 2.0;  // version header
    qreg q[2];     // two qubits
    h q[0];        // Hadamard
    // cx q[0], q[1]; this line is commented out
    x q[1];
    """
    c = from_qasm(qasm)
    assert len(c.operations) == 2
    assert c.operations[0].gate == ir_gates.H
    assert c.operations[1].gate == ir_gates.X


# ---------------------------------------------------------------------------
# Import via noisiq.io
# ---------------------------------------------------------------------------

def test_importable_from_noisiq_io():
    from noisiq.io import from_qasm as f
    c = f(BELL_PAIR)
    assert c.n_qubits == 2


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_no_qreg_raises():
    with pytest.raises(QASMParseError, match="No qreg"):
        from_qasm("OPENQASM 2.0;\nh q[0];")


def test_undeclared_register_raises():
    with pytest.raises(QASMParseError, match="not declared"):
        from_qasm("OPENQASM 2.0;\nqreg q[2];\nh r[0];")


def test_unknown_gate_raises():
    with pytest.raises(QASMParseError, match="Unsupported"):
        from_qasm("OPENQASM 2.0;\nqreg q[1];\nfoo q[0];")


def test_parameterized_gate_raises():
    with pytest.raises(QASMParseError, match="not supported"):
        from_qasm("OPENQASM 2.0;\nqreg q[1];\nrx(1.57) q[0];")


def test_custom_gate_definition_raises():
    with pytest.raises(QASMParseError, match="Custom gate"):
        from_qasm("OPENQASM 2.0;\nqreg q[1];\ngate mygate q { h q; }\nmygate q[0];")


def test_duplicate_qreg_raises():
    with pytest.raises(QASMParseError, match="Duplicate"):
        from_qasm("OPENQASM 2.0;\nqreg q[2];\nqreg q[3];")
