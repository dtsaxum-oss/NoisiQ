"""
M5 tests — Classical IR integration.

Covers:
  - ClassicalBit / ClassicalRegister construction and indexing
  - Measurement dataclass (t field required, properties)
  - ConditionalOp properties (.t and .qubits forward to inner)
  - Circuit.add_classical_register() — allocation and duplicate guard
  - Circuit.measure() — scheduling and out-of-bounds guard
  - Circuit.c_if() — ConditionalOp wrapping
  - Circuit.validate() — qubit range checks for Measurement and ConditionalOp
  - Circuit.add_gate() scheduling with mixed op types
  - _compute_wire_segments does not crash on Measurement ops
  - QASM creg + measure round-trip
  - QASM if statement round-trip
"""

import pytest

from noisiq.ir import Circuit, gates
from noisiq.ir.circuit import Operation, _get_qubits
from noisiq.ir.classical import (
    ClassicalBit,
    ClassicalRegister,
    ConditionalOp,
    Measurement,
)


# ===========================================================================
# ClassicalBit
# ===========================================================================

def test_classical_bit_fields():
    b = ClassicalBit(name="m0", index=3)
    assert b.name == "m0"
    assert b.index == 3


def test_classical_bit_is_frozen():
    b = ClassicalBit(name="m0", index=0)
    with pytest.raises((AttributeError, TypeError)):
        b.index = 1  # type: ignore[misc]


# ===========================================================================
# ClassicalRegister
# ===========================================================================

def test_classical_register_fields():
    reg = ClassicalRegister(name="c", size=4, offset=0)
    assert reg.name == "c"
    assert reg.size == 4
    assert reg.offset == 0


def test_classical_register_getitem_valid():
    reg = ClassicalRegister(name="c", size=3, offset=10)
    bit = reg[0]
    assert bit.name == "c[0]" and bit.index == 10
    bit2 = reg[2]
    assert bit2.name == "c[2]" and bit2.index == 12


def test_classical_register_getitem_out_of_range():
    reg = ClassicalRegister(name="c", size=2, offset=0)
    with pytest.raises(IndexError, match="out of range"):
        _ = reg[2]
    with pytest.raises(IndexError):
        _ = reg[-1]


def test_classical_register_offset():
    reg = ClassicalRegister(name="s", size=2, offset=5)
    assert reg[0].index == 5
    assert reg[1].index == 6


# ===========================================================================
# Measurement
# ===========================================================================

def test_measurement_has_t_field():
    reg = ClassicalRegister("c", 1)
    m = Measurement(qubit=0, cbit=reg[0], t=3)
    assert m.t == 3


def test_measurement_fields():
    reg = ClassicalRegister("c", 2)
    m = Measurement(qubit=1, cbit=reg[1], t=0, basis="Z", reset=True)
    assert m.qubit == 1
    assert m.cbit == reg[1]
    assert m.basis == "Z"
    assert m.reset is True


def test_measurement_is_frozen():
    reg = ClassicalRegister("c", 1)
    m = Measurement(qubit=0, cbit=reg[0], t=0)
    with pytest.raises((AttributeError, TypeError)):
        m.t = 5  # type: ignore[misc]


# ===========================================================================
# ConditionalOp
# ===========================================================================

def test_conditional_op_t_property():
    reg = ClassicalRegister("c", 1)
    inner = Operation(gate=gates.X, qubits=(0,), t=7)
    cond = ConditionalOp(inner=inner, condition=reg[0], value=1)
    assert cond.t == 7


def test_conditional_op_qubits_property():
    reg = ClassicalRegister("c", 1)
    inner = Operation(gate=gates.CNOT, qubits=(0, 1), t=2)
    cond = ConditionalOp(inner=inner, condition=reg[0], value=1)
    assert cond.qubits == (0, 1)


def test_conditional_op_fields():
    reg = ClassicalRegister("c", 1)
    bit = reg[0]
    inner = Operation(gate=gates.Z, qubits=(1,), t=4)
    cond = ConditionalOp(inner=inner, condition=bit, value=0)
    assert cond.condition is bit
    assert cond.value == 0


# ===========================================================================
# _get_qubits helper
# ===========================================================================

def test_get_qubits_operation():
    op = Operation(gate=gates.CNOT, qubits=(0, 2), t=0)
    assert _get_qubits(op) == (0, 2)


def test_get_qubits_measurement():
    reg = ClassicalRegister("c", 1)
    m = Measurement(qubit=3, cbit=reg[0], t=0)
    assert _get_qubits(m) == (3,)


def test_get_qubits_conditional_op():
    reg = ClassicalRegister("c", 1)
    inner = Operation(gate=gates.H, qubits=(1,), t=0)
    cond = ConditionalOp(inner=inner, condition=reg[0])
    assert _get_qubits(cond) == (1,)


# ===========================================================================
# Circuit.add_classical_register
# ===========================================================================

def test_add_classical_register_basic():
    c = Circuit(3)
    reg = c.add_classical_register("syndrome", 2)
    assert isinstance(reg, ClassicalRegister)
    assert reg.name == "syndrome"
    assert reg.size == 2
    assert reg.offset == 0
    assert c.classical_registers == [reg]


def test_add_classical_register_offsets_accumulate():
    c = Circuit(4)
    r0 = c.add_classical_register("a", 2)
    r1 = c.add_classical_register("b", 3)
    assert r0.offset == 0
    assert r1.offset == 2


def test_add_classical_register_duplicate_raises():
    c = Circuit(2)
    c.add_classical_register("c", 2)
    with pytest.raises(ValueError, match="already exists"):
        c.add_classical_register("c", 1)


# ===========================================================================
# Circuit.measure
# ===========================================================================

def test_measure_appends_measurement_op():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.measure(0, reg[0])
    assert len(c.operations) == 1
    m = c.operations[0]
    assert isinstance(m, Measurement)
    assert m.qubit == 0 and m.cbit == reg[0]


def test_measure_scheduling_after_gate():
    c = Circuit(2)
    reg = c.add_classical_register("c", 2)
    c.h(0)                 # t=0 on q0
    c.cnot(0, 1)           # t=1 on q0,q1
    c.measure(0, reg[0])   # q0 last at t=1 → measure at t=2
    m = c.operations[-1]
    assert isinstance(m, Measurement)
    assert m.t == 2


def test_measure_scheduling_empty_circuit():
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.measure(0, reg[0])
    assert c.operations[0].t == 0


def test_measure_out_of_bounds_raises():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    with pytest.raises(ValueError, match="out of bounds"):
        c.measure(5, reg[0])


def test_measure_returns_circuit_for_chaining():
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    result = c.h(0).measure(0, reg[0])
    assert result is c


def test_measure_scheduling_accounts_for_earlier_measurement():
    c = Circuit(1)
    reg = c.add_classical_register("c", 2)
    c.h(0)
    c.measure(0, reg[0])   # t=1
    c.measure(0, reg[1])   # t=2 (accounts for first measurement)
    assert c.operations[-1].t == 2


# ===========================================================================
# Circuit.c_if
# ===========================================================================

def test_c_if_wraps_gate_in_conditional_op():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.h(0).cnot(0, 1).measure(0, reg[0])
    c.c_if(reg[0], value=1).x(1)
    last = c.operations[-1]
    assert isinstance(last, ConditionalOp)
    assert last.condition == reg[0]
    assert last.value == 1
    assert last.inner.gate == gates.X
    assert last.inner.qubits == (1,)


def test_c_if_default_value_is_1():
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.h(0).measure(0, reg[0])
    c.c_if(reg[0]).z(0)
    cond = c.operations[-1]
    assert isinstance(cond, ConditionalOp)
    assert cond.value == 1


def test_c_if_schedules_after_qubits():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.h(0)            # t=0 q0
    c.cnot(0, 1)      # t=1 q0,q1
    c.measure(0, reg[0])   # t=2 q0
    c.c_if(reg[0]).x(1)   # q1 last at t=1 → conditional X at t=2
    cond = c.operations[-1]
    assert isinstance(cond, ConditionalOp)
    assert cond.t == 2


def test_c_if_value_zero():
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.h(0).measure(0, reg[0])
    c.c_if(reg[0], value=0).z(0)
    cond = c.operations[-1]
    assert cond.value == 0


def test_c_if_builder_returns_circuit():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    result = c.c_if(reg[0]).x(0)
    assert result is c


# ===========================================================================
# Circuit.validate with Measurement and ConditionalOp
# ===========================================================================

def test_validate_passes_with_measurement():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.h(0).measure(0, reg[0])
    c.validate()  # must not raise


def test_validate_passes_with_conditional_op():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.h(0).cnot(0, 1).measure(0, reg[0])
    c.c_if(reg[0]).x(1)
    c.validate()  # must not raise


def test_validate_catches_measurement_qubit_out_of_range():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.h(0)
    # Force a bad qubit directly (bypass measure() bounds check)
    bad_m = Measurement(qubit=99, cbit=reg[0], t=1)
    c.operations.append(bad_m)
    with pytest.raises(ValueError, match="qubit index out of range|out of bounds"):
        c.validate()


def test_validate_catches_conditional_op_qubit_out_of_range():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    inner = Operation(gate=gates.X, qubits=(99,), t=0)
    c.operations.append(ConditionalOp(inner=inner, condition=reg[0]))
    with pytest.raises(ValueError, match="[Qq]ubit index out of range"):
        c.validate()


def test_validate_catches_measurement_negative_t():
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.operations.append(Measurement(qubit=0, cbit=reg[0], t=-1))
    with pytest.raises(ValueError, match="t must be >= 0"):
        c.validate()


# ===========================================================================
# add_gate scheduling with mixed op types
# ===========================================================================

def test_add_gate_after_measurement_respects_qubit():
    """A gate placed after a Measurement on the same qubit must land at t >= m.t+1."""
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.measure(0, reg[0])  # t=0
    c.h(0)                # must land at t=1
    gate_op = c.operations[-1]
    assert isinstance(gate_op, Operation)
    assert gate_op.t == 1


def test_add_gate_collision_with_measurement_raises():
    """Placing a gate at the same t as an existing Measurement must raise."""
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.measure(0, reg[0], t=0)  # explicit t=0
    with pytest.raises(ValueError, match="already used"):
        c.h(0, t=0)  # same t, same qubit → collision


# ===========================================================================
# _compute_wire_segments guard
# ===========================================================================

def test_wire_segments_no_crash_on_measurement():
    from noisiq.visualization.charts.heatmap import _compute_wire_segments
    c = Circuit(2)
    reg = c.add_classical_register("c", 1)
    c.h(0).cnot(0, 1).measure(0, reg[0])
    segs = _compute_wire_segments(c, None)
    assert isinstance(segs, list)  # no crash; empty list expected (no IDLE ops)


# ===========================================================================
# QASM round-trip tests (creg / measure / if)
# ===========================================================================

def test_qasm_creg_populates_classical_registers():
    from noisiq.io.qasm import from_qasm
    qasm = "OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh q[0];"
    c = from_qasm(qasm)
    assert len(c.classical_registers) == 1
    assert c.classical_registers[0].name == "c"
    assert c.classical_registers[0].size == 2


def test_qasm_measure_creates_measurement_op():
    from noisiq.io.qasm import from_qasm
    qasm = "OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];"
    c = from_qasm(qasm)
    assert len(c.operations) == 2
    m = c.operations[1]
    assert isinstance(m, Measurement)
    assert m.qubit == 0
    assert m.cbit.index == 0


def test_qasm_multiple_measures():
    from noisiq.io.qasm import from_qasm
    qasm = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    c = from_qasm(qasm)
    measurements = [op for op in c.operations if isinstance(op, Measurement)]
    assert len(measurements) == 2
    assert measurements[0].qubit == 0
    assert measurements[1].qubit == 1


def test_qasm_if_creates_conditional_op():
    from noisiq.io.qasm import from_qasm
    qasm = """
    OPENQASM 2.0;
    qreg q[1];
    creg s[1];
    h q[0];
    measure q[0] -> s[0];
    if (s==1) x q[0];
    """
    c = from_qasm(qasm)
    cond_ops = [op for op in c.operations if isinstance(op, ConditionalOp)]
    assert len(cond_ops) == 1
    cond = cond_ops[0]
    assert cond.value == 1
    assert cond.inner.gate == gates.X
    assert cond.condition.name == "s[0]"


def test_qasm_if_value_zero():
    from noisiq.io.qasm import from_qasm
    qasm = """
    OPENQASM 2.0;
    qreg q[1];
    creg s[1];
    h q[0];
    measure q[0] -> s[0];
    if (s==0) z q[0];
    """
    c = from_qasm(qasm)
    cond = next(op for op in c.operations if isinstance(op, ConditionalOp))
    assert cond.value == 0
    assert cond.inner.gate == gates.Z


def test_qasm_if_multi_bit_creg_raises():
    from noisiq.io.qasm import from_qasm, QASMParseError
    qasm = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    h q[0];
    if (c==1) x q[1];
    """
    with pytest.raises(QASMParseError, match="multi-bit"):
        from_qasm(qasm)


def test_qasm_measure_undeclared_creg_raises():
    from noisiq.io.qasm import from_qasm, QASMParseError
    qasm = "OPENQASM 2.0;\nqreg q[1];\nmeasure q[0] -> c[0];"
    with pytest.raises(QASMParseError, match="[Cc]lassical register"):
        from_qasm(qasm)


# ===========================================================================
# IR validation — new Phase 7 guards
# ===========================================================================

def test_add_classical_register_zero_size_raises():
    c = Circuit(1)
    with pytest.raises(ValueError, match="size must be >= 1"):
        c.add_classical_register("c", 0)


def test_add_classical_register_negative_size_raises():
    c = Circuit(1)
    with pytest.raises(ValueError, match="size must be >= 1"):
        c.add_classical_register("c", -3)


def test_measure_unowned_cbit_raises():
    """measure() rejects a ClassicalBit not allocated by the circuit."""
    c = Circuit(1)
    c.add_classical_register("c", 1)
    foreign_bit = ClassicalBit(name="foreign", index=99)
    with pytest.raises(ValueError, match="not allocated"):
        c.measure(0, foreign_bit)


def test_measure_owned_cbit_passes():
    """measure() accepts a bit that came from the circuit's own register."""
    c = Circuit(1)
    reg = c.add_classical_register("c", 2)
    c.h(0)
    c.measure(0, reg[1])  # must not raise
    assert len(c.operations) == 2


def test_measure_explicit_t_collision_raises():
    """measure() with explicit t= must reject collisions like add_gate does."""
    c = Circuit(1)
    reg = c.add_classical_register("c", 2)
    c.measure(0, reg[0], t=0)
    with pytest.raises(ValueError, match="already used"):
        c.measure(0, reg[1], t=0)


def test_measure_explicit_t_no_collision_passes():
    c = Circuit(2)
    reg = c.add_classical_register("c", 2)
    c.measure(0, reg[0], t=0)
    c.measure(1, reg[1], t=0)  # different qubit — must not raise
    assert len(c.operations) == 2


def test_c_if_unowned_condition_raises():
    """c_if() rejects a condition ClassicalBit not allocated by the circuit."""
    c = Circuit(1)
    c.add_classical_register("c", 1)
    foreign_bit = ClassicalBit(name="foreign", index=99)
    with pytest.raises(ValueError, match="not allocated"):
        c.c_if(foreign_bit).x(0)


def test_c_if_owned_condition_passes():
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.h(0)
    c.measure(0, reg[0])
    c.c_if(reg[0]).x(0)  # must not raise
    assert any(isinstance(op, ConditionalOp) for op in c.operations)


def test_c_if_explicit_t_collision_raises():
    """c_if() with explicit t= must detect qubit collision."""
    c = Circuit(1)
    reg = c.add_classical_register("c", 1)
    c.h(0, t=0)
    with pytest.raises(ValueError, match="already used"):
        c.c_if(reg[0]).x(0, t=0)
