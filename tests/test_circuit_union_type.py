"""
Phase 2 tests — CircuitOp union type annotation in ir/circuit.py.
"""

from noisiq.ir import Circuit, gates
from noisiq.ir.circuit import Operation, CircuitOp


def test_circuitop_alias_is_importable():
    assert CircuitOp is not None


def test_operations_list_accepts_non_operation_objects():
    """operations is typed List[CircuitOp]; the runtime list accepts any object."""
    c = Circuit(1)
    c.h(0)
    sentinel = object()
    c.operations.append(sentinel)
    assert sentinel in c.operations
    c.operations.pop()


def test_circuit_validate_still_passes_on_pure_operation_circuit():
    c = Circuit(2)
    c.h(0).cnot(0, 1)
    c.validate()  # must not raise


def test_operations_field_default_is_empty_list():
    c = Circuit(3)
    assert c.operations == []
