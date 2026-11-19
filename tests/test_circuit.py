import numpy as np
import pytest

import noisiq as nq
from noisiq.ir import Barrier, Condition, Measure, Operation, Reset
from noisiq.ir import gates as G
from tests.conftest import random_circuit


def test_builder_chaining_and_asap_scheduling():
    c = nq.Circuit(3).h(0).cx(0, 1).cx(1, 2).h(0)
    layers = [i.layer for i in c]
    assert layers == [0, 1, 2, 2]
    assert c.depth == 3
    assert c.count_ops() == {"H": 2, "CX": 2}


def test_broadcast_single_qubit_builders():
    c = nq.Circuit(4).h(range(4)).rz(0.3, [1, 2])
    assert c.count_ops() == {"H": 4, "RZ": 2}
    assert c.depth == 2


def test_explicit_layers_and_collisions():
    c = nq.Circuit(2).x(0, layer=5)
    assert c[0].layer == 5
    with pytest.raises(ValueError, match="busy"):
        c.y(0, layer=5)
    c.y(1, layer=5)
    assert c.depth == 6


@pytest.mark.parametrize(
    "bad",
    [
        lambda c: c.cx(0, 0),
        lambda c: c.h(5),
        lambda c: c.h(-1),
        lambda c: c.append(G.CX, [0]),
        lambda c: c.append("H", [0]),
        lambda c: c.measure(0, 7),
        lambda c: c.c_if(3).x(0),
    ],
)
def test_invalid_operations_raise(bad):
    c = nq.Circuit(3)
    with pytest.raises((ValueError, TypeError)):
        bad(c)


def test_circuit_size_validation():
    with pytest.raises(ValueError):
        nq.Circuit(0)


def test_measure_allocates_clbits_and_bases():
    c = nq.Circuit(2)
    c.h(0).measure(0).measure(1, basis="X", reset=True)
    assert c.n_clbits == 2
    m = [i for i in c if isinstance(i, Measure)]
    assert m[1].basis == "X" and m[1].reset
    with pytest.raises(ValueError):
        c.measure(0, basis="W")


def test_measure_all_uses_fresh_bits_and_barrier():
    c = nq.Circuit(3).h(0).measure_all()
    assert c.n_clbits == 3
    assert any(isinstance(i, Barrier) for i in c)
    assert {i.layer for i in c if isinstance(i, Measure)} == {1}


def test_conditionals_any_gate_and_scheduling_after_measure():
    c = nq.Circuit(2)
    c.h(0).measure(0)
    c.c_if(0).rx(0.3, 1)
    c.c_if(0, 0).t(1)
    ops = [i for i in c if isinstance(i, Operation) and i.condition]
    assert ops[0].layer > c[1].layer  # after the measurement that writes the bit
    assert ops[1].condition == Condition((0,), 0)


def test_register_conditions():
    c = nq.Circuit(3)
    s = c.add_register("s", 2)
    c.measure(0, s[0]).measure(1, s[1])
    c.c_if(s, 0b10).x(2)
    op = c[-1]
    assert op.condition.clbits == (0, 1) and op.condition.value == 2
    assert op.condition.is_met([0, 1]) and not op.condition.is_met([1, 0])
    with pytest.raises(ValueError):
        Condition((0, 1), 4)


def test_inverse_composes_to_identity(rng):
    from noisiq.backends.statevector import apply_matrix

    for _ in range(5):
        c = random_circuit(3, 15, rng, clifford=False)
        full = c.copy().compose(c.inverse())
        psi = np.zeros(8, complex)
        psi[0] = 1
        for _, inst in full.ordered():
            psi = apply_matrix(psi, inst.gate.matrix, inst.qubits, 3)
        assert abs(psi[0]) == pytest.approx(1.0, abs=1e-9)


def test_inverse_rejects_non_unitary_parts():
    c = nq.Circuit(1).h(0).measure(0)
    with pytest.raises(ValueError):
        c.inverse()
    c = nq.Circuit(1).reset(0)
    with pytest.raises(ValueError):
        c.inverse()


def test_compose_with_qubit_map_and_add():
    a = nq.Circuit(3).h(0)
    b = nq.Circuit(2).cx(0, 1)
    a.compose(b, qubits=[2, 0])
    assert a[-1].qubits == (2, 0)
    s = nq.Circuit(2).h(0) + nq.Circuit(2).x(1)
    assert s.count_ops() == {"H": 1, "X": 1}
    with pytest.raises(ValueError):
        nq.Circuit(2) + nq.Circuit(3)


def test_copy_is_independent():
    a = nq.Circuit(2).h(0)
    b = a.copy()
    b.x(1)
    assert len(a) == 1 and len(b) == 2
    assert a != b and a == a.copy()


def test_reset_delay_barrier_and_layers():
    c = nq.Circuit(2).h(0).reset(1).delay(0, 1e-6).barrier().cx(0, 1)
    kinds = [type(i).__name__ for i in c]
    assert kinds == ["Operation", "Reset", "Delay", "Barrier", "Operation"]
    assert c[-1].layer == 2
    assert len(c.layers()) == 3
    with pytest.raises(ValueError):
        nq.Circuit(1).delay(0, -1)


def test_is_clifford_and_repr():
    assert nq.Circuit(2).h(0).cx(0, 1).is_clifford()
    assert not nq.Circuit(1).t(0).is_clifford()
    assert "Circuit(" in repr(nq.Circuit(2).h(0))
    assert "H" in str(nq.Circuit(2).h(0))


def test_mcx_and_unitary_builders():
    c = nq.Circuit(4).mcx([0, 1, 2], 3).unitary(G.H.matrix, [1], name="myH")
    assert c[0].gate.num_controls == 3
    assert c[1].gate.name == "myH"


def test_repeat_and_without_measurements():
    c = nq.Circuit(1).h(0).measure(0)
    r = nq.Circuit(1).h(0).repeat(3)
    assert r.count_ops() == {"H": 3}
    assert c.without_measurements().count_ops() == {"H": 1}


def test_ordered_respects_layers_over_insertion():
    c = nq.Circuit(2).x(0, layer=3).h(1, layer=0)
    assert [idx for idx, _ in c.ordered()] == [1, 0]


def test_reset_instruction_type():
    c = nq.Circuit(1).reset(0)
    assert isinstance(c[0], Reset) and c[0].qubits == (0,)
