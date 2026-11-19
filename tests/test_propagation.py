"""Exact Pauli propagation must agree with Stim's tableau conjugation."""

import numpy as np
import pytest
import stim

import noisiq as nq
from noisiq.backends import IncompatibleCircuitError
from tests.conftest import random_circuit


def _stim_conjugate(circuit: nq.Circuit, pauli: str) -> str:
    sim = stim.TableauSimulator()
    sim.set_num_qubits(circuit.n_qubits)
    for _, inst in circuit.ordered():
        sim.do_tableau(inst.gate.tableau, list(inst.qubits))
    t = sim.current_inverse_tableau().inverse()
    out = t(stim.PauliString(pauli.replace("I", "_")))
    return str(out)[1:].replace("_", "I")


@pytest.mark.parametrize("seed", range(10))
def test_propagation_matches_stim(seed):
    rng = np.random.default_rng(seed)
    c = random_circuit(5, 30, rng)
    for _ in range(5):
        p = "".join(rng.choice(list("IXYZ"), 5))
        assert nq.analysis.conjugate_pauli(c, p) == _stim_conjugate(c, p)


def test_textbook_cnot_rules():
    c = nq.Circuit(2).cx(0, 1)
    assert nq.analysis.conjugate_pauli(c, "XI") == "XX"
    assert nq.analysis.conjugate_pauli(c, "IZ") == "ZZ"
    assert nq.analysis.conjugate_pauli(c, "ZI") == "ZI"
    assert nq.analysis.conjugate_pauli(c, "IX") == "IX"
    assert nq.analysis.conjugate_pauli(nq.Circuit(1).h(0), "X") == "Z"
    assert nq.analysis.conjugate_pauli(nq.Circuit(1).s(0), "X") == "Y"


def test_inject_after_instruction_and_layer():
    c = nq.algorithms.ghz(4)
    t = nq.propagate(c, "X", 0, after=0)
    assert t.pauli_strings() == ["IIII", "XIII", "XXII", "XXXI", "XXXX"]
    t2 = nq.propagate(c, "Z", 3, layer=3)
    assert t2.pauli_strings()[-1] == "IIIZ"
    t3 = nq.propagate(c, "XIII")
    assert t3.pauli_string(0) == "XIII"
    assert t3.events[0].kind == "injected"


def test_measurement_flips_are_reported():
    c = nq.Circuit(2).cx(0, 1).measure_all()
    t = nq.propagate(c, "X", 0)
    assert t.measurement_flips[0].tolist() == [True, True]
    t = nq.propagate(c, "Z", 0)
    assert t.measurement_flips[0].tolist() == [False, False]


def test_error_through_feedforward_correction():
    """In teleportation, an X error on the Bell-measured qubit flips its readout, the Z correction
    fires wrongly, and the teleported |+> state is corrupted (visible in an X-basis readout)."""
    c = nq.algorithms.teleportation(measure_basis="X")
    last_h = [i for i, inst in enumerate(c) if inst.name == "H"][-1]
    t = nq.propagate(c, "X", 0, after=last_h)
    assert t.measurement_flips[0].tolist() == [True, False, True]
    # The same error right before a Z readout of qubit 0 but as a Z error is harmless.
    t = nq.propagate(c, "Z", 0, after=last_h)
    assert t.measurement_flips[0].tolist() == [False, False, False]


def test_propagation_rejects_non_clifford_and_bad_args():
    with pytest.raises(IncompatibleCircuitError):
        nq.propagate(nq.Circuit(1).t(0), "X", 0)
    c = nq.Circuit(2).h(0)
    for bad in (
        dict(pauli="Q", qubit=0),
        dict(pauli="X", qubit=5),
        dict(pauli="XX", qubit=0),
        dict(pauli="X"),
        dict(pauli="X", qubit=0, after=0, layer=0),
        dict(pauli="X", qubit=0, after=9),
        dict(pauli="X", qubit=0, layer=9),
    ):
        with pytest.raises(ValueError):
            nq.propagate(c, **bad)


def test_fault_spread_weights():
    c = nq.algorithms.ghz(3, measure=True)
    w_inst, w_idle = nq.analysis.fault_spread(c)
    # X after the H on qubit 0 is copied to all three outputs; Z after H flips nothing measurable
    assert w_inst[0, 0].tolist() == [3, 3, 0]
    # X on qubit 2 after the last CX flips only qubit 2
    assert w_inst[2, 2, 0] == 1
    assert w_idle.shape == (c.depth, 3, 3)


def test_trace_helpers():
    t = nq.propagate(nq.algorithms.ghz(3), "X", 0, after=0)
    assert t.weight(t.n_layers) == 3
    assert t.first_faulty_shot() == 0
    assert "PropagationTrace" in repr(t)
