"""
Gate conjugation identity tests for PauliFrame.

Verifies that the PauliFrame tracker correctly propagates errors through
Clifford gates using the fundamental identities:
    H X H = Z
    H Z H = X
    S X S† = Y
    CNOT spreads X forward and Z backward
"""

import pytest
from noisiq.visualization.pauli_frame_tracker import PauliFrame


# ---------------------------------------------------------------------------
# Single-qubit conjugation identities
# ---------------------------------------------------------------------------

def _propagate(initial_pauli: str, gates: list[str]) -> str:
    """
    Inject *initial_pauli* on qubit 0, apply *gates* in order, return final Pauli.
    """
    frame = PauliFrame(num_qubits=1)
    frame.inject_error(0, initial_pauli)
    for gate in gates:
        frame.apply_gate(gate, [0])
    return frame.get_pauli_string()[0]


def test_hxh_equals_z():
    assert _propagate("X", ["H", "H"]) == "X"   # H H = I sanity check
    frame = PauliFrame(1)
    frame.inject_error(0, "X")
    frame.apply_h(0)
    assert frame.get_pauli_string() == "Z"


def test_hzh_equals_x():
    frame = PauliFrame(1)
    frame.inject_error(0, "Z")
    frame.apply_h(0)
    assert frame.get_pauli_string() == "X"


def test_hyh_equals_y():
    # H Y H = -Y  (phase ignored by PauliFrame, so Y stays Y)
    frame = PauliFrame(1)
    frame.inject_error(0, "Y")
    frame.apply_h(0)
    assert frame.get_pauli_string() == "Y"


def test_sxs_dag_equals_y():
    # S X S† = Y  (X acquires a Z component under S)
    frame = PauliFrame(1)
    frame.inject_error(0, "X")
    frame.apply_s(0)
    assert frame.get_pauli_string() == "Y"


def test_szs_dag_equals_z():
    # S Z S† = Z  (Z commutes with S up to phase)
    frame = PauliFrame(1)
    frame.inject_error(0, "Z")
    frame.apply_s(0)
    assert frame.get_pauli_string() == "Z"


def test_double_h_is_identity():
    for pauli in ("X", "Y", "Z"):
        frame = PauliFrame(1)
        frame.inject_error(0, pauli)
        frame.apply_h(0)
        frame.apply_h(0)
        assert frame.get_pauli_string()[0] == pauli, f"H² ≠ I for {pauli}"


def test_four_s_is_identity():
    for pauli in ("X", "Y", "Z"):
        frame = PauliFrame(1)
        frame.inject_error(0, pauli)
        for _ in range(4):
            frame.apply_s(0)
        assert frame.get_pauli_string()[0] == pauli, f"S⁴ ≠ I for {pauli}"


# ---------------------------------------------------------------------------
# Two-qubit CNOT propagation
# ---------------------------------------------------------------------------

def test_cnot_spreads_x_from_control_to_target():
    # CNOT (X_ctrl ⊗ I) CNOT† = X_ctrl ⊗ X_tgt
    frame = PauliFrame(2)
    frame.inject_error(0, "X")   # X on control
    frame.apply_cnot(0, 1)
    ps = frame.get_pauli_string()
    assert ps[0] == "X"
    assert ps[1] == "X"


def test_cnot_spreads_z_from_target_to_control():
    # CNOT (I ⊗ Z_tgt) CNOT† = Z_ctrl ⊗ Z_tgt
    frame = PauliFrame(2)
    frame.inject_error(1, "Z")   # Z on target
    frame.apply_cnot(0, 1)
    ps = frame.get_pauli_string()
    assert ps[0] == "Z"
    assert ps[1] == "Z"


def test_cnot_x_on_target_unchanged():
    # CNOT (I ⊗ X_tgt) CNOT† = I ⊗ X_tgt  (X on target commutes with CNOT)
    frame = PauliFrame(2)
    frame.inject_error(1, "X")
    frame.apply_cnot(0, 1)
    ps = frame.get_pauli_string()
    assert ps[0] == "I"
    assert ps[1] == "X"


def test_cnot_z_on_control_unchanged():
    # CNOT (Z_ctrl ⊗ I) CNOT† = Z_ctrl ⊗ I  (Z on control commutes with CNOT)
    frame = PauliFrame(2)
    frame.inject_error(0, "Z")
    frame.apply_cnot(0, 1)
    ps = frame.get_pauli_string()
    assert ps[0] == "Z"
    assert ps[1] == "I"


def test_double_cnot_is_identity():
    for ctrl_p in ("X", "Y", "Z"):
        frame = PauliFrame(2)
        frame.inject_error(0, ctrl_p)
        frame.apply_cnot(0, 1)
        frame.apply_cnot(0, 1)
        ps = frame.get_pauli_string()
        assert ps[0] == ctrl_p, f"CNOT² ≠ I for control={ctrl_p}"
        assert ps[1] == "I"


# ---------------------------------------------------------------------------
# Clifford group closure spot checks
# ---------------------------------------------------------------------------

def test_hsh_sequence():
    # H S H is a valid Clifford; verify it does not raise and gives a Pauli
    frame = PauliFrame(1)
    frame.inject_error(0, "X")
    frame.apply_h(0)
    frame.apply_s(0)
    frame.apply_h(0)
    result = frame.get_pauli_string()[0]
    assert result in ("I", "X", "Y", "Z")


def test_no_error_stays_identity():
    frame = PauliFrame(3)
    for gate in ["H", "S", "H"]:
        frame.apply_gate(gate, [0])
    frame.apply_gate("CX", [0, 1])
    frame.apply_gate("CZ", [1, 2])
    assert frame.get_pauli_string() == "III"
