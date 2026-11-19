"""Branch-by-branch unit tests for the MEK injection gadget and scheduling.

Tests in this file verify circuit physics rather than formula values:

1. test_conditional_auto_schedules_after_measurement
       The ConditionalOp auto-scheduler must place the c_if gate strictly
       after the Measurement that writes the conditioning bit.

2. test_injection_sign_plus_branch_by_branch (parametrized)
       inject_y_pi_over_4(sign=+1) must implement RY(+π/4) on the target
       in *every* measurement branch after feed-forward correction.

3. test_injection_sign_minus_branch_by_branch (parametrized)
       inject_y_pi_over_4(sign=-1) must implement RY(-π/4) in every branch.

4. test_controlled_h_via_injection_branch_by_branch (parametrized)
       apply_controlled_h_via_mek_injection must implement CH (controlled-H)
       in every measurement branch after feed-forward.
"""

from __future__ import annotations

import numpy as np
import pytest

from noisiq.ir import Circuit
from noisiq.ir.classical import ConditionalOp, Measurement
from noisiq.protocols.mek_exact_branchsum import branch_statevector_mcm
from noisiq.protocols.mek_magic_state_distillation import (
    apply_controlled_h_via_mek_injection,
    inject_y_pi_over_4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _build_injection_circuit(sign: int, input_angle: float) -> Circuit:
    """2-qubit isolation circuit: q0=resource, q1=target.

    Initial state: |0⟩_R ⊗ RY(input_angle)|0⟩_T.
    After injection: each branch should end in |0⟩_R ⊗ RY(sign*π/4)|ψ⟩_T.
    """
    c = Circuit(n_qubits=2, name="inj_test")
    reg = c.add_classical_register("m", 1)
    c.ry(1, input_angle)  # prepare non-trivial target state
    inject_y_pi_over_4(
        c, target=1, resource=0, cbit=reg[0],
        sign=sign, resource_site="r",
    )
    return c


def _expected_state_injection(sign: int, input_angle: float) -> np.ndarray:
    """Expected 4-element statevector: q0=|0⟩, q1=RY(sign*π/4)·RY(input_angle)|0⟩."""
    psi = _ry(input_angle) @ np.array([1.0, 0.0], dtype=complex)
    target_out = _ry(sign * np.pi / 4) @ psi
    # Tensor product: q0=|0⟩ ⊗ q1=target_out
    # Index order: |q0 q1⟩ at position 2*q0 + q1
    return np.array([target_out[0], target_out[1], 0.0, 0.0], dtype=complex)


# ---------------------------------------------------------------------------
# Test 1: conditional scheduling
# ---------------------------------------------------------------------------

def test_conditional_auto_schedules_after_measurement():
    c = Circuit(2)
    reg = c.add_classical_register("m", 1)
    c.h(0)
    c.measure(0, reg[0])
    c.c_if(reg[0], 1).x(1)

    meas_t = next(op.t for op in c.operations if isinstance(op, Measurement))
    cond_t = next(op.t for op in c.operations if isinstance(op, ConditionalOp))

    assert cond_t > meas_t, (
        f"ConditionalOp at t={cond_t} must be strictly after Measurement at t={meas_t}."
    )


# ---------------------------------------------------------------------------
# Test 2 & 3: injection gadget — sign=+1 and sign=-1
# ---------------------------------------------------------------------------

_INPUT_ANGLES = [0.0, np.pi / 5, np.pi / 3, np.pi / 2]


@pytest.mark.parametrize("input_angle", _INPUT_ANGLES)
def test_injection_sign_plus_branch_by_branch(input_angle):
    """Every measurement branch of inject_y_pi_over_4(sign=+1) implements RY(+π/4)."""
    c = _build_injection_circuit(sign=+1, input_angle=input_angle)
    branches = branch_statevector_mcm(c)
    expected = _expected_state_injection(+1, input_angle)

    assert branches, "branch engine returned no branches"
    total_weight = sum(br.weight for br in branches)
    assert abs(total_weight - 1.0) < 1e-10, (
        f"branch weights sum to {total_weight}, expected 1.0"
    )

    for br in branches:
        fidelity = abs(np.dot(br.state.conj(), expected)) ** 2
        assert fidelity > 1.0 - 1e-10, (
            f"sign=+1 injection: branch weight={br.weight:.4f}, "
            f"fidelity with RY(+π/4)|ψ⟩ = {fidelity:.8f}. "
            f"input_angle={input_angle:.4f}. "
            "Check CY interaction direction and feed-forward in inject_y_pi_over_4."
        )


@pytest.mark.parametrize("input_angle", _INPUT_ANGLES)
def test_injection_sign_minus_branch_by_branch(input_angle):
    """Every measurement branch of inject_y_pi_over_4(sign=-1) implements RY(-π/4)."""
    c = _build_injection_circuit(sign=-1, input_angle=input_angle)
    branches = branch_statevector_mcm(c)
    expected = _expected_state_injection(-1, input_angle)

    assert branches, "branch engine returned no branches"
    total_weight = sum(br.weight for br in branches)
    assert abs(total_weight - 1.0) < 1e-10

    for br in branches:
        fidelity = abs(np.dot(br.state.conj(), expected)) ** 2
        assert fidelity > 1.0 - 1e-10, (
            f"sign=-1 injection: branch weight={br.weight:.4f}, "
            f"fidelity with RY(-π/4)|ψ⟩ = {fidelity:.8f}. "
            f"input_angle={input_angle:.4f}. "
            "Check CY interaction direction and feed-forward in inject_y_pi_over_4."
        )


# ---------------------------------------------------------------------------
# Test 4: controlled-H via injection
# ---------------------------------------------------------------------------

# 4-qubit index for a 2-qubit (control × target) subspace with resources reset:
# |q0 q1 q2 q3⟩ at index 8*q0 + 4*q1 + 2*q2 + q3
# With q2=q3=|0⟩ (resource wires reset): only indices 0,4,8,12 are non-zero.
_CH = (1 / np.sqrt(2)) * np.array(
    [
        [np.sqrt(2), 0,          0,          0         ],
        [0,          np.sqrt(2), 0,          0         ],
        [0,          0,          1,          1         ],
        [0,          0,          1,         -1         ],
    ],
    dtype=complex,
)  # CH[control ⊗ target] ordering


@pytest.mark.parametrize("control_angle,target_angle", [
    (0.0,      0.0),
    (np.pi,    0.0),          # |1⟩ control — H fires on target
    (np.pi / 3, np.pi / 5),
    (np.pi / 2, np.pi / 4),
])
def test_controlled_h_via_injection_branch_by_branch(control_angle, target_angle):
    """Y(-π/4)·CZ·Y(+π/4) on target implements CH(control, target) in every branch."""
    # 4 qubits: q0=control, q1=target, q2=resource_a, q3=resource_b
    c = Circuit(n_qubits=4, name="ch_test")
    inj_reg = c.add_classical_register("inj", 2)

    c.ry(0, control_angle)
    c.ry(1, target_angle)

    apply_controlled_h_via_mek_injection(
        c, control=0, target=1,
        resource_a=2, resource_b=3,
        cbits=(inj_reg[0], inj_reg[1]),
        site_prefix="ch_test",
    )

    branches = branch_statevector_mcm(c)

    # Build expected 4-qubit state: CH on (q0,q1), resources (q2,q3) reset to |0⟩
    psi_c = _ry(control_angle) @ np.array([1.0, 0.0], dtype=complex)
    psi_t = _ry(target_angle) @ np.array([1.0, 0.0], dtype=complex)
    psi_ct = np.kron(psi_c, psi_t)           # 4-element two-qubit state
    out_ct = _CH @ psi_ct                     # CH applied

    # Embed into 4-qubit space: resources are |0⟩ (index 0 in 4-element |q2 q3⟩ space)
    expected = np.zeros(16, dtype=complex)
    expected[[0, 4, 8, 12]] = out_ct          # positions with q2=q3=0

    assert branches, "branch engine returned no branches"
    total_weight = sum(br.weight for br in branches)
    assert abs(total_weight - 1.0) < 1e-10, (
        f"branch weights sum to {total_weight}"
    )

    for br in branches:
        fidelity = abs(np.dot(br.state.conj(), expected)) ** 2
        assert fidelity > 1.0 - 1e-10, (
            f"CH gadget: branch weight={br.weight:.4f}, "
            f"fidelity={fidelity:.8f}. "
            f"control_angle={control_angle:.4f}, target_angle={target_angle:.4f}. "
            "Check Y(±π/4) injection gadgets — verify CY interaction and feed-forward."
        )
