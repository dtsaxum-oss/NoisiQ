import pytest

from noisiq.protocols.mek_magic_state_distillation import (
    H_STATE_ERROR_SITES,
    build_mek_10to2_paper_exact_circuit,
    build_mek_10to2_scaffold_circuit,
    build_mek_resource_noise,
    find_resource_prep_ops,
)
from noisiq.protocols.mek_exact_branchsum import evaluate_mek_exact_branchsum


def test_scaffold_has_ten_resource_sites_when_ry_patch_is_applied():
    try:
        build = build_mek_10to2_scaffold_circuit(use_resource_reset=True)
    except RuntimeError as exc:
        pytest.skip(f"RY patch not applied yet: {exc}")

    site_to_op = find_resource_prep_ops(build.circuit)
    assert tuple(site_to_op.keys()) == H_STATE_ERROR_SITES
    assert len(site_to_op) == 10


def test_resource_noise_only_targets_resource_prep_ops_when_ry_patch_is_applied():
    try:
        build = build_mek_10to2_scaffold_circuit(use_resource_reset=True)
    except RuntimeError as exc:
        pytest.skip(f"RY patch not applied yet: {exc}")

    site_to_op = find_resource_prep_ops(build.circuit)
    noise = build_mek_resource_noise(build.circuit, p=0.01)
    assert set(noise) == set(site_to_op.values())
    assert len(noise) == 10
    assert all(ch.p_x == 0.0 and ch.p_y == 0.01 and ch.p_z == 0.0 for ch in noise.values())


def test_scaffold_uses_measure_reset_on_resource_wires_when_ry_patch_is_applied():
    try:
        build = build_mek_10to2_scaffold_circuit(use_resource_reset=True)
    except RuntimeError as exc:
        pytest.skip(f"RY patch not applied yet: {exc}")

    reset_measurements = [
        op for op in build.circuit.operations
        if getattr(op, "reset", False) is True
    ]
    assert len(reset_measurements) == 8


@pytest.mark.xfail(
    reason=(
        "The [[4,2,2]] encoding/decoding CNOT network and CH-gadget target mapping "
        "do not yet match arXiv:1204.4221 Fig. 2/4.  At p=0 the acceptance probability "
        "is ~0.5625 instead of 1.0 — roughly 7/16 of branches are rejected because "
        "the syndrome qubits are left in mixed states after the circuit.  "
        "Remove xfail once the encoding network is corrected."
    ),
    strict=True,
)
def test_mek_p0_all_branches_have_accepted_syndrome():
    """Sentinel: at p=0 every measurement branch must pass post-selection.

    A paper-exact MEK circuit initialises its syndrome qubits deterministically
    and the encoding/decoding CNOT network must return them to a definite
    eigenstate before each acceptance measurement.  If any branch is rejected
    the circuit structure is wrong.
    """
    build = build_mek_10to2_paper_exact_circuit()  # noqa: F841 — triggers circuit validation
    exact = evaluate_mek_exact_branchsum(0.0, return_branch_diagnostics=True)

    assert exact.acceptance == pytest.approx(1.0, abs=1e-10), (
        f"p=0 acceptance = {exact.acceptance:.6f}, expected 1.0. "
        f"rejected_probability = {exact.rejected_probability:.6f}. "
        "The encoding/decoding network or acceptance measurement basis is wrong."
    )
    assert exact.rejected_probability == pytest.approx(0.0, abs=1e-10), (
        f"rejected_probability = {exact.rejected_probability:.6f}, expected 0.0."
    )
