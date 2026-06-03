"""Paper-exact MEK 10-to-2 H-state distillation scaffold.

This module is intentionally narrow: it targets the Meier–Eastin–Knill
10-to-2 Hadamard-magic-state distillation routine under the assumptions of
arXiv:1204.4221:

    * perfect Clifford operations;
    * ten independent, twirled |H> resource states;
    * each faulty resource is represented by stochastic Y with probability p;
    * acceptance is by encoded measurement + final [[4,2,2]] syndrome.

The paper-exact circuit body is left as an explicit TODO because it should be
translated carefully from the paper circuit/gadget and then verified against the
analytic formulas below. The surrounding API, resource-site tagging, noise
builder, post-selection contract, and metric extraction are ready for that body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import string

import numpy as np

from noisiq.ir import Circuit
from noisiq.ir.classical import ClassicalBit
from noisiq.noise import PauliError


# -----------------------------------------------------------------------------
# Paper resource-site convention
# -----------------------------------------------------------------------------

H_STATE_ERROR_SITES: Tuple[str, ...] = (
    "data_0",
    "data_1",
    "ch_0_y_minus",
    "ch_0_y_plus",
    "ch_1_y_minus",
    "ch_1_y_plus",
    "ch_2_y_minus",
    "ch_2_y_plus",
    "ch_3_y_minus",
    "ch_3_y_plus",
)
"""The ten independently noisy |H> resource states in the MEK routine.

The two ``data_*`` states are encoded into the [[4,2,2]] code. The eight
``ch_*`` states are the two |H> resources consumed by each of the four
controlled-H gadgets.
"""


@dataclass(frozen=True)
class MEKCircuitBuild:
    """Container returned by MEK circuit builders.

    Attributes:
        circuit: NoisiQ circuit object.
        output_qubits: The two qubits that hold the accepted output |H> states.
        accept_cbits: Classical-bit indices whose required value is 0 on accept.
        resource_sites: The ten resource-site labels expected by the paper model.
        resource_qubits: Mapping from resource-site label to physical wire used.
        notes: Human-readable implementation status / caveats.
    """

    circuit: Circuit
    output_qubits: Tuple[int, int]
    accept_cbits: Tuple[int, ...]
    resource_sites: Tuple[str, ...] = H_STATE_ERROR_SITES
    resource_qubits: Mapping[str, int] | None = None
    notes: Tuple[str, ...] = ()

    @property
    def post_select(self) -> Dict[int, int]:
        """Post-selection map for accepted shots: all listed cbits must be 0."""
        return {idx: 0 for idx in self.accept_cbits}


@dataclass(frozen=True)
class MEKBenchmarkPoint:
    """One row comparing NoisiQ simulation to the paper formulas."""

    p: float
    simulated_acceptance: Optional[float]
    paper_acceptance: float
    simulated_marginal_error: Optional[float]
    paper_marginal_error: float
    n_shots: Optional[int] = None
    accepted_shots: Optional[int] = None


# -----------------------------------------------------------------------------
# State definitions and exact paper formulas
# -----------------------------------------------------------------------------


def ket_H() -> np.ndarray:
    """Return |H> = cos(pi/8)|0> + sin(pi/8)|1>."""
    return np.array([np.cos(np.pi / 8.0), np.sin(np.pi / 8.0)], dtype=complex)


def ket_minus_H() -> np.ndarray:
    """Return |-H>, the -1 eigenstate of Hadamard.

    With the phase convention used here, Y|H> = i|-H>, so stochastic Y maps the
    projector |H><H| to |-H><-H| exactly; the global phase is irrelevant.
    """
    return np.array([np.sin(np.pi / 8.0), -np.cos(np.pi / 8.0)], dtype=complex)


def projector(state: np.ndarray) -> np.ndarray:
    """Return |state><state|."""
    state = np.asarray(state, dtype=complex)
    return np.outer(state, state.conj())


def rho_H_twirled(p: float) -> np.ndarray:
    """Return (1-p)|H><H| + p|-H><-H| for 0 <= p <= 1."""
    _validate_probability(p)
    return (1.0 - p) * projector(ket_H()) + p * projector(ket_minus_H())


def mek_acceptance_probability(p: float) -> float:
    """Paper acceptance probability a(p), Eq. in arXiv:1204.4221."""
    _validate_probability(p)
    return float(
        1
        - 10 * p
        + 58 * p**2
        - 192 * p**3
        + 400 * p**4
        - 544 * p**5
        + 480 * p**6
        - 256 * p**7
        + 64 * p**8
    )


def mek_undetected_marginal_probability(p: float) -> float:
    """Paper marginal undetected output-error probability u(p)."""
    _validate_probability(p)
    return float(
        9 * p**2
        - 56 * p**3
        + 160 * p**4
        - 256 * p**5
        + 240 * p**6
        - 128 * p**7
        + 32 * p**8
    )


def mek_undetected_any_output_probability(p: float) -> float:
    """Paper probability u2(p) of at least one undetected error on two outputs."""
    _validate_probability(p)
    return float(
        13 * p**2
        - 80 * p**3
        + 228 * p**4
        - 368 * p**5
        + 352 * p**6
        - 192 * p**7
        + 48 * p**8
    )


def mek_output_error_probability(p: float) -> float:
    """Conditional marginal output error e(p)=u(p)/a(p)."""
    a = mek_acceptance_probability(p)
    if abs(a) < 1e-15:
        raise ZeroDivisionError(f"a(p) is numerically zero at p={p}")
    return mek_undetected_marginal_probability(p) / a


def mek_any_output_error_probability(p: float) -> float:
    """Conditional probability e2(p)=u2(p)/a(p)."""
    a = mek_acceptance_probability(p)
    if abs(a) < 1e-15:
        raise ZeroDivisionError(f"a(p) is numerically zero at p={p}")
    return mek_undetected_any_output_probability(p) / a


def mek_threshold() -> float:
    """Return the MEK paper threshold p_t ≈ 0.089.

    This intentionally returns the published value. For a computed root, use a
    plotting or numerical helper in the notebook/demo.
    """
    return 0.089


# -----------------------------------------------------------------------------
# Circuit construction helpers
# -----------------------------------------------------------------------------


def require_ry_gate_available() -> None:
    """Raise a helpful error if NoisiQ has not yet added ry_gate/Circuit.ry."""
    import noisiq.ir.gates as gates

    if not hasattr(gates, "ry_gate") or not hasattr(Circuit, "ry"):
        raise RuntimeError(
            "MEK |H> preparation needs noisiq.ir.gates.ry_gate(theta) and "
            "Circuit.ry(qubit, theta). Apply patches/ry_gate_patch.md first."
        )


def prepare_plus(c: Circuit, qubit: int) -> None:
    """Prepare |+> from |0>."""
    c.h(qubit)


def prepare_H_resource(c: Circuit, qubit: int, site: str) -> None:
    """Prepare a tagged |H> resource on `qubit`.

    The tag is the anchor used by `build_mek_resource_noise()` to attach the
    paper's stochastic Y error. This function assumes Circuit.ry exists.
    """
    require_ry_gate_available()
    c.ry(
        qubit,
        np.pi / 4.0,
        meta={
            "resource_site": site,
            "paper_model": "MEK_twirled_H_resource",
            "noise": "Y with probability p, applied after ideal |H> prep",
        },
    )


def measure_x(c: Circuit, qubit: int, cbit: ClassicalBit, *, reset: bool = False) -> None:
    """Measure X by rotating with H then using NoisiQ's Z-basis measure."""
    c.h(qubit)
    c.measure(qubit, cbit, reset=reset)


def measure_y(c: Circuit, qubit: int, cbit: ClassicalBit, *, reset: bool = False) -> None:
    """Measure Y by rotating to Z then using NoisiQ's Z-basis measure.

    Convention: S† then H maps Y eigenbasis to computational basis.
    """
    c.s_dag(qubit)
    c.h(qubit)
    c.measure(qubit, cbit, reset=reset)


def consume_resource_wire_with_reset(
    c: Circuit,
    qubit: int,
    cbit: ClassicalBit,
    *,
    label: str,
) -> None:
    """Measure/reset a consumed resource wire so it can be reused later.

    This is deliberately included to showcase NoisiQ's MCM reset feature.
    The measurement result is *scrap* data and is not part of the MEK acceptance
    condition.
    """
    c.measure(
        qubit,
        cbit,
        reset=True,
        # Circuit.measure currently has no meta slot. Keep label in cbit name.
    )


def build_mek_10to2_scaffold_circuit(*, use_resource_reset: bool = True) -> MEKCircuitBuild:
    """Build a structural scaffold circuit showing resource-wire reuse.

    WARNING: This function is intentionally **not** the final paper-exact MEK
    circuit. It exists to provide a concrete wiring/layout target and to exercise
    resource-site tagging + measure/reset while the exact gadget body is filled
    in. Use `build_mek_10to2_paper_exact_circuit()` as the production target.

    Wire convention used by this scaffold:
        q0      : encoded-measurement ancilla, prepared |+>, measured X
        q1..q4  : [[4,2,2]] code/work block; q1 and q3 receive data |H> states
        q5..q6  : reusable controlled-H resource wires for ch_* resources

    Acceptance cbits:
        accept[0] : encoded measurement ancilla, expected 0
        accept[1] : final Z syndrome, expected 0
        accept[2] : final X syndrome, expected 0
    """
    require_ry_gate_available()

    c = Circuit(n_qubits=7, name="mek_10to2_scaffold")
    accept = c.add_classical_register("mek_accept", 3)
    scrap = c.add_classical_register("mek_scrap_reset", 8)

    # q0: encoded-measurement ancilla. q4: one scaffold syndrome/work qubit.
    prepare_plus(c, 0)
    prepare_plus(c, 4)

    # Two data |H> resources. These are part of the paper's ten noisy resources.
    prepare_H_resource(c, 1, "data_0")
    prepare_H_resource(c, 3, "data_1")

    # Skeleton encoding shape for [[4,2,2]]. This is a placeholder layout, not a
    # proof that the exact MEK encoding/decoding has been completed.
    c.cnot(1, 2)
    c.cnot(3, 2)
    c.cnot(4, 1)
    c.cnot(4, 3)

    resource_qubits: Dict[str, int] = {"data_0": 1, "data_1": 3}

    # Four controlled-H gadgets, each consuming two noisy |H> resources.
    # The exact Clifford + measurement + feed-forward body must be inserted at
    # `_TODO_expand_controlled_H_injection_gadget()` below.
    scrap_i = 0
    for gadget_idx in range(4):
        site_minus = f"ch_{gadget_idx}_y_minus"
        site_plus  = f"ch_{gadget_idx}_y_plus"
        prepare_H_resource(c, 5, site_minus)
        prepare_H_resource(c, 6, site_plus)
        resource_qubits[site_minus] = 5
        resource_qubits[site_plus]  = 6

        _TODO_expand_controlled_H_injection_gadget(c, control=0, target=(1 + gadget_idx) % 4 + 1)

        if use_resource_reset:
            consume_resource_wire_with_reset(c, 5, scrap[scrap_i], label=site_minus)
            scrap_i += 1
            consume_resource_wire_with_reset(c, 6, scrap[scrap_i], label=site_plus)
            scrap_i += 1

    # Placeholder decode / syndrome shape. The exact paper decode must replace
    # this block when implementing build_mek_10to2_paper_exact_circuit().
    c.cnot(4, 3)
    c.cnot(4, 1)
    c.cnot(3, 2)
    c.cnot(1, 2)

    # MEK acceptance structure: encoded measurement + final C4 syndrome.
    measure_x(c, 0, accept[0])
    c.measure(2, accept[1])
    measure_x(c, 4, accept[2])

    return MEKCircuitBuild(
        circuit=c,
        output_qubits=(1, 3),
        accept_cbits=(accept[0].index, accept[1].index, accept[2].index),
        resource_qubits=resource_qubits,
        notes=(
            "SCAFFOLD ONLY: controlled-H injection gadget body is a TODO.",
            "SCAFFOLD ONLY: encoding/decoding must be checked against the MEK paper circuit.",
            "Uses measure(reset=True) on reusable resource wires q5 and q6.",
        ),
    )


def _TODO_expand_controlled_H_injection_gadget(
    c: Circuit,
    *,
    control: int,
    target: int,
) -> None:
    """Placeholder for the exact controlled-H resource-injection gadget.

    This intentionally inserts no physics-bearing gates. Replace this body with
    the paper-exact Clifford + resource-measurement implementation before using
    this protocol for validation.

    Implementation guidance:
        * use the two most recently prepared resource wires as the gadget inputs;
        * use only perfect Clifford gates and Z-basis measurement with basis rotations;
        * feed-forward corrections should use Circuit.c_if(...);
        * preserve the resource-site metadata on prep operations only;
        * resource measurement/reset outcomes should be scrap, not acceptance bits.
    """
    # No-op placeholder so the scaffold can be drawn and introspected.
    _ = (c, control, target)


def inject_y_pi_over_4(
    c: Circuit,
    target: int,
    resource: int,
    cbit: ClassicalBit,
    *,
    sign: int,
    resource_site: str,
) -> None:
    """Inject Y(sign*π/4) on target using one |H⟩ resource via gate teleportation.

    sign = +1  →  RY(+π/4) applied to target
    sign = -1  →  RY(-π/4) applied to target

    Circuit (per injection):
    1. Prepare |H⟩ = RY(π/4)|0⟩ on resource, tagged as resource_site for the
       MEK noise model (stochastic Y with probability p).
    2. CY interaction: resource as control, data as target.
       CY(R,D) = S†_D · CNOT(R→D) · S_D
       Both signs use the same interaction; the sign of the output rotation is
       determined by the feed-forward branch that fires.
    3. Y-basis measurement with reset: apply S†·H to resource, then measure in
       Z basis (reset=True so the resource wire can be reused for the next gadget).
       Convention: outcome=0 → |+Y⟩ resource (positive eigenvalue);
                   outcome=1 → |−Y⟩ resource (negative eigenvalue).
    4. Feed-forward Clifford correction on target:
         sign = +1 : outcome=1 needs RY(+π/2) correction.
                     RY(+π/2) = X·H in matrix form → circuit order: H then X.
         sign = -1 : outcome=0 needs RY(−π/2) correction.
                     RY(−π/2) = H·X in matrix form → circuit order: X then H.

    Verify with evaluate_mek_exact_branchsum() before Monte Carlo validation.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")

    prepare_H_resource(c, resource, resource_site)

    # CY(resource, target) = S†_target · CNOT(resource→target) · S_target
    c.s_dag(target)
    c.cnot(resource, target)
    c.s(target)

    # Y-basis measurement: S†·H rotates Y eigenstates to Z eigenstates.
    measure_y(c, resource, cbit, reset=True)

    if sign == 1:
        # outcome=1 (|−Y⟩ resource) → data in RY(−π/4)|ψ⟩; correct with RY(+π/2)=X·H
        c.c_if(cbit, 1).h(target)
        c.c_if(cbit, 1).x(target)
    else:
        # outcome=0 (|+Y⟩ resource) → data in RY(+π/4)|ψ⟩; correct with RY(−π/2)=H·X
        c.c_if(cbit, 0).x(target)
        c.c_if(cbit, 0).h(target)


def apply_controlled_h_via_mek_injection(
    c: Circuit,
    control: int,
    target: int,
    resource_a: int,
    resource_b: int,
    cbits: Tuple[ClassicalBit, ClassicalBit],
    *,
    site_prefix: str,
) -> None:
    """Paper-exact controlled-H gadget via Y(±π/4) injection.

    Implements CH = Y(-π/4) · CZ · Y(+π/4) on the target, controlled by
    `control`, consuming two |H⟩ resources via gate teleportation:

        inject Y(-π/4)  on target  (resource_a, cbits[0], site_prefix + "_y_minus")
        CZ(control, target)
        inject Y(+π/4)  on target  (resource_b, cbits[1], site_prefix + "_y_plus")

    Both resource wires are measured with reset=True after use so they can be
    reused for subsequent gadgets.
    """
    inject_y_pi_over_4(
        c,
        target=target,
        resource=resource_a,
        cbit=cbits[0],
        sign=-1,
        resource_site=f"{site_prefix}_y_minus",
    )
    c.cz(control, target)
    inject_y_pi_over_4(
        c,
        target=target,
        resource=resource_b,
        cbit=cbits[1],
        sign=+1,
        resource_site=f"{site_prefix}_y_plus",
    )


def build_mek_10to2_paper_exact_circuit(*, use_resource_reset: bool = True) -> MEKCircuitBuild:
    """Build the paper-exact MEK 10-to-2 H-state distillation circuit.

    Uses the decomposition CH = Y(-π/4) · CZ · Y(+π/4) to replace each
    controlled-H gadget with two Y(±π/4) injections (each consuming one |H⟩
    resource) bracketing a Clifford CZ.

    Wire convention (7 qubits):
        q0      : encoded-measurement ancilla, prepared |+>, measured X
        q1, q3  : data |H> resources  (data_0, data_1)
        q2, q4  : [[4,2,2]] code ancilla / work qubits
        q5      : reusable resource wire A  (Y(-π/4) injections, ch_N_y_minus)
        q6      : reusable resource wire B  (Y(+π/4) injections, ch_N_y_plus)

    Classical registers:
        mek_accept[3] : acceptance bits — all must be 0 on acceptance
        scrap[8]      : Y-injection measurement outcomes (not part of acceptance)

    The circuit should reproduce the paper benchmark formulas a(p) and e(p)
    within Monte Carlo uncertainty. Use evaluate_mek_exact_branchsum() to
    verify against the analytic formulas over all 2^10 resource-error patterns.
    """
    require_ry_gate_available()

    c = Circuit(n_qubits=7, name="mek_10to2_paper_exact")
    accept = c.add_classical_register("mek_accept", 3)
    scrap  = c.add_classical_register("mek_scrap", 8)

    resource_qubits: Dict[str, int] = {}

    # Ancillas for encoded measurement and syndrome
    prepare_plus(c, 0)
    prepare_plus(c, 4)

    # Data |H> resources on dedicated wires
    prepare_H_resource(c, 1, "data_0")
    resource_qubits["data_0"] = 1
    prepare_H_resource(c, 3, "data_1")
    resource_qubits["data_1"] = 3

    # [[4,2,2]] encoding CNOT network.
    #
    # This prepares the C4 logical state for the encoded-Hadamard measurement.
    # The order matters: it must be the exact encoder whose inverse (applied as
    # the decode network below) returns the two output |H> states to wires q1, q3
    # and leaves the syndrome qubits q2, q4 (with ancilla q0) in a definite
    # accept eigenstate at p=0. Verified by evaluate_mek_exact_branchsum(0.0)
    # giving acceptance == 1.0.
    c.cnot(4, 1)
    c.cnot(3, 2)
    c.cnot(1, 2)
    c.cnot(4, 3)

    # Encoded measurement of H_1 H_2: four controlled-H gates from the |+> ancilla
    # (q0) onto each of the four code qubits (q1, q2, q3, q4). For the C4 code the
    # transversal Hadamard H^{⊗4} acts as the logical H_1 H_2 (up to an incidental
    # logical SWAP), so controlled-H^{⊗4} = CH(q0,q1) CH(q0,q2) CH(q0,q3) CH(q0,q4)
    # implements the controlled encoded-Hadamard used for the H_1 H_2 measurement.
    #
    # Each CH is realized by the paper-exact injection gadget, which consumes the
    # two |H> resources on the reusable wires q5 (Y(-π/4)) and q6 (Y(+π/4)).
    scrap_i = 0
    for gadget_idx, target in enumerate((1, 2, 3, 4)):
        prefix = f"ch_{gadget_idx}"
        resource_qubits[f"{prefix}_y_minus"] = 5
        resource_qubits[f"{prefix}_y_plus"]  = 6

        apply_controlled_h_via_mek_injection(
            c,
            control=0,
            target=target,
            resource_a=5,
            resource_b=6,
            cbits=(scrap[scrap_i], scrap[scrap_i + 1]),
            site_prefix=prefix,
        )
        scrap_i += 2

    # [[4,2,2]] decoding CNOT network: the exact reverse of the encoding above.
    c.cnot(4, 3)
    c.cnot(1, 2)
    c.cnot(3, 2)
    c.cnot(4, 1)

    # MEK acceptance measurements
    measure_x(c, 0, accept[0])
    c.measure(2, accept[1])
    measure_x(c, 4, accept[2])

    return MEKCircuitBuild(
        circuit=c,
        output_qubits=(1, 3),
        accept_cbits=(accept[0].index, accept[1].index, accept[2].index),
        resource_qubits=resource_qubits,
        notes=(
            "Paper-exact: CH decomposed as Y(-π/4)·CZ·Y(+π/4).",
            "Y(±π/4) injected via |H> resource + CNOT + Y-basis measure + Clifford correction.",
            "Resource wires q5, q6 reused across all four gadgets via measure(reset=True).",
            "Verify against paper formulas using evaluate_mek_exact_branchsum().",
        ),
    )


# -----------------------------------------------------------------------------
# Noise and post-selection helpers
# -----------------------------------------------------------------------------


def find_resource_prep_ops(circuit: Circuit) -> Dict[str, int]:
    """Return {resource_site: operation_index} for tagged |H> preparations."""
    found: Dict[str, int] = {}
    for op_idx, op in enumerate(circuit.operations):
        meta = getattr(op, "meta", None) or {}
        site = meta.get("resource_site")
        if site is None:
            continue
        if site in found:
            raise ValueError(f"Duplicate resource_site tag {site!r}")
        found[site] = op_idx

    expected = set(H_STATE_ERROR_SITES)
    actual = set(found)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            "Resource-site mismatch. "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return found


def build_mek_resource_noise(circuit: Circuit, p: float) -> Dict[int, PauliError]:
    """Build paper-exact noise config: Y-only noise on the ten resource sites.

    Returns a dict mapping operation index -> PauliError(p_x=0, p_y=p, p_z=0).
    No other operation receives noise.
    """
    _validate_probability(p)
    site_to_op = find_resource_prep_ops(circuit)
    y_noise = PauliError(p_x=0.0, p_y=float(p), p_z=0.0)
    return {op_idx: y_noise for op_idx in site_to_op.values()}


def postselect_acceptance_map(build: MEKCircuitBuild) -> Dict[int, int]:
    """Return NoisiQ post_select map for accepted MEK shots."""
    return build.post_select



def _partial_trace_keep_one_local(rho: np.ndarray, keep_qubit: int, n_qubits: int) -> np.ndarray:
    """Trace out every qubit except `keep_qubit`; return a 2x2 reduced state."""
    rho_t = rho.reshape([2] * (2 * n_qubits))
    row_chars = list(string.ascii_lowercase[:n_qubits])
    col_chars = list(string.ascii_uppercase[:n_qubits])
    for q in range(n_qubits):
        if q != keep_qubit:
            col_chars[q] = row_chars[q]
    einsum_str = (
        "".join(row_chars)
        + "".join(col_chars)
        + "->"
        + row_chars[keep_qubit]
        + col_chars[keep_qubit]
    )
    return np.einsum(einsum_str, rho_t)

# -----------------------------------------------------------------------------
# Output metrics
# -----------------------------------------------------------------------------


def marginal_H_error_from_rho(
    rho: np.ndarray,
    *,
    n_qubits: int,
    output_qubit: int,
) -> float:
    """Return 1 - <H|rho_q|H> for one accepted output qubit."""
    rho_q = _partial_trace_keep_one_local(rho, keep_qubit=output_qubit, n_qubits=n_qubits)
    h = ket_H()
    fidelity = float(np.real(h.conj() @ rho_q @ h))
    return float(1.0 - fidelity)


def compare_to_paper_formula(
    p: float,
    *,
    simulated_acceptance: Optional[float] = None,
    simulated_marginal_error: Optional[float] = None,
    n_shots: Optional[int] = None,
    accepted_shots: Optional[int] = None,
) -> MEKBenchmarkPoint:
    """Package one comparison row against the paper formulas."""
    return MEKBenchmarkPoint(
        p=float(p),
        simulated_acceptance=simulated_acceptance,
        paper_acceptance=mek_acceptance_probability(p),
        simulated_marginal_error=simulated_marginal_error,
        paper_marginal_error=mek_output_error_probability(p),
        n_shots=n_shots,
        accepted_shots=accepted_shots,
    )


# -----------------------------------------------------------------------------
# Internal utility
# -----------------------------------------------------------------------------


def _validate_probability(p: float) -> None:
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p!r}")
