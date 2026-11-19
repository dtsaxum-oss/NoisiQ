"""
Phase 4B closeout tests.

Covers the routing, MCM semantics, ManyShotRunner aggregation, IR validation,
QASM reset handling, and MetricReport coercion changes from the Phase 4B patch.
"""

import pytest
import numpy as np

from noisiq.ir import Circuit, gates
from noisiq.ir.classical import ClassicalBit, ClassicalRegister, Measurement, ConditionalOp
from noisiq.ir.circuit import Operation
from noisiq.noise import PauliError, depolarizing_error
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.noise.kraus_channels import KrausChannel, CombinedChannel
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.backends.backend_selector import BackendSelector
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.backends.tsim_backend import TsimBackend
from noisiq.backends.trajectory_backend import TrajectoryBackend
from noisiq.backends.many_shot_runner import ManyShotRunner, AggregateResult
from noisiq.io.qasm import from_qasm, QASMParseError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clifford_circuit_2q():
    c = Circuit(n_qubits=2)
    c.h(0)
    c.cnot(0, 1)
    return c


def _mcm_circuit():
    """1-qubit circuit with a mid-circuit measurement."""
    c = Circuit(n_qubits=1)
    c.classical_registers.append(ClassicalRegister(name="c", size=1, offset=0))
    cbit = ClassicalBit(name="c[0]", index=0)
    c.h(0)
    c.operations.append(Measurement(qubit=0, cbit=cbit, t=1))
    return c


def _mcm_conditional_t_circuit():
    """Circuit with MCM + conditional T gate (non-Clifford inner op)."""
    c = Circuit(n_qubits=1)
    c.classical_registers.append(ClassicalRegister(name="c", size=1, offset=0))
    cbit = ClassicalBit(name="c[0]", index=0)
    c.h(0)
    c.operations.append(Measurement(qubit=0, cbit=cbit, t=1))
    t_op = Operation(gate=gates.T, qubits=(0,), t=2)
    c.operations.append(ConditionalOp(inner=t_op, condition=cbit, value=1))
    return c


def _mcm_with_t_circuit():
    """Circuit with both MCM and a plain T gate (non-Clifford + MCM)."""
    c = Circuit(n_qubits=1)
    c.classical_registers.append(ClassicalRegister(name="c", size=1, offset=0))
    cbit = ClassicalBit(name="c[0]", index=0)
    c.h(0)
    c.operations.append(Measurement(qubit=0, cbit=cbit, t=1))
    c.tgate(0)
    return c


# ---------------------------------------------------------------------------
# BackendSelector routing
# ---------------------------------------------------------------------------

class TestBackendSelectorRouting:

    def test_clifford_correlated_pauli_routes_to_stim(self):
        """CorrelatedPauliError is a Pauli-compatible channel → StimTableauBackend."""
        c = _clifford_circuit_2q()
        noise = CorrelatedPauliError({'ZZ': 0.01})
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, StimTableauBackend)

    def test_clifford_all_pauli_combined_channel_routes_to_stim(self):
        """CombinedChannel of all PauliErrors is Pauli-compatible → StimTableauBackend."""
        c = _clifford_circuit_2q()
        inner_channels = [depolarizing_error(0.01), depolarizing_error(0.005)]
        noise = CombinedChannel(inner_channels)
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, StimTableauBackend)

    def test_clifford_kraus_routes_to_trajectory(self):
        """Non-Pauli channel (AmplitudeDamping) → TrajectoryBackend."""
        c = Circuit(n_qubits=1)
        c.h(0)
        noise = AmplitudeDamping(T1=1.0, t=1.0)
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, TrajectoryBackend)

    def test_mcm_clifford_pauli_routes_to_stim(self):
        """MCM + Clifford + PauliError → StimTableauBackend."""
        c = _mcm_circuit()
        noise = depolarizing_error(0.01)
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, StimTableauBackend)

    def test_mcm_non_clifford_routes_to_trajectory(self):
        """MCM + plain T gate → TrajectoryBackend, not Stim."""
        c = _mcm_with_t_circuit()
        backend = BackendSelector.select(c)
        assert isinstance(backend, TrajectoryBackend)
        assert not isinstance(backend, StimTableauBackend)

    def test_mcm_conditional_non_clifford_inner_routes_to_trajectory(self):
        """MCM + ConditionalOp wrapping T gate → TrajectoryBackend.

        The selector must inspect ConditionalOp.inner; if it only checks
        isinstance(op, Operation) at the top level it will miss the T gate
        and incorrectly route to StimTableauBackend.
        """
        c = _mcm_conditional_t_circuit()
        backend = BackendSelector.select(c)
        assert isinstance(backend, TrajectoryBackend)
        assert not isinstance(backend, StimTableauBackend)

    def test_non_clifford_no_mcm_routes_to_tsim(self):
        """Non-Clifford without MCM still routes to TsimBackend (Tsim not yet removed)."""
        c = Circuit(n_qubits=1)
        c.tgate(0)
        backend = BackendSelector.select(c)
        assert isinstance(backend, TsimBackend)

    # ------------------------------------------------------------------
    # Non-Clifford + noise channel routing
    # ------------------------------------------------------------------

    def test_non_clifford_correlated_pauli_routes_to_trajectory(self):
        """T gate + CorrelatedPauliError must not route to TsimBackend.

        TsimBackend only handles plain PauliError; CorrelatedPauliError is
        Pauli-compatible but TsimBackend cannot run it.  BackendSelector must
        use has_tsim_incompatible_pauli_noise to fall through to TrajectoryBackend.
        CorrelatedPauliError requires ≥2-qubit strings, so we use a 2-qubit
        circuit with a T gate on qubit 0 and H on qubit 1.
        """
        c = Circuit(n_qubits=2)
        c.tgate(0)
        c.h(1)
        noise = {0: CorrelatedPauliError({'ZZ': 0.01})}
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, TrajectoryBackend)
        assert not isinstance(backend, TsimBackend)

    def test_non_clifford_pauli_only_combined_channel_routes_to_trajectory(self):
        """T gate + all-Pauli CombinedChannel must route to TrajectoryBackend.

        CombinedChannel([PauliError, PauliError]) is Pauli-compatible but
        TsimBackend cannot process it — only TrajectoryBackend handles it.
        """
        c = Circuit(n_qubits=1)
        c.tgate(0)
        noise = {0: CombinedChannel([depolarizing_error(0.01), depolarizing_error(0.005)])}
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, TrajectoryBackend)
        assert not isinstance(backend, TsimBackend)

    def test_non_clifford_non_pauli_combined_channel_routes_to_trajectory(self):
        """T gate + CombinedChannel containing a KrausChannel → TrajectoryBackend.

        has_non_pauli_noise=True fires the first routing condition directly.
        """
        c = Circuit(n_qubits=1)
        c.tgate(0)
        noise = {0: CombinedChannel([AmplitudeDamping(T1=1.0, t=1e-6), depolarizing_error(0.01)])}
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, TrajectoryBackend)

    def test_mcm_non_clifford_non_pauli_noise_routes_to_trajectory(self):
        """MCM + non-Clifford + non-Pauli noise → TrajectoryBackend via BackendSelector.

        has_non_pauli_noise fires the first routing condition before the MCM
        and non-Clifford checks even matter.
        """
        c = _mcm_with_t_circuit()
        noise = {0: AmplitudeDamping(T1=1.0, t=1e-6)}
        backend = BackendSelector.select(c, noise)
        assert isinstance(backend, TrajectoryBackend)


# ---------------------------------------------------------------------------
# ManyShotRunner — non-Pauli and MCM routing
# ---------------------------------------------------------------------------

class TestManyShotRunnerRouting:

    def test_mcm_clifford_non_pauli_noise_falls_back_to_trajectory(self):
        """MCM + Clifford + non-Pauli noise: ManyShotRunner routes to TrajectoryBackend.

        AmplitudeDamping is not Pauli-compatible; ManyShotRunner detects this
        before the shot loop and delegates to TrajectoryBackend.run_aggregate().
        The result must be a valid AggregateResult with measurement records.
        """
        c = Circuit(n_qubits=1)
        reg = c.add_classical_register("c", 1)
        c.h(0)
        c.measure(0, reg[0])
        noise = {0: AmplitudeDamping(T1=1e-3, t=1e-6)}
        result = ManyShotRunner().run(c, n_shots=20, noise_config=noise, seed=7)
        assert isinstance(result, AggregateResult)
        assert result.n_shots == 20

    def test_mcm_non_clifford_non_pauli_noise_raises_in_many_shot_runner(self):
        """MCM + non-Clifford + non-Pauli noise via ManyShotRunner raises NotImplementedError.

        ManyShotRunner is Stim-based (Clifford-only).  The non-Clifford gate
        check fires before the noise fallback, so a non-Clifford circuit always
        raises regardless of noise type.  Users must route these circuits through
        BackendSelector → TrajectoryBackend directly.
        """
        c = _mcm_with_t_circuit()
        noise = {0: AmplitudeDamping(T1=1e-3, t=1e-6)}
        with pytest.raises(NotImplementedError):
            ManyShotRunner().run(c, n_shots=5, noise_config=noise, seed=0)


# ---------------------------------------------------------------------------
# StimTableauBackend MCM semantics
# ---------------------------------------------------------------------------

class TestStimMCMSemantics:

    def test_non_mcm_circuit_populates_final_pauli_frames(self):
        """Non-MCM Clifford circuit: final_pauli_frames has one string per shot."""
        c = _clifford_circuit_2q()
        backend = StimTableauBackend()
        result = backend.run(c, n_shots=10, seed=42)
        stim_res = result.meta["stim_result"]
        assert len(stim_res.final_pauli_frames) == 10
        assert len(stim_res.sampled_error_products) == 0
        for frame in stim_res.final_pauli_frames:
            assert len(frame) == c.n_qubits
            assert all(ch in "IXYZ" for ch in frame)

    def test_mcm_circuit_has_no_final_pauli_frames(self):
        """MCM circuit: final_pauli_frames is empty; sampled_error_products is populated."""
        c = _mcm_circuit()
        noise = depolarizing_error(0.05)
        backend = StimTableauBackend()
        result = backend.run(c, noise_model=noise, n_shots=20, seed=0)
        stim_res = result.meta["stim_result"]
        assert len(stim_res.final_pauli_frames) == 0, (
            "MCM circuits must not populate final_pauli_frames"
        )
        assert len(stim_res.sampled_error_products) == 20

    def test_no_noise_non_mcm_all_identity_frames(self):
        """Noiseless non-MCM circuit: every shot's output-frame Pauli is all-I."""
        c = _clifford_circuit_2q()
        backend = StimTableauBackend()
        result = backend.run(c, n_shots=30, seed=7)
        stim_res = result.meta["stim_result"]
        for frame in stim_res.final_pauli_frames:
            assert frame == "I" * c.n_qubits, f"Expected all-I frame, got {frame!r}"


# ---------------------------------------------------------------------------
# ManyShotRunner — CorrelatedPauliError and MCM aggregation
# ---------------------------------------------------------------------------

class TestManyShotRunnerPhase4B:

    def test_correlated_pauli_stays_on_stim_path(self):
        """CorrelatedPauliError no longer triggers the TrajectoryBackend fallback."""
        c = _clifford_circuit_2q()
        noise = {1: CorrelatedPauliError({'ZZ': 0.01})}
        result = ManyShotRunner().run(c, n_shots=10, noise_config=noise, seed=42)
        assert isinstance(result, AggregateResult)
        assert result.n_shots == 10

    def test_correlated_pauli_p1_counts_both_qubits(self):
        """ZZ at p=1.0 fires on every shot: both qubits show Z errors."""
        c = _clifford_circuit_2q()
        # Gate index 1 is CNOT — apply deterministic ZZ error after it.
        noise = {1: CorrelatedPauliError({'ZZ': 1.0})}
        n = 20
        result = ManyShotRunner().run(c, n_shots=n, noise_config=noise, seed=0)
        # Both qubit 0 and qubit 1 must accumulate exactly n Z errors at op 1.
        assert result.counts_matrix[0, 1] == n, (
            f"Qubit 0 should have {n} errors at op 1, got {result.counts_matrix[0, 1]}"
        )
        assert result.counts_matrix[1, 1] == n, (
            f"Qubit 1 should have {n} errors at op 1, got {result.counts_matrix[1, 1]}"
        )

    def test_correlated_pauli_identity_sample_not_counted(self):
        """Identity-heavy correlated channel: near-zero noise → nearly no error counts."""
        c = _clifford_circuit_2q()
        # II has probability 0.9999; ZZ only 0.0001 — expect 0 counts in 50 shots.
        noise = {1: CorrelatedPauliError({'ZZ': 0.0001})}
        result = ManyShotRunner().run(c, n_shots=50, noise_config=noise, seed=0)
        assert result.counts_matrix.sum() == 0, (
            "Near-zero probability channel should produce no errors in 50 shots"
        )

    def test_mcm_aggregate_has_paired_sim_fields(self):
        """MCM AggregateResult: Phase 4B populates final_pauli_frames and branch_diverged.

        Phase 4B re-enables final_pauli_frames for MCM circuits via the paired
        ideal/noisy simulation path.  Both final_pauli_frames (Option A) and
        branch_diverged (Option B) must be present and have one entry per shot.
        sampled_error_products is still collected for debug purposes.
        """
        # ManyShotRunner only supports Clifford circuits, so use a Clifford MCM circuit.
        c = _mcm_circuit()
        noise = {0: depolarizing_error(0.05)}
        result = ManyShotRunner().run(c, n_shots=15, noise_config=noise, seed=1)

        # Phase 4B: final_pauli_frames now populated via paired-sim (not None).
        assert result.final_pauli_frames is not None, (
            "Phase 4B must populate final_pauli_frames for MCM circuits via run_paired()"
        )
        assert len(result.final_pauli_frames) == 15

        # branch_diverged: new Phase 4B field, one bool per shot.
        assert result.branch_diverged is not None, (
            "Phase 4B must populate branch_diverged for MCM circuits"
        )
        assert len(result.branch_diverged) == 15

        # sampled_error_products still collected as a debug-only fallback.
        assert result.sampled_error_products is not None
        assert len(result.sampled_error_products) == 15

    def test_non_mcm_aggregate_has_final_pauli_frames(self):
        """Non-MCM AggregateResult: final_pauli_frames set; sampled_error_products absent."""
        c = _clifford_circuit_2q()
        result = ManyShotRunner().run(c, n_shots=10, seed=5)
        assert result.final_pauli_frames is not None
        assert len(result.final_pauli_frames) == 10
        assert result.sampled_error_products is None


# ---------------------------------------------------------------------------
# IR validation
# ---------------------------------------------------------------------------

class TestIRValidation:

    def test_conditional_op_value_0_valid(self):
        c = Circuit(n_qubits=1)
        c.classical_registers.append(ClassicalRegister(name="c", size=1, offset=0))
        cbit = ClassicalBit(name="c[0]", index=0)
        op = Operation(gate=gates.X, qubits=(0,), t=1)
        cond = ConditionalOp(inner=op, condition=cbit, value=0)
        assert cond.value == 0

    def test_conditional_op_value_1_valid(self):
        c = Circuit(n_qubits=1)
        c.classical_registers.append(ClassicalRegister(name="c", size=1, offset=0))
        cbit = ClassicalBit(name="c[0]", index=0)
        op = Operation(gate=gates.X, qubits=(0,), t=1)
        cond = ConditionalOp(inner=op, condition=cbit, value=1)
        assert cond.value == 1

    def test_conditional_op_invalid_value_raises(self):
        cbit = ClassicalBit(name="c[0]", index=0)
        op = Operation(gate=gates.X, qubits=(0,), t=1)
        with pytest.raises(ValueError, match="0 or 1"):
            ConditionalOp(inner=op, condition=cbit, value=2)

    def test_measurement_z_basis_valid(self):
        cbit = ClassicalBit(name="c[0]", index=0)
        m = Measurement(qubit=0, cbit=cbit, t=0, basis="Z")
        assert m.basis == "Z"

    def test_measurement_non_z_basis_raises(self):
        cbit = ClassicalBit(name="c[0]", index=0)
        with pytest.raises(ValueError, match="basis must be 'Z'"):
            Measurement(qubit=0, cbit=cbit, t=0, basis="X")

    def test_measurement_negative_qubit_raises(self):
        cbit = ClassicalBit(name="c[0]", index=0)
        with pytest.raises(ValueError, match="non-negative integer"):
            Measurement(qubit=-1, cbit=cbit, t=0)


# ---------------------------------------------------------------------------
# QASM reset handling
# ---------------------------------------------------------------------------

class TestQASMReset:

    def test_qasm_reset_raises_parse_error(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        h q[0];
        reset q[0];
        measure q[0] -> c[0];
        """
        with pytest.raises(QASMParseError, match="reset"):
            from_qasm(qasm)

    def test_qasm_without_reset_parses_ok(self):
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        """
        circuit = from_qasm(qasm)
        assert circuit.n_qubits == 1


# ---------------------------------------------------------------------------
# MetricReport coercion
# ---------------------------------------------------------------------------

class TestMetricReportCoercion:

    def test_list_limitations_coerced_to_tuple(self):
        from noisiq.results.metrics import MetricKind, MetricReport
        report = MetricReport(
            name="test",
            kind=MetricKind.ZERO_ERROR_FRACTION,
            value=0.5,
            uncertainty=0.01,
            n_shots=100,
            backend="StimTableauBackend",
            noise_model="depolarizing",
            method="test",
            definition="test definition",
            limitations=["a limitation"],
        )
        assert isinstance(report.limitations, tuple)
        assert report.limitations == ("a limitation",)

    def test_list_assumptions_coerced_to_tuple(self):
        from noisiq.results.metrics import MetricKind, MetricReport
        report = MetricReport(
            name="test",
            kind=MetricKind.PROCESS_FIDELITY_PAULI,
            value=0.9,
            uncertainty=None,
            n_shots=None,
            backend="StimTableauBackend",
            noise_model="depolarizing",
            method="test",
            definition="test definition",
            assumptions=["an assumption", "another"],
        )
        assert isinstance(report.assumptions, tuple)
        assert len(report.assumptions) == 2
