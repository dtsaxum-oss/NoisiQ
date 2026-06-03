"""
Tests for M4: Modal-error GIF from many-shot runs.

Covers:
- counts_by_pauli field on AggregateResult
- modal_pauli_matrix property
- PauliFrame.apply_swap / apply_s_dag / apply_gate fixes
- build_modal_trajectory_frames helper
"""

import numpy as np
import pytest

from noisiq.ir import Circuit, gates
from noisiq.backends import ManyShotRunner, AggregateResult
from noisiq.noise import PauliError, bit_flip_error, depolarizing_error
from noisiq.visualization.pauli_frame_tracker import (
    PauliFrame,
    build_modal_trajectory_frames,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_x_circuit():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.X, (0,))
    return c


def _bell_circuit():
    c = Circuit(n_qubits=2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    return c


def _h_circuit():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.H, (0,))
    return c


def _run_deterministic_x(n_shots=50):
    """Single qubit, bit-flip p=1.0 on the only gate — every shot gets X."""
    circuit = _single_x_circuit()
    noise_config = {0: bit_flip_error(1.0)}
    return ManyShotRunner().run(circuit, n_shots=n_shots,
                                noise_config=noise_config, seed=0)


def _run_no_noise(n_shots=20):
    circuit = _bell_circuit()
    return ManyShotRunner().run(circuit, n_shots=n_shots)


# ---------------------------------------------------------------------------
# counts_by_pauli — shape, dtype, and values
# ---------------------------------------------------------------------------

class TestCountsByPauli:
    def test_shape_single_qubit(self):
        result = ManyShotRunner().run(_single_x_circuit(), n_shots=10)
        assert result.counts_by_pauli is not None
        assert result.counts_by_pauli.shape == (1, 1, 3)

    def test_shape_bell(self):
        result = _run_no_noise()
        assert result.counts_by_pauli is not None
        assert result.counts_by_pauli.shape == (2, 2, 3)

    def test_dtype_is_int64(self):
        result = ManyShotRunner().run(_single_x_circuit(), n_shots=5)
        assert result.counts_by_pauli.dtype == np.int64

    def test_no_noise_all_zeros(self):
        result = _run_no_noise()
        assert result.counts_by_pauli.sum() == 0

    def test_deterministic_x_error_accumulates_x_column(self):
        n_shots = 50
        result = _run_deterministic_x(n_shots)
        # Axis 2 index 0 = X; every shot fires X on qubit 0 at op 0
        assert result.counts_by_pauli[0, 0, 0] == n_shots   # X count
        assert result.counts_by_pauli[0, 0, 1] == 0          # Y count
        assert result.counts_by_pauli[0, 0, 2] == 0          # Z count

    def test_reproducible_with_seed(self):
        circuit = _bell_circuit()
        noise = {0: depolarizing_error(0.2), 1: depolarizing_error(0.2)}
        r1 = ManyShotRunner().run(circuit, n_shots=100, noise_config=noise, seed=7)
        r2 = ManyShotRunner().run(circuit, n_shots=100, noise_config=noise, seed=7)
        np.testing.assert_array_equal(r1.counts_by_pauli, r2.counts_by_pauli)

    def test_x_plus_y_counts_match_total_counts(self):
        """counts_by_pauli[:, :, 0..2].sum(axis=2) should equal counts_matrix."""
        result = _run_deterministic_x(30)
        np.testing.assert_array_equal(
            result.counts_by_pauli.sum(axis=2),
            result.counts_matrix,
        )

    def test_counts_by_pauli_sum_matches_counts_matrix_depolarizing(self):
        circuit = _bell_circuit()
        noise = {0: depolarizing_error(0.3), 1: depolarizing_error(0.3)}
        result = ManyShotRunner().run(circuit, n_shots=500,
                                     noise_config=noise, seed=42)
        np.testing.assert_array_equal(
            result.counts_by_pauli.sum(axis=2),
            result.counts_matrix,
        )


# ---------------------------------------------------------------------------
# modal_pauli_matrix property
# ---------------------------------------------------------------------------

class TestModalPauliMatrix:
    def test_raises_when_counts_by_pauli_is_none(self):
        circuit = _single_x_circuit()
        counts = np.zeros((1, 1), dtype=np.int64)
        result = AggregateResult(
            counts_matrix=counts,
            n_shots=1,
            circuit=circuit,
            zero_error_shots=np.array([True]),
            counts_by_pauli=None,
        )
        with pytest.raises(RuntimeError, match="counts_by_pauli"):
            _ = result.modal_pauli_matrix

    def test_no_noise_all_identity(self):
        result = _run_no_noise()
        modal = result.modal_pauli_matrix
        assert modal.shape == (2, 2)
        assert (modal == 'I').all()

    def test_deterministic_x_returns_x(self):
        result = _run_deterministic_x(n_shots=20)
        modal = result.modal_pauli_matrix
        assert modal[0, 0] == 'X'

    def test_shape_matches_counts_matrix(self):
        result = _run_no_noise()
        assert result.modal_pauli_matrix.shape == result.counts_matrix.shape

    def test_dominant_pauli_wins(self):
        """With Z-only noise (p_z=1), modal Pauli should be Z."""
        circuit = _single_x_circuit()
        noise = {0: PauliError(p_x=0.0, p_y=0.0, p_z=1.0)}
        result = ManyShotRunner().run(circuit, n_shots=30,
                                     noise_config=noise, seed=0)
        assert result.modal_pauli_matrix[0, 0] == 'Z'


# ---------------------------------------------------------------------------
# PauliFrame — new gates (bug fixes)
# ---------------------------------------------------------------------------

class TestPauliFrameNewGates:
    def test_apply_swap_exchanges_x(self):
        frame = PauliFrame(2)
        frame.inject_error(0, 'X')  # X on qubit 0
        frame.apply_swap(0, 1)
        assert frame.get_pauli_string() == 'IX'  # X moved to qubit 1

    def test_apply_swap_exchanges_z(self):
        frame = PauliFrame(2)
        frame.inject_error(1, 'Z')  # Z on qubit 1
        frame.apply_swap(0, 1)
        assert frame.get_pauli_string() == 'ZI'  # Z moved to qubit 0

    def test_apply_swap_identity_both_clean(self):
        frame = PauliFrame(2)
        frame.apply_swap(0, 1)
        assert frame.get_pauli_string() == 'II'

    def test_apply_swap_via_apply_gate(self):
        frame = PauliFrame(2)
        frame.inject_error(0, 'X')
        frame.apply_gate('SWAP', [0, 1])
        assert frame.get_pauli_string() == 'IX'

    def test_apply_s_dag_x_becomes_y(self):
        """S† maps X → Y (up to phase; Pauli frame ignores phase)."""
        frame = PauliFrame(1)
        frame.inject_error(0, 'X')
        frame.apply_s_dag(0)
        assert frame.get_pauli_string() == 'Y'

    def test_apply_s_dag_z_unchanged(self):
        frame = PauliFrame(1)
        frame.inject_error(0, 'Z')
        frame.apply_s_dag(0)
        assert frame.get_pauli_string() == 'Z'

    def test_apply_s_dag_via_apply_gate(self):
        frame = PauliFrame(1)
        frame.inject_error(0, 'X')
        frame.apply_gate('S_DAG', [0])
        assert frame.get_pauli_string() == 'Y'

    def test_apply_s_dag_y_becomes_x(self):
        """S† maps Y → X (up to phase)."""
        frame = PauliFrame(1)
        frame.inject_error(0, 'Y')   # Y = X + Z bits both set
        frame.apply_s_dag(0)
        assert frame.get_pauli_string() == 'X'


# ---------------------------------------------------------------------------
# build_modal_trajectory_frames
# ---------------------------------------------------------------------------

class TestBuildModalTrajectoryFrames:
    def test_returns_dict_keyed_by_layer(self):
        circuit = _bell_circuit()
        result = _run_no_noise()
        frames = build_modal_trajectory_frames(circuit, result)
        assert isinstance(frames, dict)
        # Bell circuit has 2 layers (t=0 and t=1)
        assert set(frames.keys()) == {0, 1}

    def test_no_noise_gives_identity_frames(self):
        circuit = _bell_circuit()
        result = _run_no_noise()
        frames = build_modal_trajectory_frames(circuit, result)
        for t, frame in frames.items():
            assert frame.get_pauli_string() == 'II', (
                f"Expected identity frame at t={t}, got {frame.get_pauli_string()}"
            )

    def test_raises_when_counts_by_pauli_none(self):
        circuit = _single_x_circuit()
        counts = np.zeros((1, 1), dtype=np.int64)
        result = AggregateResult(
            counts_matrix=counts,
            n_shots=1,
            circuit=circuit,
            zero_error_shots=np.array([True]),
            counts_by_pauli=None,
        )
        with pytest.raises(RuntimeError, match="counts_by_pauli"):
            build_modal_trajectory_frames(circuit, result)

    def test_deterministic_x_single_gate_propagates_through_x_gate(self):
        """X error at t=0, then X gate.  X commutes with X so frame stays X."""
        circuit = _single_x_circuit()
        result = _run_deterministic_x(n_shots=20)
        frames = build_modal_trajectory_frames(circuit, result)
        # Single op at t=0; after propagation through X (identity on Pauli frame)
        assert 0 in frames
        assert frames[0].get_pauli_string() == 'X'

    def test_x_error_propagates_through_h(self):
        """X error at t=0 occurs after H fires; cumulative frame captures error at injection point."""
        circuit = _h_circuit()
        noise = {0: bit_flip_error(1.0)}
        result = ManyShotRunner().run(circuit, n_shots=30,
                                     noise_config=noise, seed=0)
        frames = build_modal_trajectory_frames(circuit, result)
        # Cumulative semantics: H is applied first (I→I), then X is injected.
        # frames[0] is the error state at t=0, which is X.
        assert frames[0].get_pauli_string() == 'X'

    def test_frame_is_pauli_frame_instance(self):
        circuit = _bell_circuit()
        result = _run_no_noise()
        frames = build_modal_trajectory_frames(circuit, result)
        for frame in frames.values():
            assert isinstance(frame, PauliFrame)

    def test_each_frame_has_correct_n_qubits(self):
        circuit = _bell_circuit()
        result = _run_no_noise()
        frames = build_modal_trajectory_frames(circuit, result)
        for frame in frames.values():
            assert frame.num_qubits == 2
