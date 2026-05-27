import numpy as np
import pytest

from noisiq.ir import Circuit
from noisiq.ir import gates as ir_gates
from noisiq.backends import AggregateResult, ManyShotRunner, TrajectoryBackend
from noisiq.noise import (
    AmplitudeDamping,
    CoherentRotation,
    CombinedChannel,
    CorrelatedPauliError,
    PauliError,
    depolarizing_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _x_circuit():
    c = Circuit(1)
    c.x(0)
    return c

def _bell_circuit():
    c = Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    return c


# ---------------------------------------------------------------------------
# 7.1  Correctness: depolarizing error rate matches theory
# ---------------------------------------------------------------------------

def test_depolarizing_error_rate():
    # depolarizing_error(0.1) sets p_x=p_y=p_z=0.1/3, so P(non-I) = 0.1.
    noise = depolarizing_error(0.1)
    result = TrajectoryBackend().run_aggregate(
        _x_circuit(), noise_model=noise, n_shots=10_000, seed=42
    )
    measured = result.counts_matrix[0, 0] / result.n_shots
    assert abs(measured - 0.1) < 0.02, f"Expected ~0.1, got {measured:.4f}"


# ---------------------------------------------------------------------------
# 7.2  Correctness: final_state matches run() for the same seed
# ---------------------------------------------------------------------------

def test_final_state_matches_run():
    noise = depolarizing_error(0.1)
    circuit = _bell_circuit()
    ref = TrajectoryBackend().run(circuit, noise_model=noise, n_shots=200, seed=7)
    agg = TrajectoryBackend().run_aggregate(circuit, noise_model=noise, n_shots=200, seed=7)
    np.testing.assert_allclose(agg.final_state, ref.final_state, atol=1e-10)


# ---------------------------------------------------------------------------
# 7.3  Correctness: noiseless circuit produces all-zero counts
# ---------------------------------------------------------------------------

def test_noiseless_zero_counts():
    result = TrajectoryBackend().run_aggregate(_bell_circuit(), noise_model=None, n_shots=200)
    assert result.counts_matrix.sum() == 0
    assert result.zero_error_fraction == 1.0


# ---------------------------------------------------------------------------
# 7.4  Correctness: zero_error_shots consistency (boundary tests)
# ---------------------------------------------------------------------------

def test_zero_error_shots_all_true_when_no_noise():
    result = TrajectoryBackend().run_aggregate(_x_circuit(), noise_model=None, n_shots=50)
    assert result.zero_error_shots.all()
    assert result.counts_matrix.sum() == 0

def test_zero_error_shots_all_false_when_deterministic_noise():
    # p_x=1.0 → non-I Pauli fires every shot on every gate.
    noise = PauliError(p_x=1.0, p_y=0.0, p_z=0.0)
    result = TrajectoryBackend().run_aggregate(
        _x_circuit(), noise_model=noise, n_shots=50, seed=0
    )
    assert not result.zero_error_shots.any()
    assert result.counts_matrix[0, 0] == 50


# ---------------------------------------------------------------------------
# 7.5  Backwards compatibility: ManyShotRunner still returns final_state=None
# ---------------------------------------------------------------------------

def test_many_shot_runner_final_state_is_none():
    result = ManyShotRunner().run(_bell_circuit(), n_shots=10)
    assert result.final_state is None


# ---------------------------------------------------------------------------
# 7.6  Kraus channel: AmplitudeDamping counts match decay probability
# ---------------------------------------------------------------------------

def test_amplitude_damping_error_rate():
    # Start in |1⟩ (X gate), then apply T1 noise.
    # γ = 1 - exp(-0.5) ≈ 0.393 → expected error rate ≈ γ.
    c = Circuit(1)
    c.x(0)
    gamma = 1.0 - np.exp(-5e-6 / 10e-6)
    noise = {0: AmplitudeDamping(T1=10e-6, t=5e-6)}
    result = TrajectoryBackend().run_aggregate(c, noise_model=noise, n_shots=10_000, seed=42)
    measured = result.counts_matrix[0, 0] / result.n_shots
    assert abs(measured - gamma) < 0.02, f"Expected ~{gamma:.3f}, got {measured:.4f}"


# ---------------------------------------------------------------------------
# 7.7  CombinedChannel: both sub-channels contribute to counts
# ---------------------------------------------------------------------------

def test_combined_channel_nonzero_counts():
    noise = CombinedChannel([
        AmplitudeDamping(T1=10e-6, t=5e-6),
        depolarizing_error(0.1),
    ])
    c = Circuit(1)
    c.x(0)
    result = TrajectoryBackend().run_aggregate(c, noise_model=noise, n_shots=500, seed=0)
    assert result.counts_matrix.sum() > 0


# ---------------------------------------------------------------------------
# 7.8  CCZ circuit: no exception, counts non-zero
# ---------------------------------------------------------------------------

def test_ccz_circuit_no_exception():
    c = Circuit(3)
    c.h(0)
    c.h(1)
    c.ccz(0, 1, 2)
    noise = depolarizing_error(0.05)
    result = TrajectoryBackend().run_aggregate(c, noise_model=noise, n_shots=200, seed=42)
    assert result.counts_matrix.sum() > 0
    assert result.final_state.shape == (8, 8)


# ---------------------------------------------------------------------------
# 7.9  CoherentRotation: errors are detected (identity-check path)
# ---------------------------------------------------------------------------

def test_coherent_rotation_errors_counted():
    # CoherentRotation has one Kraus operator, so k is always 0.
    # Without the identity check, every shot would be counted as error-free.
    # With the fix, a non-trivial rotation (epsilon != 0) registers every shot.
    c = Circuit(1)
    c.x(0)
    noise = {0: CoherentRotation(axis='Z', epsilon=np.pi / 4)}
    result = TrajectoryBackend().run_aggregate(c, noise_model=noise, n_shots=100, seed=0)
    # Every shot has the coherent error applied.
    assert result.counts_matrix[0, 0] == 100
    assert result.zero_error_fraction == 0.0

def test_coherent_rotation_zero_epsilon_not_counted():
    # epsilon=0 → U_err = I → identity check → no error recorded.
    c = Circuit(1)
    c.x(0)
    noise = {0: CoherentRotation(axis='Z', epsilon=0.0)}
    result = TrajectoryBackend().run_aggregate(c, noise_model=noise, n_shots=50, seed=0)
    assert result.counts_matrix.sum() == 0
    assert result.zero_error_fraction == 1.0


# ---------------------------------------------------------------------------
# Shape and validation
# ---------------------------------------------------------------------------

def test_counts_matrix_shape():
    result = TrajectoryBackend().run_aggregate(_bell_circuit(), n_shots=10)
    assert result.counts_matrix.shape == (2, 2)

def test_final_state_shape():
    result = TrajectoryBackend().run_aggregate(_bell_circuit(), n_shots=10)
    assert result.final_state.shape == (4, 4)

def test_n_shots_zero_raises():
    with pytest.raises(ValueError, match="n_shots must be >= 1"):
        TrajectoryBackend().run_aggregate(_x_circuit(), n_shots=0)

def test_reproducibility():
    noise = depolarizing_error(0.2)
    r1 = TrajectoryBackend().run_aggregate(_bell_circuit(), noise_model=noise, n_shots=100, seed=55)
    r2 = TrajectoryBackend().run_aggregate(_bell_circuit(), noise_model=noise, n_shots=100, seed=55)
    np.testing.assert_array_equal(r1.counts_matrix, r2.counts_matrix)
    np.testing.assert_array_equal(r1.zero_error_shots, r2.zero_error_shots)
