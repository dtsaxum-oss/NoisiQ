import pytest
import numpy as np
from noisiq.ir import Circuit, gates
from noisiq.backends import ManyShotRunner, AggregateResult
from noisiq.noise import PauliError, bit_flip_error, depolarizing_error


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_n_shots_zero_raises():
    runner = ManyShotRunner()
    with pytest.raises(ValueError, match="n_shots must be >= 1"):
        runner.run(_single_x_circuit(), n_shots=0)

def test_n_shots_negative_raises():
    runner = ManyShotRunner()
    with pytest.raises(ValueError, match="n_shots must be >= 1"):
        runner.run(_single_x_circuit(), n_shots=-5)


# ---------------------------------------------------------------------------
# Shape and structure
# ---------------------------------------------------------------------------

def test_counts_matrix_shape_single_qubit():
    circuit = _single_x_circuit()
    result = ManyShotRunner().run(circuit, n_shots=10)
    assert result.counts_matrix.shape == (1, 1)

def test_counts_matrix_shape_bell():
    circuit = _bell_circuit()
    result = ManyShotRunner().run(circuit, n_shots=10)
    # 2 qubits, 2 operations
    assert result.counts_matrix.shape == (2, 2)

def test_counts_matrix_shape_ghz():
    c = Circuit(n_qubits=3)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    c.add_gate(gates.CNOT, (1, 2))
    result = ManyShotRunner().run(c, n_shots=10)
    assert result.counts_matrix.shape == (3, 3)

def test_aggregate_result_properties():
    circuit = _bell_circuit()
    n_shots = 20
    result = ManyShotRunner().run(circuit, n_shots=n_shots)
    assert result.n_qubits == 2
    assert result.n_timesteps == 2
    assert result.n_shots == n_shots
    assert result.circuit is circuit

def test_counts_matrix_dtype_is_integer():
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=10)
    assert np.issubdtype(result.counts_matrix.dtype, np.integer)


# ---------------------------------------------------------------------------
# No-noise baseline
# ---------------------------------------------------------------------------

def test_no_noise_config_zero_counts():
    result = ManyShotRunner().run(_bell_circuit(), n_shots=100)
    assert result.counts_matrix.sum() == 0

def test_no_noise_error_rate_matrix_all_zero():
    result = ManyShotRunner().run(_bell_circuit(), n_shots=50)
    assert np.all(result.error_rate_matrix == 0.0)


# ---------------------------------------------------------------------------
# Deterministic noise (p=1.0)
# ---------------------------------------------------------------------------

def test_deterministic_bitflip_counts_equal_n_shots():
    # p_x=1.0 on the only gate → every shot records an X error on qubit 0
    n_shots = 50
    noise_config = {0: bit_flip_error(1.0)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=n_shots,
                                  noise_config=noise_config, seed=42)
    assert result.counts_matrix[0, 0] == n_shots

def test_deterministic_noise_both_qubits_cnot():
    # Apply p_x=1.0 on CNOT (op index 1); CNOT targets both qubits
    n_shots = 30
    noise_config = {1: bit_flip_error(1.0)}
    circuit = _bell_circuit()
    result = ManyShotRunner().run(circuit, n_shots=n_shots,
                                  noise_config=noise_config, seed=0)
    # Both qubits touched by the CNOT should accumulate errors
    assert result.counts_matrix[:, 1].sum() == n_shots * 2


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_identical_counts():
    circuit = _bell_circuit()
    noise_config = {0: depolarizing_error(0.15), 1: depolarizing_error(0.15)}
    r1 = ManyShotRunner().run(circuit, n_shots=200, noise_config=noise_config, seed=42)
    r2 = ManyShotRunner().run(circuit, n_shots=200, noise_config=noise_config, seed=42)
    np.testing.assert_array_equal(r1.counts_matrix, r2.counts_matrix)
    np.testing.assert_array_equal(r1.zero_error_shots, r2.zero_error_shots)

def test_different_seeds_produce_different_counts():
    noise_config = {0: depolarizing_error(0.3)}
    r1 = ManyShotRunner().run(_single_x_circuit(), n_shots=200,
                               noise_config=noise_config, seed=0)
    r2 = ManyShotRunner().run(_single_x_circuit(), n_shots=200,
                               noise_config=noise_config, seed=99)
    assert not np.array_equal(r1.counts_matrix, r2.counts_matrix)


# ---------------------------------------------------------------------------
# error_rate_matrix property
# ---------------------------------------------------------------------------

def test_error_rate_matrix_equals_counts_over_shots():
    n_shots = 40
    noise_config = {0: bit_flip_error(1.0)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=n_shots,
                                  noise_config=noise_config, seed=42)
    np.testing.assert_allclose(
        result.error_rate_matrix,
        result.counts_matrix / n_shots,
    )

def test_error_rate_matrix_bounded_zero_to_one():
    noise_config = {0: depolarizing_error(0.5)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=500,
                                  noise_config=noise_config, seed=7)
    assert result.error_rate_matrix.min() >= 0.0
    assert result.error_rate_matrix.max() <= 1.0


# ---------------------------------------------------------------------------
# Statistical convergence
# ---------------------------------------------------------------------------

def test_statistical_convergence_bitflip():
    # p_x=0.2 on a single gate → measured rate should be close to 0.2
    noise_config = {0: PauliError(p_x=0.2, p_y=0.0, p_z=0.0)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=10_000,
                                  noise_config=noise_config, seed=42)
    measured = result.error_rate_matrix[0, 0]
    assert abs(measured - 0.2) < 0.02, f"Expected ~0.2, got {measured:.4f}"

# ---------------------------------------------------------------------------
# zero_error_shots and zero_error_fraction
# ---------------------------------------------------------------------------

def test_zero_error_shots_shape():
    n_shots = 15
    result = ManyShotRunner().run(_bell_circuit(), n_shots=n_shots)
    assert result.zero_error_shots.shape == (n_shots,)

def test_zero_error_shots_dtype_is_bool():
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=10)
    assert result.zero_error_shots.dtype == bool

def test_zero_error_shots_all_true_when_no_noise():
    result = ManyShotRunner().run(_bell_circuit(), n_shots=50)
    assert result.zero_error_shots.all()

def test_zero_error_shots_all_false_when_deterministic_noise():
    # p_x=1.0 → every shot has an error
    noise_config = {0: bit_flip_error(1.0)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=30,
                                  noise_config=noise_config, seed=0)
    assert not result.zero_error_shots.any()

def test_zero_error_fraction_no_noise_is_one():
    result = ManyShotRunner().run(_bell_circuit(), n_shots=50)
    assert result.zero_error_fraction == 1.0

def test_zero_error_fraction_deterministic_noise_is_zero():
    noise_config = {0: bit_flip_error(1.0)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=30,
                                  noise_config=noise_config, seed=0)
    assert result.zero_error_fraction == 0.0

def test_zero_error_fraction_bounded():
    noise_config = {0: depolarizing_error(0.5)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=200,
                                  noise_config=noise_config, seed=7)
    assert 0.0 <= result.zero_error_fraction <= 1.0


def test_statistical_convergence_depolarizing():
    # Depolarizing p=0.3 → each Pauli (X/Y/Z) has p=0.1 → total error rate = 0.3
    noise_config = {0: depolarizing_error(0.3)}
    result = ManyShotRunner().run(_single_x_circuit(), n_shots=10_000,
                                  noise_config=noise_config, seed=42)
    measured = result.error_rate_matrix[0, 0]
    assert abs(measured - 0.3) < 0.02, f"Expected ~0.3, got {measured:.4f}"
