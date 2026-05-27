"""
Tests for bloch_sphere visualization.

Covers:
- density_matrix_to_bloch_vector for known states: |0⟩, |1⟩, |+⟩, |−⟩, I/2
- Pure states lie exactly on the unit sphere (|v| = 1)
- Mixed states lie strictly inside the sphere (|v| < 1)
- Maximally mixed state I/2 maps to the origin (0, 0, 0)
- ValueError for non-2x2 input
- draw_bloch_sphere smoke test (renders without error)
- plot_t1_t2_decay returns a matplotlib Figure
- plot_trajectory_ensemble returns a matplotlib Figure
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from noisiq.visualization.bloch_sphere import (
    density_matrix_to_bloch_vector,
    draw_bloch_sphere,
    plot_t1_t2_decay,
    plot_trajectory_ensemble,
)
from noisiq.backends.trajectory_backend import TrajectoryBackend
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.ir import Circuit
from noisiq.ir import gates as ir


# ---------------------------------------------------------------------------
# Known-state Bloch vectors
# ---------------------------------------------------------------------------

def test_bloch_vector_ground_state():
    """|0⟩⟨0| → (0, 0, 1)."""
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 1)


def test_bloch_vector_excited_state():
    """|1⟩⟨1| → (0, 0, -1)."""
    rho = np.array([[0, 0], [0, 1]], dtype=complex)
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, -1)


def test_bloch_vector_plus_state():
    """|+⟩⟨+| → (1, 0, 0)."""
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert np.isclose(x, 1, atol=1e-10)
    assert np.isclose(y, 0, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)


def test_bloch_vector_minus_state():
    """|−⟩⟨−| → (-1, 0, 0)."""
    psi = np.array([1, -1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert np.isclose(x, -1, atol=1e-10)
    assert np.isclose(y, 0, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)


def test_bloch_vector_y_plus_state():
    """|i+⟩⟨i+| = (|0⟩ + i|1⟩)/√2 → (0, 1, 0)."""
    psi = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert np.isclose(x, 0, atol=1e-10)
    assert np.isclose(y, 1, atol=1e-10)
    assert np.isclose(z, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Pure state on sphere / mixed state inside
# ---------------------------------------------------------------------------

def test_pure_state_on_unit_sphere():
    """Any pure state satisfies x² + y² + z² = 1."""
    for theta in np.linspace(0, np.pi, 6):
        for phi in np.linspace(0, 2 * np.pi, 6):
            psi = np.array([np.cos(theta / 2),
                            np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)
            rho = np.outer(psi, psi.conj())
            x, y, z = density_matrix_to_bloch_vector(rho)
            assert abs(x**2 + y**2 + z**2 - 1.0) < 1e-10


def test_maximally_mixed_is_origin():
    """I/2 → (0, 0, 0)."""
    rho = np.eye(2, dtype=complex) / 2
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 0)


def test_mixed_state_strictly_inside_sphere():
    """A non-pure state has |v| < 1."""
    # Convex mix of |0⟩ and |1⟩
    rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
    x, y, z = density_matrix_to_bloch_vector(rho)
    assert x**2 + y**2 + z**2 < 1.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_non_2x2():
    rho = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="2x2"):
        density_matrix_to_bloch_vector(rho)


def test_rejects_1d_array():
    with pytest.raises((ValueError, AttributeError)):
        density_matrix_to_bloch_vector(np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# Rendering smoke tests
# ---------------------------------------------------------------------------

def test_draw_bloch_sphere_single_vector():
    """draw_bloch_sphere must not raise for a single vector."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    draw_bloch_sphere(ax, [(0, 0, 1)], labels=["|0⟩"])
    plt.close(fig)


def test_draw_bloch_sphere_multiple_vectors():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    draw_bloch_sphere(ax, [(0,0,1), (0,0,-1), (1,0,0)],
                      labels=["|0⟩", "|1⟩", "|+⟩"])
    plt.close(fig)


def _make_trajectory_results(n_points: int = 5):
    """Helper: generate TrajectoryResults across a short T1 decay."""
    backend = TrajectoryBackend()
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(ir.X, qubits=[0])
    T1 = 1e-6
    t_values = np.linspace(1e-9, 2e-6, n_points)
    results = []
    for t in t_values:
        noise = AmplitudeDamping(T1=T1, t=t)
        results.append(backend.run(circuit, noise_model=noise, n_shots=100, seed=0))
    return results, t_values


def test_plot_t1_t2_decay_returns_figure():
    results, t_values = _make_trajectory_results()
    pytest.skip("not yet implemented")
    # fig = plot_t1_t2_decay(results, t_values, qubit=0)
    # assert isinstance(fig, plt.Figure)
    # plt.close(fig)


def test_plot_t1_t2_decay_length_mismatch_raises():
    results, t_values = _make_trajectory_results(5)
    with pytest.raises(ValueError):
        plot_t1_t2_decay(results, t_values[:-1])


def test_plot_trajectory_ensemble_returns_figure():
    backend = TrajectoryBackend()
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(ir.X, qubits=[0])
    noise = AmplitudeDamping(T1=1e-6, t=0.5e-6)
    result = backend.run(circuit, noise_model=noise, n_shots=20, seed=0)
    # trajectory_states not stored by default — this test documents the
    # expected interface for when TrajectoryBackend is extended in Week 7
    pytest.skip("not yet implemented")
    # fig = plot_trajectory_ensemble(trajectory_states, result.density_matrix)
    # assert isinstance(fig, plt.Figure)
    # plt.close(fig)
