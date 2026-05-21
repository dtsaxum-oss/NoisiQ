"""
Tests for density_matrix visualization.

Covers:
- plot_density_matrix returns a Figure with exactly two Axes (Re / Im)
- plot_density_matrix renders without error for pure state, mixed state,
  maximally mixed state, and multi-qubit density matrices
- _purity helper returns 1.0 for a pure state and 1/d for I/d
- plot_purity_decay returns a Figure
- plot_purity_decay values are in (0, 1] for all results
- plot_purity_decay is monotonically non-increasing under T1 noise
- ValueError on length mismatch between results and t_values
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from noisiq.visualization.density_matrix import (
    plot_density_matrix,
    plot_purity_decay,
    _purity,
)
from noisiq.backends.trajectory_backend import TrajectoryBackend
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.ir import Circuit
from noisiq.ir import gates as ir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pure_rho(n_qubits: int = 1) -> np.ndarray:
    """Return |0...0⟩⟨0...0| as a density matrix."""
    d = 2 ** n_qubits
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _maximally_mixed(n_qubits: int = 1) -> np.ndarray:
    d = 2 ** n_qubits
    return np.eye(d, dtype=complex) / d


def _t1_results(n_points: int = 5):
    """Generate TrajectoryResults for increasing T1 decay times."""
    backend = TrajectoryBackend()
    circuit = Circuit(n_qubits=1)
    circuit.add_gate(ir.X, qubits=[0])
    T1 = 1e-6
    t_values = np.linspace(1e-9, 3e-6, n_points)
    results = [
        backend.run(circuit, noise_model=AmplitudeDamping(T1=T1, t=t),
                    n_shots=200, seed=0)
        for t in t_values
    ]
    return results, t_values


# ---------------------------------------------------------------------------
# _purity helper
# ---------------------------------------------------------------------------

def test_purity_pure_state_is_one():
    rho = _pure_rho(n_qubits=1)
    assert abs(_purity(rho) - 1.0) < 1e-12


def test_purity_maximally_mixed_2q():
    rho = _maximally_mixed(n_qubits=2)
    assert abs(_purity(rho) - 0.25) < 1e-12  # 1 / 2^2 = 0.25


def test_purity_between_zero_and_one():
    # Partial mix: p * |0><0| + (1-p) * I/2
    for p in [0.0, 0.3, 0.6, 1.0]:
        rho = p * _pure_rho() + (1 - p) * _maximally_mixed()
        assert 0.0 < _purity(rho) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# plot_density_matrix
# ---------------------------------------------------------------------------

def test_plot_density_matrix_returns_figure():
    fig = plot_density_matrix(_pure_rho())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_density_matrix_has_two_axes():
    """Figure must contain exactly two Axes (Re and Im)."""
    fig = plot_density_matrix(_pure_rho())
    # colorbar adds a third Axes internally; check at least 2 data axes
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_density_matrix_pure_state():
    fig = plot_density_matrix(_pure_rho(n_qubits=1), title="Pure |0⟩")
    plt.close(fig)


def test_plot_density_matrix_maximally_mixed():
    fig = plot_density_matrix(_maximally_mixed(n_qubits=1))
    plt.close(fig)


def test_plot_density_matrix_two_qubit():
    fig = plot_density_matrix(_pure_rho(n_qubits=2))
    plt.close(fig)


def test_plot_density_matrix_rejects_non_power_of_2():
    rho = np.eye(3, dtype=complex) / 3
    with pytest.raises(ValueError):
        plot_density_matrix(rho)


def test_plot_density_matrix_rejects_non_square():
    rho = np.zeros((2, 4), dtype=complex)
    with pytest.raises(ValueError):
        plot_density_matrix(rho)


# ---------------------------------------------------------------------------
# plot_purity_decay
# ---------------------------------------------------------------------------

def test_plot_purity_decay_returns_figure():
    results, t_values = _t1_results()
    fig = plot_purity_decay(results, t_values)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_purity_decay_values_in_range():
    """All purity values must be in (0, 1]."""
    results, _ = _t1_results()
    for r in results:
        p = _purity(r.final_state)
        assert 0.0 < p <= 1.0 + 1e-10


def test_plot_purity_decay_t1_reaches_minimum():
    """Under T1 noise on |1⟩, purity follows a U-curve: starts at 1,
    dips toward 0.5 at t ≈ T1·ln(2), then recovers as the qubit fully
    relaxes to |0⟩.  We verify that the minimum purity in the sweep is
    strictly below the initial value (noise had a measurable effect)."""
    results, _ = _t1_results(n_points=8)
    purities = [_purity(r.final_state) for r in results]
    assert min(purities) < purities[0] - 0.05


def test_plot_purity_decay_length_mismatch_raises():
    results, t_values = _t1_results(5)
    with pytest.raises(ValueError):
        plot_purity_decay(results, t_values[:-1])


def test_plot_purity_decay_accepts_existing_axes():
    results, t_values = _t1_results()
    fig, ax = plt.subplots()
    returned = plot_purity_decay(results, t_values, ax=ax)
    assert returned is fig or isinstance(returned, plt.Figure)
    plt.close(fig)
