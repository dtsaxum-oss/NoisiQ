import matplotlib
matplotlib.use("Agg")  # headless — no display needed in CI

import pytest
import numpy as np
import matplotlib.pyplot as plt

from noisiq.ir import Circuit, gates
from noisiq.backends.many_shot_runner import AggregateResult
from noisiq.visualization.charts.heatmap import plot_error_heatmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(counts: np.ndarray, circuit: Circuit, n_shots: int = 100) -> AggregateResult:
    return AggregateResult(
        counts_matrix=counts,
        n_shots=n_shots,
        circuit=circuit,
        zero_error_shots=np.ones(n_shots, dtype=bool),
        seed=None,
    )

def _single_gate_circuit():
    c = Circuit(n_qubits=1)
    c.add_gate(gates.X, (0,))
    return c

def _bell_circuit():
    c = Circuit(n_qubits=2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    return c

def _ghz_circuit():
    c = Circuit(n_qubits=3)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CNOT, (0, 1))
    c.add_gate(gates.CNOT, (1, 2))
    return c


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_matplotlib_figure():
    circuit = _single_gate_circuit()
    result = _make_result(np.array([[10]]), circuit)
    fig = plot_error_heatmap(result, circuit)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_accepts_existing_axes():
    circuit = _single_gate_circuit()
    result = _make_result(np.array([[5]]), circuit)
    fig, ax = plt.subplots()
    returned = plot_error_heatmap(result, circuit, ax=ax)
    assert returned is fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# Halo paths: zero errors, partial, full saturation
# ---------------------------------------------------------------------------

def test_zero_counts_renders_without_error():
    circuit = _bell_circuit()
    counts = np.zeros((2, 2), dtype=np.int64)
    result = _make_result(counts, circuit)
    fig = plot_error_heatmap(result, circuit)
    assert fig is not None
    plt.close(fig)

def test_max_counts_renders_without_error():
    # All errors on every gate — max intensity (bright red halo)
    circuit = _bell_circuit()
    counts = np.full((2, 2), fill_value=100, dtype=np.int64)
    result = _make_result(counts, circuit, n_shots=100)
    fig = plot_error_heatmap(result, circuit)
    assert fig is not None
    plt.close(fig)

def test_partial_counts_renders_without_error():
    circuit = _bell_circuit()
    counts = np.array([[30, 0], [0, 5]], dtype=np.int64)
    result = _make_result(counts, circuit)
    fig = plot_error_heatmap(result, circuit)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gate type rendering paths
# ---------------------------------------------------------------------------

def test_single_qubit_gates_render():
    c = Circuit(n_qubits=1)
    for g in (gates.H, gates.X, gates.Y, gates.Z, gates.S, gates.T):
        c.add_gate(g, (0,))
    counts = np.zeros((1, 6), dtype=np.int64)
    result = _make_result(counts, c)
    fig = plot_error_heatmap(result, c)
    assert fig is not None
    plt.close(fig)

def test_cnot_renders_without_error():
    circuit = _bell_circuit()
    counts = np.zeros((2, 2), dtype=np.int64)
    result = _make_result(counts, circuit)
    fig = plot_error_heatmap(result, circuit)
    assert fig is not None
    plt.close(fig)

def test_cz_renders_without_error():
    c = Circuit(n_qubits=2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.CZ, (0, 1))
    counts = np.zeros((2, 2), dtype=np.int64)
    result = _make_result(counts, c)
    fig = plot_error_heatmap(result, c)
    assert fig is not None
    plt.close(fig)

def test_multiqubit_ghz_renders():
    circuit = _ghz_circuit()
    counts = np.zeros((3, 3), dtype=np.int64)
    result = _make_result(counts, circuit)
    fig = plot_error_heatmap(result, circuit)
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

def test_default_title_includes_n_shots():
    circuit = _single_gate_circuit()
    result = _make_result(np.array([[0]]), circuit, n_shots=500)
    fig = plot_error_heatmap(result, circuit)
    title_text = fig.axes[0].get_title()
    assert "500" in title_text
    plt.close(fig)

def test_custom_title_appears():
    circuit = _single_gate_circuit()
    result = _make_result(np.array([[0]]), circuit)
    fig = plot_error_heatmap(result, circuit, title="My Test Heatmap")
    title_text = fig.axes[0].get_title()
    assert "My Test Heatmap" in title_text
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cleanup: close all figures after module runs
# ---------------------------------------------------------------------------

def teardown_module(module):
    plt.close("all")
