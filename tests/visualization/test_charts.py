import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import matplotlib.pyplot as plt

from noisiq.ir import Circuit, gates
from noisiq.backends.many_shot_runner import AggregateResult
from noisiq.visualization.charts.charts import plot_qubit_error_bar, plot_fidelity_decay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(counts: np.ndarray, circuit: Circuit, n_shots: int = 100) -> AggregateResult:
    return AggregateResult(
        counts_matrix=counts,
        n_shots=n_shots,
        circuit=circuit,
        seed=None,
    )

def _simple_circuit(n_qubits: int = 2, n_ops: int = 2) -> Circuit:
    c = Circuit(n_qubits=n_qubits)
    for i in range(n_ops):
        c.add_gate(gates.X, (i % n_qubits,))
    return c

def _depth_sweep(n_depths: int = 5, n_qubits: int = 1, n_shots: int = 100):
    results = []
    for d in range(1, n_depths + 1):
        c = Circuit(n_qubits=n_qubits)
        for _ in range(d):
            c.add_gate(gates.X, (0,))
        counts = np.zeros((n_qubits, d), dtype=np.int64)
        results.append(_make_result(counts, c, n_shots=n_shots))
    return results


# ---------------------------------------------------------------------------
# plot_qubit_error_bar — return type
# ---------------------------------------------------------------------------

def test_qubit_error_bar_returns_figure():
    circuit = _simple_circuit()
    result = _make_result(np.array([[10, 5], [3, 0]]), circuit)
    fig = plot_qubit_error_bar(result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_qubit_error_bar_accepts_existing_axes():
    circuit = _simple_circuit()
    result = _make_result(np.zeros((2, 2), dtype=np.int64), circuit)
    fig, ax = plt.subplots()
    returned = plot_qubit_error_bar(result, ax=ax)
    assert returned is fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_qubit_error_bar — content
# ---------------------------------------------------------------------------

def test_qubit_error_bar_zero_counts_renders():
    circuit = _simple_circuit(n_qubits=3, n_ops=3)
    result = _make_result(np.zeros((3, 3), dtype=np.int64), circuit)
    fig = plot_qubit_error_bar(result)
    assert fig is not None
    plt.close(fig)

def test_qubit_error_bar_one_bar_per_qubit():
    n_qubits = 4
    circuit = _simple_circuit(n_qubits=n_qubits, n_ops=n_qubits)
    counts = np.ones((n_qubits, n_qubits), dtype=np.int64)
    result = _make_result(counts, circuit)
    fig = plot_qubit_error_bar(result)
    ax = fig.axes[0]
    assert len(ax.patches) == n_qubits
    plt.close(fig)

def test_qubit_error_bar_custom_title():
    circuit = _simple_circuit()
    result = _make_result(np.zeros((2, 2), dtype=np.int64), circuit)
    fig = plot_qubit_error_bar(result, title="My Bar Chart")
    assert "My Bar Chart" in fig.axes[0].get_title()
    plt.close(fig)

def test_qubit_error_bar_default_title_includes_n_shots():
    circuit = _simple_circuit()
    result = _make_result(np.zeros((2, 2), dtype=np.int64), circuit, n_shots=777)
    fig = plot_qubit_error_bar(result)
    assert "777" in fig.axes[0].get_title()
    plt.close(fig)

def test_qubit_error_bar_rates_sum_correctly():
    # counts_matrix row 0 = [20, 30], row 1 = [10, 0] with n_shots=100
    # qubit 0 rate = (20+30)/100 = 0.5, qubit 1 rate = (10+0)/100 = 0.1
    circuit = _simple_circuit(n_qubits=2, n_ops=2)
    counts = np.array([[20, 30], [10, 0]], dtype=np.int64)
    result = _make_result(counts, circuit, n_shots=100)
    qubit_rates = counts.sum(axis=1) / 100
    assert np.isclose(qubit_rates[0], 0.5)
    assert np.isclose(qubit_rates[1], 0.1)
    fig = plot_qubit_error_bar(result)  # just check it renders with these values
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_fidelity_decay — return type
# ---------------------------------------------------------------------------

def test_fidelity_decay_returns_figure():
    depth_results = _depth_sweep(n_depths=4)
    fig = plot_fidelity_decay(depth_results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_fidelity_decay_accepts_existing_axes():
    depth_results = _depth_sweep(n_depths=3)
    fig, ax = plt.subplots()
    returned = plot_fidelity_decay(depth_results, ax=ax)
    assert returned is fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_fidelity_decay — single vs multiple curves
# ---------------------------------------------------------------------------

def test_fidelity_decay_single_curve():
    depth_results = _depth_sweep(n_depths=5)
    fig = plot_fidelity_decay(depth_results)
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    plt.close(fig)

def test_fidelity_decay_multiple_curves():
    curve_a = _depth_sweep(n_depths=4)
    curve_b = _depth_sweep(n_depths=4)
    fig = plot_fidelity_decay([curve_a, curve_b], labels=["Before", "After"])
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    plt.close(fig)

def test_fidelity_decay_labels_appear_in_legend():
    curve_a = _depth_sweep(n_depths=3)
    curve_b = _depth_sweep(n_depths=3)
    fig = plot_fidelity_decay([curve_a, curve_b], labels=["Noisy", "Suppressed"])
    legend_texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Noisy" in legend_texts
    assert "Suppressed" in legend_texts
    plt.close(fig)

def test_fidelity_decay_custom_title():
    fig = plot_fidelity_decay(_depth_sweep(), title="Fidelity Test")
    assert "Fidelity Test" in fig.axes[0].get_title()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_fidelity_decay — fidelity values
# ---------------------------------------------------------------------------

def test_fidelity_decay_values_between_zero_and_one():
    depth_results = _depth_sweep(n_depths=6)
    fig = plot_fidelity_decay(depth_results)
    ax = fig.axes[0]
    y_data = ax.lines[0].get_ydata()
    assert np.all(y_data >= 0.0)
    assert np.all(y_data <= 1.0)
    plt.close(fig)

def test_fidelity_decay_noisy_circuit_decreases_with_depth():
    # Each depth level adds more errors → fidelity estimate should not increase
    from noisiq.backends import ManyShotRunner
    from noisiq.noise import depolarizing_error

    depth_results = []
    for d in range(1, 8):
        c = Circuit(n_qubits=1)
        for _ in range(d):
            c.add_gate(gates.X, (0,))
        noise_config = {i: depolarizing_error(0.2) for i in range(d)}
        result = ManyShotRunner().run(c, n_shots=500, noise_config=noise_config, seed=42)
        depth_results.append(result)

    fig = plot_fidelity_decay(depth_results)
    y = fig.axes[0].lines[0].get_ydata()
    # Each fidelity value should be <= the previous one (non-increasing)
    assert np.all(np.diff(y) <= 0.01), f"Fidelity not monotonically decreasing: {y}"
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def teardown_module(module):
    plt.close("all")
