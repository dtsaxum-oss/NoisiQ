import matplotlib
matplotlib.use("Agg")  # headless — no display needed in CI

import pytest
import numpy as np
import matplotlib.pyplot as plt

from noisiq.ir import Circuit, gates
from noisiq.backends.many_shot_runner import AggregateResult
from noisiq.visualization.charts.heatmap import (
    plot_error_heatmap,
    _SWAP_PROPAGATION_TABLE,
    _propagate_pauli_through_gate,
    _compute_downstream_impact,
)


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
# SWAP gate support
# ---------------------------------------------------------------------------

def test_swap_renders_without_error():
    c = Circuit(n_qubits=2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.SWAP, (0, 1))
    counts = np.zeros((2, 2), dtype=np.int64)
    result = _make_result(counts, c)
    fig = plot_error_heatmap(result, c)
    assert fig is not None
    plt.close(fig)


def test_swap_halo_with_nonzero_errors():
    c = Circuit(n_qubits=2)
    c.add_gate(gates.SWAP, (0, 1))
    counts = np.array([[50], [50]], dtype=np.int64)
    result = _make_result(counts, c, n_shots=100)
    fig = plot_error_heatmap(result, c)
    assert fig is not None
    plt.close(fig)


def test_swap_propagation_table_swaps_labels():
    assert _SWAP_PROPAGATION_TABLE[('X', 'I')] == ('I', 'X')
    assert _SWAP_PROPAGATION_TABLE[('I', 'Z')] == ('Z', 'I')
    assert _SWAP_PROPAGATION_TABLE[('Y', 'Y')] == ('Y', 'Y')
    assert _SWAP_PROPAGATION_TABLE[('Z', 'X')] == ('X', 'Z')


def test_swap_propagation_is_self_inverse():
    c = Circuit(2)
    c.add_gate(gates.SWAP, (0, 1))
    swap_op = c.operations[0]

    initial = {0: 'X', 1: 'Z'}
    after_first = _propagate_pauli_through_gate(initial, swap_op)
    assert after_first == {0: 'Z', 1: 'X'}

    after_second = _propagate_pauli_through_gate(after_first, swap_op)
    assert after_second == initial


def test_swap_propagates_impact_to_downstream_gate():
    # H(q0) → SWAP(q0,q1) → X(q1): error on q0 at H reaches X(q1) via SWAP
    c = Circuit(2)
    c.add_gate(gates.H, (0,))
    c.add_gate(gates.SWAP, (0, 1))
    c.add_gate(gates.X, (1,))
    impact = _compute_downstream_impact(c)
    assert impact[0] > 0  # H error propagates through SWAP to X
    assert impact[1] > 0  # SWAP error reaches X


# ---------------------------------------------------------------------------
# Cleanup: close all figures after module runs
# ---------------------------------------------------------------------------

def teardown_module(module):
    plt.close("all")
