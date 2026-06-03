"""
Phase 1 guard tests.

Uses a minimal _FakeMeasurement stub (no .gate, no .qubits) to verify that
the isinstance(op, Operation) guards added in Phase 1 prevent AttributeError
crashes. Real Measurement / ConditionalOp types arrive in Phase 3.
"""

import pytest
from dataclasses import dataclass

from noisiq.ir import Circuit, gates
from noisiq.ir.circuit import Operation
from noisiq.backends.backend_selector import BackendSelector
from noisiq.backends.pauli_frame import StimTableauBackend


@dataclass(frozen=True)
class _FakeMeasurement:
    """Minimal stand-in for Measurement — no .gate, no .qubits attributes."""
    qubit: int
    t: int = 0


def _circuit_with_fake_measurement() -> Circuit:
    c = Circuit(2)
    c.h(0)
    c.operations.append(_FakeMeasurement(qubit=0, t=1))
    return c


# ---------------------------------------------------------------------------
# BackendSelector
# ---------------------------------------------------------------------------

def test_backend_selector_no_crash_on_measurement_circuit():
    c = _circuit_with_fake_measurement()
    backend = BackendSelector.select(c)
    assert isinstance(backend, StimTableauBackend)


def test_backend_selector_measurement_only_circuit():
    c = Circuit(1)
    c.operations.append(_FakeMeasurement(qubit=0, t=0))
    backend = BackendSelector.select(c)
    assert isinstance(backend, StimTableauBackend)


# ---------------------------------------------------------------------------
# ManyShotRunner — Clifford whitelist loop
# ---------------------------------------------------------------------------

def test_many_shot_runner_whitelist_no_crash_on_measurement_op():
    """The whitelist loop must not AttributeError on _FakeMeasurement."""
    _CLIFFORD_GATES = frozenset({'H', 'X', 'Y', 'Z', 'S', 'S_DAG', 'CNOT', 'CX', 'CZ', 'SWAP', 'I', 'IDLE'})
    c = _circuit_with_fake_measurement()
    for op in c.operations:
        if not isinstance(op, Operation):
            continue
        assert op.gate.name.upper() in _CLIFFORD_GATES


# ---------------------------------------------------------------------------
# circuit.validate()
# ---------------------------------------------------------------------------

def test_validate_no_crash_with_fake_measurement():
    c = _circuit_with_fake_measurement()
    c.validate()  # must not raise


# ---------------------------------------------------------------------------
# drawer.draw_circuit_with_labels
# ---------------------------------------------------------------------------

def test_drawer_no_crash_with_measurement_op():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from noisiq.visualization.drawer import draw_circuit_with_labels

    c = _circuit_with_fake_measurement()
    fig, ax = plt.subplots()
    draw_circuit_with_labels(ax, c)
    plt.close(fig)


# ---------------------------------------------------------------------------
# circuit_diagram.draw_circuit
# ---------------------------------------------------------------------------

def test_circuit_diagram_no_crash_with_measurement_op():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from noisiq.visualization.circuit_diagram import draw_circuit

    c = _circuit_with_fake_measurement()
    fig, ax = plt.subplots()
    draw_circuit(ax, c)
    plt.close(fig)


# ---------------------------------------------------------------------------
# heatmap._compute_downstream_impact
# ---------------------------------------------------------------------------

def test_heatmap_downstream_impact_no_crash_with_measurement_op():
    from noisiq.visualization.charts.heatmap import _compute_downstream_impact

    c = _circuit_with_fake_measurement()
    impact = _compute_downstream_impact(c)
    assert len(impact) == len(c.operations)
    # The fake measurement slot should have impact=0 (no propagation model yet).
    assert impact[1] == 0


def test_heatmap_downstream_impact_pure_gate_circuit_unchanged():
    """Guard must not affect impact values for all-Operation circuits."""
    from noisiq.visualization.charts.heatmap import _compute_downstream_impact

    c = Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    impact = _compute_downstream_impact(c)
    assert len(impact) == 2
