"""Smoke + structure tests for every plot, on arbitrary circuits."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import noisiq as nq
from noisiq.noise import NoiseModel
from noisiq.visualization import CircuitLayout
from tests.conftest import random_circuit

CIRCUITS = {
    "ghz": lambda: nq.algorithms.ghz(4, measure=True),
    "qft": lambda: nq.algorithms.qft(3),
    "grover": lambda: nq.algorithms.grover(3, "011"),
    "repetition": lambda: nq.algorithms.repetition_code(3, 2),
    "teleport": lambda: nq.algorithms.teleportation(),
    "custom": lambda: nq.Circuit(3).unitary(np.eye(4), [0, 2], name="MyGate").ccx(0, 1, 2).delay(1, 1e-6).reset(0),
}


@pytest.mark.parametrize("name", CIRCUITS)
def test_draw_any_circuit(name):
    c = CIRCUITS[name]()
    d = nq.draw_circuit(c)
    assert set(d.layout.placed) == {i for i, inst in enumerate(c) if inst.name != "BARRIER"}
    assert d.ax.get_title() == (c.name or "")


def test_layout_separates_overlapping_gates_in_a_layer():
    c = nq.Circuit(3).cx(0, 2).h(1)  # same layer, CX spans over qubit 1
    lay = CircuitLayout.from_circuit(c)
    assert lay.placed[0].column != lay.placed[1].column
    assert lay.placed[0].layer == lay.placed[1].layer == 0


def test_layout_widths_grow_with_labels_and_export():
    c = nq.Circuit(1).h(0).rz(0.123456, 0)
    lay = CircuitLayout.from_circuit(c)
    assert lay.placed[1].width > lay.placed[0].width
    d = lay.to_dict()
    assert d["instructions"][1]["label"].startswith("RZ")
    assert lay.boundary_x(0) < lay.boundary_x(1) < lay.boundary_x(2)


@pytest.mark.parametrize("name", ["ghz", "repetition", "teleport", "qft", "custom"])
@pytest.mark.parametrize("metric", ["rate", "burden"])
def test_heatmap(name, metric):
    c = CIRCUITS[name]()
    nm = NoiseModel.uniform_depolarizing(0.02, readout=0.01)
    nm.add_idle_error(lambda d: nq.noise.phase_flip(0.01))
    r = nq.simulate(c, nm, shots=500, seed=1)
    if metric == "burden" and not c.is_clifford():
        with pytest.raises(nq.backends.IncompatibleCircuitError):
            nq.plot_error_heatmap(r, metric=metric)
        return
    d = nq.plot_error_heatmap(r, metric=metric, annotate=True, scale="log")
    assert len(d.fig.axes) == 2  # circuit + colorbar


def test_heatmap_depends_on_noise_model():
    """Regression for the old structure-only halos: different noise → different halos."""
    c = nq.algorithms.ghz(3, measure=True)
    a = nq.simulate(c, NoiseModel().add_gate_error(nq.noise.depolarizing(0.1), gates="H"), shots=2000, seed=1)
    b = nq.simulate(c, NoiseModel().add_gate_error(nq.noise.depolarizing(0.1, 2), gates="CX"), shots=2000, seed=1)
    va, _ = nq.visualization.heat_values(a)
    vb, _ = nq.visualization.heat_values(b)
    assert va[0].sum() > 0 and va[1:3].sum() == 0
    assert vb[0].sum() == 0 and vb[1:3].sum() > 0


def test_propagation_plots_and_animation(tmp_path: Path):
    t = nq.propagate(nq.algorithms.ghz(4, measure=True), "X", 1, after=1)
    for step in range(t.n_layers + 1):
        nq.plot_propagation(t, step)
    anim = nq.animate_propagation(t, interval=50)
    gif = nq.save_animation(anim, tmp_path / "prop.gif", fps=5)
    html = nq.save_animation(nq.animate_propagation(t), tmp_path / "prop.html")
    assert gif.stat().st_size > 1000 and html.read_text(encoding="utf-8").count("<script") >= 1
    with pytest.raises(ValueError):
        nq.save_animation(anim, tmp_path / "x.mp3")


def test_deviation_trace_from_statevector():
    c = nq.algorithms.qft(3)
    r = nq.simulate(c, nq.get_hardware("ibm_eagle_r3").noise_model(), shots=20, seed=1, trace_shots=5)
    assert r.trace.kind == "deviation"
    assert r.trace.deviation.shape == (c.depth + 1, 3, 5)
    assert np.all((0 <= r.trace.deviation) & (r.trace.deviation <= 1))
    nq.plot_propagation(r)
    nq.animate_propagation(r)


def test_frame_trace_from_noisy_run():
    r = nq.simulate(
        nq.algorithms.repetition_code(3, 1), NoiseModel.uniform_depolarizing(0.05), shots=200, seed=1, trace_shots=10
    )
    assert r.trace.kind == "pauli" and r.trace.n_shots == 10
    shot = r.trace.first_faulty_shot()
    assert shot is not None
    nq.plot_propagation(r, shot=shot)
    with pytest.raises(ValueError):
        nq.plot_propagation(r, shot=99)
    with pytest.raises(ValueError):
        nq.plot_propagation(nq.simulate(nq.algorithms.bell(), shots=5))


def test_charts():
    c = nq.algorithms.ghz(3, measure=True)
    nm = NoiseModel.uniform_depolarizing(0.02, readout=0.02)
    r = nq.simulate(c, nm, shots=1000, seed=1)
    ideal = nq.simulate(c, shots=1000, seed=1)
    nq.plot_error_budget(r, top=2)
    nq.plot_error_budget(r, metric="burden")
    nq.plot_qubit_errors(r)
    nq.plot_counts(r, ideal)
    nq.plot_counts(r, {"000": 1, "111": 1})
    nq.plot_comparison({"noisy": r, "ideal": ideal}, "hellinger", ideal=ideal)
    nq.plot_comparison({"noisy": r}, "output_error")
    sv = nq.simulate(nq.algorithms.ghz(3), nm, shots=200, backend="statevector", seed=1)
    nq.plot_comparison({"sv": sv}, "fidelity")
    nq.plot_density_matrix(sv)
    nq.plot_bloch(sv)
    nq.plot_bloch(np.array([[1, 0], [0, 0]]))
    with pytest.raises(ValueError):
        nq.plot_comparison({"noisy": r}, "fidelity")
    with pytest.raises(ValueError):
        nq.plot_density_matrix(r)


def test_hardware_comparison_chart():
    fig = nq.visualization.plot_hardware_comparison(
        nq.algorithms.ghz(3, measure=True), ["ibm_eagle_r3", "ionq_aria"], shots=300
    )
    assert len(fig.axes[0].patches) == 2


def test_style_override():
    old = nq.visualization.get_style()
    try:
        nq.visualization.set_style(fontsize=14, heat_cmap="viridis")
        assert nq.visualization.get_style().fontsize == 14
        nq.draw_circuit(nq.algorithms.bell())
    finally:
        nq.visualization.set_style(**{k: getattr(old, k) for k in ("fontsize", "heat_cmap")})


@pytest.mark.parametrize("seed", range(6))
def test_random_circuits_render_everything(seed):
    rng = np.random.default_rng(seed)
    clifford = seed % 2 == 0
    c = random_circuit(4, 25, rng, clifford=clifford, mid_measure=True, conditionals=clifford, resets=True, bases=True)
    nm = NoiseModel.uniform_depolarizing(0.02, readout=0.02)
    r = nq.simulate(c, nm if clifford else nm, shots=100, seed=seed, trace_shots=3)
    nq.draw_circuit(c)
    nq.plot_error_heatmap(r)
    nq.plot_propagation(r)
    nq.plot_error_budget(r)
    plt.close("all")
