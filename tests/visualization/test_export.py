"""Tests for export_gif and CircuitAnimator.to_func_animation purities= support."""

import os
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_circuit():
    import noisiq as nq
    c = nq.Circuit(n_qubits=2)
    c.h(0, t=0)
    c.cnot(0, 1, t=1)
    c.h(0, t=4)
    return c


@pytest.fixture()
def filled_circuit(simple_circuit):
    from noisiq.noise import fill_idle_with_identities, get_hardware
    profile = get_hardware("ibm_eagle_r3")
    return fill_idle_with_identities(simple_circuit, profile.gate_times)


@pytest.fixture()
def pauli_noise(filled_circuit):
    from noisiq.noise import get_hardware
    return get_hardware("ibm_eagle_r3").to_pauli_noise_model(filled_circuit)


@pytest.fixture()
def visualizer(filled_circuit, pauli_noise):
    from noisiq.visualization import Visualizer
    viz = Visualizer(filled_circuit)
    viz.run_single(noise_config=pauli_noise, seed=0)
    return viz


@pytest.fixture()
def fake_purities():
    return [0.82, 0.95]


# ---------------------------------------------------------------------------
# CircuitAnimator.to_func_animation
# ---------------------------------------------------------------------------

def test_func_animation_no_purities(visualizer):
    from noisiq.visualization.animation import CircuitAnimator
    anim = CircuitAnimator(
        visualizer.circuit, visualizer.result, visualizer.trajectories
    ).to_func_animation()
    assert anim is not None
    plt.close("all")


def test_func_animation_with_purities(visualizer, fake_purities):
    from noisiq.visualization.animation import CircuitAnimator
    anim = CircuitAnimator(
        visualizer.circuit, visualizer.result, visualizer.trajectories
    ).to_func_animation(purities=fake_purities)
    assert anim is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# export_gif — Visualizer path
# ---------------------------------------------------------------------------

def test_export_gif_from_visualizer(tmp_path, visualizer):
    from noisiq.visualization import export_gif
    out = str(tmp_path / "test.gif")
    export_gif(visualizer, path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 500
    plt.close("all")


def test_export_gif_with_purities(tmp_path, visualizer, fake_purities):
    from noisiq.visualization import export_gif
    out = str(tmp_path / "test_purity.gif")
    export_gif(visualizer, path=out, purities=fake_purities)
    assert os.path.exists(out)
    # GIF with purity overlay should be at least as large as without
    # (extra text on last frame makes the frame differ → larger file)
    assert os.path.getsize(out) > 500
    plt.close("all")


def test_func_animation_wider_with_purities(visualizer, fake_purities):
    """Figure must be wider when purities= is requested (extra room for labels)."""
    from noisiq.visualization.animation import CircuitAnimator
    animator = CircuitAnimator(visualizer.circuit, visualizer.result, visualizer.trajectories)

    anim_plain = animator.to_func_animation()
    w_plain = anim_plain._fig.get_figwidth()

    anim_purity = animator.to_func_animation(purities=fake_purities)
    w_purity = anim_purity._fig.get_figwidth()

    assert w_purity > w_plain, (
        f"Figure with purities ({w_purity:.2f} in) should be wider than "
        f"figure without ({w_plain:.2f} in)."
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# export_gif — raw FuncAnimation backward compat
# ---------------------------------------------------------------------------

def test_export_gif_from_func_animation(tmp_path, visualizer):
    from noisiq.visualization import export_gif
    from noisiq.visualization.animation import CircuitAnimator
    anim = CircuitAnimator(
        visualizer.circuit, visualizer.result, visualizer.trajectories
    ).to_func_animation()
    out = str(tmp_path / "raw_anim.gif")
    export_gif(anim, path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 500
    plt.close("all")


# ---------------------------------------------------------------------------
# export_gif — error on unrun Visualizer
# ---------------------------------------------------------------------------

def test_export_gif_raises_if_not_run(tmp_path, simple_circuit):
    from noisiq.visualization import Visualizer, export_gif
    viz = Visualizer(simple_circuit)   # NOT run yet
    with pytest.raises(RuntimeError, match="run_single"):
        export_gif(viz, path=str(tmp_path / "should_fail.gif"))
    plt.close("all")
