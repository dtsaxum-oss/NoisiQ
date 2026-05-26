"""
GIF and HTML export utilities for NoisiQ animations.

Usage::

    from noisiq.visualization.export import export_gif, export_html

    # From a Visualizer (recommended — supports purities=):
    viz = Visualizer(circuit)
    viz.run_single(noise_config=noise, seed=7)
    purities = per_qubit_purities(rho, n_qubits)
    export_gif(viz, "output/circuit.gif", purities=purities)

    # From a raw FuncAnimation (legacy):
    anim = animator.to_func_animation()
    export_gif(anim, "output/circuit.gif")
    export_html(anim, "output/circuit.html")
"""

from __future__ import annotations

from typing import List, Optional, Union


def export_gif(
    viz_or_anim,
    path: str,
    fps: int = 5,
    dpi: int = 100,
    purities: Optional[List[float]] = None,
) -> None:
    """Save an animation as an animated GIF using Pillow.

    Parameters
    ----------
    viz_or_anim : Visualizer or FuncAnimation.
        When a :class:`Visualizer` is passed, the animator is built
        automatically from its stored result and trajectories.  This is the
        recommended form because it supports the ``purities=`` overlay.
        A raw ``FuncAnimation`` is accepted for backward compatibility; in
        that case ``purities=`` is ignored (the animation was already built).
    path        : Output file path (should end in .gif).
    fps         : Frames per second.
    dpi         : Output resolution in dots per inch.
    purities    : Optional list of per-qubit Tr(ρ_q²) values — one float per
                  qubit, in qubit-index order.  When provided (and
                  ``viz_or_anim`` is a :class:`Visualizer`), the values are
                  overlaid at the right edge of each qubit wire on the last
                  frame of the GIF.  Obtain via
                  ``noisiq.visualization.purity_overlay.per_qubit_purities``.
    """
    from matplotlib.animation import FuncAnimation

    if isinstance(viz_or_anim, FuncAnimation):
        anim = viz_or_anim
    else:
        # Assume Visualizer — build CircuitAnimator then FuncAnimation.
        from .animation import CircuitAnimator
        viz = viz_or_anim
        result = viz.result if viz.result is not None else viz.many_shot_result
        if result is None:
            raise RuntimeError(
                "Visualizer has no result — call run_single() or run_many() first."
            )
        animator = CircuitAnimator(viz.circuit, result, viz.trajectories)
        anim = animator.to_func_animation(
            interval_ms=int(1000 / fps),
            purities=purities,
        )

    anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    print(f"GIF saved → {path}")


def export_html(anim, path: str, fps: int = 5) -> None:
    """Save *anim* as a self-contained HTML file with embedded JS controls.

    Parameters
    ----------
    anim : FuncAnimation to save.
    path : Output file path (should end in .html).
    fps  : Frames per second for the embedded player.
    """
    html_str = anim.to_jshtml(fps=fps, default_mode="loop")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"HTML saved → {path}")
