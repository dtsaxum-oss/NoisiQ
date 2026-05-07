"""
GIF and HTML export utilities for NoisiQ animations.

Usage::

    from noisiq.visualization.export import export_gif, export_html

    anim = animator.to_func_animation()
    export_gif(anim, "output/circuit.gif")
    export_html(anim, "output/circuit.html")
"""

from __future__ import annotations

from matplotlib.animation import FuncAnimation


def export_gif(anim: FuncAnimation, path: str, fps: int = 5, dpi: int = 100) -> None:
    """
    Save *anim* as an animated GIF using Pillow.

    Parameters
    ----------
    anim : FuncAnimation to save.
    path : Output file path (should end in .gif).
    fps  : Frames per second.
    dpi  : Output resolution.
    """
    anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    print(f"GIF saved → {path}")


def export_html(anim: FuncAnimation, path: str, fps: int = 5) -> None:
    """
    Save *anim* as a self-contained HTML file with embedded JS controls.

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
