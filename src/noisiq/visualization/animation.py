"""
Interactive animation controls for NoisiQ circuit visualizations.

Provides CircuitAnimator — a wrapper around ipywidgets Play/step controls
and the circuit drawer. Supports both single-shot (StimTableauResult) and
many-shot (AggregateResult) results.

Usage::

    animator = CircuitAnimator(circuit, result, trajectories)
    animator.show()          # interactive widget in Jupyter
    anim = animator.to_func_animation()  # FuncAnimation for export
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display
import ipywidgets as widgets

from ..ir import Circuit
from .circuit_diagram import draw_circuit
from .pauli_frame_tracker import PauliFrame


class CircuitAnimator:
    """
    Interactive step-through animator for a NoisiQ circuit simulation result.

    Parameters
    ----------
    circuit      : The Circuit that was simulated.
    result       : A StimTableauResult (single-shot) or AggregateResult (many-shot).
    trajectories : Per-step PauliFrame list returned by compute_error_trajectories;
                   required for single-shot mode, ignored for many-shot.
    """

    def __init__(
        self,
        circuit: Circuit,
        result,
        trajectories: Optional[List[PauliFrame]] = None,
    ) -> None:
        from ..backends.pauli_frame import StimTableauResult
        from ..backends.many_shot_runner import AggregateResult

        self.circuit = circuit
        self.result = result
        self._is_many_shot = isinstance(result, AggregateResult)
        self._trajectories = trajectories or []

        # Ordered unique layer indices
        self._layers: List[int] = (
            sorted(set(op.t for op in circuit.operations))
            if circuit.operations
            else [0]
        )

        # Build layer → trajectory frame mapping (single-shot only)
        self._layer_to_frame: dict[int, PauliFrame] = {}
        if not self._is_many_shot and self._trajectories:
            for step in result.steps:
                idx = step.time_step
                if idx < len(self._trajectories):
                    self._layer_to_frame[step.operation.t] = self._trajectories[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Display the interactive animation widget in a Jupyter cell."""
        n_frames = len(self._layers)
        if n_frames == 0:
            print("Nothing to animate — circuit has no operations.")
            return

        # --- Controls ---------------------------------------------------
        play_btn = widgets.ToggleButton(
            value=False,
            description="▶  Play",
            button_style="success",
            layout=widgets.Layout(width="90px"),
        )
        step_back_btn = widgets.Button(
            description="◀ Step",
            layout=widgets.Layout(width="85px"),
        )
        step_fwd_btn = widgets.Button(
            description="Step ▶",
            layout=widgets.Layout(width="85px"),
        )
        speed_slider = widgets.IntSlider(
            value=3,
            min=1,
            max=10,
            step=1,
            description="FPS:",
            continuous_update=False,
            layout=widgets.Layout(width="220px"),
        )
        frame_label = widgets.Label(value=self._frame_label(0))
        output = widgets.Output()

        # Internal play widget drives the auto-advance loop
        _play = widgets.Play(
            value=0,
            min=0,
            max=n_frames - 1,
            step=1,
            interval=self._fps_to_ms(speed_slider.value),
        )
        # Keep Play widget hidden — we control it via our ToggleButton
        _play.layout.display = "none"
        _frame_slider = widgets.IntSlider(
            value=0, min=0, max=n_frames - 1, step=1
        )
        _frame_slider.layout.display = "none"
        widgets.jslink((_play, "value"), (_frame_slider, "value"))

        def render(frame_idx: int) -> None:
            t = self._layers[frame_idx]
            frame = self._get_frame(t)
            with output:
                output.clear_output(wait=True)
                fig, ax = plt.subplots(
                    figsize=(max(6, 1.1 * len(self._layers)), 0.9 * self.circuit.n_qubits + 1.2)
                )
                draw_circuit(ax, self.circuit, pauli_frame=frame, highlight_t=t)
                plt.tight_layout()
                display(fig)
                plt.close(fig)
            frame_label.value = self._frame_label(frame_idx)

        def on_frame_change(change):
            render(change["new"])

        def on_play_toggle(change):
            if change["new"]:
                play_btn.description = "⏸  Pause"
                _play.playing = True
            else:
                play_btn.description = "▶  Play"
                _play.playing = False

        def on_step_fwd(_btn):
            nv = min(_frame_slider.value + 1, n_frames - 1)
            _frame_slider.value = nv

        def on_step_back(_btn):
            nv = max(_frame_slider.value - 1, 0)
            _frame_slider.value = nv

        def on_speed_change(change):
            _play.interval = self._fps_to_ms(change["new"])

        _frame_slider.observe(on_frame_change, names="value")
        play_btn.observe(on_play_toggle, names="value")
        step_fwd_btn.on_click(on_step_fwd)
        step_back_btn.on_click(on_step_back)
        speed_slider.observe(on_speed_change, names="value")

        render(0)

        controls = widgets.HBox(
            [play_btn, step_back_btn, step_fwd_btn, speed_slider, frame_label]
        )
        display(widgets.VBox([controls, _play, _frame_slider, output]))

    def to_func_animation(self, interval_ms: int = 300) -> FuncAnimation:
        """
        Return a matplotlib FuncAnimation (for export via export.py).

        Parameters
        ----------
        interval_ms : Milliseconds between frames.
        """
        fig, ax = plt.subplots(
            figsize=(max(6, 1.1 * len(self._layers)), 0.9 * self.circuit.n_qubits + 1.2)
        )

        def update(frame_idx: int):
            ax.clear()
            t = self._layers[frame_idx]
            frame = self._get_frame(t)
            draw_circuit(ax, self.circuit, pauli_frame=frame, highlight_t=t)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self._layers),
            interval=interval_ms,
            repeat=True,
        )
        return anim

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_frame(self, t: int) -> Optional[PauliFrame]:
        """Return the PauliFrame for layer t, or None in many-shot mode."""
        if self._is_many_shot:
            return None
        return self._layer_to_frame.get(t)

    def _frame_label(self, frame_idx: int) -> str:
        t = self._layers[frame_idx]
        return f"Step {frame_idx + 1} / {len(self._layers)}  (t={t})"

    @staticmethod
    def _fps_to_ms(fps: int) -> int:
        return max(50, int(1000 / fps))
