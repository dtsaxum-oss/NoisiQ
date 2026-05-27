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

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display
import ipywidgets as widgets

from ..ir import Circuit
from .circuit_diagram import draw_circuit
from .pauli_frame_tracker import PauliFrame
from .purity_overlay import annotate_axes_with_purities
from .theme import (
    CIRCUIT_WIDTH_PER_LAYER,
    CIRCUIT_HEIGHT_PER_QUBIT,
    CIRCUIT_MIN_WIDTH,
    CIRCUIT_BASE_HEIGHT_PAD,
)


class CircuitAnimator:
    """
    Interactive step-through animator for a NoisiQ circuit simulation result.

    Parameters
    ----------
    circuit          : The Circuit that was simulated.
    result           : A StimTableauResult (single-shot) or AggregateResult (many-shot).
    trajectories     : Per-step PauliFrame list returned by compute_error_trajectories;
                       required for single-shot mode, ignored for many-shot.
    show_annotations : If True (default), show per-qubit summary text on the
                       right margin and an overall summary box. Set to False
                       to render the bare circuit diagram (legacy behavior).
    """

    def __init__(
        self,
        circuit: Circuit,
        result,
        trajectories: Optional[Dict[int, PauliFrame]] = None,
        show_annotations: bool = True,
    ) -> None:
        from ..backends.pauli_frame import StimTableauResult
        from ..backends.many_shot_runner import AggregateResult

        self.circuit = circuit
        self.result = result
        self._is_many_shot = isinstance(result, AggregateResult)
        self._trajectories = trajectories or {}
        self._show_annotations = show_annotations

        # Ordered unique layer indices
        self._layers: List[int] = (
            sorted(set(op.t for op in circuit.operations))
            if circuit.operations
            else [0]
        )

        # Build layer → trajectory frame mapping (single-shot only).
        # trajectories is now a Dict[int, PauliFrame] keyed by time-step t,
        # so we can use it directly without rebuilding from index lookups.
        self._layer_to_frame: dict[int, PauliFrame] = {}
        if not self._is_many_shot and self._trajectories:
            self._layer_to_frame = dict(self._trajectories)

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
                # Slightly wider figure when annotations are shown so the
                # right-margin text has room without crowding the gates.
                width_pad = 2.4 if self._show_annotations else 0.0
                fig, ax = plt.subplots(
                    figsize=(
                        max(CIRCUIT_MIN_WIDTH, CIRCUIT_WIDTH_PER_LAYER * len(self._layers)) + width_pad,
                        CIRCUIT_HEIGHT_PER_QUBIT * self.circuit.n_qubits + CIRCUIT_BASE_HEIGHT_PAD,
                    )
                )
                annotations, summary = self._build_annotations(frame_idx)
                draw_circuit(
                    ax, self.circuit,
                    pauli_frame=frame,
                    highlight_t=t,
                    per_qubit_annotations=annotations,
                    summary_text=summary,
                )
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

    def to_func_animation(
        self,
        interval_ms: int = 300,
        purities: Optional[List[float]] = None,
    ) -> FuncAnimation:
        """
        Return a matplotlib FuncAnimation (for export via export.py).

        Parameters
        ----------
        interval_ms : Milliseconds between frames.
        purities    : Optional list of per-qubit Tr(ρ_q²) values.  When
                      provided, purity labels are drawn at the right edge of
                      each qubit wire on the **final frame only**.  Obtain
                      this list from
                      ``noisiq.visualization.purity_overlay.per_qubit_purities``.
        """
        width_pad = 2.4 if self._show_annotations else 0.0
        if purities is not None:
            width_pad += 3.0   # room for "Tr(ρ²)=0.xxx" labels on last frame
        fig, ax = plt.subplots(
            figsize=(
                max(CIRCUIT_MIN_WIDTH, CIRCUIT_WIDTH_PER_LAYER * len(self._layers)) + width_pad,
                CIRCUIT_HEIGHT_PER_QUBIT * self.circuit.n_qubits + CIRCUIT_BASE_HEIGHT_PAD,
            )
        )

        last_frame = len(self._layers) - 1

        def update(frame_idx: int):
            ax.clear()
            t = self._layers[frame_idx]
            frame = self._get_frame(t)
            annotations, summary = self._build_annotations(frame_idx)
            draw_circuit(
                ax, self.circuit,
                pauli_frame=frame,
                highlight_t=t,
                per_qubit_annotations=annotations,
                summary_text=summary,
            )
            if purities is not None and frame_idx == last_frame:
                annotate_axes_with_purities(ax, purities)

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

    def _build_annotations(
        self, frame_idx: int
    ) -> tuple[Optional[list[str]], Optional[str]]:
        """
        Compute (per_qubit_annotations, summary_text) for the current frame.

        Single-shot mode:
            per-qubit  → final cumulative Pauli at the LAST timestep
                         (constant across frames — that's the "final" in
                         "final state"; the per-step Pauli is already
                         shown by the existing floating error labels).
            summary    → total error events injected so far in this shot.

        Many-shot mode:
            per-qubit  → empirical error rate over all shots
                         (constant across frames — it's an aggregate stat).
            summary    → zero-error fraction + total shot count.

        Returns (None, None) when show_annotations is False.
        """
        if not self._show_annotations:
            return None, None

        if self._is_many_shot:
            return self._build_many_shot_annotations()
        return self._build_single_shot_annotations(frame_idx)

    def _build_single_shot_annotations(
        self, frame_idx: int
    ) -> tuple[list[str], str]:
        """Annotations from a StimTableauResult / per-step PauliFrame trajectory."""
        n_qubits = self.circuit.n_qubits

        # Final cumulative Pauli on each wire: take the trajectory at the
        # LAST recorded timestep. This is the "final state" Pauli per qubit.
        if self._layer_to_frame:
            last_t = max(self._layer_to_frame.keys())
            final_frame = self._layer_to_frame[last_t]
            final_pauli_str = final_frame.get_pauli_string()
        else:
            final_pauli_str = "I" * n_qubits

        per_qubit = [
            (f"final: {p}" if p != "I" else "final: I (clean)")
            for p in final_pauli_str
        ]

        # Summary: total number of error events injected this shot, up to
        # the currently-displayed frame. We walk result.steps and count
        # ErrorEvents whose gate_index ≤ frame_idx.
        n_events_so_far = sum(
            len(step.errors)
            for step in self.result.steps[: frame_idx + 1]
        )
        n_events_total = sum(len(step.errors) for step in self.result.steps)
        summary = (
            f"Shot stats (single-shot)\n"
            f"  Errors injected: {n_events_so_far} / {n_events_total}\n"
            f"  Final frame: {final_pauli_str}"
        )
        return per_qubit, summary

    def _build_many_shot_annotations(self) -> tuple[list[str], str]:
        """Annotations from an AggregateResult."""
        # error_rate_matrix shape: (n_qubits, n_timesteps); per-qubit rate
        # is the *fraction of shots* in which ANY error happened on that
        # qubit at ANY timestep.  We approximate this by summing per-qubit
        # error counts across timesteps and dividing by shot count.
        #
        # Note: this slightly overcounts because two errors on the same
        # qubit in the same shot count twice. For small error rates the
        # overcounting is negligible (<5% relative). A more rigorous
        # "any-error per qubit" requires per-shot reduction, which
        # AggregateResult doesn't currently expose. Worth a follow-up.
        import numpy as np
        rate_matrix = self.result.error_rate_matrix       # (n_qubits, n_timesteps)
        per_qubit_rate = 1.0 - np.prod(1.0 - rate_matrix, axis=1) # (n_qubits,)
        per_qubit = [f"err: {r * 100:5.2f}%" for r in per_qubit_rate]

        zero_err = self.result.zero_error_fraction
        n_shots = self.result.n_shots
        summary = (
            f"Aggregate stats ({n_shots} shots)\n"
            f"  Zero-error fraction: {zero_err * 100:.2f}%\n"
            f"  Mean err rate (overall): {per_qubit_rate.mean() * 100:.2f}%"
        )
        return per_qubit, summary

    def _frame_label(self, frame_idx: int) -> str:
        t = self._layers[frame_idx]
        return f"Step {frame_idx + 1} / {len(self._layers)}  (t={t})"

    @staticmethod
    def _fps_to_ms(fps: int) -> int:
        return max(50, int(1000 / fps))
