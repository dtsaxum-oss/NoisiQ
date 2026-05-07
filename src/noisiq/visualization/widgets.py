"""
Interactive widgets for visualizing error propagation.
"""

from __future__ import annotations

from typing import Dict, Optional

import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from ..ir import Circuit
from ..backends.pauli_frame import StimTableauBackend, StimTableauResult
from ..backends.many_shot_runner import AggregateResult, ManyShotRunner
from ..noise.pauli_error import PauliError
from .drawer import draw_circuit_with_labels
from .pauli_frame_tracker import compute_error_trajectories
from .animation import CircuitAnimator
from .gate_info import GateInfoExtractor


class Visualizer:
    """
    Interactive visualizer for NoisiQ circuits and simulation results.
    """

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.backend = StimTableauBackend()
        self.result: Optional[StimTableauResult] = None
        self.many_shot_result: Optional[AggregateResult] = None
        self.trajectories = []

    def simulate(self, noise_config=None, seed=None) -> StimTableauResult:
        """Run a single-shot simulation and store the result."""
        self.result = self.backend.run_single_shot(self.circuit, noise_config, seed)
        self.trajectories = compute_error_trajectories(self.circuit, self.result)
        self.many_shot_result = None
        return self.result

    def run_many(
        self,
        n_shots: int,
        noise_config: Optional[Dict[int, PauliError]] = None,
        seed: Optional[int] = None,
    ) -> AggregateResult:
        """Run n_shots simulations and store the aggregate result."""
        runner = ManyShotRunner(self.backend)
        self.many_shot_result = runner.run(self.circuit, n_shots, noise_config, seed)
        return self.many_shot_result

    def show(self) -> None:
        """Display interactive step-through widget (single-shot)."""
        if self.result is None:
            print("Please run simulate() first.")
            return

        layers = sorted(set(op.t for op in self.circuit.operations))
        output = widgets.Output()

        if not layers:
            with output:
                fig, ax = plt.subplots(figsize=(10, 0.8 * self.circuit.n_qubits + 1))
                draw_circuit_with_labels(ax, self.circuit, pauli_frame=None, highlight_t=0)
                display(fig)
                plt.close(fig)
                print("Empty circuit — no operations to step through.")
            display(output)
            return

        layer_to_frame = {}
        for step in self.result.steps:
            t = step.operation.t
            idx = step.time_step
            if idx < len(self.trajectories):
                layer_to_frame[t] = self.trajectories[idx]

        step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(layers) - 1,
            step=1,
            description="Layer:",
            continuous_update=False,
        )

        def render(layer_idx: int) -> None:
            t = layers[layer_idx]
            frame = layer_to_frame.get(t)
            with output:
                output.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(10, 0.8 * self.circuit.n_qubits + 1))
                draw_circuit_with_labels(
                    ax, self.circuit, pauli_frame=frame, highlight_t=t,
                )
                display(fig)
                plt.close(fig)

        step_slider.observe(lambda change: render(change["new"]), names="value")
        render(0)

        display(widgets.VBox([step_slider, output]))

    def animate(self) -> None:
        """Display the full animation with play/pause/step controls."""
        if self.result is None:
            print("Please run simulate() first.")
            return
        animator = CircuitAnimator(self.circuit, self.result, self.trajectories)
        animator.show()

    def export_animation(self, filename: str, fps: int = 5) -> None:
        """Export the animation to a GIF or HTML file."""
        if self.result is None:
            print("Please run simulate() first.")
            return
        from .export import export_gif, export_html

        animator = CircuitAnimator(self.circuit, self.result, self.trajectories)
        anim = animator.to_func_animation(interval_ms=int(1000 / fps))

        if filename.endswith(".html"):
            export_html(anim, filename, fps=fps)
        else:
            export_gif(anim, filename, fps=fps)
        plt.close("all")
