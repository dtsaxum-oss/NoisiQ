"""
Interactive widgets for visualizing error propagation.
"""

from __future__ import annotations

from typing import Optional

import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from ..ir import Circuit
from ..backends.pauli_frame import StimTableauBackend, StimTableauResult
from .drawer import draw_circuit_with_labels
from .pauli_frame_tracker import compute_error_trajectories


from matplotlib.animation import FuncAnimation

class Visualizer:
    """
    Interactive visualizer for NoisiQ circuits and simulation results.
    """
    
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.backend = StimTableauBackend()
        self.result: Optional[StimTableauResult] = None
        self.trajectories = []
        
    def simulate(self, noise_config=None, seed=None):
        """Run simulation and store result."""
        self.result = self.backend.run_single_shot(self.circuit, noise_config, seed)
        self.trajectories = compute_error_trajectories(self.circuit, self.result)
        return self.result
        
    def show(self):
        """Display interactive widget."""
        if self.result is None:
            print("Please run simulate() first.")
            return

        # Step through unique layers (op.t values) not individual operations.
        layers = sorted(set(op.t for op in self.circuit.operations))
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
            description='Layer:',
            continuous_update=False,
        )

        output = widgets.Output()

        def update_plot(change):
            layer_idx = change['new']
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

                layer_errors = [
                    err
                    for step in self.result.steps
                    if step.operation.t == t
                    for err in step.errors
                ]
                if layer_errors:
                    print(f"Errors at t={t}:")
                    for err in layer_errors:
                        print(f"  - {err.pauli} error on qubit {err.qubit}")
                else:
                    print(f"No errors at t={t}")

        step_slider.observe(update_plot, names='value')
        with output:
            update_plot({'new': 0})
        display(widgets.VBox([step_slider, output]))

    def export_animation(self, filename: str, interval_ms: int = 650):
        """Export the step-by-step visualization as an animation (GIF or MP4)."""
        if self.result is None:
            print("Please run simulate() first.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 0.8 * self.circuit.n_qubits + 1))
        
        def update(i):
            ax.clear()
            frame = self.trajectories[i] if i < len(self.trajectories) else None
            draw_circuit_with_labels(
                ax, 
                self.circuit, 
                pauli_frame=frame, 
                highlight_t=i
            )
            
        anim = FuncAnimation(fig, update, frames=len(self.trajectories), interval=interval_ms, repeat=True)
        anim.save(filename)
        plt.close(fig)
        print(f"Animation saved to {filename}")
