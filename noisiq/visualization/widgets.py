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
            
        step_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(self.result.steps) - 1,
            step=1,
            description='Step:',
            continuous_update=False
        )
        
        output = widgets.Output()
        
        def update_plot(change):
            step_idx = change['new']
            with output:
                output.clear_output(wait=True)
                
                fig, ax = plt.subplots(figsize=(10, 0.8 * self.circuit.n_qubits + 1))
                
                # Get the frame for this step
                # Note: step_idx represents the state *after* the operation at step_idx
                # We can highlight the gate at step_idx
                frame = self.trajectories[step_idx] if step_idx < len(self.trajectories) else None
                
                draw_circuit_with_labels(
                    ax, 
                    self.circuit, 
                    pauli_frame=frame, 
                    highlight_t=step_idx
                )
                
                display(fig)
                plt.close(fig) # Close to prevent memory leak and hangs
                
                # Print errors at this step
                step = self.result.steps[step_idx]
                if step.errors:
                    print(f"Errors at step {step_idx}:")
                    for err in step.errors:
                        print(f"  - {err.pauli} error on qubit {err.qubit}")
                else:
                    print(f"No errors at step {step_idx}")
        
        step_slider.observe(update_plot, names='value')
        
        # Initial plot
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
