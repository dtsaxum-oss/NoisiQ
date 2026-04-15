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
from .heatmap import draw_error_heatmap


class Visualizer:
    """
    Interactive visualizer for NoisiQ circuits and simulation results.
    """
    
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.backend = StimTableauBackend()
        self.result: Optional[StimTableauResult] = None
        self._qc_cache = {}
        
    def simulate(self, noise_config=None, seed=None):
        """Run simulation and store result."""
        self.result = self.backend.run_single_shot(self.circuit, noise_config, seed)
        self._qc_cache = {} # Clear cache on new simulation
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
                fig = draw_error_heatmap(
                    self.circuit, 
                    self.result, 
                    step_index=step_idx,
                    qc_cache=self._qc_cache
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
