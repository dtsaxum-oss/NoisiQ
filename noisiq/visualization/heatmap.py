"""
Heatmap visualization for quantum circuits using Qiskit's circuit drawer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

from ..ir import Circuit as NoisiqCircuit
from ..backends.pauli_frame import StimTableauResult, ErrorEvent


def to_qiskit_circuit(noisiq_circuit: NoisiqCircuit) -> QuantumCircuit:
    """Convert NoisiQ circuit to Qiskit QuantumCircuit."""
    qc = QuantumCircuit(noisiq_circuit.n_qubits)
    
    for op in noisiq_circuit.operations:
        name = op.gate.name.upper()
        if name == 'H':
            qc.h(op.qubits[0])
        elif name == 'X':
            qc.x(op.qubits[0])
        elif name == 'Y':
            qc.y(op.qubits[0])
        elif name == 'Z':
            qc.z(op.qubits[0])
        elif name == 'S':
            qc.s(op.qubits[0])
        elif name == 'T':
            qc.t(op.qubits[0])
        elif name == 'CNOT':
            qc.cx(op.qubits[0], op.qubits[1])
        elif name == 'CZ':
            qc.cz(op.qubits[0], op.qubits[1])
        elif name == 'I':
            qc.id(op.qubits[0])
        else:
            # For unknown gates, we can add a generic instruction or skip
            pass
            
    return qc


def draw_error_heatmap(
    circuit: NoisiqCircuit,
    result: Optional[StimTableauResult] = None,
    step_index: Optional[int] = None,
    aggregate: bool = False,
    ax: Optional[plt.Axes] = None,
    qc_cache: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Draw circuit with error heatmap.
    
    Args:
        circuit: The NoisiQ circuit.
        result: Simulation result.
        step_index: If provided, show errors for this specific step.
        aggregate: If True, show aggregate error frequency across all steps.
        ax: Matplotlib axes to draw on.
        qc_cache: Optional dictionary to cache the converted Qiskit circuit.
    """
    if qc_cache is not None and 'qc' in qc_cache:
        qc = qc_cache['qc']
    else:
        qc = to_qiskit_circuit(circuit)
        if qc_cache is not None:
            qc_cache['qc'] = qc
    
    # Use qiskit's mpl drawer with some speed-up options
    fig = circuit_drawer(qc, output='mpl', ax=ax, plot_barriers=False)
    if ax is None:
        ax = fig.gca()
        
    # Data pipeline: Map (step, qubit) to error severity
    error_map = {} # (op_idx, qubit) -> severity
    
    if result is not None:
        if step_index is not None:
            # Show errors for a single step
            if 0 <= step_index < len(result.steps):
                step = result.steps[step_index]
                for err in step.errors:
                    error_map[(err.gate_index, err.qubit)] = 1.0
        elif aggregate:
            # Aggregate errors across all shots (if result had multiple shots, 
            # but StimTableauResult currently is single shot).
            # For single shot, aggregate is same as showing all errors that happened.
            for step in result.steps:
                for err in step.errors:
                    error_map[(err.gate_index, err.qubit)] = error_map.get((err.gate_index, err.qubit), 0) + 1.0
            
            # Normalize if needed (for single shot it's just 0 or 1)
    
    # Patch the Qiskit drawing
    # Qiskit's MPL drawer creates patches for gates. 
    # We match patches to (op_idx, qubit) based on their coordinates.
    
    # This mapping is approximate and depends on Qiskit's default spacing.
    # In Qiskit MPL, qubits are at y = 0, 1, 2...
    # Operations are at x = 1, 2, 3... (usually)
    
    for patch in ax.patches:
        if isinstance(patch, patches.Rectangle):
            x, y = patch.get_xy()
            w, h = patch.get_width(), patch.get_height()
            
            # Center of the patch
            cx, cy = x + w/2, y + h/2
            
            # Approximate mapping:
            # qubit_idx is round(cy)
            # op_idx is round(cx) - 1 (since first gate starts around x=1)
            q_idx = int(round(cy))
            op_idx = int(round(cx)) - 1
            
            if (op_idx, q_idx) in error_map:
                severity = error_map[(op_idx, q_idx)]
                # Color red with intensity based on severity
                patch.set_facecolor((1.0, 1.0 - severity, 1.0 - severity))
                patch.set_edgecolor('red')
                patch.set_linewidth(2)

    return fig
