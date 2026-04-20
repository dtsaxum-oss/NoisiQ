import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from ..ir import Circuit
from .pauli_frame_tracker import PauliFrame

def draw_circuit_with_labels(
    ax: plt.Axes, 
    circuit: Circuit, 
    pauli_frame: Optional[PauliFrame] = None, 
    highlight_t: Optional[int] = None, 
    note: str = ""
):
    """
    Draws a custom Matplotlib circuit timeline that accurately visualizes Pauli error propagation.
    """
    n_qubits = circuit.n_qubits
    # Determine number of time steps (we assume one operation per time step for simplicity in this visualization, 
    # or use the op index as time step)
    T = len(circuit.operations) if circuit.operations else 1

    # Wires
    for q in range(n_qubits):
        y = (n_qubits - 1 - q)
        ax.plot([-0.5, T-0.5], [y, y], linewidth=2, color="k")

    # Gates
    for t, op in enumerate(circuit.operations):
        x = t
        name = op.gate.name.upper()
        
        # Color based on whether it's highlighted
        edge = "k" if highlight_t is None or t != highlight_t else "tab:red"
        lw = 2 if edge == "k" else 3
        
        if name in ("H", "S", "X", "Y", "Z", "T"):
            (q,) = op.qubits
            y = (n_qubits - 1 - q)
            rect = plt.Rectangle((x-0.30, y-0.25), 0.60, 0.50, fill=True, facecolor="white", linewidth=lw, edgecolor=edge)
            ax.add_patch(rect)
            ax.text(x, y, name, ha="center", va="center", fontsize=10, color=edge)
            
        elif name == "CZ":
            q1, q2 = op.qubits
            y1 = (n_qubits - 1 - q1)
            y2 = (n_qubits - 1 - q2)
            ax.plot([x, x], [y1, y2], linewidth=lw, color=edge)
            ax.scatter([x, x], [y1, y2], s=70, c=edge, zorder=3)
            
        elif name in ("CX", "CNOT"):
            q1, q2 = op.qubits # q1 control, q2 target
            y1 = (n_qubits - 1 - q1)
            y2 = (n_qubits - 1 - q2)
            ax.plot([x, x], [y1, y2], linewidth=lw, color=edge)
            ax.scatter([x], [y1], s=70, c=edge, zorder=3) # Control dot
            
            # Target cross
            circle = plt.Circle((x, y2), 0.15, fill=True, facecolor="white", edgecolor=edge, linewidth=lw, zorder=3)
            ax.add_patch(circle)
            ax.plot([x, x], [y2-0.15, y2+0.15], linewidth=lw, color=edge, zorder=4)
            ax.plot([x-0.15, x+0.15], [y2, y2], linewidth=lw, color=edge, zorder=4)
            
        elif name == "I":
            pass # Draw nothing for identity

    # Qubit labels + current Pauli error labels
    x_text = (-0.75 if highlight_t is None else highlight_t - 0.45)
    
    pauli_string = pauli_frame.get_pauli_string() if pauli_frame else "I" * n_qubits
    
    for q in range(n_qubits):
        y = (n_qubits - 1 - q)
        # Qubit index on the far left
        ax.text(-0.9, y, f"q{q}", va="center", ha="right", fontsize=12, color="k")
        
        # Pauli error label
        p_label = pauli_string[q]
        ax.text(x_text, y+0.32, p_label, va="center", ha="center",
                fontsize=11, color=("tab:red" if p_label != "I" else "0.35"))

    if note:
        ax.set_title(note, fontsize=12)
        
    ax.set_xlim(-1.2, T-0.2)
    ax.set_ylim(-1, n_qubits)
    ax.set_xticks(range(T))
    ax.set_yticks([])
    ax.set_frame_on(False)
