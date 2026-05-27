import pytest
import numpy as np
from noisiq.ir.circuit import Circuit
from noisiq.ir import gates
from noisiq.backends.qiskit_backend import QiskitAerBackend
from noisiq.visualization.bloch_sphere import draw_bloch_sphere, density_matrix_to_bloch_vector
from noisiq.visualization.density_matrix import plot_density_matrix

def test_qiskit_visualization_no_errors():
    circuit = Circuit(1)
    circuit.add_gate(gates.H, [0])
    
    backend = QiskitAerBackend()
    result = backend.run(circuit, n_shots=10)
    
    # Test Bloch Sphere
    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    vec = density_matrix_to_bloch_vector(result.final_state)
    draw_bloch_sphere(ax, [vec])
    assert fig1 is not None
    
    # Test Density Matrix
    circuit2 = Circuit(2)
    circuit2.add_gate(gates.H, [0])
    circuit2.add_gate(gates.CNOT, [0, 1])
    result2 = backend.run(circuit2, n_shots=10)
    
    fig2 = plot_density_matrix(result2.final_state)
    assert fig2 is not None
