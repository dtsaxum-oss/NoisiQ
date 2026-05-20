import numpy as np
from typing import List, Dict, Any, Optional

from ..ir import Circuit
from ..backends.pauli_frame import StimTableauResult

class PauliFrame:
    """
    Tracks multi-qubit Pauli errors through Clifford gates using symplectic arrays.
    Ignores global phases and signs, focusing only on the Pauli operators.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.x = np.zeros(num_qubits, dtype=bool)
        self.z = np.zeros(num_qubits, dtype=bool)

    def copy(self) -> 'PauliFrame':
        new_frame = PauliFrame(self.num_qubits)
        new_frame.x = self.x.copy()
        new_frame.z = self.z.copy()
        return new_frame

    def inject_error(self, qubit: int, pauli: str):
        """Inject a Pauli error at a specific qubit."""
        if pauli == 'X':
            self.x[qubit] ^= True
        elif pauli == 'Z':
            self.z[qubit] ^= True
        elif pauli == 'Y':
            self.x[qubit] ^= True
            self.z[qubit] ^= True

    def get_pauli_string(self) -> str:
        """Returns the current Pauli string."""
        chars = []
        for i in range(self.num_qubits):
            if self.x[i] and self.z[i]:
                chars.append('Y')
            elif self.x[i]:
                chars.append('X')
            elif self.z[i]:
                chars.append('Z')
            else:
                chars.append('I')
        return "".join(chars)

    def apply_h(self, target: int):
        # H X H = Z, H Z H = X
        x_val = self.x[target]
        z_val = self.z[target]
        self.x[target] = z_val
        self.z[target] = x_val

    def apply_s(self, target: int):
        # S X S^dag = Y = XZ, S Z S^dag = Z
        # If X is present, it becomes XZ, meaning Z flips.
        if self.x[target]:
            self.z[target] ^= True

    def apply_cnot(self, control: int, target: int):
        # CNOT (X_c) CNOT = X_c X_t -> control X spreads to target X
        # CNOT (Z_t) CNOT = Z_c Z_t -> target Z spreads to control Z
        if self.x[control]:
            self.x[target] ^= True
        if self.z[target]:
            self.z[control] ^= True

    def apply_cz(self, control: int, target: int):
        # CZ (X_c) CZ = X_c Z_t -> control X spreads to target Z
        # CZ (X_t) CZ = Z_c X_t -> target X spreads to control Z
        if self.x[control]:
            self.z[target] ^= True
        if self.x[target]:
            self.z[control] ^= True

    def apply_x(self, target: int):
        # Commutes with X, anti-commutes with Z/Y (adds phase, but we ignore phase)
        pass

    def apply_y(self, target: int):
        pass

    def apply_z(self, target: int):
        pass

    def apply_gate(self, gate_name: str, qubits: List[int]):
        """
        Propagate the Pauli frame through an ideal gate.
        """
        gate = gate_name.upper()
        if gate == 'H':
            self.apply_h(qubits[0])
        elif gate == 'S':
            self.apply_s(qubits[0])
        elif gate == 'CX' or gate == 'CNOT':
            self.apply_cnot(qubits[0], qubits[1])
        elif gate == 'CZ':
            self.apply_cz(qubits[0], qubits[1])
        elif gate in ['X', 'Y', 'Z', 'I']:
            pass
        elif gate == 'T':
            raise NotImplementedError("T gate is not a Clifford gate and cannot be tracked efficiently by PauliFrame.")
        else:
            raise NotImplementedError(f"Gate {gate} not supported in PauliFrame tracker.")

def compute_error_trajectories(circuit: Circuit, result: StimTableauResult) -> List[PauliFrame]:
    """
    Computes the accumulated PauliFrame for each time step.
    Returns a list of PauliFrames where the i-th frame is the state of errors 
    AFTER the i-th step (ideal gate + injected errors).
    """
    frame = PauliFrame(circuit.n_qubits)
    trajectories = []

    for step in result.steps:
        # 1. Propagate existing errors through the ideal operation
        frame.apply_gate(step.operation.gate.name, step.operation.qubits)
        
        # 2. Inject any new errors that occurred at this step
        for error in step.errors:
            frame.inject_error(error.qubit, error.pauli)
            
        # 3. Record the frame state
        trajectories.append(frame.copy())
        
    return trajectories

