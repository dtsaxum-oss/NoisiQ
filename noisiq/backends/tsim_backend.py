import numpy as np
from typing import Dict, Optional, List, Union
import tsim

from ..ir import Circuit
from ..noise import PauliError
from ..noise.kraus_channels import KrausChannel
from .pauli_frame import StimTableauResult

class TsimBackend:
    """
    Backend using tsim for universal quantum circuit simulation.
    """

    GATE_MAP = {
        'H': 'H',
        'X': 'X',
        'Y': 'Y',
        'Z': 'Z',
        'S': 'S',
        'S_DAG': 'S_DAG',
        'T': 'T',
        'T_DAG': 'T_DAG',
        'CNOT': 'CX',
        'CX': 'CX',
        'CZ': 'CZ',
        'I': 'I',
    }

    def run(
        self,
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]], None] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Run n_shots using tsim.
        """
        circuit.validate()

        # Handle different noise_model formats
        noise_config = {}
        if isinstance(noise_model, dict):
            noise_config = noise_model
        elif noise_model is not None:
            for i in range(len(circuit.operations)):
                noise_config[i] = noise_model

        tsim_str = self._build_tsim_circuit(circuit, noise_config)
        tsim_circuit = tsim.Circuit(tsim_str)

        sampler = tsim_circuit.compile_sampler(seed=seed)
        samples = sampler.sample(shots=n_shots)

        # Note: bloqade-tsim is a sampler based on ZX-calculus stabilizer rank decomposition.
        # It does not natively support extraction of the full state vector. For step-by-step
        # visualization of the state vector, TrajectoryBackend should be used as a fallback.
        from ..results import SimulationResult
        
        counts = {}
        for sample in samples:
            bitstring = "".join(str(int(b)) for b in sample)
            counts[bitstring] = counts.get(bitstring, 0) + 1
            
        return SimulationResult(final_state=None, counts=counts)

    def _build_tsim_circuit(self, circuit: Circuit, noise_config: Optional[Dict[int, Union[KrausChannel, PauliError]]] = None) -> str:
        lines = []
        noise_config = noise_config or {}

        for op_idx, op in enumerate(circuit.operations):
            # Map gate
            name = op.gate.name.upper()
            if name not in self.GATE_MAP:
                raise ValueError(f"Gate {name} not supported by TsimBackend")

            tsim_name = self.GATE_MAP[name]
            qubits_str = " ".join(map(str, op.qubits))
            lines.append(f"{tsim_name} {qubits_str}")

            # Apply noise
            if op_idx in noise_config:
                noise_model = noise_config[op_idx]
                if isinstance(noise_model, KrausChannel):
                    raise TypeError("TsimBackend does not support non-Pauli noise models (e.g., KrausChannel).")
                for qubit in op.qubits:
                    p_x = noise_model.p_x
                    p_y = noise_model.p_y
                    p_z = noise_model.p_z
                    
                    if p_x > 0 or p_y > 0 or p_z > 0:
                        lines.append(f"PAULI_CHANNEL_1({p_x}, {p_y}, {p_z}) {qubit}")

        # Add measurements to all qubits at the end if we want
        for q in range(circuit.n_qubits):
            lines.append(f"M {q}")

        return "\n".join(lines)
