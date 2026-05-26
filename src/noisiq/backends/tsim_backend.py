import numpy as np
from typing import Dict, Optional, List, Union
import tsim

from noisiq.ir import Circuit
from noisiq.noise import PauliError
from noisiq.noise.kraus_channels import KrausChannel
from noisiq.noise.pauli_channels import PauliChannel
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
        'SWAP': 'SWAP',
        'CS': 'CS',
        'CS_DAG': 'CS_DAG',
        'CCZ': 'CCZ',
        'I': 'I',
        'IDLE': 'I',
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
        from noisiq.results import SimulationResult
        
        counts = {}
        for sample in samples:
            bitstring = "".join(str(int(b)) for b in sample)
            counts[bitstring] = counts.get(bitstring, 0) + 1
            
        return SimulationResult(final_state=None, counts=counts)

    def _build_tsim_circuit(self, circuit: Circuit, noise_config: Optional[Dict[int, Union[KrausChannel, PauliError]]] = None) -> str:
        lines = []
        noise_config = noise_config or {}

        for op_idx, op in sorted(enumerate(circuit.operations),
                                 key=lambda kv: (kv[1].t, kv[0])):
            name = op.gate.name.upper()
            qubits_str = " ".join(map(str, op.qubits))

            # Parameterized phase gates: extract angle from gate name "P(θ)" / "RZ(θ)"
            if name.startswith('P(') or name.startswith('RZ('):
                try:
                    theta = float(op.gate.name[op.gate.name.index('(') + 1:-1])
                except (ValueError, IndexError):
                    raise ValueError(f"Cannot parse angle from gate name '{op.gate.name}'")
                tsim_prefix = 'RZ' if name.startswith('RZ(') else 'RZ'
                lines.append(f"{tsim_prefix}({theta}) {qubits_str}")
            elif name not in self.GATE_MAP:
                raise ValueError(f"Gate {name} not supported by TsimBackend")
            else:
                tsim_name = self.GATE_MAP[name]
                lines.append(f"{tsim_name} {qubits_str}")

            # Apply noise
            if op_idx in noise_config:
                noise_model = noise_config[op_idx]
                if isinstance(noise_model, KrausChannel):
                    raise TypeError("TsimBackend does not support non-Pauli noise models (e.g., KrausChannel).")
                if isinstance(noise_model, PauliChannel):
                    noise_model = noise_model.to_pauli_error()
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
