"""
Qiskit Aer simulation backend.
"""

from typing import Dict, Optional, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel, quantum_error, pauli_error as qiskit_pauli_error
from qiskit.quantum_info import Kraus

from ..ir.circuit import Circuit
from ..noise.kraus_channels import KrausChannel, CombinedChannel
from ..noise.pauli_error import PauliError
from ..noise.correlated_errors import CorrelatedPauliError
from ..noise.coherent_errors import CoherentRotation, StochasticCoherentRotation
from ..results import SimulationResult
from .base import Backend


class QiskitAerBackend(Backend):
    """
    Simulation backend using Qiskit Aer for universal circuit simulation.
    Supports all gates and complex noise models via Qiskit Aer's density matrix simulator.
    """

    def run(
        self,
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel, Dict[int, Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel]], None] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        
        # Build QuantumCircuit
        qc = QuantumCircuit(circuit.n_qubits)
        
        # Prepare noise dict
        _channel_types = (KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel)
        noise_dict: Dict[int, object] = {}
        if isinstance(noise_model, _channel_types):
            noise_dict = {i: noise_model for i in range(len(circuit.operations))}
        elif isinstance(noise_model, dict):
            noise_dict = noise_model

        # Helper to apply noise to QuantumCircuit directly
        def apply_noise_to_qc(channel, qubits):
            if isinstance(channel, CombinedChannel):
                for inner in channel.channels:
                    apply_noise_to_qc(inner, qubits)
            elif isinstance(channel, KrausChannel):
                n_chan_qubits = int(round(np.log2(channel.operators[0].shape[0])))
                if n_chan_qubits == 1:
                    for q in qubits:
                        qc.append(Kraus(channel.operators), [q])
                else:
                    target_qubits = list(qubits)[:n_chan_qubits]
                    qc.append(Kraus(channel.operators), target_qubits)
            elif isinstance(channel, CorrelatedPauliError):
                # Construct qiskit quantum_error from probs
                probs_and_paulis = []
                p_i = max(0.0, 1.0 - sum(channel.probs.values()))
                probs_and_paulis.append(('I' * channel.num_qubits, p_i))
                for p_str, p_val in channel.probs.items():
                    probs_and_paulis.append((p_str, p_val))
                q_err = qiskit_pauli_error(probs_and_paulis)
                qc.append(q_err, list(qubits)[:channel.num_qubits])
            elif isinstance(channel, PauliError):
                p_i = max(0.0, 1.0 - (channel.p_x + channel.p_y + channel.p_z))
                q_err = qiskit_pauli_error([
                    ('I', p_i),
                    ('X', channel.p_x),
                    ('Y', channel.p_y),
                    ('Z', channel.p_z)
                ])
                for q in qubits:
                    qc.append(q_err, [q])
            elif isinstance(channel, StochasticCoherentRotation):
                raise TypeError(
                    f"StochasticCoherentRotation is not supported by QiskitAerBackend — "
                    f"it samples a random angle per trajectory shot, which has no "
                    f"equivalent in Qiskit Aer's static noise model. Convert to a "
                    f"PauliError approximation first via channel.to_pauli_error()."
                )

        for op_idx, op in sorted(enumerate(circuit.operations), key=lambda kv: (kv[1].t, kv[0])):
            name = op.gate.name.upper()
            qubits = list(op.qubits)
            
            if name == 'I' or name == 'IDLE':
                qc.id(qubits[0])
            elif name == 'X':
                qc.x(qubits[0])
            elif name == 'Y':
                qc.y(qubits[0])
            elif name == 'Z':
                qc.z(qubits[0])
            elif name == 'H':
                qc.h(qubits[0])
            elif name == 'S':
                qc.s(qubits[0])
            elif name == 'S_DAG':
                qc.sdg(qubits[0])
            elif name == 'T':
                qc.t(qubits[0])
            elif name == 'T_DAG':
                qc.tdg(qubits[0])
            elif name in ('CNOT', 'CX'):
                qc.cx(qubits[0], qubits[1])
            elif name == 'CZ':
                qc.cz(qubits[0], qubits[1])
            elif name == 'SWAP':
                qc.swap(qubits[0], qubits[1])
            elif name.startswith('P('):
                # extract angle
                import re
                match = re.match(r'P\((.*?)\)', name)
                if match:
                    angle = float(match.group(1))
                    qc.p(angle, qubits[0])
            elif name.startswith('RZ('):
                import re
                match = re.match(r'RZ\((.*?)\)', name)
                if match:
                    angle = float(match.group(1))
                    qc.rz(angle, qubits[0])
            else:
                # Custom gate
                qc.append(op.gate.matrix, qubits)

            if op_idx in noise_dict:
                apply_noise_to_qc(noise_dict[op_idx], qubits)

        # Save density matrix
        qc.save_density_matrix()

        # Run simulation
        simulator = AerSimulator(method='density_matrix')
        result = simulator.run(qc, shots=n_shots, seed_simulator=seed).result()
        
        # The result density matrix
        rho = result.data().get('density_matrix')
        if rho is not None:
            rho = np.array(rho)
        else:
            dim = 2 ** circuit.n_qubits
            rho = np.zeros((dim, dim), dtype=complex)
            rho[0, 0] = 1.0

        return SimulationResult(
            final_state=rho,
            meta={"n_shots": n_shots, "n_qubits": circuit.n_qubits, "seed": seed, "backend": "QiskitAerBackend"},
        )
