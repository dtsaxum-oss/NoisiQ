import numpy as np
from typing import Dict, List, Optional, Union
import functools

from noisiq.ir import Circuit, Operation
from noisiq.noise import PauliError
from noisiq.noise.kraus_channels import KrausChannel
from noisiq.results import SimulationResult

class TrajectoryBackend:
    """
    Monte Carlo Quantum Trajectory simulator for non-Pauli noise (Kraus channels).
    Executes N independent shots (trajectories) to approximate the density matrix evolution.
    """
    
    def run(
        self,
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]]] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        circuit.validate()
        
        if circuit.n_qubits > 15:
            # Memory warning: a 15 qubit complex128 state vector is 2^15 * 16 bytes = 524 KB
            # But the simulation is compute intensive.
            import warnings
            warnings.warn("Simulating more than 15 qubits with TrajectoryBackend is memory and compute intensive.")
            
        rng = np.random.default_rng(seed)
        
        # Determine the noise configuration map (index -> channel)
        noise_config = {}
        if isinstance(noise_model, dict):
            noise_config = noise_model
        elif noise_model is not None:
            # Apply to all operations if not a dict
            for i in range(len(circuit.operations)):
                noise_config[i] = noise_model
                
        # To accumulate density matrix if we wanted it. But wait, returning state_vector directly?
        # For mixed states, we can either return a density matrix or just counts.
        # Week 6 says: "Average density matrix over N trajectories: rho = (1/N) sum |psi_i><psi_i|"
        
        n = circuit.n_qubits
        dim = 2**n
        
        # Only accumulate density matrix if requested or n <= 10
        # (10 qubits = 1024x1024 complex128 matrix = 16MB)
        # (15 qubits = 32768x32768 complex128 matrix = 16GB)
        accumulate_rho = n <= 10
        
        rho = np.zeros((dim, dim), dtype=np.complex128) if accumulate_rho else None
        counts = {}
        
        for _ in range(n_shots):
            psi = np.zeros(dim, dtype=np.complex128)
            psi[0] = 1.0
            
            # Step by step evolution
            for op_idx, op in enumerate(circuit.operations):
                # 1. Apply ideal unitary gate
                psi = self._apply_gate(psi, op, n)
                
                # 2. Apply noise
                if op_idx in noise_config:
                    channel = noise_config[op_idx]
                    if isinstance(channel, KrausChannel):
                        for q in op.qubits:
                            psi = self._apply_kraus_sampling(psi, channel.operators, q, n, rng)
                    elif isinstance(channel, PauliError):
                        for q in op.qubits:
                            p = channel.sample(rng)
                            if p != 'I':
                                op_pauli = Operation(gate=channel.get_operator(p), qubits=(q,), t=0)
                                psi = self._apply_gate(psi, op_pauli, n)

            # Accumulate density matrix
            if accumulate_rho:
                rho += np.outer(psi, psi.conj())
            
            # Sample measurement
            # Calculate probabilities of each computational basis state
            probs = np.abs(psi)**2
            # Handle float precision
            probs = np.clip(probs, 0, 1)
            probs /= np.sum(probs)
            
            # Sample bitstring
            measured_idx = rng.choice(dim, p=probs)
            bitstring = format(measured_idx, f'0{n}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
            
        if accumulate_rho:
            rho /= n_shots
        
        return SimulationResult(final_state=rho, counts=counts)

    def _apply_gate(self, psi: np.ndarray, op: Operation, n: int) -> np.ndarray:
        """Apply a unitary gate to the state vector using reshape and transpose."""
        matrix = op.gate.matrix
        qubits = op.qubits
        return self._apply_matrix_to_qubits(psi, matrix, qubits, n)
        
    def _apply_kraus_sampling(self, psi: np.ndarray, operators: List[np.ndarray], qubit: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Apply Kraus operator sampling."""
        probabilities = []
        new_states = []
        
        for K in operators:
            # apply K to psi
            psi_k = self._apply_matrix_to_qubits(psi, K, (qubit,), n)
            # calculate probability p_k = <psi | K^dagger K | psi> = norm(K |psi>)^2
            p_k = np.linalg.norm(psi_k)**2
            probabilities.append(p_k.real)
            new_states.append(psi_k)
            
        # Preserve the raw branch probabilities for renormalization
        raw_probabilities = np.array(probabilities)
        
        # Use a normalized/clipped copy only for sampling
        sampling_probabilities = np.clip(raw_probabilities, 0, 1)
        # Normalize in case of tiny floating point issues
        sum_p = np.sum(sampling_probabilities)
        if sum_p > 0:
            sampling_probabilities /= sum_p
        else:
            # Fallback if probability is 0 (shouldn't happen with trace-preserving operations on valid states)
            return psi
            
        idx = rng.choice(len(operators), p=sampling_probabilities)
        
        # Renormalize chosen state using the original unnormalized branch probability
        if raw_probabilities[idx] > 0:
            return new_states[idx] / np.sqrt(raw_probabilities[idx])
        return new_states[idx]

    def _apply_matrix_to_qubits(self, state: np.ndarray, matrix: np.ndarray, qubits: tuple, n: int) -> np.ndarray:
        """Apply a matrix to specific qubits in a state vector."""
        # Reshape state to tensor
        tensor = state.reshape([2] * n)
        
        # Determine the number of qubits the matrix acts on
        num_act = len(qubits)
        
        # Reshape the matrix to match the tensor contraction
        matrix_tensor = matrix.reshape([2] * num_act * 2)
        
        # Construct the axes for tensordot
        # We contract the last 'num_act' axes of the matrix with the 'qubits' axes of the state tensor
        # matrix_tensor shape: [out_0, out_1, ..., in_0, in_1, ...]
        
        axes_state = list(qubits)
        axes_matrix = list(range(num_act, 2 * num_act))
        
        # Perform tensor contraction
        new_tensor = np.tensordot(matrix_tensor, tensor, axes=(axes_matrix, axes_state))
        
        # The result has axes: [out_0, out_1, ..., remaining_state_axes...]
        # We need to transpose it back to the original order
        
        # Calculate the mapping of old axes to new axes
        remaining_axes = [i for i in range(n) if i not in qubits]
        
        # After tensordot, the new axes are:
        # [0, 1, ..., num_act - 1] corresponding to the acted qubits
        # [num_act, ...] corresponding to the remaining_axes
        
        transpose_map = [0] * n
        for i, q in enumerate(qubits):
            transpose_map[q] = i
        for i, r in enumerate(remaining_axes):
            transpose_map[r] = num_act + i
            
        new_tensor = new_tensor.transpose(transpose_map)
        
        # Flatten back to a 1D state vector
        return new_tensor.flatten()
