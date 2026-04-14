"""
Pauli frame simulation via state vector propagation.

Provides exact simulation for circuits with Pauli noise channels only. 
Useful for circuits ~20 qubits. 

Classes:
    ErrorEvent: Record of a single error occurrence
    PauliFrameResult: Container for simulation results
    PauliFrameBackend: Main simulation engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..ir import Circuit, gates
from ..noise import PauliError


# Type alias for noise configuration
NoiseConfig = Dict[int, PauliError]  # gate_index -> PauliError


@dataclass
class ErrorEvent:
    """  
    Attributes:
        gate_index: Index of the operation where error occurred (0, 1, 2, ...)
        gate_name: Name of the gate being executed (e.g., "H", "CNOT")
        qubit: Which qubit the error affected (0-indexed)
        pauli: Which Pauli operator was applied ('I', 'X', 'Y', 'Z')
        time_step: Sequential time step in the circuit (same as gate_index)
    """
    gate_index: int
    gate_name: str
    qubit: int
    pauli: str
    time_step: int
    
    def __repr__(self) -> str:
        return (
            f"ErrorEvent(t={self.time_step}, gate={self.gate_name}, "
            f"qubit={self.qubit}, pauli={self.pauli})"
        )


@dataclass
class PauliFrameResult:
    """
    Results from a single-shot Pauli frame simulation.
    
    Attributes:
        final_state: Final state vector (complex array of size 2^n)
        error_history: List of all errors that occurred during simulation
        intermediate_states: Optional list of states after each operation
                           (only stored if track_history=True)
        n_qubits: Number of qubits in the circuit
        seed: Random seed used for reproducibility (None if not specified)
    """
    final_state: np.ndarray
    error_history: List[ErrorEvent]
    intermediate_states: Optional[List[np.ndarray]]
    n_qubits: int
    seed: Optional[int]
    
    def __repr__(self) -> str:
        parts = [
            f"n_qubits={self.n_qubits}",
            f"errors={len(self.error_history)}",
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        if self.intermediate_states is not None:
            parts.append(f"history_length={len(self.intermediate_states)}")
        return f"PauliFrameResult({', '.join(parts)})"


class PauliFrameBackend:
    """
    State vector simulation backend with Pauli noise sampling.
    
    The simulation follows this algorithm for each operation:
        1. Expand gate to full Hilbert space
        2. Apply gate to state vector
        3. For each qubit the gate acts on:
           a. Sample which Pauli error occurs
           b. If not identity, apply the error
           c. Record the error event

    Methods:
        run_single_shot: Simulate one execution of the circuit with noise
    """
    
    def __init__(self):
        """
        Initialize the Pauli frame backend.
        
        The backend is stateless between runs - each call to run_single_shot
        creates a fresh state vector.
        """
        pass  # No persistent state needed for now
    
    def run_single_shot(
        self,
        circuit: Circuit,
        noise_config: Optional[NoiseConfig] = None,
        seed: Optional[int] = None,
        track_history: bool = False,
        validate_norm: bool = False,
    ) -> PauliFrameResult:
        """
        Run a single shot of the circuit with Pauli noise.
        
        Simulates the circuit once, sampling Pauli errors from the noise model
        at each specified location.
        
        Args:
            circuit: The quantum circuit to simulate
            noise_config: Dictionary mapping gate indices to PauliError models.
                         If None or empty, runs noiseless simulation.
            seed: Random seed for reproducibility. If None, uses random seed.
            track_history: If True, store state after each gate and error.
            validate_norm: If True, check state normalization after each step.
                          
        Returns:
            PauliFrameResult containing final state, error history, and
            optionally intermediate states
            
        Raises:
            ValueError: If circuit validation fails
            ValueError: If noise_config contains invalid gate indices
            ValueError: If state normalization check fails (when validate_norm=True)
        """
        # Validate inputs
        circuit.validate()
        self._validate_noise_config(circuit, noise_config)
        
        # Handle empty noise config
        if noise_config is None:
            noise_config = {}
        
        # Initialize random number generator
        rng = np.random.default_rng(seed=seed)
        
        # Initialize state vector to |00...0⟩
        state = self._init_state_vector(circuit.n_qubits)
        
        # Initialize tracking structures
        error_history: List[ErrorEvent] = []
        intermediate_states: Optional[List[np.ndarray]] = (
            [] if track_history else None
        )
        
        # Handle empty circuit edge case
        if len(circuit.operations) == 0:
            return PauliFrameResult(
                final_state=state,
                error_history=error_history,
                intermediate_states=intermediate_states,
                n_qubits=circuit.n_qubits,
                seed=seed,
            )
        
        # Main simulation loop - iterate through operations
        for op_idx, operation in enumerate(circuit.operations):
            # 1. Apply the gate
            state = self._apply_gate(
                state=state,
                gate=operation.gate,
                target_qubits=operation.qubits,
                n_qubits=circuit.n_qubits,
            )
            
            # Optional: validate normalization after gate
            if validate_norm:
                self._check_normalization(state, f"after gate {operation.gate.name}")
            
            # Optional: store state after gate application
            if track_history:
                intermediate_states.append(state.copy())
            
            # 2. Apply noise if configured for this operation
            if op_idx in noise_config:
                noise_model = noise_config[op_idx]
                
                # Apply noise independently to each qubit the gate acts on
                for qubit in operation.qubits:
                    # Sample which Pauli error occurs
                    sampled_pauli = noise_model.sample(rng)
                    
                    # Apply the error if it's not identity
                    if sampled_pauli != 'I':
                        state = self._apply_pauli_error(
                            state=state,
                            pauli=sampled_pauli,
                            target_qubit=qubit,
                            n_qubits=circuit.n_qubits,
                        )
                        
                        # Optional: validate normalization after error
                        if validate_norm:
                            self._check_normalization(
                                state, 
                                f"after {sampled_pauli} error on qubit {qubit}"
                            )
                    
                    # Record the error event (even if it was identity)
                    error_event = ErrorEvent(
                        gate_index=op_idx,
                        gate_name=operation.gate.name,
                        qubit=qubit,
                        pauli=sampled_pauli,
                        time_step=op_idx,  # Same as gate_index for sequential execution
                    )
                    error_history.append(error_event)
                    
                    # Optional: store state after each error
                    if track_history and sampled_pauli != 'I':
                        intermediate_states.append(state.copy())
        
        # Return results
        return PauliFrameResult(
            final_state=state,
            error_history=error_history,
            intermediate_states=intermediate_states,
            n_qubits=circuit.n_qubits,
            seed=seed,
        )
    
    def _init_state_vector(self, n_qubits: int) -> np.ndarray:
        """
        Initialize state vector to |00...0⟩.
        
        Creates a state vector of size 2^n with the first element set to 1
        (representing the all-zeros computational basis state) and all other
        elements set to 0.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            Complex array of size 2^n initialized to |00...0⟩
            
        Example:
            >>> state = self._init_state_vector(2)
            >>> # Returns [1+0j, 0+0j, 0+0j, 0+0j] for |00⟩
        """
        dim = 2 ** n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0  # |00...0⟩ state
        return state
    
    def _apply_gate(
        self,
        state: np.ndarray,
        gate: gates.Gate,
        target_qubits: tuple,
        n_qubits: int,
    ) -> np.ndarray:
        """
        Apply a quantum gate to the state vector.
        
        Expands the gate to act on the full n-qubit Hilbert space and
        applies it via matrix-vector multiplication.
        
        Args:
            state: Current state vector (2^n complex amplitudes)
            gate: Gate to apply
            target_qubits: Which qubits the gate acts on
            n_qubits: Total number of qubits
            
        Returns:
            Updated state vector after gate application
        """
        # Expand gate to full Hilbert space
        full_gate = gates.expand_gate_to_full_space(
            gate=gate,
            target_qubits=target_qubits,
            n_qubits=n_qubits,
        )
        
        # Apply gate: |ψ'⟩ = U|ψ⟩
        new_state = full_gate @ state
        
        return new_state
    
    def _apply_pauli_error(
        self,
        state: np.ndarray,
        pauli: str,
        target_qubit: int,
        n_qubits: int,
    ) -> np.ndarray:
        """
        Apply a single-qubit Pauli error to the state.
        
        Args:
            state: Current state vector
            pauli: Which Pauli to apply ('I', 'X', 'Y', 'Z')
            target_qubit: Which qubit to apply error to
            n_qubits: Total number of qubits
            
        Returns:
            Updated state vector after error application
        """
        # Get the Pauli gate object
        pauli_gate_map = {
            'I': gates.I,
            'X': gates.X,
            'Y': gates.Y,
            'Z': gates.Z,
        }
        pauli_gate = pauli_gate_map[pauli]
        
        # Expand to full space (same as regular gate application)
        full_pauli = gates.expand_gate_to_full_space(
            gate=pauli_gate,
            target_qubits=(target_qubit,),
            n_qubits=n_qubits,
        )
        
        # Apply error: |ψ'⟩ = P|ψ⟩
        new_state = full_pauli @ state
        
        return new_state
    
    def _validate_noise_config(
        self,
        circuit: Circuit,
        noise_config: Optional[NoiseConfig],
    ) -> None:
        """
        Validate that noise configuration is consistent with circuit.
        
        Args:
            circuit: The circuit being simulated
            noise_config: Noise configuration to validate
            
        Raises:
            ValueError: If any gate index is out of bounds
            ValueError: If any value is not a PauliError instance
        """
        if noise_config is None:
            return
        
        n_operations = len(circuit.operations)
        
        for gate_idx, noise_model in noise_config.items():
            # Check gate index is valid
            if not 0 <= gate_idx < n_operations:
                raise ValueError(
                    f"Noise config has invalid gate index {gate_idx}. "
                    f"Circuit has {n_operations} operations (indices 0-{n_operations-1})."
                )
            
            # Check noise model is a PauliError
            if not isinstance(noise_model, PauliError):
                raise ValueError(
                    f"Noise config at gate {gate_idx} must be a PauliError instance, "
                    f"got {type(noise_model).__name__}"
                )
    
    def _check_normalization(self, state: np.ndarray, context: str) -> None:
        """
        Check that state vector is properly normalized.
        
        Validates that ⟨ψ|ψ⟩ = 1 within numerical tolerance.
        
        Args:
            state: State vector to check
            context: Description of when this check is happening (for error message)
            
        Raises:
            ValueError: If state is not normalized within tolerance
            
        Notes:
            Only called when validate_norm=True (debugging mode).
        """
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError(
                f"State normalization check failed {context}: "
                f"||ψ|| = {norm:.12f} (expected 1.0)"
            )
