"""
Stim-based Tableau simulation for Clifford circuits with step-by-step noise tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import stim

from ..ir import Circuit, Operation
from ..noise import PauliError


@dataclass
class ErrorEvent:
    """Record of a single error occurrence."""
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
class StepResult:
    """Results from a single step (layer) of simulation."""
    time_step: int
    operation: Operation
    errors: List[ErrorEvent] = field(default_factory=list)
    # We can store the tableau or some representation of the state here
    # For now, let's store the Pauli frame or stabilizers if needed for visualization
    tableau: Optional[stim.Tableau] = None


@dataclass
class StimTableauResult:
    """Results from a full simulation run."""
    steps: List[StepResult]
    n_qubits: int
    seed: Optional[int]
    final_tableau: stim.Tableau
    
    def __repr__(self) -> str:
        return (
            f"StimTableauResult(n_qubits={self.n_qubits}, "
            f"steps={len(self.steps)}, seed={self.seed})"
        )


class StimTableauBackend:
    """
    Backend using stim.TableauSimulator for step-by-step simulation.
    """
    
    def run_single_shot(
        self,
        circuit: Circuit,
        noise_config: Optional[Dict[int, PauliError]] = None,
        seed: Optional[int] = None,
    ) -> StimTableauResult:
        """
        Run a single shot of the circuit with step-by-step noise tracking.
        """
        circuit.validate()
        
        rng = np.random.default_rng(seed=seed)
        sim = stim.TableauSimulator()
        
        # Ensure we have enough qubits (stim handles this dynamically but we might want to be explicit)
        # stim.TableauSimulator doesn't have an explicit 'set_n_qubits' but it grows as needed.
        
        steps: List[StepResult] = []
        noise_config = noise_config or {}
        
        for op_idx, op in enumerate(circuit.operations):
            # 1. Apply the ideal gate
            self._apply_gate_to_sim(sim, op)
            
            # 2. Sample and apply noise
            step_errors: List[ErrorEvent] = []
            if op_idx in noise_config:
                noise_model = noise_config[op_idx]
                for qubit in op.qubits:
                    sampled_pauli = noise_model.sample(rng)
                    if sampled_pauli != 'I':
                        if sampled_pauli == 'X':
                            sim.x(qubit)
                        elif sampled_pauli == 'Y':
                            sim.y(qubit)
                        elif sampled_pauli == 'Z':
                            sim.z(qubit)
                        
                        error_event = ErrorEvent(
                            gate_index=op_idx,
                            gate_name=op.gate.name,
                            qubit=qubit,
                            pauli=sampled_pauli,
                            time_step=op_idx
                        )
                        step_errors.append(error_event)
            
            # 3. Record step result
            # Capture a copy of the tableau for this step if needed for animation
            # Warning: this might be slow for very large circuits
            steps.append(StepResult(
                time_step=op_idx,
                operation=op,
                errors=step_errors,
                tableau=sim.current_inverse_tableau().inverse() # Get current tableau
            ))
            
        return StimTableauResult(
            steps=steps,
            n_qubits=circuit.n_qubits,
            seed=seed,
            final_tableau=sim.current_inverse_tableau().inverse()
        )

    def _apply_gate_to_sim(self, sim: stim.TableauSimulator, op: Operation):
        """Map NoisiQ gates to stim instructions."""
        name = op.gate.name.upper()
        if name == 'H':
            sim.h(*op.qubits)
        elif name == 'X':
            sim.x(*op.qubits)
        elif name == 'Y':
            sim.y(*op.qubits)
        elif name == 'Z':
            sim.z(*op.qubits)
        elif name == 'S':
            sim.s(*op.qubits)
        elif name == 'CNOT':
            sim.cnot(*op.qubits)
        elif name == 'CZ':
            sim.cz(*op.qubits)
        elif name == 'I':
            pass # Identity does nothing in TableauSimulator
        else:
            raise ValueError(f"Gate {name} not supported by StimTableauBackend")
