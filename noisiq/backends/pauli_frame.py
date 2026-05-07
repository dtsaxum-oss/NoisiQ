"""
Stim-based Tableau simulation for Clifford circuits with step-by-step noise tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

import numpy as np
import stim

from noisiq.ir import Circuit, Operation
from noisiq.noise import PauliError
from noisiq.noise.kraus_channels import KrausChannel
from noisiq.results import SimulationResult


class NonCliffordError(Exception):
    """Raised when a non-Clifford gate is encountered in a Clifford-only backend."""
    pass

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
    
    def run(
        self,
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]], None] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run n_shots of the circuit with step-by-step noise tracking.
        Returns a SimulationResult with StimTableauResult inside its meta.
        """
        circuit.validate()
        
        # Handle different noise_model formats
        noise_config = {}
        if isinstance(noise_model, dict):
            noise_config = noise_model
        elif noise_model is not None:
            for i in range(len(circuit.operations)):
                noise_config[i] = noise_model
                
        # Type check
        for k, v in noise_config.items():
            if isinstance(v, KrausChannel):
                raise TypeError("StimTableauBackend does not support non-Pauli noise models (e.g., KrausChannel).")
        
        rng = np.random.default_rng(seed=seed)
        
        counts = {}
        all_steps = []
        final_tableau = None
        
        for shot in range(n_shots):
            sim = stim.TableauSimulator()
            steps: List[StepResult] = []
            
            for op_idx, op in enumerate(circuit.operations):
                # 1. Apply the ideal gate
                self._apply_gate_to_sim(sim, op)
                
                # 2. Sample and apply noise
                step_errors: List[ErrorEvent] = []
                if op_idx in noise_config:
                    model = noise_config[op_idx]
                    for qubit in op.qubits:
                        sampled_pauli = model.sample(rng)
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
                
                # 3. Record step result (only for the first shot to save memory, or maybe all?)
                # Actually, the original run_single_shot returned steps. We can return the first shot's steps in meta.
                if shot == 0:
                    steps.append(StepResult(
                        time_step=op_idx,
                        operation=op,
                        errors=step_errors,
                        tableau=sim.current_inverse_tableau().inverse()
                    ))
            
            if shot == 0:
                all_steps = steps
                final_tableau = sim.current_inverse_tableau().inverse()
                
            # Measure all qubits
            measurement = []
            for q in range(circuit.n_qubits):
                measurement.append(str(int(sim.measure(q))))
            bitstring = "".join(measurement)
            counts[bitstring] = counts.get(bitstring, 0) + 1
            
        stim_res = StimTableauResult(
            steps=all_steps,
            n_qubits=circuit.n_qubits,
            seed=seed,
            final_tableau=final_tableau
        )
            
        return SimulationResult(
            final_state=None,
            counts=counts,
            meta={"stim_result": stim_res}
        )

    def run_single_shot(
        self,
        circuit: Circuit,
        noise_config: Optional[Dict[int, PauliError]] = None,
        seed: Optional[int] = None,
    ) -> StimTableauResult:
        """
        Legacy method for single shot.
        """
        res = self.run(circuit, noise_model=noise_config, n_shots=1, seed=seed)
        return res.meta["stim_result"]

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
            raise NonCliffordError(
                f"Gate {name} not supported by StimTableauBackend. "
                f"Use BackendSelector to automatically route to a universal backend."
            )
