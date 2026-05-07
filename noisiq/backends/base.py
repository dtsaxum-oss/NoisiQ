from typing import Protocol, Optional, Union, Dict
from ..ir import Circuit
from ..noise import PauliError
from ..noise.kraus_channels import KrausChannel
from ..results import SimulationResult

class Backend(Protocol):
    """
    Protocol for quantum circuit simulation backends.
    """
    def run(
        self,
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]], None] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run a simulation of the given circuit with the specified noise model.
        
        Args:
            circuit: The quantum circuit to simulate.
            noise_model: Optional noise model to apply. Can be a single noise channel applied
                         to all operations, or a dictionary mapping operation indices to noise channels.
            n_shots: Number of independent shots to run.
            seed: Optional random seed for reproducibility.
            
        Returns:
            SimulationResult containing the aggregated counts and potentially the final state or metadata.
        """
        ...
