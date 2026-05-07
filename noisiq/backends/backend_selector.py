from typing import Dict, Union, Any

from ..ir import Circuit
from ..noise.kraus_channels import KrausChannel
from ..noise import PauliError
from .pauli_frame import StimTableauBackend
from .tsim_backend import TsimBackend
from .trajectory_backend import TrajectoryBackend

class BackendSelector:
    """
    Automatically selects the appropriate simulation backend based on the circuit's gates and noise model.
    """

    @staticmethod
    def select(
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]]] = None,
    ) -> Any:
        """
        Analyzes the circuit and noise model to determine the best backend.
        """
        has_non_pauli_noise = False
        
        if isinstance(noise_model, dict):
            for op_idx, noise in noise_model.items():
                if isinstance(noise, KrausChannel):
                    has_non_pauli_noise = True
                    break
        elif isinstance(noise_model, KrausChannel):
            has_non_pauli_noise = True

        has_non_clifford_gates = False
        for op in circuit.operations:
            name = op.gate.name.upper()
            if name in ['T', 'T_DAG', 'RX', 'RY', 'RZ', 'U3']:
                has_non_clifford_gates = True
                break

        if has_non_pauli_noise:
            return TrajectoryBackend()
        elif has_non_clifford_gates:
            return TsimBackend()
        else:
            return StimTableauBackend()
