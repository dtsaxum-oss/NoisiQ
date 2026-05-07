from typing import Dict, Union, Any

from noisiq.ir import Circuit
from noisiq.noise import PauliError
from noisiq.noise.kraus_channels import KrausChannel
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.backends.tsim_backend import TsimBackend
from noisiq.backends.trajectory_backend import TrajectoryBackend

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
        has_unsupported_tsim_gates = False
        
        for op in circuit.operations:
            name = op.gate.name.upper()
            if name in ['T', 'T_DAG', 'RX', 'RY', 'RZ', 'U3'] or name not in ['H', 'X', 'Y', 'Z', 'S', 'S_DAG', 'CNOT', 'CX', 'CZ', 'I']:
                has_non_clifford_gates = True
            if name not in TsimBackend.GATE_MAP:
                has_unsupported_tsim_gates = True

        if has_non_pauli_noise or (has_non_clifford_gates and has_unsupported_tsim_gates):
            return TrajectoryBackend()
        elif has_non_clifford_gates:
            return TsimBackend()
        else:
            return StimTableauBackend()
