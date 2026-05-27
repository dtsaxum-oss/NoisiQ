from typing import Dict, Union, Any

from noisiq.ir import Circuit
from noisiq.noise import PauliError
from noisiq.noise.kraus_channels import KrausChannel, CombinedChannel
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.backends.base import Backend
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.backends.qiskit_backend import QiskitAerBackend
from noisiq.backends.trajectory_backend import TrajectoryBackend

class BackendSelector:
    """
    Automatically selects the appropriate simulation backend based on the circuit's gates and noise model.
    """

    @staticmethod
    def select(
        circuit: Circuit,
        noise_model: Union[KrausChannel, PauliError, Dict[int, Union[KrausChannel, PauliError]]] = None,
    ) -> Backend:
        """
        Analyzes the circuit and noise model to determine the best backend.
        """
        _trajectory_types = (KrausChannel, CombinedChannel, CorrelatedPauliError)
        has_non_pauli_noise = False

        if isinstance(noise_model, dict):
            for noise in noise_model.values():
                if isinstance(noise, _trajectory_types):
                    has_non_pauli_noise = True
                    break
        elif isinstance(noise_model, _trajectory_types):
            has_non_pauli_noise = True

        _CLIFFORD_GATES = frozenset({
            'H', 'X', 'Y', 'Z', 'S', 'S_DAG',
            'CNOT', 'CX', 'CZ', 'SWAP',
            'I', 'IDLE',
        })

        has_non_clifford_gates = False

        for op in circuit.operations:
            name = op.gate.name.upper()
            # Parameterized phase gates are non-Clifford for general angles
            is_parametric = name.startswith('P(') or name.startswith('RZ(')
            if is_parametric or name not in _CLIFFORD_GATES:
                has_non_clifford_gates = True

        if has_non_pauli_noise or has_non_clifford_gates:
            try:
                import qiskit
                return QiskitAerBackend()
            except ImportError:
                if circuit.n_qubits > 13:
                    raise ImportError("Qiskit is required for non-Clifford/non-Pauli circuits with >13 qubits. Run `pip install qiskit qiskit-aer`.")
                return TrajectoryBackend()
        else:
            return StimTableauBackend()
