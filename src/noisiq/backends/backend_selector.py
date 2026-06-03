from typing import Dict, Union

from noisiq.ir import Circuit, Operation
from noisiq.ir.classical import ConditionalOp, Measurement
from noisiq.noise import PauliError
from noisiq.noise.kraus_channels import KrausChannel, CombinedChannel
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.backends.base import Backend
from noisiq.backends.pauli_frame import StimTableauBackend
from noisiq.backends.pauli_frame import _is_pauli_compatible_channel
from noisiq.backends.tsim_backend import TsimBackend
from noisiq.backends.trajectory_backend import TrajectoryBackend

class BackendSelector:
    """
    Automatically selects the appropriate simulation backend based on the circuit's gates and noise model.
    """

    @staticmethod
    def select(
        circuit: Circuit,
        noise_model: Union[
            KrausChannel,
            PauliError,
            CorrelatedPauliError,
            CombinedChannel,
            Dict[int, Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel]],
            None,
        ] = None,
    ) -> Backend:
        """
        Analyzes the circuit and noise model to determine the best backend.
        """
        noise_channels = []
        if isinstance(noise_model, dict):
            noise_channels = list(noise_model.values())
        elif noise_model is not None:
            noise_channels = [noise_model]

        has_non_pauli_noise = False
        if noise_channels:
            has_non_pauli_noise = any(
                not _is_pauli_compatible_channel(noise)
                for noise in noise_channels
            )

        has_tsim_incompatible_pauli_noise = any(
            not isinstance(noise, PauliError)
            for noise in noise_channels
        )

        _CLIFFORD_GATES = frozenset({
            'H', 'X', 'Y', 'Z', 'S', 'S_DAG',
            'CNOT', 'CX', 'CZ', 'SWAP',
            'I', 'IDLE',
        })

        has_non_clifford_gates = False
        has_unsupported_tsim_gates = False
        has_measurement_ops = False

        for op in circuit.operations:
            if isinstance(op, Measurement):
                has_measurement_ops = True
                continue
            if isinstance(op, ConditionalOp):
                has_measurement_ops = True
                # Inspect the inner gate so a conditional T or Rz is not missed.
                inner = op.inner
                if not isinstance(inner, Operation):
                    continue
                name = inner.gate.name.upper()
                is_parametric = name.startswith('P(') or name.startswith('RZ(')
                if is_parametric or name not in _CLIFFORD_GATES:
                    has_non_clifford_gates = True
                tsim_key = name if not is_parametric else 'RZ'
                if tsim_key not in TsimBackend.GATE_MAP and not is_parametric:
                    has_unsupported_tsim_gates = True
                continue
            if not isinstance(op, Operation):
                # Unknown op type (e.g. test stub) — treat conservatively as MCM-like.
                has_measurement_ops = True
                continue
            # Plain Operation
            name = op.gate.name.upper()
            is_parametric = name.startswith('P(') or name.startswith('RZ(')
            if is_parametric or name not in _CLIFFORD_GATES:
                has_non_clifford_gates = True
            tsim_key = name if not is_parametric else 'RZ'
            if tsim_key not in TsimBackend.GATE_MAP and not is_parametric:
                has_unsupported_tsim_gates = True

        # Non-Pauli noise or non-Clifford + unsupported-by-Tsim gates → Trajectory.
        # MCM with non-Clifford gates also routes to Trajectory: StimTableauBackend
        # is Clifford-only and would raise NonCliffordError; Tsim does not support MCM.
        if has_non_pauli_noise or (has_non_clifford_gates and has_unsupported_tsim_gates):
            return TrajectoryBackend()
        if has_measurement_ops:
            if has_non_clifford_gates:
                # MCM + non-Clifford (e.g. T gate): Stim cannot handle it;
                # Tsim does not support MCM; Trajectory is the only safe option.
                return TrajectoryBackend()
            return StimTableauBackend()
        elif has_non_clifford_gates:
            # TsimBackend only handles plain PauliError noise.  Any other channel
            # (CorrelatedPauliError, CombinedChannel) requires TrajectoryBackend.
            if has_tsim_incompatible_pauli_noise:
                return TrajectoryBackend()
            return TsimBackend()
        else:
            return StimTableauBackend()
