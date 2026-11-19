import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ..ir import Circuit
from ..ir.circuit import Operation
from ..ir.classical import Measurement as _Measurement, ConditionalOp as _ConditionalOp
from ..backends.pauli_frame import StimTableauResult

if TYPE_CHECKING:
    from ..backends.many_shot_runner import AggregateResult

class PauliFrame:
    """
    Tracks multi-qubit Pauli errors through Clifford gates using symplectic arrays.
    Ignores global phases and signs, focusing only on the Pauli operators.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.x = np.zeros(num_qubits, dtype=bool)
        self.z = np.zeros(num_qubits, dtype=bool)

    def copy(self) -> 'PauliFrame':
        new_frame = PauliFrame(self.num_qubits)
        new_frame.x = self.x.copy()
        new_frame.z = self.z.copy()
        return new_frame

    def inject_error(self, qubit: int, pauli: str):
        """Inject a Pauli error at a specific qubit."""
        if pauli == 'X':
            self.x[qubit] ^= True
        elif pauli == 'Z':
            self.z[qubit] ^= True
        elif pauli == 'Y':
            self.x[qubit] ^= True
            self.z[qubit] ^= True

    def get_pauli_string(self) -> str:
        """Returns the current Pauli string."""
        chars = []
        for i in range(self.num_qubits):
            if self.x[i] and self.z[i]:
                chars.append('Y')
            elif self.x[i]:
                chars.append('X')
            elif self.z[i]:
                chars.append('Z')
            else:
                chars.append('I')
        return "".join(chars)

    def apply_h(self, target: int):
        # H X H = Z, H Z H = X
        x_val = self.x[target]
        z_val = self.z[target]
        self.x[target] = z_val
        self.z[target] = x_val

    def apply_s(self, target: int):
        # S X S^dag = Y = XZ, S Z S^dag = Z
        # If X is present, it becomes XZ, meaning Z flips.
        if self.x[target]:
            self.z[target] ^= True

    def apply_cnot(self, control: int, target: int):
        # CNOT (X_c) CNOT = X_c X_t -> control X spreads to target X
        # CNOT (Z_t) CNOT = Z_c Z_t -> target Z spreads to control Z
        if self.x[control]:
            self.x[target] ^= True
        if self.z[target]:
            self.z[control] ^= True

    def apply_cz(self, control: int, target: int):
        # CZ (X_c) CZ = X_c Z_t -> control X spreads to target Z
        # CZ (X_t) CZ = Z_c X_t -> target X spreads to control Z
        if self.x[control]:
            self.z[target] ^= True
        if self.x[target]:
            self.z[control] ^= True

    def apply_swap(self, q0: int, q1: int):
        # SWAP exchanges the full Pauli state of two qubits
        self.x[q0], self.x[q1] = bool(self.x[q1]), bool(self.x[q0])
        self.z[q0], self.z[q1] = bool(self.z[q1]), bool(self.z[q0])

    def apply_s_dag(self, target: int):
        # S† maps X→-Y (up to phase: same x/z flip as S), Z→Z.
        # Pauli frame tracking ignores phases, so S† behaves identically to S.
        if self.x[target]:
            self.z[target] ^= True

    def apply_x(self, target: int):
        # Commutes with X, anti-commutes with Z/Y (adds phase, but we ignore phase)
        pass

    def apply_y(self, target: int):
        pass

    def apply_z(self, target: int):
        pass

    def apply_measurement(self, qubit: int) -> bool:
        """Propagate the Pauli frame through an ideal Z-basis measurement.

        Returns True when the frame on *qubit* has an X or Y component —
        i.e. when the error would flip the measurement outcome.  After
        the call the qubit frame is cleared (both x and z set to False)
        because the qubit has been projected; the same state is correct
        for a freshly reset ancilla if the circuit uses measure-and-reset.
        """
        flipped = bool(self.x[qubit])  # X or Y anti-commutes with Z meas.
        self.x[qubit] = False
        self.z[qubit] = False
        return flipped

    def apply_gate(self, gate_name: str, qubits: List[int]):
        """
        Propagate the Pauli frame through an ideal gate.
        """
        gate = gate_name.upper()
        if gate == 'H':
            self.apply_h(qubits[0])
        elif gate == 'S':
            self.apply_s(qubits[0])
        elif gate == 'S_DAG':
            self.apply_s_dag(qubits[0])
        elif gate == 'CX' or gate == 'CNOT':
            self.apply_cnot(qubits[0], qubits[1])
        elif gate == 'CZ':
            self.apply_cz(qubits[0], qubits[1])
        elif gate == 'SWAP':
            self.apply_swap(qubits[0], qubits[1])
        elif gate in ['X', 'Y', 'Z', 'I', 'IDLE']:
            pass
        elif gate == 'T':
            raise NotImplementedError("T gate is not a Clifford gate and cannot be tracked efficiently by PauliFrame.")
        else:
            raise NotImplementedError(f"Gate {gate} not supported in PauliFrame tracker.")

def build_modal_trajectory_frames(
    circuit: Circuit,
    result: "AggregateResult",
) -> Dict[int, PauliFrame]:
    """
    Build a Dict[t, PauliFrame] for many-shot modal-error animation.

    Strategy — single dominant injection
    ─────────────────────────────────────
    Across all layers, find the ONE layer whose noise channel injected the
    most errors (highest total count in counts_by_pauli).  A single Pauli
    error is injected at that layer only; all other layers just propagate
    the accumulated frame forward through their gates with no new injection.

    This mirrors how a typical low-noise shot behaves: only one or two
    gates actually fire an error.  Injecting at every layer (the previous
    approach) caused two visual artefacts:

      1. **CNOT cancellation** — if the accumulated X on ctrl and the CNOT
         layer's own modal X injection both land on the same qubit, the XOR
         cancels them (X⊕X = I) and the error appears to vanish at the CNOT.

      2. **SWAP X → Y** — X propagated from a prior layer combines with a Z
         injected at the SWAP layer (X⊕Z = Y), so the animation shows Y
         coming out of a SWAP even though X went in.

    Single-injection semantics fixes both: the dominant error appears at its
    source layer and then propagates cleanly through every subsequent gate
    with no further interference.

    Frames before the injection layer are identity (clean — no error has
    occurred yet).  Frames from the injection layer onwards show the error
    as it is conjugated by each gate.

    Requires result.counts_by_pauli to be set; raises RuntimeError otherwise.
    """
    from ..backends.many_shot_runner import _IDX_PAULI

    if result.counts_by_pauli is None:
        raise RuntimeError(
            "build_modal_trajectory_frames requires counts_by_pauli. "
            "Re-run ManyShotRunner with the current version."
        )

    from ..ir.classical import Measurement as _ModalMeasurement

    n_qubits = circuit.n_qubits
    ops = circuit.operations

    # Map time-layer → list of op indices at that layer (Operation only —
    # used for counts_by_pauli dominant-error lookup).
    layer_to_op_indices: Dict[int, List[int]] = {}
    # Map time-layer → list of Measurement ops (for frame collapse).
    meas_at_layer: Dict[int, List] = {}
    # All unique time-steps across every CircuitOp type.
    all_layer_times: set = set()

    for op_idx, op in enumerate(ops):
        t = op.t
        all_layer_times.add(t)
        if isinstance(op, Operation):
            layer_to_op_indices.setdefault(t, []).append(op_idx)
        elif isinstance(op, _ModalMeasurement):
            meas_at_layer.setdefault(t, []).append(op)

    # Sorted (op_idx, op) pairs for gate application (Operations only).
    sorted_ops = sorted(
        [(op_idx, op) for op_idx, op in enumerate(ops) if isinstance(op, Operation)],
        key=lambda kv: (kv[1].t, kv[0]),
    )

    unique_layers = sorted(all_layer_times)
    modal_frames: Dict[int, PauliFrame] = {}

    # ── Pass 1: find the single most dominant injection point ────────────────
    # Scan every layer and find the (qubit, Pauli) pair with the highest total
    # count across all shots.  Only THIS layer will receive an injection;
    # all other layers just propagate gates without injecting new errors.
    dominant_layer_t: Optional[int] = None
    dominant_qubit: Optional[int] = None
    dominant_pauli: Optional[str] = None
    dominant_count: int = 0

    for layer_t, op_indices in layer_to_op_indices.items():
        layer_counts = result.counts_by_pauli[:, op_indices, :]  # (n_qubits, k, 3)
        summed = layer_counts.sum(axis=1)                        # (n_qubits, 3)
        max_c = int(summed.max())
        if max_c > dominant_count:
            dominant_count = max_c
            flat_idx = int(summed.argmax())
            dominant_layer_t = layer_t
            dominant_qubit = flat_idx // 3
            dominant_pauli = _IDX_PAULI[flat_idx % 3]

    # ── Pass 2: build cumulative frame, injecting only at dominant_layer_t ───
    # One shared frame persists across all layers so that the injected error
    # propagates forward through every gate it encounters.
    frame = PauliFrame(n_qubits)

    for layer_t in unique_layers:
        # Step 1: apply this layer's gates (conjugates the accumulated error).
        for _, op in sorted_ops:
            if op.t != layer_t:
                continue
            try:
                frame.apply_gate(op.gate.name, list(op.qubits))
            except NotImplementedError:
                pass  # skip non-Clifford gates; frame unchanged

        # Step 2: collapse frame at Measurement layers.
        for meas in meas_at_layer.get(layer_t, []):
            frame.apply_measurement(meas.qubit)

        # Step 3: inject only at the single dominant gate layer.
        if layer_t == dominant_layer_t and dominant_count > 0:
            frame.inject_error(dominant_qubit, dominant_pauli)

        # Snapshot: error state at this moment.
        modal_frames[layer_t] = frame.copy()

    return modal_frames


def compute_error_trajectories(circuit: Circuit, result: StimTableauResult) -> Dict[int, PauliFrame]:
    """
    Computes the accumulated PauliFrame after each time step (layer).

    Returns a dict mapping time-step t -> PauliFrame representing the error
    state AFTER all operations at that layer have been applied and injected
    errors accumulated.

    Steps are processed in time-step order (not circuit.operations insertion
    order). This matters whenever the operations list is not sorted by t —
    e.g. when a later-added gate lands at an earlier time slot because its
    qubit was idle. Processing out of time order would bake future-gate
    propagation into the displayed frame for earlier time steps.
    """
    frame = PauliFrame(circuit.n_qubits)
    layer_frames: Dict[int, PauliFrame] = {}

    # Sort by (t, original op index) so same-layer gates stay in insertion
    # order relative to each other while ensuring cross-layer order is correct.
    sorted_steps = sorted(result.steps, key=lambda s: (s.operation.t, s.time_step))

    for step in sorted_steps:
        op = step.operation
        if isinstance(op, _Measurement):
            # Z-basis measurement collapses the frame: X/Y on the measured
            # qubit would flip the outcome, after which the qubit is projected
            # (and reset if applicable).  Snapshot the frame so the animation
            # can show the measurement layer.
            frame.apply_measurement(op.qubit)
            layer_frames[op.t] = frame.copy()
            continue
        # ConditionalOp wraps a real Operation; unwrap it to access .gate.
        if isinstance(op, _ConditionalOp):
            op = op.inner
        frame.apply_gate(op.gate.name, op.qubits)
        for error in step.errors:
            frame.inject_error(error.qubit, error.pauli)
        # Overwrite on each op at this t; the final write captures the full
        # layer state after all gates at that time step have been applied.
        layer_frames[step.operation.t] = frame.copy()

    return layer_frames

