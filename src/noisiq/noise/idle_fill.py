"""
Idle-slot filler — inserts IDLE operations into layers where a qubit has
no active gate, so the decoherence model can tick on idle qubits.

The filler is noise-agnostic: it builds structure only. The actual noise
channel for each IDLE comes from HardwareProfile.to_pauli_noise_model /
to_noise_model, which branch on `op.gate is IDLE` to apply only T1/T2
decoherence — never gate error.

Two helpers for attaching idle noise without a HardwareProfile:
    idle_kraus(t1, t2, duration_ns)        -> CombinedChannel
    idle_pauli_twirl(t1, t2, duration_ns)  -> PauliError
"""

from __future__ import annotations

import numpy as np

from ..ir import Circuit
from ..ir import gates as ir_gates
from .amplitude_damping import AmplitudeDamping
from .t2_dephasing import Dephasing
from .kraus_channels import CombinedChannel
from .pauli_error import PauliError


def fill_idle_with_identities(
    circuit: Circuit,
    gate_times,
    *,
    fill_leading: bool = True,
    fill_trailing: bool = False,
) -> Circuit:
    """Return a new Circuit with IDLE ops in every empty (qubit, layer) slot.

    Args:
        circuit:       Source circuit (unchanged).
        gate_times:    GateTimes spec (single_qubit_ns, two_qubit_ns).
        fill_leading:  If True (default), fill layers before each qubit's
                       first active gate, starting from t=0.
        fill_trailing: If True, fill layers after each qubit's last active
                       gate up to the global max layer. Off by default.

    Returns:
        A new Circuit. Original gate set unchanged; IDLE ops added.
    """
    new_circuit = Circuit(n_qubits=circuit.n_qubits, name=circuit.name)
    for op in circuit.operations:
        new_circuit.add_gate(op.gate, op.qubits, t=op.t,
                             params=op.params, meta=op.meta)

    if not circuit.operations:
        return new_circuit

    # Per-layer duration = duration of the longest gate at that layer.
    layer_duration_ns: dict[int, float] = {}
    for op in circuit.operations:
        d = (gate_times.single_qubit_ns
             if op.gate.num_qubits == 1
             else gate_times.two_qubit_ns)
        layer_duration_ns[op.t] = max(layer_duration_ns.get(op.t, 0.0), d)

    # Per-qubit set of layers where that qubit is active.
    qubit_busy_layers: dict[int, set[int]] = {
        q: set() for q in range(circuit.n_qubits)
    }
    for op in circuit.operations:
        for q in op.qubits:
            qubit_busy_layers[q].add(op.t)

    global_max_t = max(layer_duration_ns.keys())

    for q in range(circuit.n_qubits):
        busy = qubit_busy_layers[q]
        if not busy:
            continue
        first_active = min(busy)
        last_active = max(busy)

        scan_start = 0 if fill_leading else first_active
        scan_end = global_max_t

        for t in range(scan_start, scan_end + 1):
            if t in busy:
                continue
            dur = layer_duration_ns.get(t, gate_times.single_qubit_ns)
            new_circuit.add_gate(
                ir_gates.IDLE, (q,), t=t,
                params={"duration_ns": dur},
            )

    return new_circuit


# ---------------------------------------------------------------------------
# Standalone channel builders (used by HardwareProfile and directly by users)
# ---------------------------------------------------------------------------

def idle_kraus(t1: float, t2: float, duration_ns: float) -> CombinedChannel:
    """Return CombinedChannel(AmplitudeDamping + Dephasing) for an IDLE op.

    Used by HardwareProfile.to_noise_model() in the Kraus path.
    """
    t_s = duration_ns * 1e-9
    return CombinedChannel([
        AmplitudeDamping(T1=t1, t=t_s),
        Dephasing(T2=t2, t=t_s),
    ])


def idle_pauli_twirl(t1: float, t2: float, duration_ns: float) -> PauliError:
    """Pauli-twirl approximation of T1+T2 decoherence for an IDLE op.

        p_x = p_y = γ/4              (T1 amplitude-damping twirl)
        p_z       = γ/4 + λ/2        (T1 + T2 Z-bias)

    where γ = 1 - exp(-t/T1) and λ = 1 - exp(-2t/T2).
    """
    t_s = duration_ns * 1e-9
    gamma = 1.0 - np.exp(-t_s / t1)
    p_t1 = gamma / 4.0
    p_z_t2 = (1.0 - np.exp(-2.0 * t_s / t2)) / 2.0

    p_x = p_t1
    p_y = p_t1
    p_z = p_t1 + p_z_t2

    total = p_x + p_y + p_z
    if total > 1.0:
        scale = 0.99 / total
        p_x *= scale
        p_y *= scale
        p_z *= scale

    return PauliError(p_x=p_x, p_y=p_y, p_z=p_z)
