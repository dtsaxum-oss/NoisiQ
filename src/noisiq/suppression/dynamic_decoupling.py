"""
Dynamical decoupling (DD) sequences for idle qubit noise suppression.

DD inserts periodic refocusing pulses into idle time slots on qubits to cancel
quasi-static Z errors (dephasing) that accumulate during idle periods. The pulses
form symmetric patterns that echo out low-frequency noise while the circuit runs.

Sequences
---------
hahn  : [X]                     — cancels Z dephasing only
CPMG  : [X, X]                  — Z dephasing, improved spacing
XY-4  : [X, Y, X, Y]            — universal (cancels X and Z errors)
XY-8  : [X, Y, X, Y, Y, X, Y, X] — universal, higher-order robustness

Example
-------
    from noisiq.suppression import apply_dd, list_sequences

    print(list_sequences())    # ['CPMG', 'XY-4', 'XY-8', 'hahn']

    dd_circuit = apply_dd(circuit, sequence='XY-4', idle_threshold=2)
    # dd_circuit has the same qubit count; original circuit is unchanged
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..ir import Circuit
from ..ir import gates as ir_gates


# ---------------------------------------------------------------------------
# Pulse sequences
# ---------------------------------------------------------------------------

_DD_SEQUENCES: dict[str, list] = {
    "hahn": [ir_gates.X],
    "CPMG": [ir_gates.X, ir_gates.X],
    "XY-4": [ir_gates.X, ir_gates.Y, ir_gates.X, ir_gates.Y],
    "XY-8": [
        ir_gates.X, ir_gates.Y, ir_gates.X, ir_gates.Y,
        ir_gates.Y, ir_gates.X, ir_gates.Y, ir_gates.X,
    ],
}


def list_sequences() -> List[str]:
    """Return available DD sequence names in alphabetical order."""
    return sorted(_DD_SEQUENCES)


def apply_dd(
    circuit: Circuit,
    sequence: str = "XY-4",
    idle_threshold: int = 1,
) -> Circuit:
    """Insert DD pulses into idle time slots in the circuit.

    For each qubit, idle windows between consecutive active gates are detected.
    Any window long enough to fit the chosen sequence receives evenly-spaced DD
    pulses; shorter windows are left untouched.  Original gates are never moved.

    Args:
        circuit:        Source circuit. Not modified.
        sequence:       DD sequence name — 'hahn', 'CPMG', 'XY-4', or 'XY-8'.
        idle_threshold: Minimum number of idle layers between two active gates
                        before DD pulses are inserted. The sequence still requires
                        at least len(sequence) idle layers, so the effective
                        threshold is max(idle_threshold, len(sequence)).

    Returns:
        New Circuit with DD pulses inserted. Gate count increases; the source
        circuit is unchanged.

    Raises:
        ValueError: If sequence is not a recognised DD sequence name.
    """
    if sequence not in _DD_SEQUENCES:
        raise ValueError(
            f"Unknown DD sequence {sequence!r}. "
            f"Available sequences: {sorted(_DD_SEQUENCES)}"
        )

    pulses = _DD_SEQUENCES[sequence]
    n_pulses = len(pulses)

    # Copy original circuit preserving explicit time placement
    new_circuit = Circuit(n_qubits=circuit.n_qubits, name=circuit.name)
    for op in sorted(circuit.operations, key=lambda o: (o.t, o.qubits)):
        new_circuit.add_gate(op.gate, op.qubits, t=op.t)

    if not circuit.operations:
        return new_circuit

    # Per-qubit sorted list of time steps where the qubit has a gate
    qubit_active: dict[int, list[int]] = {q: [] for q in range(circuit.n_qubits)}
    for op in circuit.operations:
        for q in op.qubits:
            qubit_active[q].append(op.t)

    for q in range(circuit.n_qubits):
        active_times = sorted(set(qubit_active[q]))
        if len(active_times) < 2:
            continue  # need at least two anchor points to bound an idle window

        for idx in range(len(active_times) - 1):
            t_prev = active_times[idx]
            t_next = active_times[idx + 1]
            window_len = t_next - t_prev - 1  # number of idle t-slots in window

            # Must satisfy both the user threshold and the sequence length
            effective_threshold = max(idle_threshold, n_pulses)
            if window_len < effective_threshold:
                continue

            # Evenly-spaced pulse positions, strictly between t_prev and t_next.
            # np.linspace(t_prev, t_next, n_pulses+2)[1:-1] gives n_pulses interior
            # points. With window_len >= n_pulses the step is >= 1, so rounded
            # positions are guaranteed to be unique integers in [t_prev+1, t_next-1].
            raw_times = np.linspace(t_prev, t_next, n_pulses + 2)[1:-1]
            pulse_times = np.round(raw_times).astype(int)

            for pulse_t, gate in zip(pulse_times, pulses):
                new_circuit.add_gate(gate, (q,), t=int(pulse_t))

    return new_circuit
