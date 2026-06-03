"""
Stabilizer-membership and process-fidelity metrics for Clifford + Pauli simulations.

All functions operate on `final_pauli_frames` from AggregateResult — a list of
n-qubit Pauli strings (one per shot) representing the net output-frame error Pauli.

Math:
    stabilizer_state_fidelity = P(frame in Stab(|ψ_ideal⟩))
    process_fidelity_pauli    = P(frame == "I...I")
    average_circuit_fidelity  = (d * F_process + 1) / (d + 1)  where d = 2^n
"""

from __future__ import annotations

from typing import List, Optional

from .metrics import MetricKind, MetricReport, binomial_standard_error


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _str_to_bits(s: str) -> tuple[int, int]:
    """Convert a Pauli string like 'XYZI' to (x_int, z_int) bitmasks.

    Qubit 0 → LSB. I=(0,0), X=(1,0), Y=(1,1), Z=(0,1).
    """
    x = z = 0
    for i, c in enumerate(s):
        if c in ('X', 'Y'):
            x |= 1 << i
        if c in ('Y', 'Z'):
            z |= 1 << i
    return x, z


def _generate_stabilizer_group(generators: List[str]) -> set[tuple[int, int]]:
    """Generate all elements of the Pauli stabilizer group from generator strings.

    Works in the quotient Pauli group (overall phases ignored). For k independent
    generators the returned set has exactly 2^k elements.
    """
    group: set[tuple[int, int]] = {(0, 0)}  # start with identity
    for gen in generators:
        gx, gz = _str_to_bits(gen)
        group = group | {(ex ^ gx, ez ^ gz) for ex, ez in group}
    return group


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def stabilizer_state_fidelity(
    final_frames: List[str],
    generators: List[str],
    *,
    backend: str = "StimTableauBackend",
    noise_model: str = "unknown",
    target: Optional[str] = None,
) -> MetricReport:
    """Estimate P(net error Pauli is in the target stabilizer group).

    For a Clifford + Pauli circuit this is a scalable proxy for state fidelity:
    if the output-frame error Pauli stabilizes the ideal final state the shot
    produces the correct quantum state (up to global phase).

    Args:
        final_frames: Per-shot output-frame Pauli strings from
                      AggregateResult.final_pauli_frames.
        generators:   Stabilizer generators of the target state as Pauli
                      strings, e.g. ["XXXX", "ZZII", "IZZI", "IIZZ"] for the
                      4-qubit GHZ state. All generators are assumed to have
                      phase +1.
        backend:      Backend name for metadata.
        noise_model:  Noise model name for metadata.
        target:       Human-readable name of the target state.

    Returns:
        MetricReport with kind=STABILIZER_STATE_FIDELITY.

    Raises:
        ValueError: If final_frames is empty.
    """
    if not final_frames:
        raise ValueError("final_frames must not be empty")

    group = _generate_stabilizer_group(generators)
    n = len(final_frames)
    successes = sum(1 for f in final_frames if _str_to_bits(f) in group)
    p = successes / n

    return MetricReport(
        name="Stabilizer state fidelity" + (f" ({target})" if target else ""),
        kind=MetricKind.STABILIZER_STATE_FIDELITY,
        value=p,
        uncertainty=binomial_standard_error(p, n),
        n_shots=n,
        backend=backend,
        noise_model=noise_model,
        method="stabilizer-membership Monte Carlo",
        definition=(
            "Fraction of shots where the output-frame net error Pauli is "
            "in the stabilizer group of the target state"
        ),
        assumptions=(
            "Clifford circuit",
            "Pauli or Pauli-twirled noise",
            "stabilizer generators have phase +1",
        ),
        limitations=(
            "does not capture coherent non-Pauli effects unless twirled into the noise model",
        ),
        target=target,
    )


def process_fidelity_pauli(
    final_frames: List[str],
    *,
    backend: str = "StimTableauBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Estimate the Pauli process fidelity: P(output-frame error is identity).

    For a Clifford + Pauli channel the process fidelity equals the probability
    that the net effective Pauli error is the identity.

    Args:
        final_frames: Per-shot output-frame Pauli strings.
        backend:      Backend name for metadata.
        noise_model:  Noise model name for metadata.

    Returns:
        MetricReport with kind=PROCESS_FIDELITY_PAULI.

    Raises:
        ValueError: If final_frames is empty.
    """
    if not final_frames:
        raise ValueError("final_frames must not be empty")

    n_qubits = len(final_frames[0])
    identity = 'I' * n_qubits
    n = len(final_frames)
    successes = sum(1 for f in final_frames if f == identity)
    p = successes / n

    return MetricReport(
        name="Process fidelity (Pauli channel)",
        kind=MetricKind.PROCESS_FIDELITY_PAULI,
        value=p,
        uncertainty=binomial_standard_error(p, n),
        n_shots=n,
        backend=backend,
        noise_model=noise_model,
        method="identity-frame fraction",
        definition=(
            "P(net output-frame error Pauli is identity) = p_I for a Pauli channel"
        ),
        assumptions=("Clifford circuit", "Pauli or Pauli-twirled noise"),
        limitations=(
            "strictly a Pauli-channel metric; coherent errors require process tomography",
        ),
    )


def average_circuit_fidelity_pauli(
    final_frames: List[str],
    n_qubits: int,
    *,
    backend: str = "StimTableauBackend",
    noise_model: str = "unknown",
) -> MetricReport:
    """Estimate average circuit fidelity from the Pauli process fidelity.

    F_avg = (d * F_process + 1) / (d + 1)  where d = 2^n_qubits.

    Uncertainty propagated as sigma_avg = d/(d+1) * sigma_process.

    Args:
        final_frames: Per-shot output-frame Pauli strings.
        n_qubits:     Number of qubits (determines d = 2^n_qubits).
        backend:      Backend name for metadata.
        noise_model:  Noise model name for metadata.

    Returns:
        MetricReport with kind=AVERAGE_CIRCUIT_FIDELITY_PAULI.
    """
    fp = process_fidelity_pauli(final_frames, backend=backend, noise_model=noise_model)
    d = 1 << n_qubits  # 2^n_qubits
    f_avg = (d * fp.value + 1) / (d + 1)
    se_avg = (d / (d + 1)) * (fp.uncertainty or 0.0)

    return MetricReport(
        name="Average circuit fidelity (Pauli channel)",
        kind=MetricKind.AVERAGE_CIRCUIT_FIDELITY_PAULI,
        value=f_avg,
        uncertainty=se_avg,
        n_shots=len(final_frames),
        backend=backend,
        noise_model=noise_model,
        method=f"F_avg = (d·F_process + 1)/(d+1), d=2^{n_qubits}={d}",
        definition=(
            "Average fidelity of the noisy circuit channel over all input states, "
            "derived from the Pauli process fidelity"
        ),
        assumptions=("Clifford circuit", "Pauli or Pauli-twirled noise"),
        limitations=("coherent errors are not captured",),
    )
