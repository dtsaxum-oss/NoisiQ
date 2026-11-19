"""Simulation backends and the :func:`simulate` entry point.

=================  ===============================  ==========================================
backend            handles                          use it for
=================  ===============================  ==========================================
``frame``          Clifford gates + Pauli noise     large circuits, many shots, heatmaps,
                   (any size)                       exact Pauli error propagation
``statevector``    any gate, any channel, ≤ 26 q    T/rotation gates, T1/T2, coherent errors,
                                                    quasi-static noise + DD
``density_matrix`` any gate, any channel, ≤ 10 q    exact reference values (no sampling)
``aer``            via Qiskit Aer (optional)        cross-checking against Qiskit
=================  ===============================  ==========================================

``backend="auto"`` picks ``frame`` whenever the circuit is Clifford and the noise is
Pauli, otherwise ``statevector``. :func:`select_backend` tells you which and why.
"""

from __future__ import annotations

from ..ir.circuit import Circuit
from ..noise.model import NoiseModel
from ..results.result import Result
from .aer import AerBackend
from .density import DensityMatrixBackend
from .frame import FrameBackend, IncompatibleCircuitError, frame_incompatibilities
from .program import Program, compile_program
from .statevector import MAX_QUBITS as STATEVECTOR_MAX_QUBITS
from .statevector import StatevectorBackend

__all__ = [
    "simulate",
    "select_backend",
    "FrameBackend",
    "StatevectorBackend",
    "DensityMatrixBackend",
    "AerBackend",
    "IncompatibleCircuitError",
    "compile_program",
    "Program",
    "BACKENDS",
]

BACKENDS = {
    "frame": FrameBackend,
    "stabilizer": FrameBackend,
    "statevector": StatevectorBackend,
    "trajectory": StatevectorBackend,
    "density_matrix": DensityMatrixBackend,
    "aer": AerBackend,
}


def select_backend(
    circuit: Circuit, noise: NoiseModel | None = None, *, program: Program | None = None
) -> tuple[str, str]:
    """Return ``(backend_name, reason)`` for ``backend="auto"``."""
    program = program or compile_program(circuit, noise)
    problems = frame_incompatibilities(program)
    if not problems:
        return "frame", "Clifford circuit with Pauli noise: exact, fast stabilizer simulation."
    if circuit.n_qubits <= STATEVECTOR_MAX_QUBITS:
        return "statevector", "Needs a universal simulator (" + "; ".join(problems) + ")."
    raise IncompatibleCircuitError(
        f"No backend can simulate this {circuit.n_qubits}-qubit circuit exactly: "
        + "; ".join(problems)
        + ". Options: Pauli-twirl the noise (noise.twirled()) if the circuit is Clifford, or reduce the size."
    )


def simulate(
    circuit: Circuit,
    noise: NoiseModel | None = None,
    shots: int = 1024,
    *,
    backend: str = "auto",
    seed: int | None = None,
    trace_shots: int = 0,
    **options,
) -> Result:
    """Simulate a noisy circuit.

    Args:
        circuit: Any :class:`~noisiq.Circuit`. If it has no measurements, all qubits are
            measured in Z at the end (classical bit k = qubit k).
        noise: A :class:`~noisiq.noise.NoiseModel` (``None`` = noiseless).
        shots: Number of shots.
        backend: ``"auto"``, ``"frame"``, ``"statevector"``, ``"density_matrix"`` or ``"aer"``.
        seed: Seed for reproducible results.
        trace_shots: Record a step-by-step :class:`~noisiq.results.PropagationTrace` for
            the first ``trace_shots`` shots (used by the animation functions).
        **options: Backend-specific options, e.g. ``approximate=True`` (frame: twirl
            non-Pauli noise) or ``density_matrix=True`` (statevector).

    Returns:
        A :class:`~noisiq.Result`.

    Example:
        >>> import noisiq as nq
        >>> c = nq.Circuit(2).h(0).cx(0, 1)
        >>> r = nq.simulate(c, nq.NoiseModel.uniform_depolarizing(0.01), shots=1000, seed=1)
        >>> sorted(r.counts)[:2]
        ['00', '01']
    """
    if not isinstance(circuit, Circuit):
        raise TypeError(f"simulate() expects a noisiq Circuit, got {type(circuit).__name__}.")
    program = compile_program(circuit, noise)
    name = backend.lower()
    reason = None
    if name == "auto":
        name, reason = select_backend(circuit, noise, program=program)
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(set(BACKENDS))} or 'auto'.")
    engine = BACKENDS[name]()
    kwargs = dict(options)
    if name in ("frame", "stabilizer", "statevector", "trajectory"):
        kwargs["trace_shots"] = trace_shots
    result = engine.run(circuit, noise, shots, seed, program=program, **kwargs)
    if reason:
        result.metadata["backend_reason"] = reason
    return result
