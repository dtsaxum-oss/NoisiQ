"""
Hardware noise profiles for real quantum devices.

Pre-loaded profiles cover IBM Heron r2, IBM Eagle r3, IonQ Forte,
IonQ Aria, and Quantinuum H2.  Each profile stores the T1/T2 times,
gate error rates, and gate durations needed to build a NoisiQ noise
model, plus any published GHZ fidelity results for that device.

Users can register custom profiles with register() and retrieve any
profile by name with get().

Classes:
    GHZResult       : One published GHZ experiment (n_qubits, fidelity, source)
    GateTimes       : Typical gate durations in nanoseconds
    HardwareProfile : Full device spec with to_noise_model() builder

Functions:
    register        : Add a HardwareProfile to the global registry
    get             : Retrieve a profile by name
    list_profiles   : Return all registered profile names

Pre-loaded profile names:
    "ibm_heron_r2", "ibm_eagle_r3", "ionq_forte", "ionq_aria", "quantinuum_h2"

Example:
    import noisiq as nq

    profile = nq.noise.get_hardware("ibm_heron_r2")
    noise   = profile.to_noise_model(circuit, mode="t2")
    result  = nq.backends.TrajectoryBackend().run(circuit, noise_model=noise)

    for ref in profile.ghz_results:
        print(f"{ref.n_qubits}q GHZ — published F = {ref.fidelity}  ({ref.source})")

    # Register your own device
    nq.noise.register_hardware(nq.noise.HardwareProfile(
        name="my_lab_trap",
        vendor="UCLA",
        system="Trapped Ion Prototype",
        t1=1.0,
        t2=0.5,
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        spam_error=0.005,
        gate_times=nq.noise.GateTimes(single_qubit_ns=50.0, two_qubit_ns=500.0),
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..ir.circuit import Circuit


# ==============================================================================
# Data classes
# ==============================================================================


@dataclass(frozen=True)
class GHZResult:
    """
    One published GHZ state experiment result for a hardware device.

    Args:
        n_qubits:  Number of qubits in the GHZ state.
        fidelity:  Reported state fidelity in [0, 1], or None if not published.
        year:      Year of publication.
        source:    Citation (arXiv ID, paper title, or press release URL).
        notes:     Optional context (e.g. "QREM applied", "logical qubits").
    """

    n_qubits: int
    fidelity: Optional[float]
    year: int
    source: str
    notes: str = ""


@dataclass(frozen=True)
class GateTimes:
    """
    Typical physical gate durations for a device, in nanoseconds.

    Args:
        single_qubit_ns: Duration of a single-qubit gate (e.g. H, X, Rz).
        two_qubit_ns:    Duration of a two-qubit gate (e.g. CNOT, CZ, ECR).
    """

    single_qubit_ns: float
    two_qubit_ns: float


@dataclass
class HardwareProfile:
    """
    Noise specification for a real quantum hardware device.

    Stores the T1, T2, gate errors, and gate durations needed to build a
    NoisiQ noise model.  Call to_noise_model() to get a ready-to-use dict
    for TrajectoryBackend.  Published GHZ fidelities are stored in
    ghz_results for direct comparison against simulation output.

    Args:
        name:               Registry key, e.g. "ibm_heron_r2".
        vendor:             Company name, e.g. "IBM".
        system:             Device model, e.g. "Heron r2".
        t1:                 Median T1 relaxation time in seconds.
        t2:                 Median T2 coherence time in seconds.
        single_qubit_error: Single-qubit gate infidelity (dimensionless).
        two_qubit_error:    Two-qubit gate infidelity (dimensionless).
        spam_error:         State prep and measurement error.
        gate_times:         Physical gate durations (GateTimes).
        ghz_results:        Published GHZ experiments on this device.
        notes:              Free-text notes (architecture, caveats, etc.).

    Example:
        profile = get_hardware("ibm_heron_r2")
        noise   = profile.to_noise_model(circuit, mode="t2")
        result  = TrajectoryBackend().run(circuit, noise_model=noise, n_shots=500)
    """

    name: str
    vendor: str
    system: str
    t1: float
    t2: float
    single_qubit_error: float
    two_qubit_error: float
    spam_error: float
    gate_times: GateTimes
    ghz_results: list[GHZResult] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------
    # Noise model builder
    # ------------------------------------------------------------------

    def to_pauli_noise_model(self, circuit: "Circuit") -> dict:
        """Build a Pauli noise model for use with ManyShotRunner.

        Converts the hardware's gate error rates, T1 relaxation, and T2
        coherence time into a per-gate PauliError dict using the standard
        Pauli-channel approximation:

            1. Gate infidelity → depolarizing noise:
               p_dep = gate_error / 3

            2. T1 relaxation (amplitude damping Pauli twirl):
               γ = 1 − exp(−t_gate / T1)
               p_t1 = γ / 4

            3. T2 dephasing per gate duration → additional Z-error:
               p_z_t2 = (1 − exp(−2·t_gate / T2)) / 2

            Combined:
               p_x = p_dep + p_t1
               p_y = p_dep + p_t1
               p_z = p_dep + p_t1 + p_z_t2

        The T1 term makes platform comparisons physically accurate: IBM Eagle
        (T1 ≈ 100 µs) and IonQ Forte (T1 >> 1 s) produce measurably different
        depolarizing contributions even for equal gate error rates.

        This follows the industry-standard Pauli-twirl approximation used by
        Stim noise models.  It introduces ~5–10% error vs exact density-matrix
        simulation but enables fast large-scale multi-shot Clifford simulation.

        Args:
            circuit: The NoisiQ Circuit to build the noise dict for.

        Returns:
            Dict mapping operation index (int) → PauliError, ready to pass
            as noise_config to ManyShotRunner.run().
        """
        import numpy as np
        from .pauli_error import PauliError

        noise: dict = {}
        for op_idx, op in enumerate(circuit.operations):
            if op.gate.num_qubits == 1:
                p_gate = self.single_qubit_error
                t_gate = self.gate_times.single_qubit_ns * 1e-9
            else:
                p_gate = self.two_qubit_error
                t_gate = self.gate_times.two_qubit_ns * 1e-9

            p_dep = p_gate / 3.0

            # T1 amplitude-damping Pauli twirl: γ/4 added to each axis
            gamma = 1.0 - np.exp(-t_gate / self.t1)
            p_t1 = gamma / 4.0

            p_z_t2 = (1.0 - np.exp(-2.0 * t_gate / self.t2)) / 2.0

            p_x = p_dep + p_t1
            p_y = p_dep + p_t1
            p_z = p_dep + p_t1 + p_z_t2

            # Clamp to physical bounds (p_x + p_y + p_z <= 1)
            total = p_x + p_y + p_z
            if total > 1.0:
                scale = 0.99 / total
                p_x *= scale
                p_y *= scale
                p_z *= scale

            noise[op_idx] = PauliError(p_x=p_x, p_y=p_y, p_z=p_z)

        return noise

    def to_noise_model(
        self,
        circuit: "Circuit",
        mode: str = "t2",
    ) -> dict:
        """Build an op-index → KrausChannel dict for TrajectoryBackend.

        Gate times are chosen per operation from gate_times: single-qubit
        ops use single_qubit_ns, two-qubit ops use two_qubit_ns.

        Args:
            circuit: The NoisiQ Circuit to build the noise dict for.
            mode:    Which decoherence channel to apply per gate:
                       "t1"  → AmplitudeDamping (energy relaxation / T1)
                       "t2"  → Dephasing      (phase decoherence / T2, default)

        Returns:
            Dict mapping operation index (int) → KrausChannel instance,
            ready to pass directly as noise_model to TrajectoryBackend.run().

        Raises:
            ValueError: If mode is not "t1" or "t2".
        """
        from .amplitude_damping import AmplitudeDamping
        from .t2_dephasing import Dephasing

        if mode not in ("t1", "t2"):
            raise ValueError(f"mode must be 't1' or 't2', got {mode!r}")

        noise: dict = {}
        for op_idx, op in enumerate(circuit.operations):
            t_gate = (
                self.gate_times.single_qubit_ns * 1e-9
                if op.gate.num_qubits == 1
                else self.gate_times.two_qubit_ns * 1e-9
            )
            if mode == "t1":
                noise[op_idx] = AmplitudeDamping(T1=self.t1, t=t_gate)
            else:
                noise[op_idx] = Dephasing(T2=self.t2, t=t_gate)

        return noise

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return a summary dict of this profile's key parameters."""
        return {
            "name": self.name,
            "vendor": self.vendor,
            "system": self.system,
            "t1_us": round(self.t1 * 1e6, 3),
            "t2_us": round(self.t2 * 1e6, 3),
            "single_qubit_error": self.single_qubit_error,
            "two_qubit_error": self.two_qubit_error,
            "spam_error": self.spam_error,
            "gate_times_ns": {
                "1q": self.gate_times.single_qubit_ns,
                "2q": self.gate_times.two_qubit_ns,
            },
            "ghz_results": [
                {
                    "n_qubits": r.n_qubits,
                    "fidelity": r.fidelity,
                    "year": r.year,
                    "source": r.source,
                }
                for r in self.ghz_results
            ],
        }

    def __repr__(self) -> str:
        return (
            f"HardwareProfile(name={self.name!r}, vendor={self.vendor!r}, "
            f"t1={self.t1 * 1e6:.1f}µs, t2={self.t2 * 1e6:.1f}µs, "
            f"2q_error={self.two_qubit_error:.4f})"
        )


# ==============================================================================
# Registry
# ==============================================================================

_REGISTRY: dict[str, HardwareProfile] = {}


def register_hardware(profile: HardwareProfile) -> None:
    """Add a HardwareProfile to the global registry.

    If a profile with the same name already exists it is overwritten.

    Args:
        profile: HardwareProfile instance to register.
    """
    _REGISTRY[profile.name] = profile


def get_hardware(name: str) -> HardwareProfile:
    """Retrieve a registered HardwareProfile by name.

    Args:
        name: Registry key, e.g. "ibm_heron_r2".

    Returns:
        The matching HardwareProfile.

    Raises:
        KeyError: If no profile with that name is registered.
                  The error message lists available names.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"No hardware profile named {name!r}. "
            f"Available profiles: {available}"
        )
    return _REGISTRY[name]


def list_hardware() -> list[str]:
    """Return a sorted list of all registered profile names.

    Returns:
        List of profile name strings in alphabetical order.
    """
    return sorted(_REGISTRY)


# ==============================================================================
# Pre-loaded profiles
# Data sourced from ghz_hardware_summary.md (David_wip/) — May 2026
# ==============================================================================

register_hardware(HardwareProfile(
    name="ibm_heron_r2",
    vendor="IBM",
    system="Heron r2",
    t1=300e-6,
    t2=200e-6,
    single_qubit_error=3e-5,
    two_qubit_error=0.003,
    spam_error=0.015,
    gate_times=GateTimes(single_qubit_ns=50.0, two_qubit_ns=200.0),
    notes=(
        "156-qubit heavy-hexagonal lattice. Tunable couplers virtually eliminate "
        "ZZ crosstalk. TLS mitigation in hardware. 3-5x improvement over Eagle."
    ),
    ghz_results=[],
))

register_hardware(HardwareProfile(
    name="ibm_eagle_r3",
    vendor="IBM",
    system="Eagle r3",
    t1=262e-6,
    t2=177e-6,
    single_qubit_error=2.4e-4,
    two_qubit_error=0.0074,
    spam_error=0.0135,
    gate_times=GateTimes(single_qubit_ns=60.0, two_qubit_ns=533.0),
    notes="127-qubit heavy-hexagonal lattice. ECR two-qubit gate.",
    ghz_results=[
        GHZResult(
            n_qubits=127,
            fidelity=0.546,
            year=2023,
            source="arXiv:2101.08946",
            notes="Quantum Readout Error Mitigation (QREM) applied. 98.6% GME confidence.",
        ),
        GHZResult(
            n_qubits=27,
            fidelity=0.546,
            year=2021,
            source="arXiv:2101.08946",
            notes="Parity verification. Heavy-hex embedding.",
        ),
        GHZResult(
            n_qubits=18,
            fidelity=0.5165,
            year=2019,
            source="arXiv:1905.05720",
            notes="Multiple quantum coherences (MQC) verification.",
        ),
    ],
))

register_hardware(HardwareProfile(
    name="ionq_forte",
    vendor="IonQ",
    system="Forte",
    t1=1.0,
    t2=1.0,
    single_qubit_error=2e-4,
    two_qubit_error=0.004,
    spam_error=0.005,
    gate_times=GateTimes(single_qubit_ns=135_000.0, two_qubit_ns=600_000.0),
    notes=(
        "36-qubit trapped-ion chain (30 production). All-to-all connectivity. "
        "Exhaustive DRB on all 435 qubit pairs. #AQ 29."
    ),
    ghz_results=[
        GHZResult(
            n_qubits=10,
            fidelity=0.80,
            year=2024,
            source="arXiv:2601.05286",
            notes=(
                "All-to-all connectivity eliminates SWAP overhead. "
                "vs superconducting < 0.15 for same size."
            ),
        ),
    ],
))

register_hardware(HardwareProfile(
    name="ionq_aria",
    vendor="IonQ",
    system="Aria",
    t1=1.0,
    t2=1.0,
    single_qubit_error=6e-4,
    two_qubit_error=0.006,
    spam_error=0.005,
    gate_times=GateTimes(single_qubit_ns=135_000.0, two_qubit_ns=600_000.0),
    notes="25-qubit trapped ion (21 production config). All-to-all. #AQ 20.",
    ghz_results=[],
))

register_hardware(HardwareProfile(
    name="quantinuum_h2",
    vendor="Quantinuum",
    system="H2-1",
    t1=1.0,
    t2=1.0,
    single_qubit_error=3e-5,
    two_qubit_error=0.0015,
    spam_error=0.0015,
    gate_times=GateTimes(single_qubit_ns=10_000.0, two_qubit_ns=100_000.0),
    notes=(
        "56-qubit QCCD all-to-all. First production device with 99.9% two-qubit "
        "fidelity ('three 9s'). 4 parallel gate zones. Microsoft Level 2 Resilient."
    ),
    ghz_results=[
        GHZResult(
            n_qubits=50,
            fidelity=None,
            year=2024,
            source="Quantinuum press release, Dec 2024",
            notes="50 LOGICAL qubits (error-corrected). World record for logical GHZ.",
        ),
    ],
))
