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

import warnings
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
    # Coherent and correlated error parameters
    # ------------------------------------------------------------------

    idle_zz_rate_hz: float = 0.0
    """ZZ crosstalk strength between nearest-neighbor qubits during idle,
    in Hz. Superconducting fixed-coupler devices can reach 22 kHz; tunable
    couplers suppress this to < 5 kHz. Ion-trap devices have effectively
    zero always-on coupling (idle_zz_rate_hz = 0). Used by to_noise_model()
    to add a per-gate ZZ coherent or Pauli-twirled error channel."""

    coherent_fraction: float = 0.0
    """Fraction of the two-qubit gate infidelity that is coherent
    (miscalibration / systematic over-rotation) rather than stochastic.
    0.0 = purely stochastic depolarizing; 1.0 = entirely coherent.
    Typical ranges: 0.3 for tunable-coupler superconducting, 0.4 for
    fixed-coupler, 0.4–0.5 for trapped-ion (laser phase noise dominant),
    0.5 for Quantinuum (documented quadratic memory error component)."""

    spectator_error_per_2q_gate: float = 0.0
    """Probability of a stray single-qubit Z error on each non-participating
    spectator qubit per two-qubit gate. Captures dispersive coupling spillover
    (superconducting) or beam-addressing crosstalk (ion-trap)."""

    mcmr_crosstalk: float = 0.0
    """Mid-circuit measurement readout crosstalk: probability of an unintended
    disturbance on a neighboring qubit during a measurement operation.
    Primarily relevant for Quantinuum-style QCCD devices (datasheet: 5e-6
    typical for H2). Zero for most superconducting and ion-trap platforms."""

    idle_coherent_epsilon: float = 0.0
    """Standard deviation (radians) of the per-shot quasi-static Z detuning
    accumulated during each IDLE slot due to low-frequency (1/f) noise that
    DD can refocus.  When non-zero, to_noise_model() appends a noise channel
    to each IDLE op alongside the Markovian T2 channel.  The channel type
    depends on ``representation``:

      - ``"coherent"``     → StochasticCoherentRotation('Z', ε): draws a
                              fresh angle from Normal(0, ε) each shot so the
                              trajectory average correctly loses purity and DD
                              demonstrates a genuine improvement.
      - ``"pauli_twirl"``  → PauliError(p_z=sin²(ε)): stochastic Z error
                              compatible with Pauli-frame backends.
      - ``None``           → CoherentRotation('Z', ε): deterministic unitary
                              (backward-compatible; contributes no purity loss).

    Typical value: 0.05–0.10 rad per ~50–60 ns IDLE slot
    (≈ 1–2 MHz quasi-static Z detuning on superconducting hardware)."""

    # ------------------------------------------------------------------
    # Noise model builder
    # ------------------------------------------------------------------

    def to_pauli_noise_model(self, circuit: "Circuit", include_spam: bool = False) -> dict:
        """Build a Pauli noise model for use with ManyShotRunner.

        Alias/wrapper for to_noise_model with representation='pauli_twirl'.
        """
        return self.to_noise_model(
            circuit, mode="t2", representation="pauli_twirl", include_spam=include_spam
        )

    def to_noise_model(
        self,
        circuit: "Circuit",
        mode: str = "t2",
        representation: "str | None" = None,
        include_spam: bool = False,
    ) -> dict:
        """Build a per-operation noise dict for TrajectoryBackend or Clifford backends.

        Args:
            circuit:        The NoisiQ Circuit to build the noise dict for.
            mode:           Which decoherence channel to apply per gate:
                              "t1"  → AmplitudeDamping (energy relaxation / T1)
                              "t2"  → Dephasing      (phase decoherence / T2, default)
                            This argument is unchanged from previous versions —
                            existing notebooks calling to_noise_model(circuit, mode='t1')
                            continue to work exactly as before.
            representation: How to express coherent and correlated errors.
                            When omitted (default None), the original behavior is
                            preserved: one KrausChannel per op, no gate-error or
                            ZZ/spectator channels added.
                              "pauli_twirl" → All error sources Pauli-twirled into
                                  PauliError / CorrelatedPauliError. Compatible with
                                  all backends. Fast and scalable, but discards
                                  coherent accumulation in deep circuits.
                              "coherent" → CoherentRotation channels preserved.
                                  Forces routing to TrajectoryBackend. Captures
                                  coherent accumulation exactly. Limited to 13 qubits.
            include_spam:   When True and representation="pauli_twirl", adds a
                            uniform depolarizing PauliError (p_x=p_y=p_z=spam_error/3)
                            at the first and last operation index for each qubit,
                            compounding with any existing gate noise at those positions.
                            Accepted but silently ignored for other representations.
                            Default False so all existing call sites remain valid.

        Returns:
            Dict mapping operation index (int) → single channel. When representation
            is None: one KrausChannel (AmplitudeDamping or Dephasing) per op.
            When representation is set: one channel per op, composed via
            CombinedChannel when multiple noise sources apply to a gate.

        Raises:
            ValueError: If mode is not "t1" or "t2", or if representation is not
                        None, "pauli_twirl", or "coherent".
        """
        from .amplitude_damping import AmplitudeDamping
        from .t2_dephasing import Dephasing

        if mode not in ("t1", "t2"):
            raise ValueError(f"mode must be 't1' or 't2', got {mode!r}")
        if representation not in (None, "pauli_twirl", "coherent"):
            raise ValueError(
                f"representation must be None, 'pauli_twirl', or 'coherent', "
                f"got {representation!r}"
            )

        # ----------------------------------------------------------------
        # Original behavior: representation not specified
        # Returns one KrausChannel per op — no gate error, ZZ, or spectator.
        # Fully backward compatible with all existing tests and notebooks.
        # ----------------------------------------------------------------
        if representation is None:
            if self.coherent_fraction > 0:
                warnings.warn(
                    f"HardwareProfile '{self.name}' has coherent_fraction="
                    f"{self.coherent_fraction} set, but this has no effect when "
                    f"representation is not specified. Pass "
                    f"representation='pauli_twirl' or representation='coherent' "
                    f"to include coherent error contributions.",
                    UserWarning,
                    stacklevel=2,
                )
            from ..ir import gates as ir_gates
            from .idle_fill import idle_kraus
            noise: dict = {}
            for op_idx, op in enumerate(circuit.operations):
                if op.gate is ir_gates.IDLE:
                    if not op.params or "duration_ns" not in op.params:
                        raise ValueError(
                            f"IDLE op at index {op_idx} is missing 'duration_ns' in "
                            f"op.params. Use Circuit.idle() or fill_idle_with_identities()."
                        )
                    noise[op_idx] = idle_kraus(
                        self.t1, self.t2, op.params["duration_ns"]
                    )
                    continue
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

        # ----------------------------------------------------------------
        # Extended behavior: representation explicitly set
        # Composes decoherence + gate error + ZZ crosstalk + spectator into
        # a single CombinedChannel per op (dict shape unchanged).
        # ----------------------------------------------------------------
        import numpy as np
        from .kraus_channels import CombinedChannel
        from .pauli_error import PauliError
        from .coherent_errors import CoherentRotation, StochasticCoherentRotation
        from .correlated_errors import CorrelatedPauliError
        from ..ir import gates as ir_gates
        from .idle_fill import idle_kraus, idle_pauli_twirl

        from ..ir.classical import Measurement as _Measurement, ConditionalOp as _ConditionalOp

        noise = {}
        for op_idx, op in enumerate(circuit.operations):
            # Measurements carry no gate noise — skip entirely
            if isinstance(op, _Measurement):
                continue
            # ConditionalOp wraps a real Operation; unwrap it for gate-noise computation
            if isinstance(op, _ConditionalOp):
                op = op.inner
            if op.gate is ir_gates.IDLE:
                if not op.params or "duration_ns" not in op.params:
                    raise ValueError(
                        f"IDLE op at index {op_idx} is missing 'duration_ns' in "
                        f"op.params. Use Circuit.idle() or fill_idle_with_identities()."
                    )
                if representation == "pauli_twirl":
                    base_channel = idle_pauli_twirl(
                        self.t1, self.t2, op.params["duration_ns"]
                    )
                else:
                    base_channel = idle_kraus(
                        self.t1, self.t2, op.params["duration_ns"]
                    )
                if self.idle_coherent_epsilon > 0:
                    if representation == "coherent":
                        # Quasi-static: sample ε ~ Normal(0, std_dev) per shot so
                        # different shots produce different states and purity correctly
                        # degrades. DD can then demonstrate a genuine improvement.
                        coherent_channel = StochasticCoherentRotation(
                            'Z', self.idle_coherent_epsilon
                        )
                    else:
                        # representation == "pauli_twirl": convert to a PauliError so
                        # Pauli-frame backends see the correct error probability.
                        coherent_channel = CoherentRotation(
                            'Z', self.idle_coherent_epsilon
                        ).to_pauli_error()
                    noise[op_idx] = CombinedChannel([base_channel, coherent_channel])
                else:
                    noise[op_idx] = base_channel
                continue

            is_2q = op.gate.num_qubits >= 2
            p_gate = self.two_qubit_error if is_2q else self.single_qubit_error
            t_gate = (
                self.gate_times.two_qubit_ns if is_2q
                else self.gate_times.single_qubit_ns
            ) * 1e-9
            coh_frac = self.coherent_fraction

            # ------------------------------------------------------------------
            # pauli_twirl path: compose ALL sources into one PauliError per op.
            # ZZ-axis coherent terms (gate ZZ, idle ZZ) are approximated as
            # independent per-qubit Z errors (first-order convolution).
            # This is the only representation compatible with Visualizer/ManyShotRunner.
            # ------------------------------------------------------------------
            if representation == "pauli_twirl":
                gamma    = 1.0 - np.exp(-t_gate / self.t1)
                p_t1     = gamma / 4.0
                p_z_t2   = (1.0 - np.exp(-2.0 * t_gate / self.t2)) / 2.0

                if mode == "t1":
                    acc_x, acc_y, acc_z = p_t1, p_t1, p_t1
                else:
                    acc_x, acc_y, acc_z = 0.0, 0.0, p_z_t2

                # Stochastic gate error (depolarizing)
                p_dep = p_gate * (1.0 - coh_frac) / 3.0
                acc_x += p_dep
                acc_y += p_dep
                acc_z += p_dep

                # Coherent gate fraction → sin²(ε) as per-qubit Z
                p_coherent = p_gate * coh_frac
                if p_coherent > 0:
                    epsilon  = np.sqrt(2.0 * p_coherent)
                    acc_z   += float(np.sin(epsilon) ** 2)

                # Idle ZZ crosstalk during 2q gates → approximate as per-qubit Z
                if is_2q and self.idle_zz_rate_hz > 0:
                    zz_angle = 2.0 * np.pi * self.idle_zz_rate_hz * t_gate
                    acc_z   += float(np.sin(zz_angle) ** 2)

                # Spectator Z during 2q gates
                if is_2q and self.spectator_error_per_2q_gate > 0:
                    acc_z += self.spectator_error_per_2q_gate

                total = acc_x + acc_y + acc_z
                if total > 1.0:
                    scale = 0.99 / total
                    acc_x *= scale
                    acc_y *= scale
                    acc_z *= scale

                noise[op_idx] = PauliError(p_x=acc_x, p_y=acc_y, p_z=acc_z)
                continue

            # ------------------------------------------------------------------
            # coherent path: preserve exact unitary + Kraus channels.
            # ZZ terms are kept as CoherentRotation; decoherence is native Kraus.
            # Returns Dict[int, CombinedChannel] (or single KrausChannel if only
            # one source).
            # ------------------------------------------------------------------
            channels = []

            if mode == "t1":
                channels.append(AmplitudeDamping(T1=self.t1, t=t_gate))
            else:
                channels.append(Dephasing(T2=self.t2, t=t_gate))

            p_stochastic = p_gate * (1.0 - coh_frac)
            p_coherent   = p_gate * coh_frac

            if p_stochastic > 0:
                p_dep = p_stochastic / 3.0
                channels.append(PauliError(p_x=p_dep, p_y=p_dep, p_z=p_dep))

            if p_coherent > 0:
                epsilon = np.sqrt(2.0 * p_coherent)
                axis    = 'ZZ' if is_2q else 'Z'
                channels.append(CoherentRotation(axis=axis, epsilon=epsilon))

            if is_2q and self.idle_zz_rate_hz > 0:
                zz_angle = 2.0 * np.pi * self.idle_zz_rate_hz * t_gate
                channels.append(CoherentRotation(axis='ZZ', epsilon=zz_angle))

            if is_2q and self.spectator_error_per_2q_gate > 0:
                channels.append(PauliError(
                    p_x=0.0, p_y=0.0, p_z=self.spectator_error_per_2q_gate
                ))

            if len(channels) == 1:
                noise[op_idx] = channels[0]
            elif len(channels) > 1:
                noise[op_idx] = CombinedChannel(channels)

        # ------------------------------------------------------------------
        # SPAM: add uniform depolarizing error at first/last op per qubit.
        # Only active for pauli_twirl; silently ignored for coherent.
        # ------------------------------------------------------------------
        if include_spam and representation == "pauli_twirl":
            from ..ir.circuit import Operation as _Operation
            spam = self.spam_error / 3.0

            first_op_idx: dict = {}
            last_op_idx: dict = {}
            for op_idx, op in enumerate(circuit.operations):
                if not isinstance(op, _Operation):
                    continue
                for q in op.qubits:
                    if q not in first_op_idx:
                        first_op_idx[q] = op_idx
                    last_op_idx[q] = op_idx

            spam_indices = set(first_op_idx.values()) | set(last_op_idx.values())
            for idx in spam_indices:
                existing = noise.get(idx)
                if isinstance(existing, PauliError):
                    new_px = existing.p_x + spam
                    new_py = existing.p_y + spam
                    new_pz = existing.p_z + spam
                    total = new_px + new_py + new_pz
                    if total > 1.0:
                        scale = 0.99 / total
                        new_px *= scale
                        new_py *= scale
                        new_pz *= scale
                    noise[idx] = PauliError(p_x=new_px, p_y=new_py, p_z=new_pz)
                elif isinstance(existing, CombinedChannel):
                    noise[idx] = CombinedChannel(
                        list(existing.channels) + [PauliError(p_x=spam, p_y=spam, p_z=spam)]
                    )
                else:
                    noise[idx] = PauliError(p_x=spam, p_y=spam, p_z=spam)

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
    idle_zz_rate_hz=5_000.0,          # < 5 kHz; tunable couplers suppress ZZ
    coherent_fraction=0.3,             # tunable-coupler default; lower coherent fraction
    spectator_error_per_2q_gate=1e-4,  # small spectator spillover; EPLG ≈ isolated 2q
    mcmr_crosstalk=0.0,
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
    idle_zz_rate_hz=22_000.0,          # 22 kHz NN ZZ; arXiv:2512.18148, arXiv:2108.04530
    coherent_fraction=0.4,             # fixed-coupler; 30-50% coherent fraction literature
    spectator_error_per_2q_gate=5e-4,  # EPLG ~1.5-2x isolated 2q error
    mcmr_crosstalk=0.0,
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
    idle_zz_rate_hz=0.0,               # no always-on coupling in ion traps
    coherent_fraction=0.4,             # laser phase noise + motional dephasing
    spectator_error_per_2q_gate=1e-3,  # beam addressing crosstalk; arXiv:2206.02703
    mcmr_crosstalk=0.0,
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
    idle_zz_rate_hz=0.0,
    coherent_fraction=0.4,
    spectator_error_per_2q_gate=1.5e-3,  # scaled with 2q error vs Forte
    mcmr_crosstalk=0.0,
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
    idle_zz_rate_hz=0.0,               # ions physically separated between gate zones
    coherent_fraction=0.5,             # documented quadratic coherent memory error component
    spectator_error_per_2q_gate=1e-5,  # 4 isolated gate zones; negligible cross-zone
    mcmr_crosstalk=5e-6,               # directly from H2 datasheet v4.00 (Sep 2025)
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
