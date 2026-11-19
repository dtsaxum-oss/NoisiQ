"""Hardware noise profiles for real devices.

A :class:`HardwareProfile` stores published device figures (T1, T2, gate
infidelities, readout error, gate durations) and turns them into a
:class:`~noisiq.noise.NoiseModel` with :meth:`HardwareProfile.noise_model`.

How the numbers become noise (see ``nq.explain("hardware_profile")``):

1. Every gate and idle period gets :class:`~noisiq.noise.ThermalRelaxation` for its
   duration (T1 and T2 combined correctly).
2. The reported gate infidelity ``r`` already *includes* that decoherence, so only the
   remainder ``r - r_thermal`` is added as depolarizing noise (optionally split into a
   coherent over-rotation via ``coherent_fraction``).
3. Readout error is a classical flip of the recorded bit.

The figures below are medians taken from the cited sources at the time of writing;
devices change quickly — please send a PR with newer numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .channels import CoherentRotation, ThermalRelaxation, depolarizing
from .model import Durations, NoiseModel

__all__ = [
    "HardwareProfile",
    "GHZResult",
    "register_hardware",
    "get_hardware",
    "list_hardware",
]


@dataclass(frozen=True)
class GHZResult:
    """A published GHZ-state experiment, for comparing simulations with hardware."""

    n_qubits: int
    fidelity: float | None
    year: int
    source: str
    notes: str = ""


@dataclass(frozen=True)
class HardwareProfile:
    """Published noise figures for one device.

    Args:
        name: Registry key, e.g. ``"ibm_heron_r3"``.
        vendor, system: Human-readable identification.
        t1, t2: Median relaxation / coherence times in seconds (``T2 <= 2 T1``).
        single_qubit_error: Average 1-qubit gate infidelity.
        two_qubit_error: Average 2-qubit gate infidelity.
        readout_error: Average measurement assignment error.
        durations: Gate/measurement durations.
        coherent_fraction: Fraction (0–1) of the non-thermal gate error modelled as a
            systematic over-rotation instead of depolarizing noise.
        notes, references: Provenance.
        ghz_results: Published GHZ fidelities for comparison plots.
    """

    name: str
    vendor: str
    system: str
    t1: float
    t2: float
    single_qubit_error: float
    two_qubit_error: float
    readout_error: float
    durations: Durations
    coherent_fraction: float = 0.0
    notes: str = ""
    references: tuple[str, ...] = ()
    ghz_results: tuple[GHZResult, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.t1 <= 0 or self.t2 <= 0 or self.t2 > 2 * self.t1:
            raise ValueError(f"{self.name}: need 0 < T2 <= 2·T1 (got T1={self.t1}, T2={self.t2}).")
        for f in ("single_qubit_error", "two_qubit_error", "readout_error", "coherent_fraction"):
            v = getattr(self, f)
            if not 0 <= v <= 1:
                raise ValueError(f"{self.name}: {f} must be in [0, 1], got {v}.")

    # ------------------------------------------------------------------
    def thermal_infidelity(self, duration: float, num_qubits: int = 1) -> float:
        """Average gate infidelity caused by T1/T2 alone during ``duration``."""
        ch = ThermalRelaxation(self.t1, self.t2, duration)
        f_pro_1q = 1.0 - ch.error_probability
        d = 2**num_qubits
        f_pro = f_pro_1q**num_qubits
        return (1.0 - f_pro) * d / (d + 1)

    def noise_model(
        self,
        representation: str = "exact",
        *,
        idle: bool = True,
        readout: bool = True,
        gate_error: bool = True,
        decoherence: bool = True,
        coherent_fraction: float | None = None,
    ) -> NoiseModel:
        """Build a :class:`NoiseModel` for this device.

        Args:
            representation: ``"exact"`` keeps non-Pauli channels (statevector backend);
                ``"pauli"`` twirls everything (fast stabilizer backend).
            idle: Apply T1/T2 to idle qubits.
            readout: Include readout error.
            gate_error: Include depolarizing/coherent gate error.
            decoherence: Include T1/T2 during gates (and idle, if ``idle``).
            coherent_fraction: Override the profile's coherent fraction.
        """
        if representation not in ("exact", "pauli"):
            raise ValueError("representation must be 'exact' or 'pauli'.")
        f_coh = self.coherent_fraction if coherent_fraction is None else coherent_fraction
        nm = NoiseModel(f"{self.vendor} {self.system}", self.durations)
        if decoherence:
            nm.add_thermal_relaxation(self.t1, self.t2, on_gates=True, on_idle=idle)
        if gate_error:
            for k, r, dur in (
                (1, self.single_qubit_error, self.durations.single_qubit),
                (2, self.two_qubit_error, self.durations.two_qubit),
            ):
                r_rest = max(0.0, r - (self.thermal_infidelity(dur, k) if decoherence else 0.0))
                if r_rest <= 0:
                    continue
                d = 2**k
                p_total = min(1.0, r_rest * (d + 1) / d)
                p_dep = p_total * (1 - f_coh)
                if p_dep > 0:
                    nm.add_gate_error(depolarizing(p_dep, k), num_qubits=k)
                if f_coh > 0:
                    angle = 2 * np.arcsin(np.sqrt(min(1.0, p_total * f_coh)))
                    nm.add_gate_error(CoherentRotation("Z" * k, angle), num_qubits=k)
        if readout and self.readout_error > 0:
            nm.add_readout_error(self.readout_error)
        return nm.twirled() if representation == "pauli" else nm

    def with_overrides(self, **changes) -> HardwareProfile:
        """Copy with some fields changed, e.g. ``profile.with_overrides(t1=50e-6)``."""
        return replace(self, **changes)

    def describe(self) -> dict:
        """Summary dictionary in human units."""
        return {
            "name": self.name,
            "device": f"{self.vendor} {self.system}",
            "T1 (us)": round(self.t1 * 1e6, 3),
            "T2 (us)": round(self.t2 * 1e6, 3),
            "1q error": self.single_qubit_error,
            "2q error": self.two_qubit_error,
            "readout error": self.readout_error,
            "1q gate (ns)": self.durations.single_qubit * 1e9,
            "2q gate (ns)": self.durations.two_qubit * 1e9,
            "measure (ns)": self.durations.measure * 1e9,
        }

    def __repr__(self) -> str:
        return (
            f"HardwareProfile({self.name!r}: T1={self.t1 * 1e6:.4g}µs, T2={self.t2 * 1e6:.4g}µs, "
            f"1q={self.single_qubit_error:.2g}, 2q={self.two_qubit_error:.2g}, ro={self.readout_error:.2g})"
        )


_REGISTRY: dict[str, HardwareProfile] = {}


def register_hardware(profile: HardwareProfile, *, overwrite: bool = False) -> None:
    """Add a profile to the registry (``overwrite=True`` to replace an existing one)."""
    if profile.name in _REGISTRY and not overwrite:
        raise ValueError(f"A profile named {profile.name!r} already exists (pass overwrite=True).")
    _REGISTRY[profile.name] = profile


def get_hardware(name: str) -> HardwareProfile:
    """Look up a registered profile by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"No hardware profile {name!r}. Available: {', '.join(list_hardware())}") from None


def list_hardware() -> list[str]:
    """Names of all registered profiles."""
    return sorted(_REGISTRY)


# =============================================================================
# Built-in profiles
# =============================================================================

register_hardware(
    HardwareProfile(
        name="ibm_heron_r3",
        vendor="IBM",
        system="Heron r3",
        t1=269e-6,
        t2=322e-6,
        single_qubit_error=1.6e-4,
        two_qubit_error=3.0e-3,
        readout_error=3.5e-3,
        durations=Durations(single_qubit=50e-9, two_qubit=200e-9, measure=1.2e-6, reset=1.2e-6),
        coherent_fraction=0.0,
        notes="156-qubit heavy-hex lattice with tunable couplers (ibm_boston medians).",
        references=("https://quantum.cloud.ibm.com/computers (ibm_boston calibration data)",),
    )
)

register_hardware(
    HardwareProfile(
        name="ibm_eagle_r3",
        vendor="IBM",
        system="Eagle r3",
        t1=263e-6,
        t2=147e-6,
        single_qubit_error=5.49e-4,
        two_qubit_error=1.16e-2,
        readout_error=1.35e-2,
        durations=Durations(single_qubit=50e-9, two_qubit=561e-9, measure=1.2e-6, reset=1.2e-6),
        notes="127-qubit heavy-hex lattice, ECR two-qubit gate (ibm_kyiv 2024 means).",
        references=("arXiv:2410.00916",),
        ghz_results=(
            GHZResult(27, 0.546, 2021, "arXiv:2101.08946", "Parity verification, heavy-hex embedding."),
            GHZResult(18, 0.5165, 2019, "arXiv:1905.05720", "Multiple-quantum-coherence verification."),
        ),
    )
)

register_hardware(
    HardwareProfile(
        name="ionq_forte",
        vendor="IonQ",
        system="Forte",
        t1=50.0,
        t2=1.0,
        single_qubit_error=2e-4,
        two_qubit_error=4e-3,
        readout_error=5e-3,
        durations=Durations(single_qubit=135e-6, two_qubit=600e-6, measure=300e-6, reset=50e-6),
        notes="36-qubit trapped-ion chain, all-to-all connectivity. T1 quoted as 10–100 s.",
        references=("https://www.ionq.com/quantum-systems/forte-enterprise",),
    )
)

register_hardware(
    HardwareProfile(
        name="ionq_aria",
        vendor="IonQ",
        system="Aria",
        t1=50.0,
        t2=1.0,
        single_qubit_error=6e-4,
        two_qubit_error=6e-3,
        readout_error=3.9e-3,
        durations=Durations(single_qubit=135e-6, two_qubit=600e-6, measure=300e-6, reset=50e-6),
        notes="25-qubit trapped ion, all-to-all.",
        references=("https://www.ionq.com/resources/ionq-aria-practical-performance",),
    )
)

register_hardware(
    HardwareProfile(
        name="quantinuum_h2",
        vendor="Quantinuum",
        system="H2-1",
        t1=100.0,
        t2=1.0,
        single_qubit_error=3e-5,
        two_qubit_error=1.5e-3,
        readout_error=2e-3,
        durations=Durations(single_qubit=5e-6, two_qubit=150e-6, measure=100e-6, reset=20e-6),
        notes="56-qubit QCCD trapped-ion processor, all-to-all.",
        references=("Quantinuum H2 product data sheet",),
    )
)
