import enum
import math
from dataclasses import dataclass
from typing import Optional


def binomial_standard_error(p: float, n: int) -> float:
    """Binomial standard error sqrt(p*(1-p)/n). Returns 0.0 for n=0."""
    if n == 0:
        return 0.0
    return math.sqrt(p * (1.0 - p) / n)


class MetricKind(enum.Enum):
    ZERO_ERROR_FRACTION = "zero_error_fraction"
    OUTPUT_SUCCESS_PROBABILITY = "output_success_probability"
    STABILIZER_STATE_FIDELITY = "stabilizer_state_fidelity"
    GHZ_STABILIZER_STATE_FIDELITY = "ghz_stabilizer_state_fidelity"
    # Fraction of shots where the noisy classical branch (which conditional
    # corrections fired) exactly matched the ideal branch.  Produced by
    # PairedSimResult.branch_success_rate_report().  This is distinct from
    # LOGICAL_SUCCESS_PROBABILITY, which is reserved for a proper QEC-decoder
    # success definition (Metrics-6, not yet implemented).
    BRANCH_SUCCESS_RATE = "branch_success_rate"
    # Reserved for Metrics-6 (QEC layer + decoder required).
    LOGICAL_SUCCESS_PROBABILITY = "logical_success_probability"
    LOGICAL_ERROR_RATE = "logical_error_rate"
    DENSITY_MATRIX_STATE_FIDELITY = "density_matrix_state_fidelity"
    PROCESS_FIDELITY_PAULI = "process_fidelity_pauli"
    AVERAGE_CIRCUIT_FIDELITY_PAULI = "average_circuit_fidelity_pauli"
    SINGLE_QUBIT_MARGINAL_PURITY = "single_qubit_marginal_purity"
    GLOBAL_PURITY = "global_purity"
    GHZ_SUBSPACE_POPULATION = "ghz_subspace_population"
    GHZ_BRANCH_COHERENCE_REAL = "ghz_branch_coherence_real"
    GHZ_BRANCH_COHERENCE_ABS = "ghz_branch_coherence_abs"


@dataclass(frozen=True)
class MetricReport:
    name: str
    kind: MetricKind
    value: float
    uncertainty: Optional[float]
    n_shots: Optional[int]
    backend: str
    noise_model: str
    method: str
    definition: str
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    target: Optional[str] = None

    def __post_init__(self) -> None:
        # Coerce assumptions/limitations to tuples so callers can pass lists
        # without breaking the frozen-dataclass immutability guarantee.
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "limitations", tuple(self.limitations))

        if not isinstance(self.kind, MetricKind):
            raise TypeError(f"kind must be a MetricKind, got {type(self.kind)}")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value}")
        if self.uncertainty is not None:
            if not math.isfinite(self.uncertainty) or self.uncertainty < 0:
                raise ValueError(
                    f"uncertainty must be finite and non-negative, got {self.uncertainty}"
                )
        if self.n_shots is not None:
            if not isinstance(self.n_shots, int) or self.n_shots <= 0:
                raise ValueError(
                    f"n_shots must be a positive integer, got {self.n_shots}"
                )
        for field_name in ("name", "backend", "noise_model", "method", "definition"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def __str__(self) -> str:
        lines = [
            f"{self.name}",
            f"  kind       : {self.kind.value}",
            f"  value      : {self.value:.6g}",
        ]
        if self.uncertainty is not None:
            lines.append(f"  uncertainty: ±{self.uncertainty:.6g}")
        if self.n_shots is not None:
            lines.append(f"  n_shots    : {self.n_shots}")
        lines += [
            f"  backend    : {self.backend}",
            f"  noise_model: {self.noise_model}",
            f"  method     : {self.method}",
            f"  definition : {self.definition}",
        ]
        if self.target is not None:
            lines.append(f"  target     : {self.target}")
        if self.assumptions:
            lines.append("  assumptions:")
            for a in self.assumptions:
                lines.append(f"    - {a}")
        if self.limitations:
            lines.append("  limitations:")
            for lim in self.limitations:
                lines.append(f"    - {lim}")
        return "\n".join(lines)
