"""The :class:`Result` returned by every backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

from ..ir.circuit import Circuit
from ..ir.instructions import Operation
from ..metrics import binomial_stderr, hellinger_fidelity, purity, total_variation_distance
from .trace import PropagationTrace

__all__ = ["Result", "Estimate"]

PAULI_COLUMNS = ("X", "Y", "Z", "other")


@dataclass(frozen=True)
class Estimate:
    """A value with its (1σ) standard error."""

    value: float
    stderr: float = 0.0

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return f"{self.value:.6g} ± {self.stderr:.2g}"


@dataclass
class Result:
    """Outcome of :func:`noisiq.simulate`.

    Not every field is available from every backend (unavailable fields are ``None``):

    ============================  =======  ===========  =============
    field                         frame    statevector  density_matrix
    ============================  =======  ===========  =============
    counts / memory               yes      yes          yes (sampled)
    probabilities (exact)         —        —            yes
    fault statistics              yes      yes          —
    measurement_errors            yes      —            —
    final_errors (Pauli frames)   yes      —            —
    fidelity                      —        yes          yes
    qubit_states / density_matrix —        yes          yes
    trace                         pauli    deviation    —
    ============================  =======  ===========  =============

    Bitstring convention: the leftmost character of a ``counts`` key is classical
    bit 0. When a circuit has no measurements, every qubit is measured at the end
    and classical bit ``k`` holds qubit ``k`` (``implicit_measurement=True``).
    """

    circuit: Circuit
    backend: str
    shots: int
    seed: int | None
    counts: dict[str, int]
    memory: np.ndarray | None = None
    implicit_measurement: bool = False
    noise_name: str = "noiseless"
    instruction_faults: np.ndarray | None = None  # (n_instructions, n_qubits, 4)
    idle_faults: np.ndarray | None = None  # (n_layers, n_qubits, 4)
    readout_faults: np.ndarray | None = None  # (n_qubits,) — all measurements
    fault_free: np.ndarray | None = None  # (shots,) bool
    measurement_errors: np.ndarray | None = None  # (shots, n_clbits) bool
    final_errors: tuple[np.ndarray, np.ndarray] | None = None  # x, z: (n_qubits, shots)
    probabilities_exact: dict[str, float] | None = None
    fidelity: Estimate | None = None
    qubit_states: np.ndarray | None = None  # (n_qubits, 2, 2)
    density_matrix: np.ndarray | None = None
    trace: PropagationTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ basics
    @property
    def n_qubits(self) -> int:
        return self.circuit.n_qubits

    @property
    def n_clbits(self) -> int:
        return len(next(iter(self.counts))) if self.counts else self.circuit.n_clbits

    def probabilities(self) -> dict[str, float]:
        """Outcome distribution (exact when the backend provides it, else from counts)."""
        if self.probabilities_exact is not None:
            return dict(self.probabilities_exact)
        total = sum(self.counts.values())
        return {k: v / total for k, v in sorted(self.counts.items())}

    def marginal_counts(self, clbits: Sequence[int]) -> dict[str, int]:
        """Counts restricted to the given classical bits (in the given order)."""
        out: dict[str, int] = {}
        for k, v in self.counts.items():
            key = "".join(k[c] for c in clbits)
            out[key] = out.get(key, 0) + v
        return dict(sorted(out.items()))

    def most_frequent(self) -> str:
        return max(self.counts, key=self.counts.get)

    # ------------------------------------------------------- fault statistics
    def _need_faults(self) -> None:
        if self.instruction_faults is None:
            raise ValueError(
                f"The {self.backend} backend does not record fault statistics; use backend='frame' or 'statevector'."
            )

    def fault_rates(self, by_pauli: bool = False) -> np.ndarray:
        """Per-instruction fault rates (see ``nq.explain("fault_rate")``).

        Returns:
            ``(n_instructions, n_qubits)`` array, or ``(n_instructions, n_qubits, 4)``
            with columns X, Y, Z, other if ``by_pauli``.
        """
        self._need_faults()
        r = self.instruction_faults / self.shots
        return r if by_pauli else r.sum(axis=2)

    def idle_fault_rates(self, by_pauli: bool = False) -> np.ndarray:
        """Per ``(layer, qubit)`` idle fault rates."""
        self._need_faults()
        r = self.idle_faults / self.shots
        return r if by_pauli else r.sum(axis=2)

    def instruction_rates(self) -> np.ndarray:
        """Total fault rate per instruction (summed over qubits)."""
        return self.fault_rates().sum(axis=1)

    @property
    def fault_free_fraction(self) -> Estimate:
        """Fraction of shots with no fault anywhere (see ``nq.explain("fault_free_fraction")``)."""
        self._need_faults()
        p = float(np.mean(self.fault_free))
        return Estimate(p, binomial_stderr(p, self.shots))

    def output_error_rate(self, clbits: Sequence[int] | None = None) -> Estimate:
        """Fraction of shots in which any (selected) measured bit differs from the ideal run.

        Frame backend only — see ``nq.explain("output_error_rate")``.
        """
        if self.measurement_errors is None:
            raise ValueError("output_error_rate needs the frame backend (Clifford circuit + Pauli noise).")
        m = self.measurement_errors if clbits is None else self.measurement_errors[:, list(clbits)]
        p = float(np.mean(m.any(axis=1))) if m.size else 0.0
        return Estimate(p, binomial_stderr(p, self.shots))

    def final_error_strings(self, max_shots: int | None = None) -> list[str]:
        """Residual Pauli error per shot before the final measurements, e.g. ``"IXZ"``."""
        if self.final_errors is None:
            raise ValueError("Pauli error frames are only available from the frame backend.")
        x, z = self.final_errors
        k = x.shape[1] if max_shots is None else min(max_shots, x.shape[1])
        chars = np.array(["I", "Z", "X", "Y"])
        idx = (x[:, :k].astype(int) << 1) | z[:, :k].astype(int)
        return ["".join(chars[idx[:, s]]) for s in range(k)]

    def error_budget(self, metric: str = "rate", include_idle: bool = True) -> list[dict]:
        """Rank where errors come from (see ``nq.explain("error_burden")``).

        Args:
            metric: ``"rate"`` — fault probability; ``"burden"`` — rate weighted by how
                many output bits/qubits the fault corrupts (Clifford circuits only).
            include_idle: Include one row per qubit for idle (decoherence) noise.

        Returns:
            Rows sorted by value, each ``{"label", "index", "layer", "qubits", "value", "share"}``
            where ``share`` is the fraction of the total (sums to 1).
        """
        self._need_faults()
        if metric not in ("rate", "burden"):
            raise ValueError("metric must be 'rate' or 'burden'.")
        inst = self.fault_rates(by_pauli=True)
        idle = self.idle_fault_rates(by_pauli=True)
        if metric == "burden":
            from ..analysis.propagation import fault_spread

            w_inst, w_idle = fault_spread(self.circuit)
            inst_v = (inst[:, :, :3] * w_inst).sum(axis=(1, 2))
            idle_v = (idle[:, :, :3] * w_idle).sum(axis=2)
        else:
            inst_v = inst.sum(axis=(1, 2))
            idle_v = idle.sum(axis=2)
        rows = []
        for i, ins in enumerate(self.circuit.instructions):
            if inst_v[i] > 0:
                label = ins.gate.display_label if isinstance(ins, Operation) else ins.name.title()
                rows.append(
                    {
                        "label": f"{label} {list(ins.qubits)} @{ins.layer}",
                        "index": i,
                        "layer": ins.layer,
                        "qubits": tuple(ins.qubits),
                        "value": float(inst_v[i]),
                    }
                )
        if self.readout_faults is not None and self.implicit_measurement:
            for q, v in enumerate(self.readout_faults / self.shots):
                if v > 0 and metric == "rate":
                    rows.append(
                        {"label": f"Readout q{q}", "index": None, "layer": None, "qubits": (q,), "value": float(v)}
                    )
        if include_idle:
            per_q = idle_v.sum(axis=0)
            for q, v in enumerate(per_q):
                if v > 0:
                    rows.append(
                        {"label": f"Idle q{q}", "index": None, "layer": None, "qubits": (q,), "value": float(v)}
                    )
        total = sum(r["value"] for r in rows) or 1.0
        for r in rows:
            r["share"] = r["value"] / total
        return sorted(rows, key=lambda r: -r["value"])

    def qubit_error_rates(self) -> np.ndarray:
        """Total fault rate per qubit, ``(n_qubits, 4)`` columns X, Y, Z, other."""
        self._need_faults()
        return (self.instruction_faults.sum(axis=0) + self.idle_faults.sum(axis=0)) / self.shots

    # --------------------------------------------------------------- states
    def qubit_purities(self) -> np.ndarray:
        """``Tr(ρ_q²)`` of each qubit's reduced state (statevector / density backends)."""
        if self.qubit_states is None:
            raise ValueError("Reduced states are only available from the statevector or density_matrix backend.")
        return np.array([purity(r) for r in self.qubit_states])

    # ------------------------------------------------------------ comparison
    def compare(self, other: Union[Result, dict]) -> dict[str, float]:
        """Distribution distances to another result (or a counts/probability dict)."""
        q = other.probabilities() if isinstance(other, Result) else other
        p = self.probabilities()
        return {
            "hellinger_fidelity": hellinger_fidelity(p, q),
            "total_variation_distance": total_variation_distance(p, q),
        }

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"Result: {self.circuit.name or 'circuit'} — {self.n_qubits} qubits, "
            f"{self.shots} shots, backend={self.backend}, noise={self.noise_name}"
        ]
        top = sorted(self.counts.items(), key=lambda kv: -kv[1])[:5]
        lines.append("  top outcomes: " + ", ".join(f"{k}: {v}" for k, v in top))
        if self.fault_free is not None:
            lines.append(f"  fault-free fraction: {self.fault_free_fraction!r}")
        if self.measurement_errors is not None:
            lines.append(f"  output error rate:   {self.output_error_rate()!r}")
        if self.fidelity is not None:
            lines.append(f"  fidelity to ideal:   {self.fidelity!r}")
        if self.instruction_faults is not None:
            budget = self.error_budget()[:3]
            if budget:
                lines.append(
                    "  largest error sources: " + ", ".join(f"{r['label']} ({100 * r['share']:.0f}%)" for r in budget)
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Result(backend={self.backend!r}, shots={self.shots}, n_qubits={self.n_qubits}, "
            f"outcomes={len(self.counts)}, noise={self.noise_name!r})"
        )

    # -------------------------------------------------------------- save/load
    def save(self, path: Union[str, Path]) -> Path:
        """Save to a ``.npz`` file (arrays + JSON metadata) for later comparison.

        The circuit is stored as OpenQASM when possible. The noise model itself is not
        stored, only its name. Traces are not saved.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        from ..io.qasm import QASMExportError, to_qasm

        try:
            qasm = to_qasm(self.circuit)
        except QASMExportError:
            qasm = ""
        meta = {
            "format": "noisiq-result-v1",
            "backend": self.backend,
            "shots": self.shots,
            "seed": self.seed,
            "counts": self.counts,
            "implicit_measurement": self.implicit_measurement,
            "noise_name": self.noise_name,
            "qasm": qasm,
            "circuit_name": self.circuit.name,
            "n_qubits": self.n_qubits,
            "fidelity": None if self.fidelity is None else [self.fidelity.value, self.fidelity.stderr],
            "probabilities_exact": self.probabilities_exact,
            "metadata": _jsonable(self.metadata),
        }
        arrays = {
            k: getattr(self, k)
            for k in (
                "memory",
                "instruction_faults",
                "idle_faults",
                "readout_faults",
                "fault_free",
                "measurement_errors",
                "qubit_states",
                "density_matrix",
            )
            if getattr(self, k) is not None
        }
        if self.final_errors is not None:
            arrays["final_errors_x"], arrays["final_errors_z"] = self.final_errors
        np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> Result:
        """Load a result written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            arrays = {k: data[k] for k in data.files if k != "__meta__"}
        if meta.get("qasm"):
            from ..io.qasm import from_qasm

            circuit = from_qasm(meta["qasm"])
            circuit.name = meta.get("circuit_name")
        else:
            circuit = Circuit(meta["n_qubits"], name=meta.get("circuit_name"))
        fe = None
        if "final_errors_x" in arrays:
            fe = (arrays.pop("final_errors_x"), arrays.pop("final_errors_z"))
        fid = meta.get("fidelity")
        return cls(
            circuit=circuit,
            backend=meta["backend"],
            shots=meta["shots"],
            seed=meta["seed"],
            counts=meta["counts"],
            implicit_measurement=meta["implicit_measurement"],
            noise_name=meta["noise_name"],
            final_errors=fe,
            fidelity=None if fid is None else Estimate(*fid),
            probabilities_exact=meta.get("probabilities_exact"),
            metadata=meta.get("metadata", {}),
            **arrays,
        )


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return {k: str(v) for k, v in obj.items()} if isinstance(obj, dict) else str(obj)
