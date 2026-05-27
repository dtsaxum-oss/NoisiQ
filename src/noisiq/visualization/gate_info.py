"""
Stateless gate information extractor for hover tooltips, plus the
WireSegment dataclass used by wire-halo rendering and hit-testing.

GateInfoExtractor.for_gate(op_idx, result) dispatches on result type and
returns a dict of display fields for the hover panel.

Single-shot fields  (StimTableauResult):
    mode, gate_type, qubits, timestep, error_occurred, errors

Many-shot fields    (AggregateResult):
    mode, gate_type, qubits, timestep, error_rates, most_likely_error,
    total_error_rate, fidelity_estimate

When circuit and noise_config are also supplied to for_gate:
    noise, per_gate_decoherence, cumulative
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# WireSegment — lives here so heatmap.py can import it without circularity
# (heatmap.py imports GateInfoExtractor; gate_info.py does not import heatmap)
# ---------------------------------------------------------------------------

@dataclass
class WireSegment:
    """One contiguous idle stretch on one qubit, between two non-IDLE ops.

    Attributes:
        qubit:         Which qubit this segment lives on.
        t_lo, t_hi:    x-extent of the segment in heatmap data coordinates.
        idle_op_idxs:  Indices into circuit.operations of the IDLE ops
                       that fill this segment.
        duration_ns:   Total wall-clock time the qubit sat idle here.
        gamma:         Per-segment T1 leakage = 1 - exp(-duration / T1).
        lam:           Per-segment T2 phase decay = 1 - exp(-2·duration / T2).
        intensity:     Combined = 1 - (1-gamma)*(1-lam). In [0, 1].
    """
    qubit: int
    t_lo: float
    t_hi: float
    idle_op_idxs: list
    duration_ns: float
    gamma: float
    lam: float
    intensity: float


# ---------------------------------------------------------------------------
# GateInfoExtractor
# ---------------------------------------------------------------------------

class GateInfoExtractor:
    """Stateless extractor — call class methods directly."""

    @classmethod
    def for_gate(
        cls,
        op_idx: int,
        result,
        *,
        circuit=None,
        noise_config: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return a display-ready info dict for the operation at op_idx.

        Parameters
        ----------
        op_idx       : Index into circuit.operations (0-based).
        result       : StimTableauResult or AggregateResult.
        circuit      : The circuit (required for cumulative noise fields).
        noise_config : Per-op noise dict (required for noise fields).

        New fields when both circuit and noise_config are supplied:
            noise, per_gate_decoherence, cumulative
        """
        from ..backends.pauli_frame import StimTableauResult
        from ..backends.many_shot_runner import AggregateResult

        if isinstance(result, AggregateResult):
            info = cls._many_shot(op_idx, result)
        elif isinstance(result, StimTableauResult):
            info = cls._single_shot(op_idx, result)
        else:
            raise TypeError(f"Unsupported result type: {type(result).__name__}")

        if circuit is not None and noise_config is not None:
            info["noise"] = cls._describe_channel(noise_config.get(op_idx))
            info["per_gate_decoherence"] = cls._per_gate_decoherence(
                noise_config.get(op_idx)
            )
            info["cumulative"] = cls._cumulative_decoherence(
                op_idx, circuit, noise_config
            )

        return info

    @classmethod
    def for_qubit_endcap(
        cls,
        qubit: int,
        circuit,
        result,
        *,
        noise_config: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Display info dict for the end-of-wire region on qubit."""
        last_idx = None
        for i, op in enumerate(circuit.operations):
            if qubit in op.qubits:
                last_idx = i

        info: dict[str, Any] = {
            "kind": "qubit_endcap",
            "qubit": qubit,
            "last_op_idx": last_idx,
        }

        if last_idx is not None and noise_config is not None:
            cum = cls._cumulative_decoherence(last_idx, circuit, noise_config)
            info["cumulative"] = cum.get(f"q{qubit}", {})

        from ..backends.many_shot_runner import AggregateResult
        if isinstance(result, AggregateResult):
            qubit_counts = result.counts_matrix[qubit, :]
            info["total_error_events"] = int(qubit_counts.sum())
            info["per_qubit_error_rate"] = float(
                qubit_counts.sum() / result.n_shots
            )

        return info

    @classmethod
    def for_wire_segment(
        cls,
        segment: WireSegment,
        cursor_x: float,
        circuit,
        result,
        noise_config: dict,
    ) -> dict[str, Any]:
        """Info dict for a wire segment hover.

        Reports per-segment γ / λ / intensity / duration and cumulative
        T1+T2 on this qubit up to cursor_x.
        """
        info: dict[str, Any] = {
            "kind": "wire_segment",
            "qubit": segment.qubit,
            "t_range": (segment.t_lo, segment.t_hi),
            "per_segment": {
                "duration_ns": segment.duration_ns,
                "gamma": segment.gamma,
                "lambda": segment.lam,
                "intensity": segment.intensity,
                "idle_op_count": len(segment.idle_op_idxs),
            },
        }
        info["cumulative_up_to_cursor"] = cls._cumulative_at_x(
            segment.qubit, cursor_x, circuit, noise_config
        )
        return info

    # ------------------------------------------------------------------
    # Private dispatch targets
    # ------------------------------------------------------------------

    @classmethod
    def _single_shot(cls, op_idx: int, result) -> dict[str, Any]:
        step = result.steps[op_idx]
        op = step.operation
        errors = [{"qubit": e.qubit, "pauli": e.pauli} for e in step.errors]
        return {
            "mode": "single_shot",
            "gate_type": op.gate.name,
            "qubits": list(op.qubits),
            "timestep": op.t,
            "error_occurred": len(errors) > 0,
            "errors": errors,
        }

    @classmethod
    def _many_shot(cls, op_idx: int, result) -> dict[str, Any]:
        op = result.circuit.operations[op_idx]
        rates_col = result.error_rate_matrix[:, op_idx]
        error_rates = {
            f"q{q}": float(rates_col[q]) for q in range(result.n_qubits)
        }
        peak_qubit = int(rates_col.argmax())
        peak_rate = float(rates_col[peak_qubit])
        most_likely = (
            {"qubit": peak_qubit, "error_rate": peak_rate}
            if peak_rate > 0 else None
        )
        return {
            "mode": "many_shot",
            "gate_type": op.gate.name,
            "qubits": list(op.qubits),
            "timestep": op.t,
            "n_shots": result.n_shots,
            "error_rates": error_rates,
            "most_likely_error": most_likely,
            "total_error_rate": float(rates_col.sum()),
            "fidelity_estimate": float(np.prod(1.0 - rates_col)),
        }

    # ------------------------------------------------------------------
    # Noise-aware private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iterate_channels(ch) -> Iterable:
        """Yield sub-channels of a CombinedChannel, or ch itself."""
        from ..noise.kraus_channels import CombinedChannel
        if ch is None:
            return
        if isinstance(ch, CombinedChannel):
            yield from ch.channels
        else:
            yield ch

    @classmethod
    def _describe_channel(cls, ch) -> dict:
        if ch is None:
            return {"type": "none", "summary": "no noise attached"}
        if hasattr(ch, "describe"):
            d = dict(ch.describe())
            d["type"] = type(ch).__name__
            return d
        return {"type": type(ch).__name__, "repr": repr(ch)}

    @classmethod
    def _per_gate_decoherence(cls, ch) -> dict:
        from ..noise.amplitude_damping import AmplitudeDamping
        from ..noise.t2_dephasing import Dephasing
        out: dict = {}
        for sub in cls._iterate_channels(ch):
            if isinstance(sub, AmplitudeDamping):
                out["t1_gamma"] = float(sub.gamma)
                out["t1_T1_s"] = float(sub.T1)
                out["t1_t_s"] = float(sub.t)
            elif isinstance(sub, Dephasing):
                out["t2_lambda"] = float(sub.lam)
                out["t2_t_s"] = float(sub.t)
                if sub.T2 is not None:
                    out["t2_T2_s"] = float(sub.T2)
                else:
                    out["t2_T1_s"] = float(sub.T1)
                    out["t2_Tphi_s"] = float(sub.Tphi)
        return out

    @classmethod
    def _cumulative_decoherence(
        cls, op_idx: int, circuit, noise_config: dict
    ) -> dict:
        from ..noise.amplitude_damping import AmplitudeDamping
        from ..noise.t2_dephasing import Dephasing
        target_op = circuit.operations[op_idx]
        out: dict = {}

        for q in target_op.qubits:
            total_t_t1_s = 0.0
            total_t_t2_s = 0.0
            T1_seen = None
            T2_seen = None

            for prev_idx in range(op_idx + 1):
                prev_op = circuit.operations[prev_idx]
                if q not in prev_op.qubits:
                    continue
                ch = noise_config.get(prev_idx)
                for sub in cls._iterate_channels(ch):
                    if isinstance(sub, AmplitudeDamping):
                        total_t_t1_s += sub.t
                        T1_seen = sub.T1
                    elif isinstance(sub, Dephasing):
                        total_t_t2_s += sub.t
                        T2_seen = (
                            sub.T2 if sub.T2 is not None
                            else 1.0 / (1.0 / (2 * sub.T1) + 1.0 / sub.Tphi)
                        )

            cum_t1 = (
                1.0 - float(np.exp(-total_t_t1_s / T1_seen))
                if T1_seen else 0.0
            )
            cum_t2 = (
                1.0 - float(np.exp(-2.0 * total_t_t2_s / T2_seen))
                if T2_seen else 0.0
            )

            out[f"q{q}"] = {
                "cumulative_t1_leakage": cum_t1,
                "cumulative_t2_phase_decay": cum_t2,
                "total_t1_exposure_ns": total_t_t1_s * 1e9,
                "total_t2_exposure_ns": total_t_t2_s * 1e9,
            }
        return out

    @classmethod
    def _cumulative_at_x(
        cls, qubit: int, cursor_x: float, circuit, noise_config: dict
    ) -> dict:
        """Sum T1/T2 exposure on qubit over all ops with op.t <= cursor_x."""
        from ..noise.amplitude_damping import AmplitudeDamping
        from ..noise.t2_dephasing import Dephasing
        total_t_t1 = 0.0
        total_t_t2 = 0.0
        T1_seen = None
        T2_seen = None

        for idx, op in enumerate(circuit.operations):
            if qubit not in op.qubits:
                continue
            if op.t > cursor_x:
                continue
            for sub in cls._iterate_channels(noise_config.get(idx)):
                if isinstance(sub, AmplitudeDamping):
                    total_t_t1 += sub.t
                    T1_seen = sub.T1
                elif isinstance(sub, Dephasing):
                    total_t_t2 += sub.t
                    T2_seen = (
                        sub.T2 if sub.T2 is not None
                        else 1.0 / (1.0 / (2 * sub.T1) + 1.0 / sub.Tphi)
                    )

        return {
            "t1_leakage": (
                1.0 - float(np.exp(-total_t_t1 / T1_seen)) if T1_seen else 0.0
            ),
            "t2_phase_decay": (
                1.0 - float(np.exp(-2.0 * total_t_t2 / T2_seen)) if T2_seen else 0.0
            ),
            "t1_exposure_ns": total_t_t1 * 1e9,
            "t2_exposure_ns": total_t_t2 * 1e9,
        }

    # ------------------------------------------------------------------
    # Formatting helpers — used by both display modes
    # ------------------------------------------------------------------

    @staticmethod
    def format_full(info: dict[str, Any]) -> str:
        """Multi-line full info string for the ipywidgets Output panel."""
        lines = [
            f"Gate         {info['gate_type']}  (t={info['timestep']}, op_idx={info.get('op_idx', '?')})",
            f"Qubits       {info['qubits']}",
        ]

        if info.get("per_gate_decoherence"):
            pg = info["per_gate_decoherence"]
            lines += ["", "── Decoherence (this gate) ──"]
            if "t1_gamma" in pg:
                lines.append(f"T1 γ (per gate)   {pg['t1_gamma']:.3%}")
            if "t2_lambda" in pg:
                lines.append(f"T2 λ (per gate)   {pg['t2_lambda']:.3%}")

        if info.get("cumulative"):
            lines += ["", "── Cumulative on these qubits ──"]
            for q_label, c in info["cumulative"].items():
                lines.append(
                    f"{q_label}: T1 leak {c['cumulative_t1_leakage']:.2%}"
                    f"  |  T2 phase decay {c['cumulative_t2_phase_decay']:.2%}"
                )

        if "noise" in info:
            lines += ["", "── Noise model attached ──"]
            n = info["noise"]
            lines.append(f"  type: {n.get('type', '?')}")
            for k, v in n.items():
                if k != "type":
                    lines.append(f"    {k} = {v}")

        if info["mode"] == "single_shot":
            lines += ["", "── This shot ──"]
            if info["error_occurred"]:
                for e in info["errors"]:
                    lines.append(f"  Error: {e['pauli']} on q{e['qubit']}")
            else:
                lines.append("  Error: none")
        else:
            lines += ["", f"── Many-shot stats (n={info['n_shots']}) ──"]
            ml = info.get("most_likely_error")
            if ml:
                lines.append(
                    f"  Most likely error: q{ml['qubit']}  ({ml['error_rate']:.3f})"
                )
            lines.append(f"  Total error rate: {info['total_error_rate']:.4f}")
            lines.append(f"  Fidelity estimate: {info['fidelity_estimate']:.4f}")

        return "\n".join(lines)

    @staticmethod
    def format_compact(info: dict[str, Any]) -> str:
        """Short version for the ax.annotate tooltip (5-7 lines max)."""
        lines = [f"{info['gate_type']} @ t={info['timestep']}, q={info['qubits']}"]
        if info.get("per_gate_decoherence"):
            pg = info["per_gate_decoherence"]
            if "t1_gamma" in pg:
                lines.append(
                    f"γ={pg['t1_gamma']:.2%}  λ={pg.get('t2_lambda', 0):.2%}"
                )
        if info.get("cumulative"):
            for q_label, c in info["cumulative"].items():
                lines.append(
                    f"{q_label}: ΣT1 {c['cumulative_t1_leakage']:.1%}"
                    f"  ΣT2 {c['cumulative_t2_phase_decay']:.1%}"
                )
        if info["mode"] == "many_shot":
            lines.append(
                f"err rate {info['total_error_rate']:.3f}, "
                f"F≈{info['fidelity_estimate']:.3f}"
            )
        return "\n".join(lines)

    @staticmethod
    def format_for_display(info: dict[str, Any]) -> str:
        """Backwards-compatible alias for format_full (used by widgets.py)."""
        lines: list[str] = []
        lines.append(f"Gate:     {info['gate_type']}  (t={info['timestep']})")
        lines.append(f"Qubits:   {info['qubits']}")

        if info["mode"] == "single_shot":
            if info["error_occurred"]:
                for e in info["errors"]:
                    lines.append(f"Error:    {e['pauli']} on q{e['qubit']}")
            else:
                lines.append("Error:    none")
        else:
            lines.append(f"Shots:    {info['n_shots']}")
            for q_label, rate in info["error_rates"].items():
                if rate > 0:
                    lines.append(f"  {q_label}: {rate:.3f}")
            ml = info["most_likely_error"]
            if ml:
                lines.append(f"Top error: q{ml['qubit']}  ({ml['error_rate']:.3f})")
            lines.append(f"Fidelity: {info['fidelity_estimate']:.4f}")

        return "\n".join(lines)
