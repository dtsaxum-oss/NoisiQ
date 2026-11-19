"""
Stateless gate information extractor for hover tooltips, plus the
WireSegment dataclass used by wire-halo rendering and hit-testing.

GateInfoExtractor.for_gate(op_idx, result) dispatches on result type and
returns a dict of display fields for the hover panel.

Single-shot fields  (StimTableauResult):
    mode, gate_type, qubits, timestep, error_occurred, errors

Many-shot fields    (AggregateResult):
    mode, gate_type, qubits, timestep, error_rates, most_likely_error,
    total_error_rate, zero_error_survival

When circuit and noise_config are also supplied to for_gate:
    noise, per_gate_decoherence, cumulative
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from ..ir.classical import Measurement as _Measurement, ConditionalOp as _ConditionalOp
from .theme import (
    HOVER_GATE_HALO_SECTION,
    HOVER_BURDEN_LABEL,
    HOVER_GATE_DECOHERENCE_SECTION,
    HOVER_GATE_CUMULATIVE_SECTION,
)


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

def _get_op_qubits(op) -> tuple:
    """Return a qubits tuple for any CircuitOp variant.

    Operation and ConditionalOp both expose .qubits (tuple).
    Measurement exposes .qubit (int, singular) — normalise it here.
    """
    from ..ir.classical import Measurement
    if isinstance(op, Measurement):
        return (op.qubit,)
    return op.qubits


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
        hover_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Return a display-ready info dict for the operation at op_idx.

        Parameters
        ----------
        op_idx        : Index into circuit.operations (0-based).
        result        : StimTableauResult or AggregateResult.
        circuit       : The circuit (required for cumulative noise fields).
        noise_config  : Per-op noise dict (required for noise fields).
        hover_context : Optional metadata dict (noise_model, run_metrics, …)
                        forwarded unchanged as info["hover_context"].

        New fields when both circuit and noise_config are supplied:
            noise, per_gate_decoherence, cumulative
        """
        from ..backends.pauli_frame import StimTableauResult
        from ..backends.many_shot_runner import AggregateResult
        from ..ir.classical import Measurement

        # Resolve the circuit reference so we can inspect the op type.
        # AggregateResult always carries result.circuit; fall back to it when
        # the caller omits the keyword argument.
        _circuit = circuit
        if _circuit is None and isinstance(result, AggregateResult):
            _circuit = getattr(result, "circuit", None)

        if _circuit is not None:
            op = _circuit.operations[op_idx]
            if isinstance(op, Measurement):
                info = cls._measurement_info(op_idx, op, _circuit, noise_config)
                info["hover_context"] = hover_context or {}
                return info

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

        info["hover_context"] = hover_context or {}
        return info

    @classmethod
    def _measurement_info(
        cls,
        op_idx: int,
        op,
        circuit,
        noise_config: Optional[dict],
    ) -> dict[str, Any]:
        """Return display dict for a Measurement op.

        Includes cumulative bit-flip probability (p_wrong_measurement) from
        all PauliError channels on the measured qubit up to this op, derived
        via _cumulative_decoherence so that the hover panel and the average-mode
        heatmap weighting use an identical computation path.
        """
        info: dict[str, Any] = {
            "mode": "measurement",
            "qubit": op.qubit,
            "cbit": op.cbit.name,
            "basis": op.basis,
            "reset": op.reset,
            "timestep": op.t,
            "op_idx": op_idx,
        }

        if noise_config is not None:
            cum = cls._cumulative_decoherence(op_idx, circuit, noise_config)
            qubit_key = f"q{op.qubit}"
            qubit_cum = cum.get(qubit_key, {})
            info["cumulative"] = qubit_cum
            info["p_wrong_measurement"] = qubit_cum.get("p_wrong_measurement", 0.0)

        return info

    @classmethod
    def for_qubit_endcap(
        cls,
        qubit: int,
        circuit,
        result,
        *,
        noise_config: Optional[dict] = None,
        hover_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Display info dict for the end-of-wire region on qubit."""
        last_idx = None
        for i, op in enumerate(circuit.operations):
            if qubit in _get_op_qubits(op):
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

        info["hover_context"] = hover_context or {}
        return info

    @classmethod
    def for_wire_segment(
        cls,
        segment: WireSegment,
        cursor_x: float,
        circuit,
        result,
        noise_config: dict,
        *,
        hover_context: Optional[dict] = None,
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
        info["hover_context"] = hover_context or {}
        return info

    # ------------------------------------------------------------------
    # Private dispatch targets
    # ------------------------------------------------------------------

    @classmethod
    def _single_shot(cls, op_idx: int, result) -> dict[str, Any]:
        step = result.steps[op_idx]
        op = step.operation
        if isinstance(op, _Measurement):
            return {"mode": "single_shot", "gate_type": "MEASURE", "qubits": [op.qubit],
                    "timestep": op.t, "error_occurred": False, "errors": []}
        if isinstance(op, _ConditionalOp):
            op = op.inner
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
        if isinstance(op, _Measurement):
            return {"mode": "many_shot", "gate_type": "MEASURE", "qubits": [op.qubit],
                    "timestep": op.t, "n_shots": result.n_shots,
                    "error_rates": {}, "most_likely_error": None,
                    "total_error_rate": 0.0, "zero_error_survival": 1.0}
        if isinstance(op, _ConditionalOp):
            op = op.inner
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
        total_events_per_shot = float(rates_col.sum())
        return {
            "mode": "many_shot",
            "gate_type": op.gate.name,
            "qubits": list(op.qubits),
            "timestep": op.t,
            "n_shots": result.n_shots,
            "error_rates": error_rates,
            "most_likely_error": most_likely,
            "total_error_rate": total_events_per_shot,          # kept for backward compat
            "total_error_events_per_shot": total_events_per_shot,
            "peak_qubit_error_events_per_shot": peak_rate,
            "zero_error_survival": float(np.prod(1.0 - rates_col)),
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
        from ..noise.pauli_error import PauliError
        out: dict = {}
        pauli_px = 0.0
        pauli_py = 0.0
        pauli_pz = 0.0
        has_pauli = False
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
            elif isinstance(sub, PauliError):
                pauli_px += float(sub.p_x)
                pauli_py += float(sub.p_y)
                pauli_pz += float(sub.p_z)
                has_pauli = True
        if has_pauli:
            out["pauli_px"] = pauli_px
            out["pauli_py"] = pauli_py
            out["pauli_pz"] = pauli_pz
            out["pauli_p_total"] = pauli_px + pauli_py + pauli_pz
        return out

    @classmethod
    def _cumulative_decoherence(
        cls, op_idx: int, circuit, noise_config: dict
    ) -> dict:
        from ..noise.amplitude_damping import AmplitudeDamping
        from ..noise.pauli_error import PauliError
        from ..noise.t2_dephasing import Dephasing
        target_op = circuit.operations[op_idx]
        out: dict = {}

        for q in _get_op_qubits(target_op):
            total_t_t1_s = 0.0
            total_t_t2_s = 0.0
            T1_seen = None
            T2_seen = None
            total_px_py = 0.0  # cumulative bit-flip probability (PauliError path)
            total_pz = 0.0     # cumulative phase-flip probability (PauliError path)

            for prev_idx in range(op_idx + 1):
                prev_op = circuit.operations[prev_idx]
                if q not in _get_op_qubits(prev_op):
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
                    elif isinstance(sub, PauliError):
                        total_px_py += sub.p_x + sub.p_y
                        total_pz += sub.p_z

            cum_t1 = (
                1.0 - float(np.exp(-total_t_t1_s / T1_seen))
                if T1_seen else 0.0
            )
            # NoisiQ convention:
            # cumulative_t2_phase_decay reports the phase-damping channel parameter λ,
            # matching Dephasing.lam = 1 - exp(-2t/T2).
            # This is not the same as off-diagonal coherence loss, 1 - exp(-t/T2).
            cum_t2 = (
                1.0 - float(np.exp(-2.0 * total_t_t2_s / T2_seen))
                if T2_seen else 0.0
            )
            coherence_remaining = (
                float(np.exp(-total_t_t2_s / T2_seen))
                if T2_seen else 1.0
            )

            # p_wrong_measurement: probability a Z-basis readout gives the wrong
            # bit due to accumulated X and Y Pauli errors.  Z errors do not flip
            # the Z-basis outcome, so they are excluded.  Capped at 0.5 (max
            # disorder); values above that are unphysical in the Pauli-twirl
            # approximation used throughout this pipeline.
            p_wrong = min(total_px_py, 0.5)

            out[f"q{q}"] = {
                "cumulative_t1_leakage": cum_t1,
                "cumulative_t2_phase_decay": cum_t2,
                "cumulative_t2_coherence_loss": 1.0 - coherence_remaining,
                "cumulative_t2_coherence_remaining": coherence_remaining,
                "total_t1_exposure_ns": total_t_t1_s * 1e9,
                "total_t2_exposure_ns": total_t_t2_s * 1e9,
                "cumulative_x_error_prob": min(total_px_py, 1.0),
                "cumulative_z_error_prob": min(total_pz, 1.0),
                "p_wrong_measurement": p_wrong,
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
            if qubit not in _get_op_qubits(op):
                continue
            if op.t > cursor_x:
                break
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
        """Aligned two-column info string for the ipywidgets Output panel."""

        W = 22  # key column width — accommodates "Error events / shot" (19 chars)

        def row(key: str, value: str) -> str:
            return f"  {key:<{W}}{value}"

        ctx   = info.get("hover_context", {})
        nm    = ctx.get("noise_model", {})
        run_m = ctx.get("run_metrics", {})
        pg    = info.get("per_gate_decoherence", {})
        noise = info.get("noise", {})

        # ── Measurement branch ────────────────────────────────────────────────
        if info.get("mode") == "measurement":
            reset_tag = "  (measure-and-reset)" if info.get("reset") else ""
            lines = [
                f"Measurement: q{info['qubit']} → {info['cbit']}{reset_tag}",
                f"Time:        t={info['timestep']}",
            ]
            if "p_wrong_measurement" in info:
                lines.append(row("P(wrong readout)", f"{info['p_wrong_measurement']:.3%}"))
            cum = info.get("cumulative", {})
            if cum:
                lines += ["", "── Accumulated before this measurement ──"]
                t1 = cum.get("cumulative_t1_leakage", 0.0)
                t2 = cum.get("cumulative_t2_phase_decay", 0.0)
                bf = cum.get("cumulative_x_error_prob", 0.0)
                pf = cum.get("cumulative_z_error_prob", 0.0)
                if t1 > 0 or t2 > 0:
                    lines.append(row("T1 leakage", f"{t1:.2%}"))
                    lines.append(row("T2 phase decay", f"{t2:.2%}"))
                elif bf > 0 or pf > 0:
                    lines.append(row("Σ bit-flip", f"{bf:.3%}  [X+Y channels]"))
                    lines.append(row("Σ phase-flip", f"{pf:.3%}  [Z channels]"))
            return "\n".join(lines)

        # ── Gate header ───────────────────────────────────────────────────────
        lines = [
            f"Gate:   {info['gate_type']}",
            f"Qubits: {info['qubits']}",
            f"Time:   t={info['timestep']}",
        ]

        # ── Noise model passed to circuit ─────────────────────────────────────
        nm_name = (
            nm.get("source")
            or (
                f"{nm['vendor']} {nm['system']}"
                if nm.get("vendor") and nm.get("system")
                else nm.get("system") or nm.get("vendor") or ""
            )
            or noise.get("type", "")
        )
        if "pauli_px" in pg:
            compat = "Pauli-compatible"
        elif "t1_gamma" in pg:
            compat = "Kraus / density-matrix"
        else:
            compat = "—"
        mode_str = "many-shot" if info.get("mode") == "many_shot" else "single-shot"

        if nm or noise:
            lines += ["", "── Noise model passed to circuit ──"]
            if nm_name:
                lines.append(row("Source", nm_name))
            if nm.get("mode"):
                lines.append(row("Mode", nm["mode"]))
            if nm.get("representation"):
                lines.append(row("Representation", nm["representation"]))
            lines.append(row("Backend target", f"{compat} / {mode_str}"))
            if nm.get("single_qubit_error") is not None:
                lines.append(row("1Q gate error", f"{float(nm['single_qubit_error']):.2e}"))
            if nm.get("two_qubit_error") is not None:
                lines.append(row("2Q gate error", f"{float(nm['two_qubit_error']):.2e}"))

        # ── Gate channel on this operation ────────────────────────────────────
        channel_type = noise.get("type") or ("PauliError" if "pauli_px" in pg else "—")
        lines += ["", "── Gate channel on this operation ──"]
        lines.append(row("Channel type", channel_type))
        if "pauli_px" in pg:
            lines.append(row("pX", f"{pg['pauli_px']:.4%}"))
            lines.append(row("pY", f"{pg['pauli_py']:.4%}"))
            lines.append(row("pZ", f"{pg['pauli_pz']:.4%}"))
            lines.append(row("pΣ", f"{pg['pauli_p_total']:.4%}"))
        elif "t1_gamma" in pg:
            lines.append(row("γ  (T1)", f"{pg['t1_gamma']:.4%}"))
            if "t2_lambda" in pg:
                lines.append(row("λ  (T2)", f"{pg['t2_lambda']:.4%}"))

        # ── Accumulated before / through this gate ────────────────────────────
        cum = info.get("cumulative", {})
        if cum:
            lines += ["", "── Accumulated before/through this gate ──"]
            for q_label, c in cum.items():
                t1 = c.get("cumulative_t1_leakage", 0.0)
                t2 = c.get("cumulative_t2_phase_decay", 0.0)
                bf = c.get("cumulative_x_error_prob", 0.0)
                pf = c.get("cumulative_z_error_prob", 0.0)
                if t1 > 0 or t2 > 0:
                    lines.append(f"  {q_label}: ΣT1 {t1:.2%}  |  ΣT2 {t2:.2%}")
                elif bf > 0 or pf > 0:
                    lines.append(f"  {q_label}: Σbit {bf:.2%}  |  Σphase {pf:.2%}")

        # ── Many-shot observed stats ──────────────────────────────────────────
        if info.get("mode") == "many_shot":
            lines += ["", "── Many-shot observed stats ──"]
            lines.append(row("Shots", str(info["n_shots"])))
            lines.append(row("Error events / shot", f"{info['total_error_events_per_shot']:.4g}"))
            lines.append(row("Zero-error survival", f"{info['zero_error_survival']:.4f}"))
            ml = info.get("most_likely_error")
            if ml:
                lines.append(row("Most likely error", f"q{ml['qubit']}  ({ml['error_rate']:.3f} / shot)"))
        elif info.get("mode") == "single_shot":
            lines += ["", "── This shot ──"]
            if info.get("error_occurred"):
                for e in info["errors"]:
                    lines.append(f"  Error: {e['pauli']} on q{e['qubit']}")
            else:
                lines.append("  No error this shot")

        # ── Gate halo quantity ────────────────────────────────────────────────
        if "gate_halo" in info:
            gh = info["gate_halo"]
            metric_word = (
                "expected" if gh.get("impact_metric") == "expected" else "worst-case"
            )
            lines += ["", HOVER_GATE_HALO_SECTION]
            lines.append(row("Metric", f"{metric_word} {HOVER_BURDEN_LABEL}"))
            lines.append(row("Raw burden", f"{gh['downstream_burden']:.4g}"))
            lines.append(row("Visual halo", f"{gh['visual_intensity']:.4f}"))

        # ── Run context ───────────────────────────────────────────────────────
        if run_m:
            lines += ["", "── Run context ──"]
            for k, v in run_m.items():
                lines.append(row(k, str(v)))

        return "\n".join(lines)

    @staticmethod
    def format_compact(info: dict[str, Any]) -> str:
        """Structured tooltip for the ax.annotate hover box."""
        if info.get("mode") == "measurement":
            lines = [
                f"Measurement: q{info['qubit']} → {info['cbit']}",
                f"Time: t={info['timestep']}",
            ]
            if "p_wrong_measurement" in info:
                lines.append(f"P(wrong readout): {info['p_wrong_measurement']:.2%}")
            return "\n".join(lines)

        ctx = info.get("hover_context", {})

        lines = [
            f"Gate: {info['gate_type']}",
            f"Qubits: {info['qubits']}",
            f"Time: t={info['timestep']}",
        ]

        # Noise model — sourced from hover_context when available
        nm = ctx.get("noise_model", {})
        if nm:
            nm_parts: list[str] = []
            name = nm.get("source") or (
                f"{nm['vendor']} {nm['system']}"
                if nm.get("vendor") and nm.get("system")
                else nm.get("system") or nm.get("vendor") or ""
            )
            if name:
                nm_parts.append(name)
            if nm.get("mode"):
                nm_parts.append(f"mode={nm['mode']}")
            if nm.get("representation"):
                nm_parts.append(nm["representation"])
            if nm_parts:
                lines.append("Noise model: " + " | ".join(nm_parts))
            # Nominal gate error rates from the hardware profile
            spec_parts: list[str] = []
            if nm.get("single_qubit_error") is not None:
                spec_parts.append(f"1Q ≤ {float(nm['single_qubit_error']):.1e}")
            if nm.get("two_qubit_error") is not None:
                spec_parts.append(f"2Q ≤ {float(nm['two_qubit_error']):.1e}")
            if spec_parts:
                lines.append("Gate specs:  " + "  |  ".join(spec_parts))

        # Per-gate channel parameters
        pg = info.get("per_gate_decoherence", {})
        if "pauli_px" in pg:
            lines.append(
                f"Gate channel: pX={pg['pauli_px']:.2%}"
                f"  pY={pg['pauli_py']:.2%}"
                f"  pZ={pg['pauli_pz']:.2%}"
                f"  pΣ={pg['pauli_p_total']:.2%}"
            )
        elif "t1_gamma" in pg:
            t2_str = f"  λ={pg['t2_lambda']:.2%}" if "t2_lambda" in pg else ""
            lines.append(f"Gate channel: γ={pg['t1_gamma']:.2%}{t2_str}")

        # Accumulated error per qubit — all qubits on one line, pipe-separated
        cum = info.get("cumulative", {})
        if cum:
            cum_parts: list[str] = []
            for q_label, c in cum.items():
                t1 = c.get("cumulative_t1_leakage", 0.0)
                t2 = c.get("cumulative_t2_phase_decay", 0.0)
                bit_flip = c.get("cumulative_x_error_prob", 0.0)
                phase_flip = c.get("cumulative_z_error_prob", 0.0)
                if t1 > 0 or t2 > 0:
                    cum_parts.append(f"{q_label} ΣT1={t1:.1%} ΣT2={t2:.1%}")
                elif bit_flip > 0 or phase_flip > 0:
                    cum_parts.append(
                        f"{q_label} Σbit={bit_flip:.1%} Σphase={phase_flip:.1%}"
                    )
            if cum_parts:
                lines.append("Accumulated: " + " | ".join(cum_parts))

        # Observed many-shot stats
        if info.get("mode") == "many_shot":
            lines.append(
                f"Observed: errors/shot={info['total_error_events_per_shot']:.3g}"
                f" | zero-error≈{info['zero_error_survival']:.3f}"
            )

        # Halo quantity — what drove the halo color
        if "gate_halo" in info:
            gh = info["gate_halo"]
            metric_word = (
                "expected" if gh.get("impact_metric") == "expected" else "worst-case"
            )
            lines.append(
                f"Halo: {metric_word} burden/shot={gh['downstream_burden']:.3g}"
                f" | visual={gh['visual_intensity']:.3f}"
            )

        return "\n".join(lines)

    @staticmethod
    def format_for_display(info: dict[str, Any]) -> str:
        """Backwards-compatible alias for format_full (used by widgets.py)."""
        if info.get("mode") == "measurement":
            lines = [
                f"Measurement: q{info['qubit']} → {info['cbit']}  (t={info['timestep']})",
            ]
            if "p_wrong_measurement" in info:
                lines.append(f"P(wrong meas): {info['p_wrong_measurement']:.3%}")
            return "\n".join(lines)

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
            lines.append(f"Error events / shot: {info.get('total_error_events_per_shot', info['total_error_rate']):.4g}")
            lines.append(f"Zero-error survival: {info['zero_error_survival']:.4f}")

        return "\n".join(lines)
