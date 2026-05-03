"""
Stateless gate information extractor for hover tooltips.

GateInfoExtractor.for_gate(op_idx, result) dispatches on result type and
returns a dict of display fields for the hover panel in widgets.py.

Single-shot fields  (StimTableauResult):
    mode, gate_type, qubits, timestep, error_occurred, errors

Many-shot fields    (AggregateResult):
    mode, gate_type, qubits, timestep, error_rates, most_likely_error,
    total_error_rate, fidelity_estimate
"""

from __future__ import annotations

from typing import Any


class GateInfoExtractor:
    """Stateless extractor — call for_gate() as a class method."""

    @classmethod
    def for_gate(cls, op_idx: int, result) -> dict[str, Any]:
        """
        Return a display-ready info dict for the operation at *op_idx*.

        Parameters
        ----------
        op_idx : Index into circuit.operations (0-based).
        result : StimTableauResult or AggregateResult.

        Raises
        ------
        IndexError  if op_idx is out of range.
        TypeError   if result type is not recognised.
        """
        from ..backends.pauli_frame import StimTableauResult
        from ..backends.many_shot_runner import AggregateResult

        if isinstance(result, AggregateResult):
            return cls._many_shot(op_idx, result)
        if isinstance(result, StimTableauResult):
            return cls._single_shot(op_idx, result)
        raise TypeError(f"Unsupported result type: {type(result).__name__}")

    # ------------------------------------------------------------------
    # Private dispatch targets
    # ------------------------------------------------------------------

    @classmethod
    def _single_shot(cls, op_idx: int, result) -> dict[str, Any]:
        step = result.steps[op_idx]
        op = step.operation
        errors = [
            {"qubit": e.qubit, "pauli": e.pauli}
            for e in step.errors
        ]
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
        # Per-qubit error rates for this operation column
        rates_col = result.error_rate_matrix[:, op_idx]  # shape (n_qubits,)
        error_rates = {
            f"q{q}": float(rates_col[q])
            for q in range(result.n_qubits)
        }

        # Most likely error: qubit with the highest rate at this op
        peak_qubit = int(rates_col.argmax())
        peak_rate = float(rates_col[peak_qubit])
        most_likely = (
            {"qubit": peak_qubit, "error_rate": peak_rate}
            if peak_rate > 0
            else None
        )

        total_error_rate = float(rates_col.sum())
        fidelity = max(0.0, 1.0 - total_error_rate)

        return {
            "mode": "many_shot",
            "gate_type": op.gate.name,
            "qubits": list(op.qubits),
            "timestep": op.t,
            "n_shots": result.n_shots,
            "error_rates": error_rates,
            "most_likely_error": most_likely,
            "total_error_rate": total_error_rate,
            "fidelity_estimate": fidelity,
        }

    # ------------------------------------------------------------------
    # Formatting helper (used by widgets.py hover panel)
    # ------------------------------------------------------------------

    @staticmethod
    def format_for_display(info: dict[str, Any]) -> str:
        """Render an info dict as a compact multi-line string."""
        lines: list[str] = []
        lines.append(f"Gate:     {info['gate_type']}  (t={info['timestep']})")
        lines.append(f"Qubits:   {info['qubits']}")

        if info["mode"] == "single_shot":
            if info["error_occurred"]:
                for e in info["errors"]:
                    lines.append(f"Error:    {e['pauli']} on q{e['qubit']}")
            else:
                lines.append("Error:    none")

        else:  # many_shot
            lines.append(f"Shots:    {info['n_shots']}")
            for q_label, rate in info["error_rates"].items():
                if rate > 0:
                    lines.append(f"  {q_label}: {rate:.3f}")
            ml = info["most_likely_error"]
            if ml:
                lines.append(
                    f"Top error: q{ml['qubit']}  ({ml['error_rate']:.3f})"
                )
            lines.append(f"Fidelity: {info['fidelity_estimate']:.4f}")

        return "\n".join(lines)
