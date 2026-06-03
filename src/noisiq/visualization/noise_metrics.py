"""
Shared visualization noise-metric helpers.

Provides channel_event_probability() and normalize_heat_values() so that the
heatmap, qubit-error bar chart, and hover panel all compute burden scores the
same way and stay consistent when new channel types are added.

These are *visualization* metrics — not logical error rates, process infidelity,
or state-fidelity loss.  The recommended naming is "estimated downstream error
burden" or "error events / shot", not "fidelity loss".
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def combine_independent_probabilities(probabilities: list[float]) -> float:
    """Combine independent event probabilities as 1 - Π(1 - p_i)."""
    p_no_event = 1.0
    for p in probabilities:
        p_no_event *= 1.0 - float(np.clip(p, 0.0, 1.0))
    return float(np.clip(1.0 - p_no_event, 0.0, 1.0))


def channel_event_probability(channel) -> float:
    """Return a visualization-level non-identity event probability for a noise channel.

    Handles PauliError, CorrelatedPauliError, CoherentRotation,
    StochasticCoherentRotation, AmplitudeDamping, Dephasing, and CombinedChannel.
    Returns 0.0 for None or any unrecognised type.
    """
    from ..noise.pauli_error import PauliError
    from ..noise.correlated_errors import CorrelatedPauliError
    from ..noise.coherent_errors import CoherentRotation, StochasticCoherentRotation
    from ..noise.amplitude_damping import AmplitudeDamping
    from ..noise.t2_dephasing import Dephasing
    from ..noise.kraus_channels import CombinedChannel

    if channel is None:
        return 0.0

    if isinstance(channel, CombinedChannel):
        return combine_independent_probabilities(
            [channel_event_probability(sub) for sub in channel.channels]
        )

    if isinstance(channel, PauliError):
        return float(np.clip(channel.p_x + channel.p_y + channel.p_z, 0.0, 1.0))

    if isinstance(channel, CorrelatedPauliError):
        # probs are joint event probabilities; exclude explicit all-identity strings.
        return float(np.clip(
            sum(p for s, p in channel.probs.items() if any(c != "I" for c in s)),
            0.0,
            1.0,
        ))

    if isinstance(channel, (CoherentRotation, StochasticCoherentRotation)):
        # Reuse the existing Pauli-twirl bridge (sin²(ε) or sin²(std_dev)).
        # This is a visualization-compatible stochastic equivalent, not a claim
        # that coherent accumulation is fully reduced to Pauli noise.
        return channel_event_probability(channel.to_pauli_error())

    if isinstance(channel, AmplitudeDamping):
        return float(np.clip(channel.gamma, 0.0, 1.0))

    if isinstance(channel, Dephasing):
        return float(np.clip(channel.lam, 0.0, 1.0))

    return 0.0


def qualitative_burden_label(value: float) -> str:
    """Order-of-magnitude label for a raw error-burden value."""
    if value <= 0.0:
        return "None"
    if value < 1e-3:
        return "Trace"
    if value < 1e-2:
        return "Low"
    if value < 1e-1:
        return "Moderate"
    return "High"


def normalize_heat_values(
    values,
    *,
    mode: Literal["relative", "absolute", "absolute_log"] = "absolute_log",
    vmin: float = 1e-4,
    vmax: float = 1e-1,
    floor: float = 0.15,
    gamma: float = 1.0,
) -> np.ndarray:
    """Convert raw error-burden values into [0, 1] visual intensities.

    Parameters
    ----------
    values  : Array-like of non-negative floats.
    mode    : Normalization strategy.
              "relative"     — hottest nonzero value in this array → 1.0.
              "absolute"     — linear scale between vmin and vmax.
              "absolute_log" — log₁₀ scale between vmin and vmax (recommended
                               for probabilities spanning orders of magnitude).
    vmin, vmax : Bounds used by "absolute" and "absolute_log" modes.
    floor   : Minimum intensity for any element where value > 0.
    gamma   : Power applied after scaling (1.0 = linear; >1 compresses high end).

    Returns
    -------
    NumPy array of floats in [0, 1].  Zero where value == 0, >= floor where value > 0.
    """
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = arr > 0

    if not mask.any():
        return out

    if mode == "relative":
        denom = float(arr[mask].max())
        scaled = np.zeros_like(arr, dtype=float)
        if denom > 0:
            scaled[mask] = arr[mask] / denom

    elif mode == "absolute":
        if vmax <= vmin:
            raise ValueError("absolute normalization requires vmax > vmin")
        scaled = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    elif mode == "absolute_log":
        if vmin <= 0 or vmax <= vmin:
            raise ValueError("absolute_log normalization requires 0 < vmin < vmax")
        log_vmin = np.log10(vmin)
        log_vmax = np.log10(vmax)
        safe = np.maximum(arr, vmin)
        scaled = np.clip(
            (np.log10(safe) - log_vmin) / (log_vmax - log_vmin), 0.0, 1.0
        )
        scaled[~mask] = 0.0

    else:
        raise ValueError(f"Unknown heat normalization mode: {mode!r}")

    out[mask] = floor + (1.0 - floor) * scaled[mask] ** gamma
    return out
