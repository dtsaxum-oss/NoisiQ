"""The single source of truth for the math behind every noise channel and metric.

``noisiq.explain("amplitude_damping")`` prints an entry; ``python -m noisiq.catalog``
regenerates ``docs/math.md`` from this file, so code, docstrings and documentation can
never drift apart. When you add a channel or metric, add an entry here.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

__all__ = ["Entry", "CATALOG", "explain", "to_markdown"]


@dataclass(frozen=True)
class Entry:
    key: str
    title: str
    kind: str  # "channel" | "metric" | "model"
    summary: str
    math: str
    parameters: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    api: str = ""
    references: tuple[str, ...] = ()


_E = [
    # ----------------------------------------------------------------- channels
    Entry(
        "pauli",
        "Pauli channel",
        "channel",
        "Applies a Pauli operator P with probability p_P; the identity otherwise.",
        """
        E(ρ) = (1 − Σ_P p_P) ρ + Σ_{P ≠ I} p_P · P ρ P
        Error probability (process infidelity): p_err = Σ_{P≠I} p_P
        Multi-qubit strings (e.g. "ZZ") model correlated errors such as crosstalk.
        """,
        {"probs": "mapping Pauli string → probability (Σ ≤ 1)"},
        "Pauli channels are simulated natively by the stabilizer (frame) backend, at any size.",
        "nq.noise.PauliChannel, nq.noise.pauli_channel(p_x, p_y, p_z)",
    ),
    Entry(
        "depolarizing",
        "Depolarizing channel",
        "channel",
        "With probability p, replace the n-qubit error by a uniformly random non-identity Pauli.",
        """
        E(ρ) = (1 − p) ρ + p / (4ⁿ − 1) · Σ_{P ≠ I} P ρ P
        p is the process infidelity. Average gate infidelity: r = p · d / (d + 1), d = 2ⁿ
        (1 qubit: r = 2p/3;  2 qubits: r = 4p/5).
        Relation to Qiskit's depolarizing_error(λ): p = λ (4ⁿ − 1) / 4ⁿ.
        """,
        {"p": "total error probability in [0, 1]", "num_qubits": "1 or more"},
        "Use NoiseModel.depolarizing_from_infidelity(r) to convert a datasheet infidelity.",
        "nq.noise.depolarizing(p, num_qubits=1)",
    ),
    Entry(
        "bit_flip",
        "Bit flip",
        "channel",
        "Flips |0⟩↔|1⟩ with probability p.",
        "E(ρ) = (1 − p) ρ + p · X ρ X",
        {"p": "flip probability"},
        "Invisible to phase (X-basis) information, detected by Z-basis measurements.",
        "nq.noise.bit_flip(p)",
    ),
    Entry(
        "phase_flip",
        "Phase flip (Pauli dephasing)",
        "channel",
        "Applies Z with probability p.",
        """
        E(ρ) = (1 − p) ρ + p · Z ρ Z
        Off-diagonal elements shrink by (1 − 2p); populations are untouched.
        """,
        {"p": "flip probability"},
        "Invisible to Z-basis measurements; detected in the X basis (measure(..., basis='X')).",
        "nq.noise.phase_flip(p)",
    ),
    Entry(
        "amplitude_damping",
        "Amplitude damping (T1)",
        "channel",
        "Energy relaxation: |1⟩ decays to |0⟩ with probability γ.",
        """
        K0 = [[1, 0], [0, √(1−γ)]],   K1 = [[0, √γ], [0, 0]]
        ρ11 → (1−γ) ρ11,   ρ01 → √(1−γ) ρ01
        From a relaxation time: γ = 1 − exp(−t / T1)
        Pauli twirl: p_X = p_Y = γ/4,  p_Z = (1 − √(1−γ))/2 − γ/4
        """,
        {"gamma": "decay probability", "t1, duration": "via AmplitudeDamping.from_t1"},
        "Not a Pauli channel: it is non-unital (pushes every state towards |0⟩).",
        "nq.noise.AmplitudeDamping(gamma), AmplitudeDamping.from_t1(t1, duration)",
        ("Nielsen & Chuang, §8.3.5",),
    ),
    Entry(
        "phase_damping",
        "Phase damping (pure dephasing, Tφ)",
        "channel",
        "Loss of phase coherence without energy exchange.",
        """
        K0 = [[1, 0], [0, √(1−λ)]],   K1 = [[0, 0], [0, √λ]]
        ρ01 → √(1−λ) ρ01;  populations unchanged.
        From a pure-dephasing time: √(1−λ) = exp(−t/Tφ)  ⇒  λ = 1 − exp(−2t/Tφ)
        Equivalent to a phase flip with p = (1 − √(1−λ)) / 2.
        """,
        {"lam": "damping parameter λ"},
        "",
        "nq.noise.PhaseDamping(lam), PhaseDamping.from_tphi(tphi, duration)",
        ("Nielsen & Chuang, §8.3.6",),
    ),
    Entry(
        "thermal_relaxation",
        "Thermal relaxation (T1 + T2)",
        "channel",
        "Combined energy relaxation and dephasing of a qubit for a time t (zero temperature).",
        """
        ρ11(t) = ρ11 · exp(−t/T1)
        ρ01(t) = ρ01 · exp(−t/T2)            (requires T2 ≤ 2·T1)
        Implemented as amplitude damping γ = 1 − exp(−t/T1) followed by phase damping with
            1/Tφ = 1/T2 − 1/(2·T1)
        so the T1 contribution to dephasing is counted exactly once.
        """,
        {"t1": "relaxation time (s)", "t2": "coherence time (s)", "duration": "exposure time (s)"},
        "Used by NoiseModel.add_thermal_relaxation for gates (gate duration) and idle qubits.",
        "nq.noise.ThermalRelaxation(t1, t2, duration)",
    ),
    Entry(
        "coherent",
        "Coherent rotation error",
        "channel",
        "A systematic over/under-rotation U = exp(−i θ P / 2) about a Pauli axis P.",
        """
        U = cos(θ/2) I − i sin(θ/2) P
        Pauli twirl: p_P = sin²(θ/2)
        N repetitions add in amplitude: error ≈ sin²(Nθ/2) ≈ N²θ²/4, versus N·θ²/4 for the
        twirled (stochastic) model — coherent errors can be much worse in deep circuits.
        """,
        {"axis": "Pauli string, e.g. 'Z' or 'ZZ'", "angle": "θ in radians"},
        "Simulated exactly by the statevector backend; twirled for the stabilizer backend.",
        "nq.noise.CoherentRotation(axis, angle)",
    ),
    Entry(
        "kraus",
        "General Kraus channel",
        "channel",
        "Any CPTP map written in operator-sum form.",
        """
        E(ρ) = Σ_k K_k ρ K_k†,   with  Σ_k K_k† K_k = I  (trace preservation)
        Exact Pauli twirl: p_P = Σ_k |Tr(P K_k)|² / d²
        Trajectory sampling: operator k is chosen with probability ‖K_k |ψ⟩‖².
        """,
        {"operators": "list of 2ⁿ×2ⁿ matrices"},
        "",
        "nq.noise.KrausChannel(operators)",
    ),
    Entry(
        "readout",
        "Readout (measurement assignment) error",
        "channel",
        "The recorded classical bit is wrong with some probability.",
        """
        P(report 1 | true 0) = p01,   P(report 0 | true 1) = p10
        Confusion matrix M[reported, true] = [[1−p01, p10], [p01, 1−p10]]
        """,
        {"p01, p10": "assignment error probabilities"},
        "Affects the reported bit only (including feed-forward), not the post-measurement state.",
        "NoiseModel.add_readout_error(p01, p10=None, qubits=None)",
    ),
    Entry(
        "quasistatic",
        "Quasi-static detuning (low-frequency noise)",
        "channel",
        "Each shot draws a fixed frequency offset δ ~ N(0, σ²); over time τ the qubit picks up RZ(δτ).",
        """
        Shot-averaged coherence after free evolution τ:  E[e^{iδτ}] = exp(−σ²τ²/2)  (Gaussian decay)
        A spin echo (X at τ/2) cancels the phase exactly because δ is constant within a shot.
        Pauli-twirled per interval: phase flip with p = (1 − exp(−σ²τ²/2)) / 2 (loses the correlation).
        """,
        {"sigma": "standard deviation of δ in rad/s"},
        "This is the noise dynamical decoupling is designed to remove; see nq.suppression.apply_dd.",
        "NoiseModel.add_quasistatic_detuning(sigma, qubits=None)",
    ),
    Entry(
        "hardware_profile",
        "From datasheet figures to a noise model",
        "model",
        "How HardwareProfile.noise_model() converts published device numbers.",
        """
        1. Every gate of duration t_g and every idle interval gets ThermalRelaxation(T1, T2, t).
        2. Datasheet gate infidelity r already includes decoherence, so the depolarizing part is
               r_dep = max(0, r − r_thermal),   r_thermal = (1 − F_pro,thermal) · d/(d+1)
           converted to a process error p = r_dep · (d+1)/d.
        3. A fraction f (coherent_fraction) of p becomes a coherent rotation with sin²(θ/2) = f·p.
        4. Readout error → symmetric classical flip.
        """,
        {},
        "",
        "nq.noise.get_hardware(name).noise_model(representation='exact'|'pauli')",
    ),
    # ------------------------------------------------------------------ metrics
    Entry(
        "fault_rate",
        "Fault rate (per instruction)",
        "metric",
        "How often a noise channel actually fired at a given gate/qubit, estimated from shots.",
        """
        rate(g, q) = (#shots in which a non-identity fault was sampled at gate g on qubit q) / shots
        Std. error: √(rate (1 − rate) / shots)
        For coherent (unitary) errors in the statevector backend the per-shot contribution is
        the deviation probability 1 − |⟨ψ|U|ψ⟩|² instead of 0/1.
        """,
        {},
        "Drives the heatmap halos (metric='rate').",
        "Result.fault_rates()",
    ),
    Entry(
        "error_burden",
        "Error burden (rate × spread)",
        "metric",
        "Expected number of output qubits corrupted by faults originating at an instruction.",
        """
        burden(g) = Σ_q Σ_P  rate_P(g, q) · w(propagate(P at g, q → end of circuit))
        w(E) = number of measured bits flipped by E, plus the weight of E on qubits that are never
               measured afterwards. Propagation through Clifford gates is exact (U P U†).
        Share(g) = burden(g) / Σ_g' burden(g')   (shown as % in the error budget chart)
        """,
        {},
        "Available for Clifford circuits (frame backend).",
        "Result.error_budget()",
    ),
    Entry(
        "fault_free_fraction",
        "Fault-free fraction",
        "metric",
        "Fraction of shots in which no fault was sampled anywhere.",
        "F_ff = (#shots with zero faults) / shots;  ≈ Π_g (1 − p_g) for independent faults",
        {},
        "A lower bound on success probability; not a state fidelity.",
        "Result.fault_free_fraction",
    ),
    Entry(
        "output_error_rate",
        "Output (logical) error rate",
        "metric",
        "Fraction of shots whose measured bits differ from the ideal run with the same measurement randomness.",
        """
        For each shot the frame backend tracks the error frame E (noisy ⊕ ideal). A measurement in
        basis B is flipped iff E anticommutes with B. output_error_rate = P(any measured bit flipped).
        """,
        {},
        "Frame backend only. Unlike the fault-free fraction it ignores harmless faults.",
        "Result.output_error_rate()",
    ),
    Entry(
        "state_fidelity",
        "State fidelity",
        "metric",
        "Overlap between the simulated state and the ideal state.",
        """
        Pure ideal |ψ⟩:  F = ⟨ψ| ρ |ψ⟩
        General:         F(ρ, σ) = (Tr √(√ρ σ √ρ))²
        Trajectory estimate: F = mean over shots of |⟨ψ_ideal|ψ_shot⟩|²
        """,
        {},
        "",
        "nq.metrics.state_fidelity(a, b), Result.fidelity",
    ),
    Entry(
        "purity",
        "Purity",
        "metric",
        "How mixed a state is.",
        """
        P(ρ) = Tr(ρ²) ∈ [1/d, 1];  1 for pure states.
        Single-qubit marginal purity uses the reduced state ρ_q = Tr_{others}(ρ); for an
        entangled pure state the marginal purity is < 1 even without noise.
        """,
        {},
        "",
        "nq.metrics.purity(rho), Result.qubit_purities()",
    ),
    Entry(
        "hellinger",
        "Hellinger fidelity / total variation distance",
        "metric",
        "Distances between two measurement distributions (e.g. noisy vs ideal counts).",
        """
        Hellinger fidelity: H(p, q) = (Σ_x √(p_x q_x))²
        Total variation distance: TVD(p, q) = ½ Σ_x |p_x − q_x|
        """,
        {},
        "",
        "nq.metrics.hellinger_fidelity, nq.metrics.total_variation_distance, Result.compare",
    ),
    Entry(
        "ghz_fidelity",
        "GHZ fidelity",
        "metric",
        "Fidelity with (|0…0⟩ + |1…1⟩)/√2.",
        """
        F = ½ (ρ_{0…0,0…0} + ρ_{1…1,1…1}) + Re ρ_{0…0,1…1}
        Populations P = ρ_{00,00} + ρ_{11,11} and coherence C = 2|ρ_{0…0,1…1}| give F = (P + C)/2
        when the coherence phase is zero.
        """,
        {},
        "",
        "nq.metrics.ghz_fidelity(rho)",
    ),
]

CATALOG: dict[str, Entry] = {e.key: e for e in _E}

_SYNONYMS = {
    "t1": "amplitude_damping",
    "t2": "thermal_relaxation",
    "dephasing": "phase_flip",
    "decoherence": "thermal_relaxation",
    "depolarising": "depolarizing",
    "spam": "readout",
    "measurement": "readout",
    "coherent_rotation": "coherent",
    "over_rotation": "coherent",
    "fidelity": "state_fidelity",
    "tvd": "hellinger",
    "burden": "error_burden",
    "pauli_channel": "pauli",
    "kraus_channel": "kraus",
    "thermalrelaxation": "thermal_relaxation",
    "amplitudedamping": "amplitude_damping",
    "phasedamping": "phase_damping",
    "1/f": "quasistatic",
}


def _lookup(name: str) -> Entry:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = _SYNONYMS.get(key, key)
    if key in CATALOG:
        return CATALOG[key]
    matches = [e for e in CATALOG.values() if key in e.key or key in e.title.lower()]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"No catalog entry for {name!r}. Try one of: {', '.join(sorted(CATALOG))}")


def _render(e: Entry) -> str:
    lines = [f"{e.title}  [{e.kind}: {e.key}]", "=" * (len(e.title) + len(e.kind) + len(e.key) + 6), e.summary, ""]
    lines.append(textwrap.dedent(e.math).strip("\n"))
    if e.parameters:
        lines += ["", "Parameters:"] + [f"  {k}: {v}" for k, v in e.parameters.items()]
    if e.notes:
        lines += ["", e.notes]
    if e.api:
        lines += ["", f"API: {e.api}"]
    if e.references:
        lines += ["", "References: " + "; ".join(e.references)]
    return "\n".join(lines)


def explain(name: object = None) -> str:
    """Print (and return) the math for a channel, metric or noise object.

    Args:
        name: A catalog key (``"amplitude_damping"``), a synonym (``"T1"``), or a
            channel instance. ``None`` lists all entries.
    """
    if name is None:
        text = "\n".join(f"{e.key:<22} {e.title}" for e in CATALOG.values())
    else:
        if not isinstance(name, str):
            name = getattr(name, "catalog_key", type(name).__name__)
        text = _render(_lookup(name))
    print(text)
    return text


def to_markdown() -> str:
    """Render the whole catalog as Markdown (used to build ``docs/math.md``)."""
    out = [
        "# Noise channels and metrics — the math",
        "",
        "_Generated from `src/noisiq/catalog.py` by `python -m noisiq.catalog`. Do not edit by hand._",
        "",
    ]
    for kind, heading in (("channel", "Noise channels"), ("model", "Noise models"), ("metric", "Metrics")):
        out += [f"## {heading}", ""]
        for e in CATALOG.values():
            if e.kind != kind:
                continue
            out += [
                f"### {e.title}",
                "",
                f"`{e.key}` — {e.summary}",
                "",
                "```text",
                textwrap.dedent(e.math).strip("\n"),
                "```",
                "",
            ]
            if e.parameters:
                out += ["| Parameter | Meaning |", "|---|---|"]
                out += [f"| `{k}` | {v} |" for k, v in e.parameters.items()] + [""]
            if e.notes:
                out += [e.notes, ""]
            if e.api:
                out += [f"**API:** `{e.api}`", ""]
            if e.references:
                out += ["**References:** " + "; ".join(e.references), ""]
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/math.md")
    target.write_text(to_markdown(), encoding="utf-8")
    print(f"wrote {target}")
