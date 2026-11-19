# Noise channels and metrics — the math

_Generated from `src/noisiq/catalog.py` by `python -m noisiq.catalog`. Do not edit by hand._

## Noise channels

### Pauli channel

`pauli` — Applies a Pauli operator P with probability p_P; the identity otherwise.

```text
E(ρ) = (1 − Σ_P p_P) ρ + Σ_{P ≠ I} p_P · P ρ P
Error probability (process infidelity): p_err = Σ_{P≠I} p_P
Multi-qubit strings (e.g. "ZZ") model correlated errors such as crosstalk.
```

| Parameter | Meaning |
|---|---|
| `probs` | mapping Pauli string → probability (Σ ≤ 1) |

Pauli channels are simulated natively by the stabilizer (frame) backend, at any size.

**API:** `nq.noise.PauliChannel, nq.noise.pauli_channel(p_x, p_y, p_z)`

### Depolarizing channel

`depolarizing` — With probability p, replace the n-qubit error by a uniformly random non-identity Pauli.

```text
E(ρ) = (1 − p) ρ + p / (4ⁿ − 1) · Σ_{P ≠ I} P ρ P
p is the process infidelity. Average gate infidelity: r = p · d / (d + 1), d = 2ⁿ
(1 qubit: r = 2p/3;  2 qubits: r = 4p/5).
Relation to Qiskit's depolarizing_error(λ): p = λ (4ⁿ − 1) / 4ⁿ.
```

| Parameter | Meaning |
|---|---|
| `p` | total error probability in [0, 1] |
| `num_qubits` | 1 or more |

Use NoiseModel.depolarizing_from_infidelity(r) to convert a datasheet infidelity.

**API:** `nq.noise.depolarizing(p, num_qubits=1)`

### Bit flip

`bit_flip` — Flips |0⟩↔|1⟩ with probability p.

```text
E(ρ) = (1 − p) ρ + p · X ρ X
```

| Parameter | Meaning |
|---|---|
| `p` | flip probability |

Invisible to phase (X-basis) information, detected by Z-basis measurements.

**API:** `nq.noise.bit_flip(p)`

### Phase flip (Pauli dephasing)

`phase_flip` — Applies Z with probability p.

```text
E(ρ) = (1 − p) ρ + p · Z ρ Z
Off-diagonal elements shrink by (1 − 2p); populations are untouched.
```

| Parameter | Meaning |
|---|---|
| `p` | flip probability |

Invisible to Z-basis measurements; detected in the X basis (measure(..., basis='X')).

**API:** `nq.noise.phase_flip(p)`

### Amplitude damping (T1)

`amplitude_damping` — Energy relaxation: |1⟩ decays to |0⟩ with probability γ.

```text
K0 = [[1, 0], [0, √(1−γ)]],   K1 = [[0, √γ], [0, 0]]
ρ11 → (1−γ) ρ11,   ρ01 → √(1−γ) ρ01
From a relaxation time: γ = 1 − exp(−t / T1)
Pauli twirl: p_X = p_Y = γ/4,  p_Z = (1 − √(1−γ))/2 − γ/4
```

| Parameter | Meaning |
|---|---|
| `gamma` | decay probability |
| `t1, duration` | via AmplitudeDamping.from_t1 |

Not a Pauli channel: it is non-unital (pushes every state towards |0⟩).

**API:** `nq.noise.AmplitudeDamping(gamma), AmplitudeDamping.from_t1(t1, duration)`

**References:** Nielsen & Chuang, §8.3.5

### Phase damping (pure dephasing, Tφ)

`phase_damping` — Loss of phase coherence without energy exchange.

```text
K0 = [[1, 0], [0, √(1−λ)]],   K1 = [[0, 0], [0, √λ]]
ρ01 → √(1−λ) ρ01;  populations unchanged.
From a pure-dephasing time: √(1−λ) = exp(−t/Tφ)  ⇒  λ = 1 − exp(−2t/Tφ)
Equivalent to a phase flip with p = (1 − √(1−λ)) / 2.
```

| Parameter | Meaning |
|---|---|
| `lam` | damping parameter λ |

**API:** `nq.noise.PhaseDamping(lam), PhaseDamping.from_tphi(tphi, duration)`

**References:** Nielsen & Chuang, §8.3.6

### Thermal relaxation (T1 + T2)

`thermal_relaxation` — Combined energy relaxation and dephasing of a qubit for a time t (zero temperature).

```text
ρ11(t) = ρ11 · exp(−t/T1)
ρ01(t) = ρ01 · exp(−t/T2)            (requires T2 ≤ 2·T1)
Implemented as amplitude damping γ = 1 − exp(−t/T1) followed by phase damping with
    1/Tφ = 1/T2 − 1/(2·T1)
so the T1 contribution to dephasing is counted exactly once.
```

| Parameter | Meaning |
|---|---|
| `t1` | relaxation time (s) |
| `t2` | coherence time (s) |
| `duration` | exposure time (s) |

Used by NoiseModel.add_thermal_relaxation for gates (gate duration) and idle qubits.

**API:** `nq.noise.ThermalRelaxation(t1, t2, duration)`

### Coherent rotation error

`coherent` — A systematic over/under-rotation U = exp(−i θ P / 2) about a Pauli axis P.

```text
U = cos(θ/2) I − i sin(θ/2) P
Pauli twirl: p_P = sin²(θ/2)
N repetitions add in amplitude: error ≈ sin²(Nθ/2) ≈ N²θ²/4, versus N·θ²/4 for the
twirled (stochastic) model — coherent errors can be much worse in deep circuits.
```

| Parameter | Meaning |
|---|---|
| `axis` | Pauli string, e.g. 'Z' or 'ZZ' |
| `angle` | θ in radians |

Simulated exactly by the statevector backend; twirled for the stabilizer backend.

**API:** `nq.noise.CoherentRotation(axis, angle)`

### General Kraus channel

`kraus` — Any CPTP map written in operator-sum form.

```text
E(ρ) = Σ_k K_k ρ K_k†,   with  Σ_k K_k† K_k = I  (trace preservation)
Exact Pauli twirl: p_P = Σ_k |Tr(P K_k)|² / d²
Trajectory sampling: operator k is chosen with probability ‖K_k |ψ⟩‖².
```

| Parameter | Meaning |
|---|---|
| `operators` | list of 2ⁿ×2ⁿ matrices |

**API:** `nq.noise.KrausChannel(operators)`

### Readout (measurement assignment) error

`readout` — The recorded classical bit is wrong with some probability.

```text
P(report 1 | true 0) = p01,   P(report 0 | true 1) = p10
Confusion matrix M[reported, true] = [[1−p01, p10], [p01, 1−p10]]
```

| Parameter | Meaning |
|---|---|
| `p01, p10` | assignment error probabilities |

Affects the reported bit only (including feed-forward), not the post-measurement state.

**API:** `NoiseModel.add_readout_error(p01, p10=None, qubits=None)`

### Quasi-static detuning (low-frequency noise)

`quasistatic` — Each shot draws a fixed frequency offset δ ~ N(0, σ²); over time τ the qubit picks up RZ(δτ).

```text
Shot-averaged coherence after free evolution τ:  E[e^{iδτ}] = exp(−σ²τ²/2)  (Gaussian decay)
A spin echo (X at τ/2) cancels the phase exactly because δ is constant within a shot.
Pauli-twirled per interval: phase flip with p = (1 − exp(−σ²τ²/2)) / 2 (loses the correlation).
```

| Parameter | Meaning |
|---|---|
| `sigma` | standard deviation of δ in rad/s |

This is the noise dynamical decoupling is designed to remove; see nq.suppression.apply_dd.

**API:** `NoiseModel.add_quasistatic_detuning(sigma, qubits=None)`

## Noise models

### From datasheet figures to a noise model

`hardware_profile` — How HardwareProfile.noise_model() converts published device numbers.

```text
1. Every gate of duration t_g and every idle interval gets ThermalRelaxation(T1, T2, t).
2. Datasheet gate infidelity r already includes decoherence, so the depolarizing part is
       r_dep = max(0, r − r_thermal),   r_thermal = (1 − F_pro,thermal) · d/(d+1)
   converted to a process error p = r_dep · (d+1)/d.
3. A fraction f (coherent_fraction) of p becomes a coherent rotation with sin²(θ/2) = f·p.
4. Readout error → symmetric classical flip.
```

**API:** `nq.noise.get_hardware(name).noise_model(representation='exact'|'pauli')`

## Metrics

### Fault rate (per instruction)

`fault_rate` — How often a noise channel actually fired at a given gate/qubit, estimated from shots.

```text
rate(g, q) = (#shots in which a non-identity fault was sampled at gate g on qubit q) / shots
Std. error: √(rate (1 − rate) / shots)
For coherent (unitary) errors in the statevector backend the per-shot contribution is
the deviation probability 1 − |⟨ψ|U|ψ⟩|² instead of 0/1.
```

Drives the heatmap halos (metric='rate').

**API:** `Result.fault_rates()`

### Error burden (rate × spread)

`error_burden` — Expected number of output qubits corrupted by faults originating at an instruction.

```text
burden(g) = Σ_q Σ_P  rate_P(g, q) · w(propagate(P at g, q → end of circuit))
w(E) = number of measured bits flipped by E, plus the weight of E on qubits that are never
       measured afterwards. Propagation through Clifford gates is exact (U P U†).
Share(g) = burden(g) / Σ_g' burden(g')   (shown as % in the error budget chart)
```

Available for Clifford circuits (frame backend).

**API:** `Result.error_budget()`

### Fault-free fraction

`fault_free_fraction` — Fraction of shots in which no fault was sampled anywhere.

```text
F_ff = (#shots with zero faults) / shots;  ≈ Π_g (1 − p_g) for independent faults
```

A lower bound on success probability; not a state fidelity.

**API:** `Result.fault_free_fraction`

### Output (logical) error rate

`output_error_rate` — Fraction of shots whose measured bits differ from the ideal run with the same measurement randomness.

```text
For each shot the frame backend tracks the error frame E (noisy ⊕ ideal). A measurement in
basis B is flipped iff E anticommutes with B. output_error_rate = P(any measured bit flipped).
```

Frame backend only. Unlike the fault-free fraction it ignores harmless faults.

**API:** `Result.output_error_rate()`

### State fidelity

`state_fidelity` — Overlap between the simulated state and the ideal state.

```text
Pure ideal |ψ⟩:  F = ⟨ψ| ρ |ψ⟩
General:         F(ρ, σ) = (Tr √(√ρ σ √ρ))²
Trajectory estimate: F = mean over shots of |⟨ψ_ideal|ψ_shot⟩|²
```

**API:** `nq.metrics.state_fidelity(a, b), Result.fidelity`

### Purity

`purity` — How mixed a state is.

```text
P(ρ) = Tr(ρ²) ∈ [1/d, 1];  1 for pure states.
Single-qubit marginal purity uses the reduced state ρ_q = Tr_{others}(ρ); for an
entangled pure state the marginal purity is < 1 even without noise.
```

**API:** `nq.metrics.purity(rho), Result.qubit_purities()`

### Hellinger fidelity / total variation distance

`hellinger` — Distances between two measurement distributions (e.g. noisy vs ideal counts).

```text
Hellinger fidelity: H(p, q) = (Σ_x √(p_x q_x))²
Total variation distance: TVD(p, q) = ½ Σ_x |p_x − q_x|
```

**API:** `nq.metrics.hellinger_fidelity, nq.metrics.total_variation_distance, Result.compare`

### GHZ fidelity

`ghz_fidelity` — Fidelity with (|0…0⟩ + |1…1⟩)/√2.

```text
F = ½ (ρ_{0…0,0…0} + ρ_{1…1,1…1}) + Re ρ_{0…0,1…1}
Populations P = ρ_{00,00} + ρ_{11,11} and coherence C = 2|ρ_{0…0,1…1}| give F = (P + C)/2
when the coherence phase is zero.
```

**API:** `nq.metrics.ghz_fidelity(rho)`
