# Concepts and conventions

## The pipeline

```
Circuit ──┐
          ├─► compile_program() ──► backend.run() ──► Result ──► plots / metrics
NoiseModel┘     (one flat list of steps)                │
                                                        └─► PropagationTrace ──► animations
```

1. **`Circuit`** (`noisiq.ir`) — instructions placed in layers.
2. **`NoiseModel`** (`noisiq.noise`) — rules saying which channels follow which instructions.
3. **`compile_program`** (`noisiq.backends.program`) — resolves the rules against the circuit
   into a flat list of steps: gates, *fault locations* (a channel on some qubits, attributed to
   an instruction or to an idle `(layer, qubit)`), measurements, resets, detuning, layer ends.
   Scheduling, durations and idle-noise logic exist only here.
4. **Backends** execute the program and all return the same **`Result`**.
5. **Visualization** reads `Result` / `PropagationTrace`, never backend internals.

## Conventions

| Topic | Convention |
|---|---|
| Qubit order in matrices | **Big-endian**: for a gate on qubits `(a, b)`, `a` is the most significant bit. `CX(control, target)` has the textbook matrix. Statevectors index `|q0 q1 … q_{n-1}⟩` with `q0` most significant. |
| Bitstrings | The leftmost character of a `counts` key is classical bit 0. |
| Implicit measurement | A circuit with no measurements is measured in Z at the end; classical bit `k` = qubit `k` (`result.implicit_measurement`). |
| Rotations | `R_P(θ) = exp(−iθP/2)`; `P(θ) = diag(1, e^{iθ})`; `U(θ, φ, λ)` is OpenQASM's `u3`. |
| Layers | ASAP scheduling; two instructions never share a qubit in a layer; execution order is `(layer, insertion order)`. |
| Time | A layer lasts as long as its slowest instruction. Qubits busy for less than the layer idle for the rest. Idle noise is applied from the start of the circuit up to each qubit's last instruction. |
| Conditions | `Condition(clbits, value)`: `clbits[0]` is the least significant bit of `value` (OpenQASM convention). Unwritten bits read as 0. |
| Faults | Noise is applied **after** the instruction it is attached to. A 1-qubit channel on a k-qubit gate acts independently on each qubit; a k-qubit channel acts jointly. |
| Readout error | Flips the *recorded* bit (also seen by feed-forward); the qubit's state is unaffected. |

## Errors, faults and frames

* A **fault** is one sampled non-identity event at a fault location (e.g. a Z after a CX on
  qubit 1). `Result.fault_rates()` counts these.
* The **error frame** (Clifford circuits) is the Pauli operator relating the noisy state to
  the ideal one. Faults are injected into it and gates conjugate it. `PropagationTrace`
  stores it after every layer — that is what the animations draw.
* A fault is **harmful** only if it eventually flips a measured bit. `Result.output_error_rate()`
  counts harmful outcomes; `fault_free_fraction` counts shots with no fault at all.
* For non-Clifford circuits a Pauli error does not stay a Pauli error, so the statevector
  backend reports a **deviation** per qubit instead: `1 − F(ρ_q, ρ_q^ideal)`.

## Exact vs. twirled noise

Every channel has an exact Kraus form and a **Pauli twirl** (same process fidelity, but only
stochastic Paulis). Twirled noise is fast and scales to large circuits, but loses coherent
build-up (`N` over-rotations: error `∝ N²` exactly vs `∝ N` twirled) and shot-to-shot
correlations (quasi-static drift that dynamical decoupling can echo away). Use
`representation="exact"` (statevector) to study those effects and `"pauli"` (frame) for
large circuits and heatmaps.
