# NoisiQ Backend Strategy: Options & Recommendation

## The Core Problem

NoisiQ currently uses dense $2^n \times 2^n$ matrix expansion to apply gates to a $2^n$-element state vector. This hits a hard wall around **12–14 qubits** (16 GB+ RAM). Your project abstract mentions wanting to scale to "larger circuits" and your schedule includes non-Pauli noise (amplitude damping, T1/T2), coherent errors, and Qiskit/Cirq adapters. The backend needs to support all of this without you having to build a general-purpose quantum simulator from scratch.

**Key insight for NoisiQ:** You are building a *visualization and teaching tool*, not a competitor to Qiskit Aer. This means you can strategically delegate heavy computation while owning the analysis, error tracking, and visualization layers.

---

## The Options

### Option A: Stim for Clifford Circuits (Pauli Frame Tracking)

**What it is:** Google's [Stim](https://github.com/quantumlib/Stim) is a purpose-built Clifford circuit simulator that tracks Pauli error propagation in polynomial time. It can simulate **millions of qubits** for Clifford-only circuits.

**How it works:** Instead of maintaining a $2^n$ state vector, Stim tracks an $n \times 2n$ stabilizer tableau. When a Clifford gate is applied, it conjugates the tableau (a simple table update). When a Pauli error occurs, it multiplies into a "Pauli frame." The `FlipSimulator` specifically tracks *only the errors*, giving you exactly the data NoisiQ needs for visualization.

**What NoisiQ would use:**
- `stim.FlipSimulator` — step through a circuit, call `peek_pauli_flips()` after each gate to see which qubits have X/Y/Z errors at each step. This is literally the data you need for your propagation animation.
- `stim.TableauSimulator` — interactive stepping for debugging and building reference states.

**Supported gates:** H, S, CNOT, CZ, X, Y, Z, SWAP, and all Clifford gates. **Does NOT support T, Toffoli, or arbitrary rotations.**

**Pros:**
- Scales to thousands of qubits trivially
- Gives you *exact* Pauli error propagation data — no sampling needed
- `peek_pauli_flips()` is literally "which qubit has what error right now" — perfect for animation
- Mature, well-tested, actively maintained by Google

**Cons:**
- No T gate support (not universal)
- No non-Pauli noise (amplitude damping, coherent errors) — you'd need something else for Weeks 6–7

**Best for:** Weeks 2–5 of your schedule (Pauli frame propagation, animation, Pauli noise channels, heatmaps). This covers the MVP beautifully.

---

### Option B: Qiskit Aer as the Computation Backend

**What it is:** Use Qiskit Aer's optimized C++ simulators for the actual state/density-matrix evolution, and have NoisiQ focus purely on analysis and visualization.

**How it works:** Qiskit Aer provides multiple simulation methods:
- **`statevector`** — full state vector, up to ~30 qubits (optimized C++)
- **`density_matrix`** — density matrix for mixed states with noise, up to ~15 qubits
- **`matrix_product_state` (MPS)** — tensor-network-based, scales to 50+ qubits for low-entanglement circuits
- **`stabilizer`** — Clifford-only, similar to Stim
- **`extended_stabilizer`** — Clifford + limited T gates

**What NoisiQ would use:**
- Convert NoisiQ circuits → Qiskit circuits (you're already building adapters in Week 8)
- Attach Qiskit noise models
- Run single-shot simulations, extract intermediate states
- NoisiQ owns the error tracking, propagation analysis, and visualization

**Supported gates:** Everything. Any unitary. Full noise model library (depolarizing, amplitude damping, T1/T2, coherent errors, SPAM).

**Pros:**
- Universal gate set — Clifford+T, arbitrary rotations, everything
- Built-in non-Pauli noise models (amplitude damping, thermal relaxation)
- MPS method gives better scaling for many practical circuits
- You're building a Qiskit adapter anyway (Week 8)
- Battle-tested, widely used, great documentation

**Cons:**
- You depend on Qiskit as a runtime dependency (it's large)
- State vector method still hits the wall at ~30 qubits even with C++
- Getting intermediate states (for step-by-step animation) requires running the circuit incrementally, which is slower
- You don't get "which Pauli error hit which qubit" tracking for free — you'd need to build that layer

**Best for:** Weeks 6–10 (non-Pauli noise, coherent errors, adapters, final package). Also a solid fallback for everything.

---

### Option C: Pauli Propagation (Heisenberg Picture)

**What it is:** Instead of evolving the state forward through the circuit (Schrödinger picture), you track how *observables* (represented as Pauli strings) evolve backward. This is the mathematical backbone for understanding how errors transform.

**How it works:** Given a Pauli string $P = X_0 Z_2$, conjugating through a Clifford gate $C$ gives you $C^\dagger P C = P'$, which is another Pauli string. For noise channels, Pauli strings get "contracted" (multiplied by a factor < 1). You track a weighted sum of Pauli strings rather than a $2^n$ state vector.

**What NoisiQ would use:**
- For each error injection point, propagate the Pauli error forward through the remaining circuit
- At each gate, the error transforms: $X \xrightarrow{H} Z$, $X \xrightarrow{\text{CNOT}} X \otimes X$, etc.
- This gives you the *exact* propagation path for visualization
- For Clifford circuits, this is exact. For non-Clifford, you truncate or approximate.

**Pros:**
- Extremely elegant for teaching — shows the physics directly
- Efficient for Clifford circuits (polynomial) 
- Natural fit for "error propagation animation" — you literally track error → gate → transformed error
- No external dependencies needed; you can implement the core in ~200 lines of NumPy

**Cons:**
- For non-Clifford gates (T), the Pauli string count can grow exponentially (a single T gate turns 1 Pauli string into 2)
- Implementing truncation strategies adds complexity
- Doesn't give you final state vectors (you'd need a separate method for measurement statistics)

**Best for:** The core "propagation animation" feature. This should be the conceptual backbone of how NoisiQ thinks about error propagation, regardless of which computation backend you choose.

---

### Option D: Tensor Networks via quimb

**What it is:** [quimb](https://quimb.readthedocs.io/) is a Python library for tensor network simulations. Instead of a monolithic $2^n$ state vector, the state is decomposed into a network of smaller tensors (typically Matrix Product States / MPS).

**How it works:** The state is represented as a chain of tensors, each handling a few qubits. Gates are applied by contracting tensors locally. The bond dimension (entanglement capacity) is bounded, so circuits with moderate entanglement can be simulated on 50–100+ qubits.

**Pros:**
- Scales much better than dense state vectors for many practical circuits
- Handles any gate set (universal)
- Can model noise via Kraus operators
- Pure Python + NumPy, integrates well

**Cons:**
- Performance degrades badly for highly entangled circuits (e.g., deep random circuits)
- More complex to implement correctly
- Intermediate state extraction for animation is non-trivial
- Overkill for your 10-week timeline

**Best for:** Future work beyond Week 10, if you want to push qubit counts without depending on Qiskit.

---

## Recommendation: Tiered Backend Architecture

Given your 10-week schedule, your focus on visualization/teaching, and your desire to support Clifford+T with future expansion, I recommend a **tiered approach** where NoisiQ selects the right backend based on the circuit:

```
┌─────────────────────────────────────────────────────┐
│                   NoisiQ User API                   │
│         (Circuit, Noise Models, Visualization)      │
├─────────────────────────────────────────────────────┤
│               Error Analysis Layer                  │
│   Pauli Propagation (Heisenberg picture tracking)   │
│   → Powers the propagation animation & heatmaps    │
├────────────┬────────────────────┬───────────────────┤
│  Backend 1 │     Backend 2      │    Backend 3      │
│   Stim     │   Qiskit Aer       │  (Future: quimb) │
│ (Clifford) │ (Universal gates)  │  (Large circuits)│
│ Weeks 2-5  │    Weeks 6-10      │    Post v1.0     │
└────────────┴────────────────────┴───────────────────┘
```

### Mapping to Your Schedule

| Week | Backend Used | Why |
|------|-------------|-----|
| 2–3 | **Stim** (`FlipSimulator`) | Pauli frame propagation + animation. `peek_pauli_flips()` gives you exactly the per-gate error data you need. |
| 4 | **Stim** + Pauli Propagation | Many-shot sampling via Stim, aggregate via your own analysis layer. Heatmaps from Pauli propagation. |
| 5 | **Both** | Cross-validate Stim results against Qiskit Aer on small Clifford circuits. |
| 6 | **Qiskit Aer** (`density_matrix`) | Amplitude damping / T1/T2 requires non-Pauli channels — Qiskit has these built-in. |
| 7 | **Qiskit Aer** (`statevector`) | Coherent errors (over/under-rotation) are non-Clifford — need Aer. |
| 8 | **Qiskit adapter** | You're building the adapter anyway — now it also serves as the non-Clifford backend path. |
| 9–10 | **All** | Suppression demos use whichever backend fits the circuit. Polish. |

### What NoisiQ Owns vs. Delegates

| NoisiQ owns | Delegated to external |
|---|---|
| Circuit IR (your `Circuit` class) | State vector / tableau evolution (Stim or Aer) |
| Noise model specification & placement | Actual noise channel math (leverage Stim/Aer implementations) |
| Error propagation analysis (Pauli tracking) | Dense matrix multiplication |
| Visualization (animation, heatmaps) | GPU acceleration, MPS |
| Adapters (Qiskit ↔ NoisiQ conversion) | — |

### The Key Abstraction

Define a simple `Backend` protocol/ABC that all backends implement:

```python
class Backend(Protocol):
    def run_single_shot(
        self, circuit: Circuit, noise_config: NoiseConfig, seed: int
    ) -> SimulationResult:
        """Run one shot, return final state + error history."""
        ...

    def get_error_propagation(
        self, circuit: Circuit, noise_config: NoiseConfig
    ) -> list[PauliFrame]:
        """Return the Pauli frame at each time step (for animation)."""
        ...
```

Your existing `PauliFrameBackend` becomes one implementation. `StimBackend` and `QiskitAerBackend` become others. The visualization layer doesn't care which one ran — it just consumes the `PauliFrame` data.

---

## Summary Table

| Criterion | Stim | Qiskit Aer | Pauli Propagation | quimb |
|---|---|---|---|---|
| **Gate support** | Clifford only | Universal | Clifford exact, non-Clifford approx | Universal |
| **Max qubits** | Millions | ~30 (SV), 50+ (MPS) | Depends on Pauli count | 50–100+ |
| **Noise types** | Pauli only | All (Pauli, amp. damp, T1/T2, coherent) | Pauli exact, others approx | All (Kraus) |
| **Error tracking** | Native (`FlipSimulator`) | Must build yourself | Native (this IS error tracking) | Must build yourself |
| **Animation data** | `peek_pauli_flips()` | Run circuit incrementally | Track Pauli string at each step | Complex |
| **Install size** | Lightweight (C++) | Heavy (Qiskit ecosystem) | Zero (pure NumPy) | Moderate |
| **Implementation effort** | Low (use API) | Low (use API) | Medium (~200 lines core) | High |
| **Timeline fit** | Weeks 2–5 | Weeks 6–10 | Core analysis layer | Post v1.0 |

> [!TIP]
> **Start with Stim for the MVP.** It gives you the fastest path to a working propagation animation with real physics. Then layer in Qiskit Aer for non-Clifford/non-Pauli features. The Pauli Propagation approach should be the *conceptual* model that unifies both — it's what your visualization layer actually renders, regardless of which backend computed it.

> [!IMPORTANT]
> **Regarding Clifford+T specifically:** For circuits with a small number of T gates in a mostly-Clifford circuit, Stim can handle the Clifford parts while you use Qiskit Aer's `extended_stabilizer` for the T-gate segments. This hybrid approach lets you push to larger qubit counts than pure state-vector simulation while still supporting the full Clifford+T gate set.
x