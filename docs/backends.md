# Backends

| backend | gates | noise | size | output highlights |
|---|---|---|---|---|
| `frame` | Clifford (any Clifford matrix, incl. custom) | Pauli (or twirled with `approximate=True`) | hundreds of qubits | exact Pauli error frames, output error rate, fault burden |
| `statevector` | any | any channel, coherent errors, quasi-static drift | ≤ 26 qubits | fidelity to ideal, reduced states, deviation traces |
| `density_matrix` | any | any Markovian channel | ≤ 10 qubits | exact probabilities (no sampling), exact density matrix |
| `aer` (optional) | any | any Markovian channel | ≤ ~25 qubits | cross-check against Qiskit Aer |

All support mid-circuit measurement in X/Y/Z, reset, readout error and feed-forward
(`frame`: classically controlled **Pauli** gates only; `aer`: single-bit conditions).

`nq.select_backend(circuit, noise)` returns the automatic choice and the reason.

## frame — vectorized Pauli frames

One noiseless reference shot is run with `stim.TableauSimulator`; every shot is then a
Pauli frame relative to it, stored as bit arrays of shape `(n_qubits, shots)` and updated
with one binary matrix multiply per gate (the gate's symplectic matrix from its Stim
tableau). Two frames are tracked: the **error frame** (what NoisiQ visualizes) and a
**randomization frame** that makes random measurement outcomes come out right ("stabilizer
randomization", as in Stim). Cost is `O(gates × shots)` bit operations: a 200-qubit GHZ with
10,000 shots runs in well under a second.

## statevector — trajectories

States of a chunk of shots are evolved together as an array `(batch, 2ⁿ)`. Kraus channels
are sampled per shot with probabilities `‖K ψ‖²` (for 1-qubit channels from the 2×2 reduced
state, then only the chosen operator is applied). An ideal copy is evolved alongside
(post-selected on the same measurement outcomes) to report fidelity and per-qubit
deviation. Memory per chunk is bounded (~4M amplitudes); the full density matrix is only
accumulated for `n ≤ 8` (or `density_matrix=True`).

## density_matrix — exact reference

Keeps one density matrix per classical record; measurements and readout errors branch
exactly. Use it to get exact numbers or to validate the sampling backends (the test suite
does this on random circuits).

## Adding a backend

Implement a class with `name` and
`run(circuit, noise, shots, seed, *, program=None, **options) -> Result`, iterate
`program.steps` (see `backends/program.py`), and register it in `BACKENDS` in
`backends/__init__.py`. Add it to the cross-backend tests in `tests/test_backends.py`.
