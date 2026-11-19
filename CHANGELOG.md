# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/);
before 1.0 minor versions may break the API.

## [0.2.0] — 2026-11-19

Complete overhaul for public release. See `docs/development/audit-2026-11.md` for the problems
fixed and `docs/development/overhaul-plan.md` for the plan. **Breaking:** the API changed throughout.

### Added
- Arbitrary circuits: full gate library (SX, RX/RY/RZ, P, U, CY, CH, CP, CRX/CRY/CRZ, iSWAP, CCX,
  CCZ, CSWAP, multi-controlled X/Z, custom unitaries), `inverse`, `compose`, `repeat`,
  X/Y/Z-basis measurement, reset, delay, barrier, feed-forward on any gate and multi-bit conditions.
- Rule-based `NoiseModel` (gate/qubit/arity/predicate rules, idle noise from gate durations,
  readout and reset errors, quasi-static detuning, explicit error injection, `twirled()`).
- Channels: generic `PauliChannel`, `KrausChannel`, `ThermalRelaxation`, `PhaseDamping`,
  `CoherentRotation`, `ReadoutError`, exact Pauli twirling for every channel.
- `simulate()` with automatic backend selection; vectorized Pauli-frame backend (hundreds of
  qubits); shot-vectorized statevector backend; exact density-matrix backend; fixed Qiskit Aer backend.
- `Result` with fault statistics, error budget, output error rate, fidelity, save/load, compare.
- `propagate()` for exact error injection/propagation; `fault_spread()`.
- One layout engine + renderer; data-driven heatmaps; propagation plots/animations (GIF/HTML);
  error budget with percentages; counts vs ideal; comparisons; Bloch/density plots.
- `noisiq.algorithms` (Bell, GHZ incl. echo/uncompute, DJ, BV, QFT, Grover, QPE, repetition
  codes, teleportation); OpenQASM 2 import/export; Qiskit adapters.
- `nq.explain()` and generated `docs/math.md`; documentation, examples, tour notebook.
- ~430 tests incl. cross-backend physics checks and fuzzing; CI with lint and multiple Pythons.

### Fixed
- T1 double counting in idle T1/T2 noise; wrong Pauli-twirl formulas; `0.99` probability rescaling.
- Hardware profiles applied either T1 or T2 (never both) and ignored decoherence in gate error.
- Hahn-echo DD flipped the logical state; DD crashed on measurements.
- Gate angles were parsed from rounded gate names; custom Clifford gates were rejected.
- Trajectory backend ran out of memory at ~15 qubits.
- Heatmap halos ignored the simulated noise.

### Removed
- TSim backend and the `bloqade-tsim` dependency; unfinished magic-state-distillation scaffold;
  course notebooks and LaTeX build artifacts (still available in git history).

## [0.1.0] — 2026-04-05
- Initial package skeleton developed during the UCLA QNT SCI 412 lab course (Spring 2026).
