# NoisiQ overhaul plan (v0.2) — status

Goal: an installable, documented package that does what the concept paper promised for
**arbitrary** circuits, with a clean architecture the next maintainers can extend.
Problems are catalogued in [audit-2026-11.md](audit-2026-11.md). The architecture is
described in [../maintainers.md](../maintainers.md).

Legend: [x] done · [ ] open

## Phase 0 — Repository cleanup *(audit §6)*
- [x] Audit + plan written.
- [x] LaTeX build artifacts removed; report sources moved to `paper/`.
- [x] Course/weekly notebooks, stale demo notebooks, `scripts/`, `theme.md`, `docs/documentation.ipynb` removed (in git history).
- [x] TSim backend + `bloqade-tsim` removed; unfinished `protocols/` (MEK distillation scaffold) removed.
- [x] `.gitignore` rewritten (LaTeX aux files, outputs, caches, stackdumps).

## Phase 1 — IR *(audit §1)*
- [x] `Gate` defined by matrix with exact `params`; Clifford detection/tableau/symplectic matrix from the matrix (verified by reconstruction); `inverse()`, `controlled()`.
- [x] Full gate library + factories; custom unitaries.
- [x] `Operation` with any classical condition (multi-bit), `Measure` (X/Y/Z, reset), `Reset`, `Delay`, `Barrier`.
- [x] `Circuit`: builders for every gate (broadcasting), scheduling, validation (duplicates, ranges), registers, `c_if` for any gate, `measure_all`, `inverse`, `compose`, `copy`, `repeat`, `depth`, `layers`, `count_ops`, `draw`, QASM.

## Phase 2 — Noise *(audit §2.5, 2.7, 2.8, §3)*
- [x] `Channel` base with `kraus()`, exact `pauli_twirl()`, `is_pauli`, `error_probability`.
- [x] Pauli/depolarizing/bit/phase flips, amplitude & phase damping, thermal relaxation (correct Tφ), coherent rotation, generic Kraus, composition, readout error.
- [x] Rule-based `NoiseModel` with durations, idle noise, readout/reset errors, quasi-static detuning, explicit injection, `twirled()`.
- [x] Hardware profiles rebuilt (thermal + residual depolarizing reproduce datasheet infidelity; readout error).
- [x] `nq.explain()` + generated `docs/math.md`.

## Phase 3 — Backends & results *(audit §2)*
- [x] Vectorized Pauli-frame backend (any Clifford, X/Y/Z measurement, reset, readout, Pauli feed-forward, traces).
- [x] Statevector backend vectorized over shots; no `4ⁿ` accumulator; reduced states; deviation traces; quasi-static noise.
- [x] Exact density-matrix backend (branching cq-state).
- [x] Aer backend fixed (imports, ordering, measurements, Kraus noise; works around an Aer issue with read-only arrays).
- [x] `simulate()` + `select_backend()` with reasons.
- [x] Unified `Result` (fault stats, budget, output error rate, fidelity, save/load, compare).

## Phase 4 — Analysis, visualization, algorithms
- [x] `propagate()`, `conjugate_pauli()`, `fault_spread()`.
- [x] `CircuitLayout` + single renderer.
- [x] Data-driven heatmap (rate / burden), idle bands.
- [x] Propagation plot, animation, GIF/HTML export, ipywidgets stepper.
- [x] Error budget with percentages, per-qubit errors, counts vs ideal, comparisons, density/Bloch, hardware comparison.
- [x] `algorithms` module.
- [x] DD rewritten (identity-preserving sequences, splits delays, safe with measurements); demonstrably refocuses quasi-static noise.
- [x] QASM 2 import/export (params, macros, broadcast, reset, if) and Qiskit adapters.

## Phase 5 — Tests *(audit §5)*
- [x] Old milestone-coupled suite replaced by ~430 behaviour/physics tests (≈93 % line coverage):
  gates, circuits, channels (CPTP, closed-form decays, twirls), noise rules, hardware,
  cross-backend agreement on random circuits (incl. MCM/reset/feed-forward), propagation vs
  Stim, algorithms, QASM/Qiskit round trips, DD, results I/O, catalog, every plot, fuzzing,
  README + examples + notebook execution.

## Phase 6 — Packaging & docs
- [x] `pyproject.toml`: lean core deps (numpy, matplotlib, stim), extras `qiskit`, `widgets`, `test`, `dev`; version 0.2.0; `py.typed`.
- [x] CI: ruff lint + format, tests on 3.10–3.13 (+Windows/macOS), full run with Qiskit, build + `twine check`.
- [x] README (tested quickstart), docs pages, maintainer guide, examples, tour notebook, CHANGELOG, CONTRIBUTING.
- [ ] Publish to PyPI (needs a maintainer PyPI account/token — see maintainers.md § Releasing).
- [ ] Host docs (e.g. GitHub Pages / Read the Docs) — markdown is ready.

## The original "to do before release" list

| Item | Status |
|---|---|
| Stim not used in visuals (OOM at 15 qubits) | Fixed: visuals use the frame (Stim-tableau) engine; statevector no longer builds `4ⁿ` matrices. |
| Abstract/standardize visuals, no hard-coded positions | Done: `CircuitLayout` + one renderer + `Style`. |
| TSim still in the branch | Removed. |
| Heat-map halos ignore the noise model | Fixed: halos come from simulated fault statistics (test `test_heatmap_depends_on_noise_model`). |
| Website with live demos for the poster | Open — `CircuitLayout.to_dict()` and the examples are the starting point. |
| arXiv paper | Open — LaTeX sources kept in `paper/`. |
| Math documentation in one place | Done: `catalog.py` → `nq.explain()`, docstrings, `docs/math.md`. |
| Same for report metrics | Done: metrics are catalog entries too. |
| Remove hard-coding; make promised features work | Done (see audit). |
| Hard-coded common algorithms (e.g. Deutsch–Jozsa(n)) | Done: `nq.algorithms`. |
| Extensive bug testing, especially mid-circuit/classical measurement | Done: cross-backend MCM/feed-forward fuzzing against an exact reference. |
| SPAM example/tests | Done: readout + reset errors, tests, examples. |
| Phase-flip correction / XY measurement | Done: X/Y-basis measurement; phase-flip repetition code. |
| Percentages on the error-burden chart | Done: `plot_error_budget`. |
| "Upside-down" gates / uncomputing GHZ | Done: `Circuit.inverse()`, `ghz(..., uncompute=True)`. |
| Later: Quirk-like web sim, more hardware platforms | Open. |
| Later: pip | Packaging ready; publishing open. |
