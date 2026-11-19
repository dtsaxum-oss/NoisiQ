# Maintainer guide

Read this before changing NoisiQ. It explains how the code is organized, *why* it is built
this way, and how to extend, test and release it.

## Layout

```
src/noisiq/
  ir/             gates.py (Gate + library), instructions.py, circuit.py
  noise/          channels.py, model.py (NoiseModel, Durations), hardware.py (profiles)
  backends/       program.py (compiler), frame.py, statevector.py, density.py, aer.py, __init__.py (simulate)
  results/        result.py (Result, Estimate), trace.py (PropagationTrace)
  analysis/       propagation.py (propagate, fault_spread)
  visualization/  layout.py, circuit.py, heatmap.py, propagation.py, charts.py, style.py
  algorithms/     textbook circuits
  suppression/    dynamical decoupling
  io/             qasm.py, qiskit.py
  catalog.py      math for every channel/metric (nq.explain, docs/math.md)
  metrics.py      fidelity, purity, distances
tests/            behaviour/physics tests (see "Testing")
examples/         runnable scripts + tour notebook (executed by tests)
docs/             user docs; development/ holds the 2026-11 audit and overhaul plan
paper/            the course final report (LaTeX sources)
```

## Design decisions (and the bugs they prevent)

1. **Gates are identified by matrix, never by name.** Clifford-ness and the propagation
   matrix come from `stim.Tableau.from_unitary_matrix` *verified by reconstruction* (Stim
   silently accepts some non-Cliffords). Names are display-only and used in user-facing noise
   rules. Never add `if gate.name == ...` to a simulator.
2. **Parameters are stored as numbers** on `Gate.params`; never parse names.
3. **Noise is rules, resolved at compile time** (`compile_program`). Old code keyed noise by
   operation index, which silently moved noise when circuits were edited.
4. **One compiler, many backends.** Scheduling, durations, idle noise, implicit measurement
   and the "snapshot" point (before terminal measurements) live in `program.py` only.
5. **Every channel can twirl itself.** Backends ask `channel.is_pauli` / `pauli_twirl()` /
   `kraus()`; there are no channel-type ladders in backends (except fast paths).
6. **One Result type.** Fields a backend cannot provide are `None` and the accessor raises a
   helpful error.
7. **One layout engine and renderer.** Overlays (heatmaps, propagation) draw on top of
   `draw_circuit` using `CircuitLayout` coordinates.
8. **Math lives in `catalog.py`.** Docstrings point there; `docs/math.md` is generated from it
   and a test fails if it is stale.

## Extending

* **A gate:** add a constant or factory in `ir/gates.py` (+ `STANDARD_GATES`/`_PARAMETRIC`,
  inverse naming), a builder method in `Circuit` (and `_Conditioned._GATE_METHODS`), QASM
  names in `io/qasm.py`, and a test in `tests/test_gates.py`. Simulators need nothing.
* **A channel:** subclass `Channel` in `noise/channels.py` (`num_qubits`, `kraus()`, optionally
  a faster `pauli_twirl()`), set `catalog_key`, add a catalog entry, regenerate
  `docs/math.md` (`python -m noisiq.catalog docs/math.md`), add CPTP/twirl tests.
* **A noise rule:** add a method to `NoiseModel` that stores a rule, and resolve it in
  `gate_faults`/`idle_faults`/`compile_program`. Keep `twirled()` and `copy()` in sync.
* **A hardware profile:** `register_hardware(HardwareProfile(...))` in `noise/hardware.py`
  with references. `tests/test_noise.py` builds every registered profile.
* **A plot:** build on `draw_circuit` + `CircuitLayout`; add a smoke test on random circuits
  in `tests/test_visualization.py`.
* **A backend:** see [backends.md](backends.md#adding-a-backend).

## Testing

```bash
pytest                       # everything (~40 s)
pytest -m "not slow"         # skip example scripts / notebook / large circuits
pytest --cov=noisiq          # coverage (≈93 %)
ruff check . && ruff format --check .
```

The most valuable tests are the **cross-backend** ones: random circuits (with mid-circuit
measurement, resets, feed-forward and noise) must produce the same distribution on the
frame, statevector and exact density-matrix engines (`tests/test_backends.py`,
`tests/test_edge_cases.py`). When you touch a backend, run those first. Statistical tests use
fixed seeds and total-variation-distance tolerances sized for their shot counts.

## Known limitations / roadmap

* Frame backend: conditional gates must be Paulis; noise must be Pauli (or twirled).
* Statevector backend: ≤ 26 qubits; Kraus sampling is per shot (vectorized over a chunk).
* Aer backend: single-bit conditions only; readout error with feed-forward unsupported.
* Not modelled yet: leakage, crosstalk between neighbouring idle qubits (needs a coupling
  map), non-zero temperature relaxation, QEC decoders beyond the d=3 lookup table.
* Interactive/web front-end: `CircuitLayout.to_dict()` is the intended hook.

## Releasing

1. Update `version` in `pyproject.toml` and `noisiq/__init__.py`, and `CHANGELOG.md`.
2. `pytest && ruff check .`
3. `python -m build && twine check dist/*`
4. Tag `vX.Y.Z`; upload with `twine upload dist/*` (PyPI token required — maintainers only).
