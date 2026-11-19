# Visualization

All circuit pictures come from one place: `CircuitLayout` (pure geometry, no matplotlib —
exportable with `layout.to_dict()` for other front-ends) and `draw_circuit` (the only
renderer). Heatmaps and animations draw on top of it, so any circuit that can be drawn can
also be heat-mapped and animated. Change the look globally with
`nq.visualization.set_style(fontsize=12, heat_cmap="viridis", ...)`.

| function | shows | needs |
|---|---|---|
| `draw_circuit(circuit)` | the circuit (any gates, measurements, resets, delays, conditions) | — |
| `plot_error_heatmap(result, metric="rate")` | glow around each gate = probability a fault fired there; bands on wires = idle faults | frame / statevector result |
| `plot_error_heatmap(result, metric="burden")` | rate × number of outputs the fault corrupts | Clifford circuit |
| `plot_error_budget(result)` | ranked error sources with % share | frame / statevector |
| `plot_qubit_errors(result)` | X/Y/Z/other faults per qubit | frame / statevector |
| `plot_propagation(trace_or_result, step)` | error frame (Clifford) or deviation (general) after `step` layers; ★ = fault, ⓧ badges = current error, red ring = flipped measurement | a trace |
| `animate_propagation(...)`, `save_animation(anim, "x.gif"/"x.html")` | the above as a movie | a trace |
| `interactive_propagation(...)` | slider in Jupyter | `noisiq[widgets]` |
| `plot_counts(result, ideal)` | outcome distribution vs ideal, with Hellinger fidelity | — |
| `plot_comparison({...}, metric)` | one metric across runs (noise models, DD on/off, devices) | — |
| `plot_hardware_comparison(circuit, profiles)` | the same circuit on several device profiles | — |
| `plot_density_matrix(result)`, `plot_bloch(result)` | states | statevector / density_matrix |

## Reading a heatmap

The halo color is a **measured** quantity from the simulation (with the noise model you
chose) — change the noise and the picture changes. With `scale="log"` small error sources
stay visible; pass the same `vmin`/`vmax` to compare several heatmaps on one scale.
`annotate=True` prints the value under each gate.

## Where traces come from

* `nq.propagate(circuit, "X", qubit, after=k)` — exact, deterministic, one error you choose.
* `nq.simulate(..., trace_shots=N)` — the first `N` sampled shots. Plotting functions pick
  the first shot that had a fault unless you pass `shot=`.
