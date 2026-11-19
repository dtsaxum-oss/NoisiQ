# Getting started

This page walks through the whole workflow. Everything here also exists as a runnable
notebook: [`examples/tour.ipynb`](../examples/tour.ipynb).

```python
import numpy as np
import noisiq as nq
```

## 1. Build a circuit

`Circuit(n_qubits)` has a builder method for every library gate. Methods return the circuit,
so calls chain; single-qubit methods accept a list/range of qubits.

```python
c = nq.Circuit(4, name="demo")
c.h(range(4))  # H on every qubit
c.cx(0, 1).cz(2, 3)  # two-qubit gates
c.rz(np.pi / 3, 1)  # rotations take the angle first
c.ccx(0, 1, 2)  # Toffoli
c.unitary(my_matrix, [0, 3], name="MYGATE")  # any unitary (big-endian in qubit order)
c.measure_all()  # measure every qubit into new classical bits
c.draw()
```

Gates are scheduled "as soon as possible" into *layers*; pass `layer=` to place one
explicitly. Other instructions: `measure(q, basis="X"|"Y"|"Z", reset=False)`, `reset(q)`,
`delay(q, seconds)`, `barrier(...)`, and classical feed-forward:

```python
c = nq.Circuit(3)
c.h(0).measure(0)  # clbit 0
c.c_if(0).x(1)  # X on qubit 1 if clbit 0 == 1 (any gate can be conditioned)
```

Ready-made circuits live in `nq.algorithms` (`bell`, `ghz`, `qft`, `grover`,
`deutsch_jozsa`, `bernstein_vazirani`, `phase_estimation`, `repetition_code`,
`teleportation`), all parameterized by size. Circuits can be inverted
(`c.inverse()`), composed (`a.compose(b, qubits=[...])`), and imported/exported with
`nq.io.from_qasm / to_qasm` or `nq.io.from_qiskit / to_qiskit`.

## 2. Describe the noise

A `NoiseModel` is a list of rules that are matched against the circuit at simulation time:

```python
noise = nq.NoiseModel("my device")
noise.add_gate_error(nq.noise.depolarizing(1e-3), num_qubits=1)  # every 1-qubit gate
noise.add_gate_error(nq.noise.depolarizing(1e-2, 2), gates="CX")  # CX gates, 2-qubit channel
noise.add_gate_error(nq.noise.CoherentRotation("Z", 0.02), qubits=[3])  # over-rotation on qubit 3
noise.add_thermal_relaxation(t1=100e-6, t2=80e-6)  # T1/T2 on gates + idle time
noise.add_readout_error(0.02)  # measurement errors
noise.add_quasistatic_detuning(2 * np.pi * 10e3)  # slow frequency drift
```

Or start from a device profile: `nq.get_hardware("ibm_heron_r3").noise_model()`
(`nq.list_hardware()` shows all). Gate durations for time-dependent noise come from
`nq.noise.Durations`. `noise.twirled()` converts every channel into its Pauli twirl so the
fast stabilizer backend can run it.

For the math of any channel: `nq.explain("amplitude_damping")` (or see [math.md](math.md)).

## 3. Simulate

```python
result = nq.simulate(c, noise, shots=5000, seed=1, trace_shots=20)
print(result.summary())
```

`backend="auto"` picks the fast **frame** engine for Clifford circuits with Pauli noise and
the universal **statevector** engine otherwise (see [backends.md](backends.md)).
`trace_shots` records the step-by-step error state of the first shots for animation.

The `Result` holds counts (`result.counts`, `result.probabilities()`), fault statistics
(`fault_rates()`, `error_budget()`, `fault_free_fraction`), output error rates (frame),
fidelity to ideal and reduced states (statevector), and can be saved with
`result.save("run.npz")` / `nq.Result.load(...)` for later comparison.

## 4. Visualize

```python
nq.plot_error_heatmap(result)  # halos = simulated fault rate per gate / idle wire
nq.plot_error_heatmap(result, metric="burden")  # weighted by how far each fault spreads
nq.plot_error_budget(result)  # ranked error sources, with % shares
nq.plot_propagation(result, step=3)  # error state after 3 layers
anim = nq.animate_propagation(result)  # the whole movie
nq.save_animation(anim, "movie.gif")  # or .html
nq.plot_counts(result, ideal_result)  # outcomes vs ideal
nq.plot_comparison({"A": r1, "B": r2}, "fidelity")
```

For a hand-crafted teaching example, inject an error explicitly:

```python
trace = nq.propagate(nq.algorithms.ghz(4), "X", qubit=0, after=0)
trace.pauli_strings()  # ['IIII', 'XIII', 'XXII', 'XXXI', 'XXXX']
nq.animate_propagation(trace)
```

In Jupyter, `nq.visualization.interactive_propagation(trace)` adds a slider
(`pip install "noisiq[widgets]"`).
