"""Dynamical decoupling refocuses slow (quasi-static) frequency noise.

A qubit in |+> waits 20 us while its frequency is randomly detuned from shot to shot.
Without DD the phase is scrambled; an echo (X ... X) undoes it, because the detuning is
constant within each shot. If the same noise is modelled as Markovian (Pauli-twirled)
noise, DD cannot help - which is why the choice of noise model matters.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

import noisiq as nq  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

c = nq.Circuit(1, name="Ramsey").h(0).delay(0, 20e-6).h(0).measure(0)
noise = nq.NoiseModel("30 kHz quasi-static detuning").add_quasistatic_detuning(2 * np.pi * 30e3)

runs = {}
for seq in (None, "X2", "XY4"):
    circ = c if seq is None else nq.suppression.apply_dd(c, seq)
    runs[seq or "no DD"] = nq.simulate(circ, noise, shots=4000, seed=1, backend="statevector")
for name, r in runs.items():
    print(f"{name:>6}: P(error) = {r.probabilities().get('1', 0):.3f}")

twirled = nq.simulate(nq.suppression.apply_dd(c, "X2"), noise.twirled(), shots=4000, seed=1)
print(f"X2 with Markovian (twirled) noise: P(error) = {twirled.probabilities().get('1', 0):.3f}")

nq.draw_circuit(nq.suppression.apply_dd(c, "XY4")).fig.savefig(OUT / "dd_circuit.png", bbox_inches="tight")
nq.plot_comparison(runs, "fidelity").savefig(OUT / "dd_comparison.png", bbox_inches="tight")
