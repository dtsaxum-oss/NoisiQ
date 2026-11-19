"""Build a circuit, add noise, simulate, and look at where errors happen.

Run:  python examples/01_quickstart.py   (figures are written to examples/output/)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import noisiq as nq  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

circuit = nq.Circuit(3, name="GHZ").h(0).cx(0, 1).cx(1, 2).measure_all()
noise = nq.NoiseModel.uniform_depolarizing(p1=0.002, p2=0.02, readout=0.01)

result = nq.simulate(circuit, noise, shots=5000, seed=1, trace_shots=20)
ideal = nq.simulate(circuit, shots=5000, seed=1)
print(result.summary())
print("distance to ideal:", result.compare(ideal))

nq.draw_circuit(circuit).fig.savefig(OUT / "quickstart_circuit.png", bbox_inches="tight")
nq.plot_error_heatmap(result, annotate=True).fig.savefig(OUT / "quickstart_heatmap.png", bbox_inches="tight")
nq.plot_error_budget(result).savefig(OUT / "quickstart_budget.png", bbox_inches="tight")
nq.plot_counts(result, ideal).savefig(OUT / "quickstart_counts.png", bbox_inches="tight")
nq.save_animation(nq.animate_propagation(result.trace), OUT / "quickstart_propagation.gif")
nq.explain("depolarizing")
