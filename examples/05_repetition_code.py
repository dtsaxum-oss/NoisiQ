"""A distance-3 repetition code with mid-circuit syndrome measurement and feed-forward.

Shows mid-circuit measurement, measure-and-reset, classically-controlled corrections,
X-basis readout (phase-flip code) and the logical error-rate improvement.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import noisiq as nq  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

p = 0.03
for basis, err in (("bit", nq.noise.bit_flip), ("phase", nq.noise.phase_flip)):
    code = nq.algorithms.repetition_code(3, rounds=1, basis=basis)
    # Errors strike the data qubits right after encoding, before the syndrome round
    encode = nq.Circuit(5).i([0, 1, 2]) if basis == "bit" else nq.Circuit(5)
    if basis == "phase":
        encode.h([0, 1, 2]).i([0, 1, 2]).h([0, 1, 2])
    noise = nq.NoiseModel(f"{basis}-flip p={p}").add_gate_error(err(p), gates="I")
    r = nq.simulate(encode.compose(code), noise, shots=20000, seed=1)
    data = r.marginal_counts(range(r.n_clbits - 3, r.n_clbits))
    logical = sum(v for k, v in data.items() if k.count("1") >= 2) / r.shots
    print(f"{basis}-flip code: logical error {logical:.4f} vs physical {p}")

code = nq.algorithms.repetition_code(3, 1)
r = nq.simulate(code, nq.NoiseModel.uniform_depolarizing(0.01, readout=0.01), shots=4000, seed=2, trace_shots=10)
nq.plot_error_heatmap(r, annotate=True).fig.savefig(OUT / "repetition_heatmap.png", bbox_inches="tight")
nq.plot_propagation(r).fig.savefig(OUT / "repetition_propagation.png", bbox_inches="tight")
nq.plot_error_budget(r, metric="burden").savefig(OUT / "repetition_budget.png", bbox_inches="tight")
