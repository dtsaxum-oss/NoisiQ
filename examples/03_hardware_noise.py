"""Compare devices and noise representations on the same algorithm.

* exact (Kraus) noise -> statevector backend: T1/T2 and coherent errors exactly
* Pauli-twirled noise -> frame backend: fast, scalable approximation
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import noisiq as nq  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

print("available profiles:", nq.list_hardware())
profile = nq.get_hardware("ibm_eagle_r3")
print(profile)

qft = nq.algorithms.qft(4, input_state="0101")
exact = nq.simulate(qft, profile.noise_model(), shots=500, seed=1, trace_shots=5)
print("QFT fidelity to ideal on", profile.system, "=", exact.fidelity)
print("per-qubit purity:", exact.qubit_purities().round(4))
nq.plot_error_heatmap(exact).fig.savefig(OUT / "hardware_qft_heatmap.png", bbox_inches="tight")
nq.plot_propagation(exact).fig.savefig(OUT / "hardware_qft_deviation.png", bbox_inches="tight")

bv = nq.algorithms.bernstein_vazirani("1011011")
fig = nq.visualization.plot_hardware_comparison(bv, shots=4000)
fig.savefig(OUT / "hardware_comparison.png", bbox_inches="tight")

# Coherent vs twirled: the same over-rotation, repeated, behaves very differently.
c = nq.Circuit(1)
for _ in range(30):
    c.i(0)
c.measure(0)
over = nq.NoiseModel().add_gate_error(nq.noise.CoherentRotation("X", 0.05))
p_coh = nq.simulate(c, over, shots=1, backend="density_matrix").probabilities().get("1", 0)
p_tw = nq.simulate(c, over.twirled(), shots=1, backend="density_matrix").probabilities().get("1", 0)
print(f"30 over-rotations: coherent P(1)={p_coh:.4f}, twirled P(1)={p_tw:.4f}")
