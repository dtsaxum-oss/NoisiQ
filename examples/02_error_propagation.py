"""Inject a Pauli error by hand and follow it through a circuit (exact, no noise).

This is the core teaching view: CNOTs copy X errors forward (control to target) and
Z errors backward (target to control); Hadamards swap X and Z.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import noisiq as nq  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

ghz = nq.algorithms.ghz(5)
for pauli in ("X", "Z"):
    trace = nq.propagate(ghz, pauli, qubit=0, after=0)
    print(f"{pauli} after the first H:", " -> ".join(trace.pauli_strings()))

# "Upside-down" GHZ: compute then uncompute. An X error in the middle spreads, then the
# uncomputation funnels it back onto a single qubit.
echo = nq.algorithms.ghz(5, uncompute=True)
trace = nq.propagate(echo, "X", qubit=2, after=2)
print("echo:", " -> ".join(trace.pauli_strings()))
nq.plot_propagation(trace).fig.savefig(OUT / "propagation_echo.png", bbox_inches="tight")
nq.save_animation(nq.animate_propagation(trace), OUT / "propagation_echo.gif", fps=2)

# How many outputs does a fault at each location corrupt? (exact, Clifford circuits)
w_inst, _ = nq.analysis.fault_spread(nq.algorithms.ghz(5, measure=True))
print("X faults after each instruction corrupt this many outputs:", w_inst[:, :, 0].max(axis=1))
