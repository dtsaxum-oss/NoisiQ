# NoisiQ

**Noise-Aware Quantum Circuit Simulation and Visualization**

[![CI](https://github.com/<your-org>/noisiq/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/noisiq/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

NoisiQ is an open-source Python library that help's visualize quantum circuit noise. Users build or import a quantum circuit, attach selectable noise models, like simple Pauli channels and non-Pauli decoherence such as amplitude damping and dephasing. Then, observe how errors propagate through the circuit in real time.

NoisiQ is designed primarily as a **teaching resource** but also has some usefulness as a **research tool**. For small circuits, it will produce step-by-step propagation animations. For larger circuits, it emphasizes aggregated statistics and heatmaps where gate-level animation is not practical.

The physics is validated against known algebraic identities and spot-checked against trusted simulators so that users can trust what they see.

EXAMPLE : 

[NoisiQ-Example](https://github.com/user-attachments/assets/f21d3b45-6cc1-46ec-916f-31b68e427df0)



---

## Features in Development

- **Circuit builder** — construct quantum circuits from standard gates (H, X, Y, Z, CNOT, etc) using NumPy matrix representations, with full unitarity validation on every gate
- **Noise model library** — attach Pauli channels (dephasing, depolarizing, bit/phase flips), non-Pauli decoherence (amplitude damping T1, dephasing T2), and coherent control errors to any gate in your circuit, or a random selection from any combination
- **Error propagation animation** — visualize how an injected error propagates through a circuit as a step-by-step animation or GIF, designed for circuits up to 16 qubits and depth 64
- **Aggregate heatmaps** — for larger circuits or many-shot runs, view which gates and qubit positions accumulate the most error
- **Interactive panel** — an ipywidgets-based interface inside Jupyter for selecting gates, error types, and probabilities without rewriting code
- **Noise suppression demonstrations** — dynamical decoupling-style suppression pipelines with before/after visual comparisons
- **Circuit adapters** — import circuits from Qiskit, Cirq, and other common frameworks
  directly into Noisiq 

## Future Goals
- A web-based app version
- Addition of more complex decoherence models
- Explore common QEC models 
- Allow decoherence models for specific qubit modalities  

---

## Installation

> **Note:** Noisiq is under active development. The instructions below will work once the initial package is published to PyPI. In the meantime, see [Development Installation](#development-installation).

```bash
pip install noisiq
```

### Development Installation

To install from source and contribute to the project:

```bash
# 1. Clone the repository
git clone https://github.com/<dtsaxum-oss>/noisiq.git
cd noisiq

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install in editable mode with development dependencies
pip install -e ".[dev]"

# 4. Confirm everything is working
pytest
```

---

## Quickstart

The following example builds a 2-qubit Bell state circuit, injects a depolarizing error on the Hadamard gate, and runs the propagation animation.

```python
import noisiq as nq

# Build a circuit
circuit = nq.Circuit(num_qubits=2)
circuit.add_gate(nq.gates.H, qubits=[0])
circuit.add_gate(nq.gates.CNOT, qubits=[0, 1])

# Attach a noise model to the Hadamard gate
circuit.add_noise(
    gate_index=0,
    model=nq.noise.Depolarizing(probability=0.05)
)

# Run the propagation animation
nq.visualize.propagation(circuit)
```

---

## Roadmap

NoisiQ is built on a 10-week development schedule. The table below reflects the current plan.

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | Mar 30 – Apr 5 | Repo skeleton, `Circuit` and `Gate` classes, CI |
| 2 | Apr 6 – Apr 12 | Pauli-frame propagation backend |
| 3 | Apr 13 – Apr 19 | Propagation animation and first demo notebook |
| 4 | Apr 20 – Apr 26 | Pauli noise channels and aggregate heatmap |
| 5 | Apr 27 – May 3 | Correctness harness and cross-simulator validation |
| 6 | May 4 – May 10 | Non-Pauli decoherence (T1 amplitude damping, T2 dephasing) |
| 7 | May 11 – May 17 | Coherent control errors (over/under-rotation) |
| 8 | May 18 – May 24 | Circuit adapters (Qiskit / Cirq import) |
| 9 | May 25 – May 31 | Noise suppression demo (dynamical decoupling) |
| 10 | Jun 1 – Jun 8 | Polish, final validation, paper, PyPI publish |

---

## Verification and Physical Soundness

NoisiQ is a tool. Users must be able to trust its outputs. Correctness is established through:

- **Gate-level unitarity checks** — every gate matrix is validated as unitary (U†U = I) at the time it is added to a circuit
- **Channel sanity checks** — all noise channels are validated for trace preservation and physical parameter bounds (Kraus operators)
- **Unit tests** — gate conjugation and projection rules, channel properties, and circuit construction are all covered by the test suite
- **Cross-simulator spot checks** — results on small circuits are compared against trusted simulators to confirm correctness

---

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. It covers our branching strategy, commit conventions, code style, docstring format, and testing requirements.

---

## Authors

- David Saxum — [GitHub](https://github.com/<dtsaxum-oss>)
- Teddy Jones— [GitHub](https://github.com/TeddyJ0nes)
- Giovanni Castorina — [GitHub](https://github.com/<gio>)

Faculty sponsors: Prof. Niknam, Prof. Ross

Completed during the QNT SCI 412 Lab Course as required via the MQST program at UCLA during the Spring quarter, 2026. 

---

## Citation

If you use NoisiQ in your research, please cite:

```bibtex
@software{noisiq2026,
  author  = {David Saxum and Teddy Jones and Giovanni Castorina},
  title   = {NoisiQ: Noise-Aware Quantum Circuit Simulation and Visualization},
  year    = {2026},
  url     = {https://github.com/<dtsaxum-oss>/noisiq},
  license = {MIT}
}
```

---

## License

NoisiQ is released under the [MIT License](LICENSE).  
Copyright (c) 2026 David Saxum, Teddy Jones, Giovanni Castorina