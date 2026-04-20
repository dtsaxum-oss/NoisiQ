## Development Roadmap
### Rules
1. Use the standards set in CONTRUBUTING.md and README.md
2. Propose architecture best practices that will help develop the minimum viable product (MVP)
3. Use the MVP to test the architecture and make improvements
4. Make sure the architecture is scalable to future plans (given later in this document)
5. Please ask questions and consult the user before making any major architectural decisions.

## TLDR:
1. Use the context provided below to get an understanding of the project and where we want to go with the timeline. Ultimately, I would like to implement as much of this timeline as possible as soon as possible so we have a product I can use to show interviewees. 
2. I want a plan to get to the MVP (Stim backend, Pauli errors, visualization core) as soon as possible. 
3. I also want a plan for going forward, how to expand the package to universal gates (Tsim?) and non Clifford + T gatesets (AerSimulator?) and beyond. This should also include how to expand the noise models and suppression/correction modules. 
4. Also let me know how having these other python packages as dependencies work. Can I just take certain functions from Aer or do I need to have the entire bulky package as a dependency (I dont want users's environments be excessively large for no reason).

## Concept Proposal

### Project Abstract 
We propose noisiq, a Python library that makes quantum circuit noise tangible. Users build or import a circuit, attach selectable noise models (from simple Pauli channels to non-Pauli decoherence such as amplitude damping and dephasing), and then observe how errors propagate through the circuit. Noisiq supports a mode that produces a step-by-step propagation animation and clear diagnostics, and a mode that scales to larger circuits by emphasizing aggregated statistics and backend efficiency. The project targets a 10-week lab-course build culminating in a working, installable package, validated against known algebraic identities (e.g., Clifford conjugation) and spot-checked against trusted simulators. 
### Project Description 
Objective & Payoff. Build an installable Python package that lets users attach realistic noise processes to quantum circuits and see how those errors propagate. The payoff is a tool that improves intuition (through propagation animation) while remaining scientifically grounded (through validated backends and reproducible statistics). 
Technical concept. Noisiq allows for circuit representation via user builds or imports, pluggable simulation backends, a noise-model library, and visualization. This minimizes rework as we expand to new modalities, models, and larger circuits. 
User workflow. 
● Create or import a circuit into the Noisiq circuit. 
● Select noise models (type, placement, and parameters) and optionally suppression/correction steps. 
● Run either as a step-by-step propagation animation or with larger circuits; where statistics are the outputs. 
Noise and suppression roadmap. Noisiq will support a staged progression from simpler, interpretable models to richer, modality-informed behavior: 
Near-term: Pauli channels (dephasing, depolarizing, bit/phase flips); non-Pauli decoherence (amplitude damping/T1 and dephasing/T2); simple coherent control errors (over/under-rotation). 
 Mid/long-term extensions: SPAM (measurement assignment, imperfect reset); correlated/crosstalk errors; leakage/loss channels; modality parameter presets (e.g., trapped ions, superconducting, neutral atoms). 
Suppression/correction (MVP + future): dynamical decoupling demonstrations; later error mitigation and introductory QEC modules as optional pipeline steps. 
Verification & measures of success. Correctness will be established via unit tests for gate conjugation/projection rules, channel sanity checks (trace preservation and parameter bounds for Kraus models), and spot checks against trusted simulators on small circuits. Success is a package that produces: (a) a visually correct propagation animation, (b) aggregate heatmaps/statistics across circuit, and (c) reproducible results. 
Deliverables. Noisiq Python package with documented API. Demo notebook(s) showing propagation animation + heatmap summaries under multiple noise models, adapter(s) from a common circuit framework into Noisiq circuit.
Program Plan 
Schedule & milestones. 
Week 
Dates 
Milestones (what we build) 
Key deliverable
1 
Mar 23–Mar 29, 
2026
Repo + Circuit skeleton 
Installable package skeleton; Circuit/Operation + validation; CI + tests running.
2 
Mar 30–Apr 05, 
2026
Pauli-frame backend 
Deterministic Pauli-frame propagation + user-specified error injection.
3 
Apr 06–Apr 12, 
2026
Propagation animation 
Circuit drawing + animation, first demo notebook reproducing propagation movie/GIF.
4 
Apr 13–Apr 19, 
2026
Pauli noise + aggregation 
Pauli channels (dephasing, depolarizing) + many-shot runner; first aggregate heatmap.
5 
Apr 20–Apr 26, 
2026
Correctness harness 
Expanded unit tests + randomized checks via small-circuit spot checks against trusted backends sims.
6 
Apr 27–May 03, 
2026
Non-Pauli decoherence MVP 
Amplitude damping (T1) + dephasing (T2/Tphi) via trajectories or small-n density matrix; comparisons to Pauli approximations.
7 
May 04–May 10, 
2026
Coherent control errors 
Over/under-rotation model on select gates; demonstrate systematic vs stochastic behavior.
8 
May 11–May 17, 
2026
Adapters (import circuits) 
Adapter functions from at least one framework (Qiskit or Cirq) into Noisiq IR; documented examples.
9 
May 18–May 24, 
2026
Noise suppression demo 
Dynamical decoupling-style suppression pipeline; before/after visual + aggregate improvement.
10 
May 25–May 31, 
2026
Polish + final package 
Docs, examples, performance cleanup, final validation runs; final presentation notebook + report.


Risks and Mitigation 
Risk 
Mitigation
Scope creep (too many models/UI features)
Lock down to: propagation animation + heatmap + 2–3 noise models; treat extras (Error trace, native QASM parsing, leakage) as stretch goals.
Correctness bugs in propagation / channels
Gate-level unit tests; randomized property tests; small-circuit cross-checks against trusted simulators; seeded reproducible runs.
Performance limits for non-Pauli models
Use trajectories / small-n density matrix only where needed; keep Pauli-frame backend for scaling; profile early.



Current Selected References
● M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, Cambridge University Press. 
● S. Aaronson and D. Gottesman, “Improved Simulation of Stabilizer Circuits,” Phys. Rev. A 70, 052328 (2004). 
● C. Gidney, “Stim: a fast stabilizer circuit simulator,” project documentation (software).

## Backend architecture.

Please evaluate this plan, propose improvements if necessary, and come up with a plan to implement this backend system design (the math behind the visualization tools).

System design: 
User interface → Simulator API → Backend (noise models, IR, visualization models, etc) 
Main working path: 
Circuit input via qasm, built in IR (built internally using our gates), etc
Circuit analysis / Backend selection (automated selection via gates found in 2.1, downselecting backend simulator between TSTIM and STIM, and automatic conversion of IR or qiskit circuit to proper format) 
Check for t-gates
if yes, use TSTIM, 
if no, check if clifford-only, 
if yes use STIM, 
if no, use AER.
Keep common user interface between simulators 
Noise Application user identified from pre-defined noise library (also allow users to upload their own noise module?) 
STIM, only use with Clifford gates (H, CNOT, CZ, SWAP, S) and Pauli noise. 
If clifford+plus T, then TSTIM. 
If neither use AER. 
Noise Styles 
Gate errors, T1/T2 errors, measurement errors, total circuit error 
Noise channels can be applied with some probability level, or with a more generic on X gate, on this qubit, at this location, apply a certain noise type 
Find common reported values to build generic profiles for certain modalities 
The simulation backend runs a noisy circuit simulation 
Visualization output of results, animated, static, error-values + a way to save results for comparison between simulations 
Static circuit diagram for no errors / static circuit diagram with heatmap of most impactful error when run with multi-shot sim 
Comparison plots when a new noise model is applied on same circuit or a error correction model is applied 
Animated error propagation, ipywidgets + matplotlib.animation representation?, allow user to step through one aspect at a time, final output is final state 
Comparison against “ideal” case? With statistics from ideal 
Other qiskit / pennylane convention that’s used too 

### Visualization Core

Originally, I planned to make the error visualization with matplotlib and ipywidgets as described in the backend architecture. This really needs to be the core of the package, as we are seeking to improve visualization tools for quantum error correction simulations to be used to convey information to non technical people and help teach students/newcomers to the field. Please let me know if this is a good way to do this or if you have suggestions for other improvements. 

### Noise Models

We need to implement the noise models as described in the backend architecture. Please let me know if this is a good way to do this or if you have suggestions for other improvements. 