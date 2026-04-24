## Visual Theme — NoisiQ

Follow algassert.com/quirk for general guidance on gate font, box size, control gate representation, and input state display.

---

### Global Color Scheme

| Element | Color |
|---------|-------|
| Qubit wires | Black |
| Clifford gates (H, X, Y, Z, S, CNOT, CZ, I) | Baby / lite blue |
| T-gates (T, T_DAG) | Traditional blue |
| Other gates (non-Clifford, non-T) | Dark blue |
| Pauli errors | Red, small font |
| Error-free halo | Light blue |
| Maximum-impact halo | Bright red |

All colors defined as constants in `noisiq/visualization/theme.py`.

---

### Circuit Representation (all views)

- Gate boxes: colored by category (see above); white label text; consistent width and spacing
- Qubit wires: thin black horizontal lines with qubit index labels on left (q0, q1, ...)
- Control gate representation: filled dot on control qubit, standard symbol on target qubit — follow Quirk style

---

### Static View — Non-interactive Heatmap

- Circuit "print-out" with gate boxes and qubit wires as drawn above
- **Halo-color effect** around each gate box (both single-qubit and multi-qubit gates):
  - Color scales from **bright red** (maximum downstream impact) → **no halo** → **light blue** (zero errors)
  - Halo intensity is normalized: the gate with the most downstream qubit interactions across all shots gets brightest red; everything else is set relative to that
  - Zero errors propagated from a gate → light blue halo
  - **Week 4–6 (proxy):** halo intensity = `error_rate_matrix` (error count / n_shots)
  - **Week 7+ (final):** halo intensity = `downstream_impact_matrix` (count of downstream non-I qubits after propagation)
- No error labels on the qubit wires in heatmap views — halo is the sole error indicator
- **n=1 simulation summary (single-shot):** counter of independent errors; output most significant gate (most downstream interactions) and error type

---

### Interactive View — Hover Tooltip

Users hover over any gate box to see an info panel. The system automatically detects whether the last run was single-shot or many-shot and displays the appropriate fields.

**Single-shot fields:**

| Field | Description |
|-------|-------------|
| Gate type | Name and category (e.g. "H — Clifford") |
| Error occurred | Exact Pauli that occurred: X / Y / Z / none |
| Incoming state | Qubit state entering the gate |
| Ideal outgoing state | Expected output assuming no error |
| Actual outgoing state | Actual output after error injection |
| Fan-out graph | Button: "Show propagation graph" → triggers static fan-out figure |

**Many-shot fields (superset of single-shot):**

| Field | Description |
|-------|-------------|
| Gate type | Name and category |
| Error probability distribution | X: %, Y: %, Z: %, I: % across all shots |
| Most likely error | Dominant error type and its probability |
| Fidelity estimate | 1 − total error rate at this gate |
| Incoming state | From tableau |
| Ideal outgoing state | Expected output |
| Fan-out graph | Button: "Show propagation graph" → triggers static fan-out figure |

**Fan-out graph (triggered from hover):**
- Opens a new static matplotlib figure
- Displays a `networkx` DAG from the selected gate forward through the circuit
- Nodes = gate boxes, directed edges = qubit paths errors can travel
- Edge weights = error propagation probability from `AggregateResult` (many-shot) or unweighted (single-shot)
- Implemented in `noisiq/visualization/fanout.py`

**Large circuit support:**
- For circuits with >24 qubits or gate depth >64: allow user scrolling/panning within the interactive panel (future version — see Schedule_update.md)

---

### Interactive View — Error Propagation Animation (.gif / ipywidgets)

**Error visual representation:**
- Pauli errors **float on the qubit wire** between gate boxes — they do not appear inside or on top of gate boxes
- After a gate applies an error, the error label **pops up immediately after the gate box** on the wire
- The label **rides the wire** until it reaches the next gate box, then disappears and a new error label pops up after that gate
- For multi-qubit gates (e.g. CNOT): the error splits according to propagation rules and **rides both respective qubit lines** forward independently

**Many-shot animation:**
- The error displayed at each gate is the **highest-probability error** from the many-shot simulation
- The probability label is shown alongside the error (e.g. "X  23%")
- Animation runs only after many-shot simulation is complete

**Animation controls (ipywidgets):**
- Play / Pause `ToggleButton`
- Step forward / Step backward `Button`
- Speed `IntSlider` (frames per second)
- All controls in a horizontal `HBox` below the animation

---

### Standalone Charts

These are callable as standalone functions and display in Jupyter notebooks automatically.

| Function | File | Description |
|----------|------|-------------|
| `plot_qubit_error_bar(result)` | `charts.py` | Horizontal bar chart: total error counts per qubit summed across all timesteps |
| `plot_fidelity_decay(depth_results)` | `charts.py` | Gate depth vs. fidelity estimate (fraction of shots with zero errors at each depth) |
| `plot_fanout_graph(circuit, op_idx, result)` | `fanout.py` | Propagation DAG from selected gate — also accessible via hover menu |

---

### Future Version Notes

- **Qubit phase metric:** Once the trajectory backend (Week 6) is stable, expose Z-expectation value per qubit (±1 for stabilizer states) or full phase angle for non-Clifford states. Display in hover tooltip. Deferred — requires wavefunction access not available in Clifford-only backend.
- **Stim QEC syndrome view:** Stabilizer check / detector visualization for QEC circuits. Deferred — requires new backend class and QEC circuit support.
