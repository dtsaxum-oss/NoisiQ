# Contributing to NoisiQ

This document defines how we work together, how we write code, how we submit it, and how we keep the project clean and credible. 

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Getting Started](#2-getting-started)
3. [Branching Strategy](#3-branching-strategy)
4. [Commit Message Convention](#4-commit-message-convention)
5. [Code Style](#5-code-style)
6. [Docstrings and Documentation](#6-docstrings-and-documentation)
7. [Testing Requirements](#7-testing-requirements)
8. [Pull Request Process](#8-pull-request-process)
9. [Weekly Workflow](#9-weekly-workflow)
10. [What Not to Commit](#10-what-not-to-commit)

---

## 1. Project Overview

NoisiQ is a Python library for noise-aware quantum circuit simulation and visualization. The goal is to create a clean, installable, pip-publishable package accompanied by a research paper. Code quality, physical correctness, and modularity are non-negotiable priorities.

**Initial Target stack:**
- Python 3.10+
- NumPy (core math)
- Matplotlib (visualization)
- pytest (testing)
- Jupyter / ipywidgets (demo notebooks and interactive tools)

---

## 2. Getting Started

Follow these steps to set up your local development environment after cloning the repo.

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-org>/noisiq.git
cd noisiq
```

### Step 2 — Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install the package in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

The `-e` flag installs the package in "editable" mode, meaning changes you make to the source files are immediately reflected without reinstalling. The `[dev]` flag installs additional tools needed for development (testing, linting, notebooks) that are not required by end users.

### Step 4 — Confirm everything is working

```bash
pytest
```

All tests should pass on a fresh clone. If they do not, stop and raise it with the team.

---

## 3. Branching Strategy

We use a **feature-branch workflow** :

| Branch | Purpose | Who |
|---|---|---|
| `main` | Always stable. Always passing tests. | No one directly — PRs only |
| `feature/<name>` | All new work happens here | Anyone |
| `fix/<name>` | Bug fixes | Anyone |
| `docs/<name>` | Documentation-only changes | Anyone |

### Rules

- **Never commit directly to `main`.** All changes arrive via Pull Request.
- **Branch names should be lowercase and hyphenated.** Examples:
  - `feature/pauli-frame-backend`
  - `feature/circuit-animation`
  - `fix/unitarity-check-tolerance`
  - `docs/readme-install-instructions`
- **One feature per branch.** Keep branches focused. 
- **Keep branches short-lived.** Ideally a branch lives for one week or less. Long-lived branches may accumulate conflicts.

### Starting a new branch

Always branch from the latest `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## 4. Commit Message Convention

Readable commit history matters — both for collaboration and for documenting the project's development in the eventual paper. Every commit message must follow this format:

```
<type>: <short description in present tense>
```

### Allowed types

| Type | Use for |
|---|---|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `test` | Adding or updating tests |
| `docs` | Documentation changes only |
| `refactor` | Code restructuring with no behavior change |
| `perf` | Performance improvements |
| `chore` | Maintenance tasks (dependency updates, CI config, etc.) |

### Examples

```
feat: add Hadamard gate definition with unitarity validation
fix: correct tolerance in unitary check for near-identity matrices
test: add CNOT gate round-trip test
docs: update README with virtual environment setup instructions
refactor: simplify Circuit.add_gate to reduce branching
perf: vectorize Pauli frame update step
chore: add numpy and matplotlib to pyproject.toml dependencies
```

### Rules

- Use the **present tense** ("add" not "added", "fix" not "fixed")
- Keep the first line **under 72 characters**
- Do not end the first line with a period
- If more context is needed, leave a blank line after the first line and write a longer description below

---

## 5. Code Style

All code in this project follows **PEP 8**, Python's official style guide. Consistency makes the codebase easier to read, review, and publish.

### Key rules

- **Indentation:** 4 spaces.
- **Line length:** Maximum 88 characters per line.
- **Naming:**
  - Classes: `CapWords` (e.g., `QuantumCircuit`, `PauliChannel`)
  - Functions and variables: `snake_case` (e.g., `add_gate`, `num_qubits`)
  - Constants: `ALL_CAPS_SNAKE_CASE` (e.g., `PAULI_X`, `DEFAULT_TOLERANCE`)
  - Private methods/attributes: prefix with single underscore (e.g., `_validate_matrix`)
- **Imports:** Group in this order, with a blank line between each group:
  1. Standard library (e.g., `import math`)
  2. Third-party libraries (e.g., `import numpy as np`)
  3. Internal noisiq modules (e.g., `from noisiq.gates import Gate`)
- **Type hints:** Required on all function and method signatures. 

### Example of correct style

```python
import numpy as np

from noisiq.gates import Gate


def apply_gate(gate: Gate, state: np.ndarray) -> np.ndarray:
    """Apply a gate to a statevector and return the result.

    Args:
        gate: The Gate object to apply.
        state: A 1D NumPy array representing the statevector.

    Returns:
        A new 1D NumPy array representing the updated statevector.
    """
    return gate.matrix @ state
```

---

## 6. Docstrings and Documentation

Every public class, method, and function must have a docstring. Follow the 
**Google docstring style** for consistency.

### Format

```python
def add_gate(self, gate: Gate, qubits: list[int], layer: int | None = None) -> None:
    """Add a gate operation to the circuit.

    Args:
        gate: The Gate object to add.
        qubits: A list of qubit indices the gate acts on. Must match
            the gate's qubit count and must be valid indices for this circuit.
        layer: The circuit layer (time step) to place the gate at.
            If None, the gate is appended to the next available layer.

    Raises:
        ValueError: If qubit indices are out of range or the number of
            qubits does not match the gate definition.
    """
```

### Rules

- Write docstrings for **what** a function does, not **how** it does it
- Always document `Args`, `Returns`, and `Raises` where applicable
- Inline comments (`#`) should explain *why* something is done, not *what* the code is doing (the code itself should be readable enough to show what)

---

## 7. Testing Requirements

**Every new feature must ship with tests.**

NoisiQ is a tool — users must be able to trust its outputs. Tests are how we guarantee that.

### Where tests live

All tests live in the `tests/` directory and mirror the structure of the `noisiq/` package:

```
tests/
├── test_circuit.py       ← tests for noisiq/circuit.py
├── test_gates.py         ← tests for noisiq/gates.py
├── noise/
│   └── test_pauli.py     ← tests for noisiq/noise/pauli.py
└── ...
```

### Running tests

```bash
# Run the full test suite
pytest

# Run a specific test file
pytest tests/test_gates.py

# Run with verbose output
pytest -v
```

### What to test

- **Happy path:** Does the function return the correct result for valid inputs?
- **Physics checks:** Does the gate matrix pass unitarity? Does the channel preserve trace? Can you verify against a known engine?
- **Edge cases:** What happens at boundary values (e.g., a 1-qubit circuit, zero probability noise)?
- **Failure cases:** Does the function correctly raise an error for invalid inputs?

### Naming convention

Test functions must start with `test_` and have a descriptive name:

```python
def test_hadamard_is_unitary():
def test_add_gate_rejects_invalid_qubit_index():
def test_depolarizing_channel_preserves_trace():
```

---

## 8. Pull Request Process

### Before opening a PR

1. Make sure your branch is up to date with `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/your-feature-name
   git merge main
   ```
2. Run the full test suite and confirm it passes:
   ```bash
   pytest
   ```
3. Review your own diff before asking others to review it.

### Opening the PR

- **Title:** Use the same format as commit messages (e.g., `feat: add Pauli frame propagation backend`)
- **Description:** Answer these three questions:
  1. What does this PR do?
  2. Why is this change needed?
  3. How was it tested?
- **Assign a reviewer:** Tag at least one teammate for review.

### Review expectations

- Leave comments on specific lines using GitHub's review interface.
- Approve only when you have actually read and understood the changes.
- If you request changes, the author addresses them and re-requests review — do not re-review until asked.

### Merging

- A PR requires **at least 1 approval** before merging.
- All CI checks (automated tests) must pass.
- **The author merges their own PR** once approved — not the reviewer.
- Use **"Squash and merge"** to keep the `main` history clean.
- Delete the branch after merging.

---

## 9. Weekly Workflow

Noisiq is built on a 10-week sprint schedule. Each week has a defined milestone (see the project concept paper and Schedule.md). During Tuesday lab meeting:

1. The team reviews the week's milestone together and creates a review deck for Thursday's meeting.
2. Next weeks tasks are divided and each team member opens their working branch.
3. At the end of the week, all PRs for that milestone should be merged into `main`.

Avoid carrying unreviewed PRs from one week to another.

---

## 10. What Not to Commit

The `.gitignore` handles most of this automatically, but as a rule of thumb:

| Do not commit | Reason |
|---|---|
| `.venv/` or any virtual environment folder | Enormous, machine-specific, reproducible via `pip install` |
| `__pycache__/` and `*.pyc` files | Auto-generated Python bytecode, not source |
| `.ipynb_checkpoints/` | Jupyter auto-save artifacts |
| `.DS_Store` | macOS folder metadata — irrelevant to the project |
| Any file containing API keys, tokens, or passwords | Security — never, ever commit secrets |
| Large binary data files | Use a data store or link externally instead |

If you accidentally commit something that shouldn't be there, tell the team immediately.

---

## Questions?

If something in this document is unclear or you think a convention should change, raise it with the team. Conventions should evolve as the project does — but changes going forward should be agreed upon by the whole team before being adopted.