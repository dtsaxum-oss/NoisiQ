"""
OpenQASM 2.0 parser for importing circuits into the NoisiQ IR.

Supported statements:
  - qreg declarations           (one or more named registers)
  - Single-qubit gates          h, x, y, z, s, sdg, t, tdg, id
  - Two-qubit gates             cx, cnot, cz, swap, cs
  - Three-qubit gates           ccz
  - barrier                     silently ignored
  - measure / reset             silently ignored
  - include / OPENQASM header   silently ignored
  - Line comments               // ...

Unsupported (raises QASMParseError):
  - Custom gate definitions     gate ...
  - Classical control           if (...)
  - Parameterized gates         rx(theta), ry(theta), rz(theta) etc.
  - creg declarations           (classical registers have no IR equivalent)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..ir import Circuit
from ..ir import gates as ir_gates


class QASMParseError(ValueError):
    """Raised when a QASM string cannot be parsed into a NoisiQ Circuit."""


# ---------------------------------------------------------------------------
# Gate name → NoisiQ Gate mapping
# ---------------------------------------------------------------------------

_GATE_MAP: Dict[str, object] = {
    "h":    ir_gates.H,
    "x":    ir_gates.X,
    "y":    ir_gates.Y,
    "z":    ir_gates.Z,
    "s":    ir_gates.S,
    "sdg":  ir_gates.S_DAG,
    "t":    ir_gates.T,
    "tdg":  ir_gates.T_DAG,
    "id":   ir_gates.I,
    "i":    ir_gates.I,
    "cx":   ir_gates.CNOT,
    "cnot": ir_gates.CNOT,
    "cz":   ir_gates.CZ,
    "swap": ir_gates.SWAP,
    "cs":   ir_gates.CS,
    "ccz":  ir_gates.CCZ,
}

# Statements that are legal but carry no information for the IR
_IGNORED_KEYWORDS = frozenset({
    "openqasm", "include", "barrier", "measure", "reset", "creg",
})

# Regex for a qubit reference: regname[index]
_QUBIT_REF = re.compile(r"(\w+)\[(\d+)\]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_qasm(qasm_str: str) -> Circuit:
    """Parse an OpenQASM 2.0 string and return a NoisiQ Circuit.

    Args:
        qasm_str: A valid OpenQASM 2.0 program as a plain string.

    Returns:
        Circuit with gates applied in the order they appear in the QASM source.

    Raises:
        QASMParseError: If the source contains unsupported statements or
                        references an undeclared qubit register.

    Example:
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[3];
        ... h q[0];
        ... cx q[0], q[1];
        ... cx q[1], q[2];
        ... '''
        >>> circuit = from_qasm(qasm)
        >>> circuit.n_qubits
        3
    """
    lines = _strip_comments(qasm_str)
    statements = _split_statements(lines)

    # First pass: collect all qreg declarations to build qubit index map
    registers: Dict[str, int] = {}   # reg_name → starting global qubit index
    total_qubits = 0

    for stmt in statements:
        tokens = stmt.split()
        if not tokens:
            continue
        keyword = tokens[0].lower()

        if keyword == "qreg":
            reg_name, size = _parse_register_decl(stmt)
            if reg_name in registers:
                raise QASMParseError(
                    f"Duplicate qreg declaration for '{reg_name}'."
                )
            registers[reg_name] = total_qubits
            total_qubits += size

    if total_qubits == 0:
        raise QASMParseError("No qreg declaration found — cannot build a Circuit.")

    circuit = Circuit(n_qubits=total_qubits)

    # Second pass: apply gates in order
    for stmt in statements:
        tokens = stmt.split()
        if not tokens:
            continue
        keyword = tokens[0].lower()

        if keyword in _IGNORED_KEYWORDS or keyword == "qreg":
            continue

        # Parameterized gate call e.g. rx(pi/2) — not supported
        if "(" in keyword:
            gate_base = keyword.split("(")[0]
            if gate_base not in _GATE_MAP:
                raise QASMParseError(
                    f"Parameterized gate '{keyword}' is not supported. "
                    f"NoisiQ currently supports fixed single- and two-qubit gates only."
                )

        if keyword not in _GATE_MAP:
            # Check for "gate" definition keyword
            if keyword == "gate":
                raise QASMParseError(
                    f"Custom gate definitions ('gate ...') are not supported. "
                    f"Decompose the gate into primitive operations before importing."
                )
            raise QASMParseError(
                f"Unsupported gate or statement: '{keyword}'. "
                f"Supported gates: {sorted(_GATE_MAP)}."
            )

        gate = _GATE_MAP[keyword]
        qubit_args = _parse_qubit_args(stmt, keyword)
        global_qubits = _resolve_qubits(qubit_args, registers, stmt)

        expected = gate.num_qubits
        if len(global_qubits) != expected:
            raise QASMParseError(
                f"Gate '{keyword}' expects {expected} qubit(s), "
                f"got {len(global_qubits)} in: '{stmt}'"
            )

        circuit.add_gate(gate, tuple(global_qubits))

    return circuit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_comments(qasm_str: str) -> str:
    """Remove // line comments from the QASM source."""
    lines = []
    for line in qasm_str.splitlines():
        if "//" in line:
            line = line[:line.index("//")]
        lines.append(line)
    return "\n".join(lines)


def _split_statements(source: str) -> List[str]:
    """Split source text into individual statements on ';' boundaries."""
    return [s.strip() for s in source.split(";") if s.strip()]


def _parse_register_decl(stmt: str) -> Tuple[str, int]:
    """Parse 'qreg name[size]' → (name, size).

    Raises:
        QASMParseError: If the declaration is malformed.
    """
    m = re.search(r"qreg\s+(\w+)\[(\d+)\]", stmt, re.IGNORECASE)
    if not m:
        raise QASMParseError(f"Malformed qreg declaration: '{stmt}'")
    return m.group(1), int(m.group(2))


def _parse_qubit_args(stmt: str, gate_name: str) -> List[Tuple[str, int]]:
    """Extract qubit references after the gate name.

    Returns list of (register_name, index) tuples in argument order.

    Raises:
        QASMParseError: If no qubit references are found.
    """
    # Everything after the gate name token
    rest = stmt[len(gate_name):].strip()
    refs = _QUBIT_REF.findall(rest)
    if not refs:
        raise QASMParseError(
            f"No qubit references found in statement: '{stmt}'"
        )
    return [(name, int(idx)) for name, idx in refs]


def _resolve_qubits(
    qubit_args: List[Tuple[str, int]],
    registers: Dict[str, int],
    stmt: str,
) -> List[int]:
    """Convert (register_name, index) pairs to global qubit indices.

    Raises:
        QASMParseError: If a register name is undeclared.
    """
    global_indices = []
    for reg_name, local_idx in qubit_args:
        if reg_name not in registers:
            raise QASMParseError(
                f"Qubit register '{reg_name}' used in '{stmt}' "
                f"was not declared with qreg."
            )
        global_indices.append(registers[reg_name] + local_idx)
    return global_indices
