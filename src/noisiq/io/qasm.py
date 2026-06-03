"""
OpenQASM 2.0 parser for importing circuits into the NoisiQ IR.

Supported statements:
  - qreg declarations           (one or more named registers)
  - creg declarations           (classical registers; wired to ClassicalRegister)
  - Single-qubit gates          h, x, y, z, s, sdg, t, tdg, id
  - Two-qubit gates             cx, cnot, cz, swap, cs
  - Three-qubit gates           ccz
  - measure q[i] -> c[j]        parsed into Measurement ops
  - if (creg==val) gate q[k]    parsed into ConditionalOp (single-bit cregs only)
  - barrier                     silently ignored
  - include / OPENQASM header   silently ignored
  - Line comments               // ...

Unsupported (raises QASMParseError):
  - reset                       raises QASMParseError (not yet implemented)
  - Custom gate definitions     gate ...
  - Parameterized gates         rx(theta), ry(theta), rz(theta) etc.
  - if with multi-bit creg comparison  (only 1-bit cond registers in M5)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..ir import Circuit
from ..ir import gates as ir_gates
from ..ir.classical import ClassicalBit, ClassicalRegister, Measurement, ConditionalOp
from ..ir.circuit import Operation, _get_qubits


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
    "openqasm", "include", "barrier",
})

# Regex for a qubit/cbit reference: regname[index]
_QUBIT_REF = re.compile(r"(\w+)\[(\d+)\]")

# Regex for a register declaration: qreg/creg name[size]
_REG_DECL = re.compile(r"(?:qreg|creg)\s+(\w+)\[(\d+)\]", re.IGNORECASE)

# Regex for a measure statement: measure qreg[i] -> creg[j]
_MEASURE_RE = re.compile(
    r"measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\]",
    re.IGNORECASE,
)

# Regex for an if statement: if (creg==value) rest
_IF_RE = re.compile(
    r"if\s*\(\s*(\w+)\s*==\s*(\d+)\s*\)\s*(.+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def from_qasm(qasm_str: str) -> Circuit:
    """Parse an OpenQASM 2.0 string and return a NoisiQ Circuit.

    Classical register declarations (``creg``) are added to
    ``circuit.classical_registers``.  ``measure`` statements produce
    ``Measurement`` operations.  Single-bit ``if`` conditions produce
    ``ConditionalOp`` operations.

    Args:
        qasm_str: A valid OpenQASM 2.0 program as a plain string.

    Returns:
        Circuit with gates, measurements, and conditional ops in source order.

    Raises:
        QASMParseError: If the source contains unsupported statements or
                        references an undeclared register.

    Example:
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q[0] -> c[0];
        ... measure q[1] -> c[1];
        ... '''
        >>> circuit = from_qasm(qasm)
        >>> circuit.n_qubits
        2
    """
    lines = _strip_comments(qasm_str)
    statements = _split_statements(lines)

    # -----------------------------------------------------------------------
    # First pass: collect qreg and creg declarations
    # -----------------------------------------------------------------------
    qregs: Dict[str, Tuple[int, int]] = {}  # reg_name → (offset, size)
    cregs: Dict[str, ClassicalRegister] = {}  # reg_name → ClassicalRegister
    total_qubits = 0
    total_cbits = 0

    for stmt in statements:
        tokens = stmt.split()
        if not tokens:
            continue
        keyword = tokens[0].lower()

        if keyword == "qreg":
            reg_name, size = _parse_reg_decl(stmt, "qreg")
            if reg_name in qregs:
                raise QASMParseError(
                    f"Duplicate qreg declaration for '{reg_name}'."
                )
            qregs[reg_name] = (total_qubits, size)
            total_qubits += size

        elif keyword == "creg":
            reg_name, size = _parse_reg_decl(stmt, "creg")
            if reg_name in cregs:
                raise QASMParseError(
                    f"Duplicate creg declaration for '{reg_name}'."
                )
            cregs[reg_name] = ClassicalRegister(
                name=reg_name, size=size, offset=total_cbits
            )
            total_cbits += size

    if total_qubits == 0:
        raise QASMParseError("No qreg declaration found — cannot build a Circuit.")

    circuit = Circuit(n_qubits=total_qubits)

    # Register classical registers on the circuit in declaration order.
    for reg in cregs.values():
        circuit.classical_registers.append(reg)

    # -----------------------------------------------------------------------
    # Second pass: apply gates, measurements, and conditionals in order
    # -----------------------------------------------------------------------
    for stmt in statements:
        tokens = stmt.split()
        if not tokens:
            continue
        keyword = tokens[0].lower()

        if keyword in _IGNORED_KEYWORDS or keyword in ("qreg", "creg"):
            continue

        # ------------------------------------------------------------------
        # reset q[i] — not yet supported; fail loudly rather than silently
        # dropping the operation (which would produce wrong circuit semantics)
        # ------------------------------------------------------------------
        if keyword == "reset":
            raise QASMParseError(
                "'reset' is not yet supported by NoisiQ. "
                "Remove or replace 'reset' statements before parsing."
            )

        # ------------------------------------------------------------------
        # measure q[i] -> c[j]
        # ------------------------------------------------------------------
        if keyword == "measure":
            _apply_measure(stmt, circuit, qregs, cregs)
            continue

        # ------------------------------------------------------------------
        # if (creg==val) gate q[k]
        # ------------------------------------------------------------------
        if keyword == "if":
            _apply_if(stmt, circuit, qregs, cregs)
            continue

        # ------------------------------------------------------------------
        # Parameterized gate call e.g. rx(pi/2) — not supported
        # ------------------------------------------------------------------
        if "(" in keyword:
            gate_base = keyword.split("(")[0]
            if gate_base not in _GATE_MAP:
                raise QASMParseError(
                    f"Parameterized gate '{keyword}' is not supported. "
                    f"NoisiQ currently supports fixed single- and two-qubit gates only."
                )

        if keyword not in _GATE_MAP:
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
        global_qubits = _resolve_qubits(qubit_args, qregs, stmt)

        expected = gate.num_qubits
        if len(global_qubits) != expected:
            raise QASMParseError(
                f"Gate '{keyword}' expects {expected} qubit(s), "
                f"got {len(global_qubits)} in: '{stmt}'"
            )

        circuit.add_gate(gate, tuple(global_qubits))

    return circuit


# ---------------------------------------------------------------------------
# Measurement and conditional helpers
# ---------------------------------------------------------------------------

def _apply_measure(
    stmt: str,
    circuit: Circuit,
    qregs: Dict[str, Tuple[int, int]],
    cregs: Dict[str, ClassicalRegister],
) -> None:
    """Parse 'measure q[i] -> c[j]' and append a Measurement to circuit."""
    m = _MEASURE_RE.search(stmt)
    if not m:
        raise QASMParseError(f"Malformed measure statement: '{stmt}'")

    qreg_name, q_local = m.group(1), int(m.group(2))
    creg_name, c_local = m.group(3), int(m.group(4))

    if qreg_name not in qregs:
        raise QASMParseError(
            f"Qubit register '{qreg_name}' used in measure statement "
            f"was not declared with qreg."
        )
    if creg_name not in cregs:
        raise QASMParseError(
            f"Classical register '{creg_name}' used in measure statement "
            f"was not declared with creg."
        )

    qreg_offset, qreg_size = qregs[qreg_name]
    if q_local >= qreg_size:
        raise QASMParseError(
            f"Qubit index {q_local} is out of range for register "
            f"'{qreg_name}' (size {qreg_size}) in: '{stmt}'"
        )
    global_qubit = qreg_offset + q_local

    creg = cregs[creg_name]
    if not 0 <= c_local < creg.size:
        raise QASMParseError(
            f"Classical bit index {c_local} out of range for "
            f"register '{creg_name}' (size {creg.size})."
        )

    cbit = creg[c_local]
    circuit.measure(global_qubit, cbit)


def _apply_if(
    stmt: str,
    circuit: Circuit,
    qregs: Dict[str, Tuple[int, int]],
    cregs: Dict[str, ClassicalRegister],
) -> None:
    """Parse 'if (creg==val) gate q[k]' and append a ConditionalOp to circuit."""
    m = _IF_RE.match(stmt)
    if not m:
        raise QASMParseError(f"Malformed if statement: '{stmt}'")

    creg_name = m.group(1)
    value = int(m.group(2))
    inner_stmt = m.group(3).strip()

    if creg_name not in cregs:
        raise QASMParseError(
            f"Classical register '{creg_name}' used in if condition "
            f"was not declared with creg."
        )
    creg = cregs[creg_name]

    # For M5 we support only 1-bit registers (condition on a single bit).
    if creg.size != 1:
        raise QASMParseError(
            f"if conditions on multi-bit registers are not yet supported "
            f"(register '{creg_name}' has size {creg.size}). "
            f"Use a 1-bit creg or decompose the comparison."
        )
    if value not in (0, 1):
        raise QASMParseError(
            f"if condition value must be 0 or 1 for a 1-bit register; "
            f"got {value} for '{creg_name}'."
        )

    cbit = creg[0]

    # Parse the inner gate statement.
    inner_tokens = inner_stmt.split()
    if not inner_tokens:
        raise QASMParseError(f"Empty inner statement in: '{stmt}'")
    inner_keyword = inner_tokens[0].lower()

    if inner_keyword not in _GATE_MAP:
        raise QASMParseError(
            f"Gate '{inner_keyword}' inside if statement is not supported."
        )

    gate = _GATE_MAP[inner_keyword]
    qubit_args = _parse_qubit_args(inner_stmt, inner_keyword)
    global_qubits = _resolve_qubits(qubit_args, qregs, inner_stmt)

    if len(global_qubits) != gate.num_qubits:
        raise QASMParseError(
            f"Gate '{inner_keyword}' expects {gate.num_qubits} qubit(s), "
            f"got {len(global_qubits)} in if body: '{inner_stmt}'"
        )

    # Schedule: find the latest t across the target qubits.
    qubit_last_t: Dict[int, int] = {}
    for op in circuit.operations:
        for q in _get_qubits(op):
            if op.t > qubit_last_t.get(q, -1):
                qubit_last_t[q] = op.t
    t = max((qubit_last_t.get(q, -1) for q in global_qubits), default=-1) + 1

    inner = Operation(gate=gate, qubits=tuple(global_qubits), t=t)
    circuit.operations.append(ConditionalOp(inner=inner, condition=cbit, value=value))


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


def _parse_reg_decl(stmt: str, kind: str) -> Tuple[str, int]:
    """Parse 'qreg/creg name[size]' → (name, size).

    Raises:
        QASMParseError: If the declaration is malformed or size < 1.
    """
    m = re.search(rf"{kind}\s+(\w+)\[(\d+)\]", stmt, re.IGNORECASE)
    if not m:
        raise QASMParseError(f"Malformed {kind} declaration: '{stmt}'")
    name, size = m.group(1), int(m.group(2))
    if size < 1:
        raise QASMParseError(
            f"{kind} '{name}' has size {size}; register size must be >= 1."
        )
    return name, size


def _parse_qubit_args(stmt: str, gate_name: str) -> List[Tuple[str, int]]:
    """Extract qubit references after the gate name.

    Returns list of (register_name, index) tuples in argument order.

    Raises:
        QASMParseError: If no qubit references are found.
    """
    rest = stmt[len(gate_name):].strip()
    refs = _QUBIT_REF.findall(rest)
    if not refs:
        raise QASMParseError(
            f"No qubit references found in statement: '{stmt}'"
        )
    return [(name, int(idx)) for name, idx in refs]


def _resolve_qubits(
    qubit_args: List[Tuple[str, int]],
    registers: Dict[str, Tuple[int, int]],
    stmt: str,
) -> List[int]:
    """Convert (register_name, index) pairs to global qubit indices.

    Raises:
        QASMParseError: If a register name is undeclared or the local index
                        exceeds the declared register size.
    """
    global_indices = []
    for reg_name, local_idx in qubit_args:
        if reg_name not in registers:
            raise QASMParseError(
                f"Qubit register '{reg_name}' used in '{stmt}' "
                f"was not declared with qreg."
            )
        offset, size = registers[reg_name]
        if local_idx >= size:
            raise QASMParseError(
                f"Qubit index {local_idx} is out of range for register "
                f"'{reg_name}' (size {size}) in: '{stmt}'"
            )
        global_indices.append(offset + local_idx)
    return global_indices
