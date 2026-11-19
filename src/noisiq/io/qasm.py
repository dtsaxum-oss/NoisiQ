"""OpenQASM 2.0 import and export.

Import supports: multiple ``qreg``/``creg``, all ``qelib1.inc`` gates (plus ``sx``,
``sxdg``, ``p``, ``cp``, ``rzz``, ``rxx``), parameter expressions (``pi/4``, ``-2*pi/3``,
``sin(...)``...), register broadcasting (``h q;``, ``measure q -> c;``), ``reset``,
``barrier``, ``if (creg == n) <op>;`` and user ``gate`` definitions (expanded inline).

Export writes one ``creg`` per user register (``Circuit.add_register``) and one 1-bit
``creg`` per remaining classical bit, so single-bit conditions survive the round trip.
X/Y-basis measurements are written as basis rotations around a Z measurement.
"""

from __future__ import annotations

import ast
import math
import operator
import re

import numpy as np

from ..ir import gates as G
from ..ir.circuit import Circuit
from ..ir.gates import Gate
from ..ir.instructions import Barrier, Condition, Delay, Measure, Operation, Reset

__all__ = ["from_qasm", "to_qasm", "QASMParseError", "QASMExportError"]


class QASMParseError(ValueError):
    """The QASM source could not be parsed."""


class QASMExportError(ValueError):
    """The circuit cannot be expressed in OpenQASM 2.0."""


# ============================================================================
# expression evaluation
# ============================================================================

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
}


def _eval(expr: str, env: dict[str, float] | None = None) -> float:
    env = {"pi": math.pi, **(env or {})}
    expr = expr.strip().replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise QASMParseError(f"Bad parameter expression {expr!r}.") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return float(env[node.id])
            raise QASMParseError(f"Unknown identifier {node.id!r} in {expr!r}.")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise QASMParseError(f"Unsupported expression {expr!r}.")

    return ev(tree)


def _split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


# ============================================================================
# gate table
# ============================================================================


def _rzz(t):
    return Gate("RZZ", np.diag(np.exp(-0.5j * t * np.array([1, -1, -1, 1]))), (t,))


def _rxx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    m = np.array([[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]])
    return Gate("RXX", m, (t,))


def _cu3(theta, phi, lam):
    g = G.controlled(G.u(theta, phi, lam))
    return Gate("CU3", g.matrix, (theta, phi, lam), None, 1, G.u(theta, phi, lam))


_BUILTIN: dict[str, tuple[int, object]] = {
    # name: (n_params, factory or Gate)
    "id": (0, G.I),
    "x": (0, G.X),
    "y": (0, G.Y),
    "z": (0, G.Z),
    "h": (0, G.H),
    "s": (0, G.S),
    "sdg": (0, G.SDG),
    "t": (0, G.T),
    "tdg": (0, G.TDG),
    "sx": (0, G.SX),
    "sxdg": (0, G.SXDG),
    "cx": (0, G.CX),
    "CX": (0, G.CX),
    "cy": (0, G.CY),
    "cz": (0, G.CZ),
    "ch": (0, G.CH),
    "swap": (0, G.SWAP),
    "ccx": (0, G.CCX),
    "cswap": (0, G.CSWAP),
    "iswap": (0, G.ISWAP),
    "ccz": (0, G.CCZ),
    "rx": (1, G.rx),
    "ry": (1, G.ry),
    "rz": (1, G.rz),
    "p": (1, G.p),
    "u1": (1, G.p),
    "phase": (1, G.p),
    "cp": (1, G.cp),
    "cu1": (1, G.cp),
    "cphase": (1, G.cp),
    "crx": (1, G.crx),
    "cry": (1, G.cry),
    "crz": (1, G.crz),
    "rzz": (1, _rzz),
    "rxx": (1, _rxx),
    "u2": (2, lambda phi, lam: G.u(math.pi / 2, phi, lam)),
    "u3": (3, G.u),
    "u": (3, G.u),
    "U": (3, G.u),
    "cu3": (3, _cu3),
}


# ============================================================================
# import
# ============================================================================

_GATE_DEF = re.compile(r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([^{]*)\{([^}]*)\}", re.S)
_REG = re.compile(r"^(qreg|creg)\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_REF = re.compile(r"^(\w+)\s*(?:\[\s*(\d+)\s*\])?$")
_IF = re.compile(r"^if\s*\(\s*(\w+)\s*==\s*(\d+)\s*\)\s*(.+)$", re.S)
_CALL = re.compile(r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s+(.+)$", re.S)


def from_qasm(source: str) -> Circuit:
    """Parse an OpenQASM 2.0 program into a :class:`~noisiq.Circuit`.

    Raises:
        QASMParseError: On unsupported or malformed statements (with the statement text).
    """
    src = re.sub(r"//[^\n]*", "", source)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    macros: dict[str, tuple[list[str], list[str], str]] = {}
    for m in _GATE_DEF.finditer(src):
        name, params, args, body = m.group(1), m.group(2) or "", m.group(3), m.group(4)
        macros[name] = (
            [p.strip() for p in params.split(",") if p.strip()],
            [a.strip() for a in args.split(",") if a.strip()],
            body,
        )
    src = _GATE_DEF.sub("", src)
    statements = [s.strip() for s in src.split(";") if s.strip()]

    qregs: dict[str, tuple[int, int]] = {}
    cregs: dict[str, tuple[int, int]] = {}
    nq = nc = 0
    for st in statements:
        m = _REG.match(st)
        if m:
            kind, name, size = m.group(1), m.group(2), int(m.group(3))
            if name in qregs or name in cregs:
                raise QASMParseError(f"Register '{name}' declared twice.")
            if kind == "qreg":
                qregs[name] = (nq, size)
                nq += size
            else:
                cregs[name] = (nc, size)
                nc += size
    if nq == 0:
        raise QASMParseError("No qreg declared.")
    circuit = Circuit(nq)
    for name, (off, size) in cregs.items():
        reg = circuit.add_register(name, size)
        assert reg.offset == off

    def refs(token: str, regs, what) -> list[int]:
        m = _REF.match(token.strip())
        if not m or m.group(1) not in regs:
            raise QASMParseError(f"Unknown {what} '{token}'.")
        off, size = regs[m.group(1)]
        if m.group(2) is None:
            return list(range(off, off + size))
        i = int(m.group(2))
        if i >= size:
            raise QASMParseError(f"Index {i} out of range for {what} '{m.group(1)}[{size}]'.")
        return [off + i]

    def apply_gate(name: str, params: list[float], qubit_lists: list[list[int]], cond, stmt: str) -> None:
        widths = {len(x) for x in qubit_lists if len(x) > 1}
        if len(widths) > 1:
            raise QASMParseError(f"Register size mismatch in '{stmt}'.")
        reps = widths.pop() if widths else 1
        for r in range(reps):
            qs = [x[r] if len(x) > 1 else x[0] for x in qubit_lists]
            if name in macros:
                pnames, anames, body = macros[name]
                if len(pnames) != len(params) or len(anames) != len(qs):
                    raise QASMParseError(f"Wrong number of arguments for gate '{name}' in '{stmt}'.")
                env = dict(zip(pnames, params))
                amap = dict(zip(anames, qs))
                for sub in [b.strip() for b in body.split(";") if b.strip()]:
                    run_call(sub, cond, env, amap)
                continue
            if name not in _BUILTIN:
                raise QASMParseError(f"Unsupported gate '{name}' in '{stmt}'.")
            npar, fac = _BUILTIN[name]
            if len(params) != npar:
                raise QASMParseError(f"Gate '{name}' expects {npar} parameter(s) in '{stmt}'.")
            gate = fac(*params) if npar else fac
            try:
                circuit.append(gate, qs, condition=cond)
            except ValueError as e:
                raise QASMParseError(f"{e} (in '{stmt}')") from e

    def run_call(stmt: str, cond, env=None, amap=None) -> None:
        if stmt.startswith("barrier"):
            if amap is None:
                args = _split_args(stmt[len("barrier") :])
                qs = [q for a in args for q in refs(a, qregs, "qubit register")]
                circuit.barrier(qs)
            return
        m = _CALL.match(stmt)
        if not m:
            raise QASMParseError(f"Cannot parse statement '{stmt}'.")
        name, ptxt, atxt = m.group(1), m.group(2), m.group(3)
        params = [_eval(p, env) for p in _split_args(ptxt)] if ptxt else []
        args = _split_args(atxt)
        if amap is not None:
            qlists = [[amap[a]] if a in amap else _fail(f"Unknown argument '{a}' in gate body") for a in args]
        else:
            qlists = [refs(a, qregs, "qubit register") for a in args]
        apply_gate(name, params, qlists, cond, stmt)

    for st in statements:
        head = st.split()[0]
        if head in ("OPENQASM", "include") or _REG.match(st):
            continue
        cond = None
        body = st
        m = _IF.match(st)
        if m:
            reg, val = m.group(1), int(m.group(2))
            if reg not in cregs:
                raise QASMParseError(f"Unknown classical register '{reg}' in '{st}'.")
            off, size = cregs[reg]
            cond = Condition(tuple(range(off, off + size)), val)
            body = m.group(3).strip()
        head = body.split()[0].split("(")[0]
        if head == "measure":
            mm = re.match(r"measure\s+(.+?)\s*->\s*(.+)$", body)
            if not mm or cond is not None:
                raise QASMParseError(f"Unsupported measure statement '{st}'.")
            qs = refs(mm.group(1), qregs, "qubit register")
            cs = refs(mm.group(2), cregs, "classical register")
            if len(qs) != len(cs):
                raise QASMParseError(f"Size mismatch in '{st}'.")
            circuit.measure(qs, cs)
        elif head == "reset":
            if cond is not None:
                raise QASMParseError(f"Conditional reset is not supported: '{st}'.")
            circuit.reset(refs(body.split(None, 1)[1], qregs, "qubit register"))
        elif head == "opaque":
            raise QASMParseError(f"Opaque gates are not supported: '{st}'.")
        else:
            run_call(body, cond)
    return circuit


def _fail(msg: str):
    raise QASMParseError(msg)


# ============================================================================
# export
# ============================================================================

_EXPORT_NAMES = {
    "I": "id",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "S": "s",
    "SDG": "sdg",
    "T": "t",
    "TDG": "tdg",
    "SX": "sx",
    "SXDG": "sxdg",
    "CX": "cx",
    "CY": "cy",
    "CZ": "cz",
    "CH": "ch",
    "SWAP": "swap",
    "CCX": "ccx",
    "CSWAP": "cswap",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "P": "u1",
    "U": "u3",
    "CP": "cu1",
    "CRZ": "crz",
    "CRX": "crx",
    "CRY": "cry",
    "RZZ": "rzz",
    "RXX": "rxx",
    "CU3": "cu3",
}


def _num(x: float) -> str:
    for den in (1, 2, 3, 4, 6, 8, 16):
        k = x * den / math.pi
        if abs(k - round(k)) < 1e-12 and round(k) != 0:
            k = int(round(k))
            num = "pi" if abs(k) == 1 else f"{abs(k)}*pi"
            s = ("-" if k < 0 else "") + num
            return s if den == 1 else f"{s}/{den}"
    return repr(float(x))


def to_qasm(circuit: Circuit) -> str:
    """Export a circuit to OpenQASM 2.0 text.

    Raises:
        QASMExportError: For custom unitary gates, delays, or multi-bit conditions that do
            not correspond to a whole register.
    """
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.n_qubits}];"]
    bitname: dict[int, str] = {}
    reg_of: dict[tuple[int, ...], str] = {}
    covered = set()
    for reg in sorted(circuit.registers, key=lambda r: r.offset):
        lines.append(f"creg {reg.name}[{reg.size}];")
        reg_of[reg.clbits] = reg.name
        for i, c in enumerate(reg):
            bitname[c] = f"{reg.name}[{i}]"
            covered.add(c)
    used = {r.name for r in circuit.registers}
    for c in range(circuit.n_clbits):
        if c not in covered:
            name = f"c{c}"
            while name in used:
                name = "_" + name
            lines.append(f"creg {name}[1];")
            bitname[c] = f"{name}[0]"
            reg_of[(c,)] = name

    body: list[str] = []
    for _, inst in circuit.ordered():
        prefix = ""
        if isinstance(inst, Operation) and inst.condition is not None:
            key = inst.condition.clbits
            if key not in reg_of:
                raise QASMExportError(
                    f"Condition on clbits {list(key)} is not a whole register; OpenQASM 2 cannot express it."
                )
            prefix = f"if({reg_of[key]}=={inst.condition.value}) "
        if isinstance(inst, Operation):
            g = inst.gate
            if g.name == "ISWAP":  # = (S⊗S) H_a CX(a,b) CX(b,a) H_b, exact
                a, b = inst.qubits
                for line in (
                    f"s q[{a}];",
                    f"s q[{b}];",
                    f"h q[{a}];",
                    f"cx q[{a}],q[{b}];",
                    f"cx q[{b}],q[{a}];",
                    f"h q[{b}];",
                ):
                    body.append(prefix + line)
                continue
            if g.name not in _EXPORT_NAMES and g.name != "CCZ":
                raise QASMExportError(f"Gate '{g.name}' has no OpenQASM 2 equivalent (custom unitary).")
            qs = ",".join(f"q[{q}]" for q in inst.qubits)
            if g.name == "CCZ":
                a, b, c = inst.qubits
                body.append(f"{prefix}h q[{c}];")
                body.append(f"{prefix}ccx q[{a}],q[{b}],q[{c}];")
                body.append(f"{prefix}h q[{c}];")
                continue
            params = f"({','.join(_num(p) for p in g.params)})" if g.params else ""
            body.append(f"{prefix}{_EXPORT_NAMES[g.name]}{params} {qs};")
        elif isinstance(inst, Measure):
            q, c = inst.qubit, bitname[inst.clbit]
            if inst.basis == "X":
                body.append(f"h q[{q}];")
            elif inst.basis == "Y":
                body += [f"sdg q[{q}];", f"h q[{q}];"]
            body.append(f"measure q[{q}] -> {c};")
            if inst.reset:
                body.append(f"reset q[{q}];")
            elif inst.basis == "X":
                body.append(f"h q[{q}];")
            elif inst.basis == "Y":
                body += [f"h q[{q}];", f"s q[{q}];"]
        elif isinstance(inst, Reset):
            body.append(f"reset q[{inst.qubit}];")
        elif isinstance(inst, Barrier):
            body.append("barrier " + ",".join(f"q[{q}]" for q in inst.qubits) + ";")
        elif isinstance(inst, Delay):
            raise QASMExportError("OpenQASM 2 has no delay instruction.")
    return "\n".join(lines + body) + "\n"
