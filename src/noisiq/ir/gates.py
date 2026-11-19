"""Quantum gates and the standard gate library.

A :class:`Gate` is defined by its unitary matrix. Everything a simulator needs is
derived from that matrix — in particular whether the gate is a Clifford (and its
stabilizer tableau), which is computed with :func:`stim.Tableau.from_unitary_matrix`.
Gate *names* are only used for display and for user-facing noise rules
("add this error to every CX"); simulators never dispatch on names.

Conventions
-----------
* Matrices are **big-endian** in the gate's qubit order: for a gate applied to
  qubits ``(a, b)``, qubit ``a`` is the most significant bit of the matrix index.
  So ``CX`` applied to ``(control, target)`` has the textbook matrix.
* Rotations follow the standard convention ``R_P(θ) = exp(-i θ P / 2)``.
* Parameters are stored as exact floats in :attr:`Gate.params`; they are never
  parsed back out of names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import stim

__all__ = [
    "Gate",
    "is_unitary",
    "unitary",
    "controlled",
    # fixed gates
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SDG",
    "T",
    "TDG",
    "SX",
    "SXDG",
    "CX",
    "CNOT",
    "CY",
    "CZ",
    "CH",
    "SWAP",
    "ISWAP",
    "CCX",
    "TOFFOLI",
    "CCZ",
    "CSWAP",
    "FREDKIN",
    # parametric factories
    "rx",
    "ry",
    "rz",
    "p",
    "u",
    "cp",
    "crx",
    "cry",
    "crz",
    "STANDARD_GATES",
]

_ATOL = 1e-9


def is_unitary(matrix: np.ndarray, atol: float = _ATOL) -> bool:
    """Return True if ``matrix`` is square and satisfies ``U†U = I``."""
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    return bool(np.allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=atol))


def _num_qubits_for_dim(dim: int) -> int:
    n = int(round(np.log2(dim))) if dim > 0 else -1
    if n < 1 or 2**n != dim:
        raise ValueError(f"Gate matrix dimension {dim} is not a power of two >= 2.")
    return n


@dataclass(frozen=True, eq=False)
class Gate:
    """A unitary quantum gate.

    Args:
        name: Identifier, e.g. ``"H"`` or ``"RZ"``. Used for display, noise rules
            and QASM export.
        matrix: Unitary matrix of shape ``(2**k, 2**k)`` (big-endian, see module docs).
        params: Numeric parameters (e.g. rotation angles) — kept exactly.
        label: Optional display label; defaults to the name (plus parameters).
        num_controls: For controlled gates, how many of the leading qubits are
            controls. Purely informational (used by the circuit drawer).
        base: For controlled gates, the gate applied to the targets.

    Raises:
        ValueError: If the matrix is not unitary or its size is not ``2**k``.

    Example:
        >>> import numpy as np
        >>> sqrt_x = Gate("SX", np.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2)
        >>> sqrt_x.is_clifford
        True
    """

    name: str
    matrix: np.ndarray
    params: tuple[float, ...] = ()
    label: str | None = None
    num_controls: int = 0
    base: Gate | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=complex)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f"Gate '{self.name}': matrix must be square, got shape {m.shape}.")
        _num_qubits_for_dim(m.shape[0])
        if not is_unitary(m):
            raise ValueError(f"Gate '{self.name}': matrix is not unitary.")
        m.setflags(write=False)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "params", tuple(float(x) for x in self.params))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Gate name must be a non-empty string.")
        if not 0 <= self.num_controls < self.num_qubits:
            raise ValueError("num_controls must be >= 0 and smaller than the number of qubits.")

    # ------------------------------------------------------------------ basics
    @property
    def num_qubits(self) -> int:
        """Number of qubits the gate acts on."""
        return _num_qubits_for_dim(self.matrix.shape[0])

    @property
    def display_label(self) -> str:
        """Label used by the drawer, e.g. ``"RZ(0.79)"``."""
        if self.label is not None:
            return self.label
        if self.params:
            return f"{self.name}({', '.join(_format_angle(x) for x in self.params)})"
        return self.name

    # --------------------------------------------------------------- Clifford
    @cached_property
    def tableau(self) -> stim.Tableau | None:
        """The stabilizer tableau of this gate, or ``None`` if it is not Clifford."""
        import stim

        try:
            tab = stim.Tableau.from_unitary_matrix(self.matrix, endian="big")
        except ValueError:
            return None
        # Stim's conversion is tolerant of non-Clifford input; verify by reconstruction
        # (equality up to a global phase).
        back = tab.to_unitary_matrix(endian="big")
        overlap = np.trace(back.conj().T @ self.matrix) / self.matrix.shape[0]
        if not np.isclose(abs(overlap), 1.0, atol=1e-8):
            return None
        return tab

    @property
    def is_clifford(self) -> bool:
        """True if the gate maps Paulis to Paulis under conjugation."""
        return self.tableau is not None

    @cached_property
    def symplectic(self) -> np.ndarray | None:
        """Binary ``2k × 2k`` matrix describing how Pauli errors propagate.

        A Pauli error with bit-vector ``v = (x_0..x_{k-1}, z_0..z_{k-1})`` placed
        *before* the gate becomes ``M @ v (mod 2)`` after it. ``None`` for
        non-Clifford gates.
        """
        tab = self.tableau
        if tab is None:
            return None
        k = self.num_qubits
        m = np.zeros((2 * k, 2 * k), dtype=np.uint8)
        for j in range(k):
            xs, zs = tab.x_output(j).to_numpy()
            m[:k, j], m[k:, j] = xs, zs
            xs, zs = tab.z_output(j).to_numpy()
            m[:k, k + j], m[k:, k + j] = xs, zs
        m.setflags(write=False)
        return m

    @property
    def is_pauli(self) -> bool:
        """True if the gate is a Pauli operator (up to global phase)."""
        s = self.symplectic
        return s is not None and bool(np.array_equal(s, np.eye(s.shape[0], dtype=np.uint8)))

    @cached_property
    def pauli_bits(self) -> tuple[np.ndarray, np.ndarray] | None:
        """For Pauli gates, the ``(x, z)`` bit vectors of the Pauli; else ``None``."""
        if not self.is_pauli:
            return None
        k = self.num_qubits
        x = np.zeros(k, dtype=np.uint8)
        z = np.zeros(k, dtype=np.uint8)
        tab = self.tableau
        for j in range(k):
            # P anticommutes with Z_j iff it has an X component on j (sign flips).
            x[j] = tab.z_output(j).sign != 1
            z[j] = tab.x_output(j).sign != 1
        return x, z

    # ------------------------------------------------------------ algebra
    def inverse(self) -> Gate:
        """Return the inverse gate ``U†`` (named sensibly for library gates)."""
        if self.name in _SELF_INVERSE:
            return self
        if self.name in _INVERSE_NAMES:
            return STANDARD_GATES[_INVERSE_NAMES[self.name]]
        if self.name in _PARAMETRIC and self.params:
            if self.name == "U":
                theta, phi, lam = self.params
                return u(-theta, -lam, -phi)
            return _PARAMETRIC[self.name](*(-x for x in self.params))
        name = self.name[:-3] if self.name.endswith("_DG") else f"{self.name}_DG"
        label = None
        if self.label is not None:
            label = self.label[:-1] if self.label.endswith("†") else f"{self.label}†"
        base = self.base.inverse() if self.base is not None else None
        return Gate(name, self.matrix.conj().T, self.params, label, self.num_controls, base)

    def controlled(self, num_controls: int = 1) -> Gate:
        """Return this gate controlled on ``num_controls`` extra leading qubits."""
        return controlled(self, num_controls)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        if self.name != other.name or self.matrix.shape != other.matrix.shape:
            return False
        if len(self.params) != len(other.params) or not np.allclose(self.params, other.params):
            return False
        return bool(np.allclose(self.matrix, other.matrix, atol=_ATOL))

    def __hash__(self) -> int:
        return hash((self.name, self.matrix.shape[0], tuple(round(x, 9) for x in self.params)))

    def __repr__(self) -> str:
        return f"Gate({self.display_label!r}, num_qubits={self.num_qubits})"


def _format_angle(x: float) -> str:
    """Format an angle compactly, recognising simple multiples of π."""
    for den in (1, 2, 3, 4, 6, 8, 16):
        num = x * den / np.pi
        if abs(num - round(num)) < 1e-9 and round(num) != 0:
            n = int(round(num))
            sign = "-" if n < 0 else ""
            n = abs(n)
            top = "π" if n == 1 else f"{n}π"
            return f"{sign}{top}" if den == 1 else f"{sign}{top}/{den}"
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.3g}"


# ============================================================================
# Factories
# ============================================================================


def unitary(matrix: np.ndarray, name: str = "U", label: str | None = None) -> Gate:
    """Create a custom gate from any unitary matrix (big-endian qubit order)."""
    return Gate(name=name, matrix=np.asarray(matrix, dtype=complex), label=label)


def controlled(gate: Gate, num_controls: int = 1) -> Gate:
    """Return ``gate`` controlled on ``num_controls`` leading qubits (all must be |1⟩)."""
    if num_controls < 1:
        raise ValueError("num_controls must be >= 1.")
    base = gate.base if gate.base is not None else gate
    total_controls = gate.num_controls + num_controls
    d = gate.matrix.shape[0]
    dim = d * 2**num_controls
    m = np.eye(dim, dtype=complex)
    m[dim - d :, dim - d :] = gate.matrix
    name = "C" * num_controls + gate.name
    label = None if gate.label is None else gate.label
    return Gate(name, m, gate.params, label, total_controls, base)


_SQ2 = 1 / np.sqrt(2)


def rx(theta: float) -> Gate:
    """``RX(θ) = exp(-iθX/2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return Gate("RX", np.array([[c, -1j * s], [-1j * s, c]]), (theta,))


def ry(theta: float) -> Gate:
    """``RY(θ) = exp(-iθY/2)``."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return Gate("RY", np.array([[c, -s], [s, c]]), (theta,))


def rz(theta: float) -> Gate:
    """``RZ(θ) = exp(-iθZ/2) = diag(e^{-iθ/2}, e^{iθ/2})``."""
    return Gate("RZ", np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]), (theta,))


def p(theta: float) -> Gate:
    """Phase gate ``P(θ) = diag(1, e^{iθ})``. ``P(π/2)=S``, ``P(π/4)=T``."""
    return Gate("P", np.diag([1, np.exp(1j * theta)]), (theta,))


def u(theta: float, phi: float, lam: float) -> Gate:
    """General single-qubit gate ``U(θ, φ, λ)`` (OpenQASM ``u3`` convention)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
    return Gate("U", m, (theta, phi, lam))


def cp(theta: float) -> Gate:
    """Controlled phase ``CP(θ) = diag(1, 1, 1, e^{iθ})``."""
    g = controlled(p(theta))
    return Gate("CP", g.matrix, (theta,), None, 1, p(theta))


def crx(theta: float) -> Gate:
    """Controlled ``RX(θ)``."""
    return Gate("CRX", controlled(rx(theta)).matrix, (theta,), None, 1, rx(theta))


def cry(theta: float) -> Gate:
    """Controlled ``RY(θ)``."""
    return Gate("CRY", controlled(ry(theta)).matrix, (theta,), None, 1, ry(theta))


def crz(theta: float) -> Gate:
    """Controlled ``RZ(θ)``."""
    return Gate("CRZ", controlled(rz(theta)).matrix, (theta,), None, 1, rz(theta))


# ============================================================================
# Fixed gates
# ============================================================================

I = Gate("I", np.eye(2))  # noqa: E741
X = Gate("X", np.array([[0, 1], [1, 0]]))
Y = Gate("Y", np.array([[0, -1j], [1j, 0]]))
Z = Gate("Z", np.array([[1, 0], [0, -1]]))
H = Gate("H", _SQ2 * np.array([[1, 1], [1, -1]]))
S = Gate("S", np.diag([1, 1j]))
SDG = Gate("SDG", np.diag([1, -1j]), label="S†")
T = Gate("T", np.diag([1, np.exp(1j * np.pi / 4)]))
TDG = Gate("TDG", np.diag([1, np.exp(-1j * np.pi / 4)]), label="T†")
SX = Gate("SX", 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]), label="√X")
SXDG = Gate("SXDG", 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]), label="√X†")

CX = Gate("CX", controlled(X).matrix, num_controls=1, base=X)
CNOT = CX
CY = Gate("CY", controlled(Y).matrix, num_controls=1, base=Y)
CZ = Gate("CZ", controlled(Z).matrix, num_controls=1, base=Z)
CH = Gate("CH", controlled(H).matrix, num_controls=1, base=H)
SWAP = Gate("SWAP", np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
ISWAP = Gate("ISWAP", np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]), label="iSWAP")
CCX = Gate("CCX", controlled(X, 2).matrix, num_controls=2, base=X)
TOFFOLI = CCX
CCZ = Gate("CCZ", controlled(Z, 2).matrix, num_controls=2, base=Z)
CSWAP = Gate("CSWAP", controlled(SWAP).matrix, num_controls=1, base=SWAP)
FREDKIN = CSWAP

STANDARD_GATES: dict[str, Gate] = {
    g.name: g for g in (I, X, Y, Z, H, S, SDG, T, TDG, SX, SXDG, CX, CY, CZ, CH, SWAP, ISWAP, CCX, CCZ, CSWAP)
}
"""Fixed (non-parametric) library gates keyed by name."""

_SELF_INVERSE = {"I", "X", "Y", "Z", "H", "CX", "CY", "CZ", "CH", "SWAP", "CCX", "CCZ", "CSWAP"}
_INVERSE_NAMES = {"S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T", "SX": "SXDG", "SXDG": "SX"}
_PARAMETRIC = {"RX": rx, "RY": ry, "RZ": rz, "P": p, "U": u, "CP": cp, "CRX": crx, "CRY": cry, "CRZ": crz}


def gate_from_name(name: str, params: Sequence[float] = ()) -> Gate:
    """Look up a library gate by (case-insensitive) name, e.g. ``("rz", [0.1])``."""
    key = name.upper()
    aliases = {
        "CNOT": "CX",
        "TOFFOLI": "CCX",
        "FREDKIN": "CSWAP",
        "S_DAG": "SDG",
        "T_DAG": "TDG",
        "SDAG": "SDG",
        "TDAG": "TDG",
        "ID": "I",
        "PHASE": "P",
        "U3": "U",
        "U1": "P",
        "CU1": "CP",
    }
    key = aliases.get(key, key)
    if key in _PARAMETRIC:
        return _PARAMETRIC[key](*params)
    if key in STANDARD_GATES:
        if params:
            raise ValueError(f"Gate '{name}' takes no parameters.")
        return STANDARD_GATES[key]
    raise KeyError(f"Unknown gate '{name}'.")
