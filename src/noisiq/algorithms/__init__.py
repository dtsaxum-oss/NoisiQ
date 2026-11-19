"""Ready-made textbook circuits, parameterized by size.

Every function returns an ordinary :class:`~noisiq.Circuit`, so the result can be
inspected, extended, combined with noise and visualized like any user circuit.

========================  ==============================================  ==========
function                  what it builds                                  Clifford?
========================  ==============================================  ==========
``bell``                  one of the four Bell states                     yes
``ghz``                   n-qubit GHZ state (chain/star/log-depth)        yes
``deutsch_jozsa``         DJ with a constant or balanced oracle           yes
``bernstein_vazirani``    BV for a secret bitstring                       yes
``qft``                   quantum Fourier transform (and inverse)         no
``grover``                Grover search for marked bitstrings             no (n>2)
``phase_estimation``      QPE of the phase gate P(2πφ)                    no
``repetition_code``       bit-/phase-flip code memory with syndrome       yes
``teleportation``         teleport a prepared state with feed-forward     yes*
========================  ==============================================  ==========

\\* Clifford when the input state is prepared with Clifford gates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Union

from ..ir import gates as G
from ..ir.circuit import Circuit

__all__ = [
    "bell",
    "ghz",
    "deutsch_jozsa",
    "bernstein_vazirani",
    "qft",
    "grover",
    "phase_estimation",
    "repetition_code",
    "teleportation",
]


def _bits(s: Union[str, Sequence[int]], n: int | None = None) -> list[int]:
    b = [int(c) for c in s] if isinstance(s, str) else [int(c) for c in s]
    if any(x not in (0, 1) for x in b):
        raise ValueError(f"Not a bitstring: {s!r}")
    if n is not None and len(b) != n:
        raise ValueError(f"Expected {n} bits, got {len(b)}.")
    return b


def bell(kind: str = "phi+", *, measure: bool = False) -> Circuit:
    """Bell state ``"phi+" | "phi-" | "psi+" | "psi-"`` on two qubits."""
    kinds = ("phi+", "phi-", "psi+", "psi-")
    if kind not in kinds:
        raise ValueError(f"kind must be one of {kinds}.")
    c = Circuit(2, name=f"bell_{kind}")
    if kind in ("psi+", "psi-"):
        c.x(1)
    if kind in ("phi-", "psi-"):
        c.x(0)
    c.h(0).cx(0, 1)
    return c.measure_all() if measure else c


def ghz(n: int, *, layout: str = "chain", uncompute: bool = False, measure: bool = False) -> Circuit:
    """GHZ state ``(|0…0⟩ + |1…1⟩)/√2``.

    Args:
        n: Number of qubits (>= 2).
        layout: ``"chain"`` (CX q→q+1, depth n), ``"star"`` (all CX from qubit 0) or
            ``"tree"`` (log-depth fan-out).
        uncompute: Append the inverse circuit ("upside-down" GHZ) so the ideal output is
            |0…0⟩ — a sensitive echo test of the noise.
        measure: Measure all qubits at the end.
    """
    if n < 2:
        raise ValueError("GHZ needs at least 2 qubits.")
    c = Circuit(n, name=f"ghz_{n}")
    c.h(0)
    if layout == "chain":
        for i in range(n - 1):
            c.cx(i, i + 1)
    elif layout == "star":
        for i in range(1, n):
            c.cx(0, i)
    elif layout == "tree":
        have = 1
        while have < n:
            for i in range(min(have, n - have)):
                c.cx(i, have + i)
            have *= 2
    else:
        raise ValueError("layout must be 'chain', 'star' or 'tree'.")
    if uncompute:
        c.barrier()
        c.compose(c.inverse())
        c.name = f"ghz_{n}_echo"
    return c.measure_all() if measure else c


def deutsch_jozsa(n: int, oracle: Union[str, Sequence[int]] = "balanced", *, measure: bool = True) -> Circuit:
    """Deutsch–Jozsa on ``n`` input qubits plus one ancilla (qubit ``n``).

    Args:
        oracle: ``"constant0"``, ``"constant1"``, ``"balanced"`` (f(x) = x_0 ⊕ … ⊕ x_{n-1}),
            or a bit mask ``s`` (f(x) = s·x mod 2, balanced for any nonzero s).
        measure: Measure the input register (all zeros ⇔ constant).
    """
    c = Circuit(n + 1, name=f"deutsch_jozsa_{n}")
    c.x(n)
    c.h(range(n + 1))
    if oracle == "constant1":
        c.x(n)
    elif oracle == "constant0":
        pass
    else:
        mask = [1] * n if oracle == "balanced" else _bits(oracle, n)
        if not any(mask):
            raise ValueError("A balanced oracle needs a nonzero mask.")
        for i, m in enumerate(mask):
            if m:
                c.cx(i, n)
    c.h(range(n))
    if measure:
        c.barrier()
        c.measure(range(n))
    return c


def bernstein_vazirani(secret: Union[str, Sequence[int]], *, measure: bool = True) -> Circuit:
    """Bernstein–Vazirani: recovers ``secret`` in one query (ancilla is the last qubit)."""
    s = _bits(secret)
    n = len(s)
    c = Circuit(n + 1, name=f"bernstein_vazirani_{''.join(map(str, s))}")
    c.x(n)
    c.h(range(n + 1))
    for i, b in enumerate(s):
        if b:
            c.cx(i, n)
    c.h(range(n))
    if measure:
        c.barrier()
        c.measure(range(n))
    return c


def qft(
    n: int,
    *,
    inverse: bool = False,
    swaps: bool = True,
    input_state: Union[str, Sequence[int]] | None = None,
    measure: bool = False,
) -> Circuit:
    """Quantum Fourier transform on ``n`` qubits (qubit 0 = most significant bit).

    Args:
        inverse: Build QFT† instead.
        swaps: Include the final bit-reversal swaps.
        input_state: Optional computational basis state to prepare first (e.g. ``"101"``).
    """
    body = Circuit(n, name=f"qft_{n}")
    for j in range(n):
        body.h(j)
        for k in range(j + 1, n):
            body.cp(math.pi / 2 ** (k - j), k, j)
    if swaps:
        for j in range(n // 2):
            body.swap(j, n - 1 - j)
    if inverse:
        body = body.inverse()
        body.name = f"iqft_{n}"
    c = Circuit(n, name=body.name)
    if input_state is not None:
        for q, b in enumerate(_bits(input_state, n)):
            if b:
                c.x(q)
    c.compose(body)
    return c.measure_all() if measure else c


def _diffuser(c: Circuit, qubits: list[int]) -> None:
    c.h(qubits).x(qubits)
    c.mcz(qubits[:-1], qubits[-1])
    c.x(qubits).h(qubits)


def grover(
    n: int, marked: Union[str, Iterable[str]], iterations: int | None = None, *, measure: bool = True
) -> Circuit:
    """Grover search over ``n`` qubits for one or more marked bitstrings.

    The oracle flips the phase of each marked state with a multi-controlled Z
    (exact matrices — fine for the statevector backend up to ~20 qubits).
    ``iterations`` defaults to ``round(π/4 · √(N/M) − 1/2)``.
    """
    marks = [marked] if isinstance(marked, str) else list(marked)
    if not marks:
        raise ValueError("Need at least one marked state.")
    for m in marks:
        _bits(m, n)
    N, M = 2**n, len(set(marks))
    if iterations is None:
        iterations = max(1, int(round(math.pi / 4 * math.sqrt(N / M) - 0.5)))
    qs = list(range(n))
    c = Circuit(n, name=f"grover_{n}")
    c.h(qs)
    for _ in range(iterations):
        for m in marks:
            zeros = [q for q, b in enumerate(_bits(m)) if b == 0]
            if zeros:
                c.x(zeros)
            c.mcz(qs[:-1], qs[-1])
            if zeros:
                c.x(zeros)
        _diffuser(c, qs)
    return c.measure_all() if measure else c


def phase_estimation(n_counting: int, phase: float, *, measure: bool = True) -> Circuit:
    """Estimate ``φ`` for the eigenstate |1⟩ of ``P(2πφ)`` with ``n_counting`` bits.

    Qubits ``0..n_counting-1`` are the counting register (qubit 0 = most significant
    bit of the estimate); the last qubit holds the eigenstate. For ``φ = k / 2ⁿ`` the
    noiseless output is exactly the binary expansion of ``k``.
    """
    n = n_counting
    c = Circuit(n + 1, name=f"qpe_{n}")
    c.x(n)
    c.h(range(n))
    for j in range(n):
        reps = 2 ** (n - 1 - j)
        c.cp(2 * math.pi * phase * reps, j, n)
    c.compose(qft(n, inverse=True), qubits=list(range(n)))
    if measure:
        c.barrier()
        c.measure(range(n))
    return c


def repetition_code(
    distance: int = 3, rounds: int = 1, *, basis: str = "bit", correct: bool = True, initial: int = 0
) -> Circuit:
    """Repetition-code memory experiment.

    Data qubits are ``0..d-1``; ancillas ``d..2d-2`` measure neighbouring parities
    (``Z_iZ_{i+1}`` for ``basis="bit"``, ``X_iX_{i+1}`` for ``basis="phase"``) each round,
    with measure-and-reset. For ``distance=3`` and ``correct=True`` the last round's
    syndrome drives a classically-controlled correction (lookup decoder). The data
    qubits are finally measured in Z (bit code) or X (phase code).

    Classical bits: syndromes first (round by round), then the ``d`` data bits.

    Args:
        initial: Logical state 0 or 1 (|0_L⟩/|1_L⟩ for the bit code, |+_L⟩/|−_L⟩ for the phase code).
    """
    if distance < 2:
        raise ValueError("distance must be >= 2.")
    if basis not in ("bit", "phase"):
        raise ValueError("basis must be 'bit' or 'phase'.")
    d = distance
    data = list(range(d))
    anc = list(range(d, 2 * d - 1))
    c = Circuit(2 * d - 1, name=f"repetition_{basis}_d{d}_r{rounds}")
    if initial:
        c.x(data)
    if basis == "phase":
        c.h(data)
    syndromes = []
    for r in range(rounds):
        reg = c.add_register(f"s{r}", d - 1)
        syndromes.append(reg)
        if basis == "phase":
            c.h(data)
        for i, a in enumerate(anc):
            c.cx(data[i], a)
            c.cx(data[i + 1], a)
        if basis == "phase":
            c.h(data)
        for i, a in enumerate(anc):
            c.measure(a, reg[i], reset=True)
    if correct and d == 3 and rounds >= 1:
        s = syndromes[-1]
        fix = G.X if basis == "bit" else G.Z
        # syndrome (s0, s1): 10 → qubit 0, 11 → qubit 1, 01 → qubit 2  (value = s0 + 2*s1)
        for value, q in ((0b01, 0), (0b11, 1), (0b10, 2)):
            c.append(fix, [q], condition=(s, value))
    c.add_register("data", d)
    c.barrier(data)
    c.measure(data, list(range(c.n_clbits - d, c.n_clbits)), basis="Z" if basis == "bit" else "X")
    return c


def teleportation(prepare: Circuit | None = None, *, measure_basis: str = "Z") -> Circuit:
    """Teleport qubit 0's state to qubit 2 with mid-circuit measurement and feed-forward.

    Args:
        prepare: A 1-qubit circuit preparing the state to send (default: H, i.e. |+⟩).
        measure_basis: Basis for the final measurement of qubit 2 (clbit 2).

    Classical bits: 0 = Bell measurement of qubit 0, 1 = of qubit 1, 2 = output.
    """
    c = Circuit(3, name="teleportation")
    if prepare is None:
        c.h(0)
    else:
        if prepare.n_qubits != 1:
            raise ValueError("prepare must be a 1-qubit circuit.")
        c.compose(prepare, qubits=[0])
    c.h(1).cx(1, 2)
    c.cx(0, 1).h(0)
    c.measure(0, c._new_clbit())
    c.measure(1, c._new_clbit())
    c.c_if(1).x(2)
    c.c_if(0).z(2)
    c.measure(2, c._new_clbit(), basis=measure_basis)
    return c
