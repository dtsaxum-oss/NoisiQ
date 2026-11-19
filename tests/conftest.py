"""Shared fixtures and random-circuit generators for the NoisiQ test suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import stim  # noqa: E402

import noisiq as nq  # noqa: E402
from noisiq.ir import gates as G  # noqa: E402

CLIFFORD_1Q = [G.I, G.X, G.Y, G.Z, G.H, G.S, G.SDG, G.SX, G.SXDG]
CLIFFORD_2Q = [G.CX, G.CY, G.CZ, G.SWAP, G.ISWAP]
PAULIS = [G.X, G.Y, G.Z]


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_clifford_gate(k: int, rng: np.random.Generator) -> G.Gate:
    tab = stim.Tableau.random(k)
    return G.unitary(tab.to_unitary_matrix(endian="big"), name=f"RC{k}")


def random_circuit(
    n: int,
    n_ops: int,
    rng: np.random.Generator,
    *,
    clifford: bool = True,
    mid_measure: bool = False,
    conditionals: bool = False,
    resets: bool = False,
    bases: bool = False,
) -> nq.Circuit:
    """A random circuit mixing library gates and custom (random) gates."""
    c = nq.Circuit(n, name="random")
    for _ in range(n_ops):
        r = rng.random()
        if mid_measure and r < 0.1:
            q = int(rng.integers(n))
            basis = str(rng.choice(["X", "Y", "Z"])) if bases else "Z"
            c.measure(q, basis=basis, reset=bool(resets and rng.random() < 0.5))
            continue
        if resets and r < 0.13:
            c.reset(int(rng.integers(n)))
            continue
        if conditionals and c.n_clbits and r < 0.25:
            bit = int(rng.integers(c.n_clbits))
            c.append(PAULIS[int(rng.integers(3))], [int(rng.integers(n))], condition=(bit, int(rng.integers(2))))
            continue
        if n >= 2 and rng.random() < 0.4:
            qs = [int(x) for x in rng.choice(n, 2, replace=False)]
            if clifford:
                g = (
                    CLIFFORD_2Q[int(rng.integers(len(CLIFFORD_2Q)))]
                    if rng.random() < 0.7
                    else random_clifford_gate(2, rng)
                )
            else:
                g = [
                    G.CX,
                    G.CZ,
                    G.cp(rng.uniform(0, 6)),
                    G.crx(rng.uniform(0, 6)),
                    G.unitary(random_unitary(4, rng), "RU2"),
                ][int(rng.integers(5))]
            c.append(g, qs)
        else:
            q = int(rng.integers(n))
            if clifford:
                g = (
                    CLIFFORD_1Q[int(rng.integers(len(CLIFFORD_1Q)))]
                    if rng.random() < 0.8
                    else random_clifford_gate(1, rng)
                )
            else:
                g = [
                    G.H,
                    G.T,
                    G.rx(rng.uniform(0, 6)),
                    G.ry(rng.uniform(0, 6)),
                    G.u(*rng.uniform(0, 6, 3)),
                    G.unitary(random_unitary(2, rng), "RU1"),
                ][int(rng.integers(6))]
            c.append(g, [q])
    return c


def tvd(p: dict, q: dict) -> float:
    return nq.metrics.total_variation_distance(p, q)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
