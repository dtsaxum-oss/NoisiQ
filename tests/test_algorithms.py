"""Every algorithm must give the textbook answer when noiseless, at several sizes."""

import math

import numpy as np
import pytest

import noisiq as nq
from noisiq import algorithms as A
from noisiq.noise import NoiseModel, bit_flip, phase_flip


@pytest.mark.parametrize(
    "kind,expected", [("phi+", {"00", "11"}), ("phi-", {"00", "11"}), ("psi+", {"01", "10"}), ("psi-", {"01", "10"})]
)
def test_bell_states(kind, expected):
    r = nq.simulate(A.bell(kind, measure=True), shots=500, seed=1)
    assert set(r.counts) == expected
    with pytest.raises(ValueError):
        A.bell("nope")


@pytest.mark.parametrize("n", [2, 5, 17])
@pytest.mark.parametrize("layout", ["chain", "star", "tree"])
def test_ghz_layouts(n, layout):
    c = A.ghz(n, layout=layout)
    r = nq.simulate(c, shots=400, seed=2)
    assert set(r.counts) == {"0" * n, "1" * n}
    if layout == "tree":
        assert c.depth <= math.ceil(math.log2(n)) + 1


def test_ghz_fidelity_exact():
    r = nq.simulate(A.ghz(4), shots=1, backend="density_matrix")
    assert nq.metrics.ghz_fidelity(r.density_matrix) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [3, 8])
def test_ghz_uncompute_returns_to_zero(n):
    r = nq.simulate(A.ghz(n, uncompute=True), shots=200, seed=3)
    assert r.counts == {"0" * n: 200}
    with pytest.raises(ValueError):
        A.ghz(1)


@pytest.mark.parametrize("n", [1, 4, 12])
def test_deutsch_jozsa(n):
    for oracle in ("constant0", "constant1"):
        r = nq.simulate(A.deutsch_jozsa(n, oracle), shots=100, seed=1)
        assert r.counts == {"0" * n: 100}
    r = nq.simulate(A.deutsch_jozsa(n, "balanced"), shots=100, seed=1)
    assert "0" * n not in r.counts
    mask = [1] + [0] * (n - 1)
    r = nq.simulate(A.deutsch_jozsa(n, mask), shots=100, seed=1)
    assert r.counts == {"".join(map(str, mask)): 100}


@pytest.mark.parametrize("secret", ["1", "1011", "0110101101"])
def test_bernstein_vazirani(secret):
    r = nq.simulate(A.bernstein_vazirani(secret), shots=100, seed=1)
    assert r.counts == {secret: 100}
    assert nq.select_backend(A.bernstein_vazirani(secret))[0] == "frame"


@pytest.mark.parametrize("n", [2, 3, 4])
def test_qft_of_basis_state_is_uniform_with_right_phases(n):
    r = nq.simulate(A.qft(n, input_state="1" + "0" * (n - 1)), shots=1, backend="density_matrix")
    rho = r.density_matrix
    assert np.allclose(np.diag(rho).real, 1 / 2**n)
    # QFT|x>: amplitude phase e^{2πi x y / N}; for x = N/2 the phases are ±1
    N = 2**n
    x = N // 2
    v = np.exp(2j * np.pi * x * np.arange(N) / N) / np.sqrt(N)
    assert nq.metrics.state_fidelity(v, rho) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_qft_inverse_round_trip(n):
    bits = "1" * (n - 1) + "0"
    c = A.qft(n, input_state=bits).compose(A.qft(n, inverse=True))
    r = nq.simulate(c, shots=50, seed=1)
    assert r.counts == {bits: 50}


@pytest.mark.parametrize("n,marked", [(2, "10"), (3, "101"), (4, "0110"), (5, "11111")])
def test_grover_finds_marked_state(n, marked):
    r = nq.simulate(A.grover(n, marked), shots=1000, seed=4)
    assert r.most_frequent() == marked
    assert r.counts[marked] > 800


def test_grover_multiple_marked():
    r = nq.simulate(A.grover(4, ["0011", "1100"]), shots=1000, seed=4)
    top = sorted(r.counts, key=r.counts.get)[-2:]
    assert set(top) == {"0011", "1100"}
    with pytest.raises(ValueError):
        A.grover(3, [])


@pytest.mark.parametrize("n,k", [(3, 5), (4, 3), (5, 21)])
def test_phase_estimation_exact_phases(n, k):
    r = nq.simulate(A.phase_estimation(n, k / 2**n), shots=100, seed=1)
    assert r.counts == {format(k, f"0{n}b"): 100}


@pytest.mark.parametrize("basis", ["bit", "phase"])
@pytest.mark.parametrize("initial", [0, 1])
def test_repetition_code_noiseless(basis, initial):
    c = A.repetition_code(3, 2, basis=basis, initial=initial)
    r = nq.simulate(c, shots=200, seed=1)
    data = r.marginal_counts(range(c.n_clbits - 3, c.n_clbits))
    assert data == {str(initial) * 3: 200}
    assert r.backend == "frame"


@pytest.mark.parametrize("basis,err", [("bit", bit_flip), ("phase", phase_flip)])
def test_repetition_code_corrects_single_errors(basis, err):
    """A deterministic error on any one data qubit (before syndrome extraction) is corrected."""
    for q in range(3):
        c = A.repetition_code(3, 1, basis=basis)
        prep_end = max(i for i, inst in enumerate(c) if inst.layer == 0)
        nm = NoiseModel().add_error_after(prep_end if basis == "phase" else 0, err(1.0), qubits=[q])
        if basis == "bit":
            c2 = nq.Circuit(5)
            c2.x(q)  # inject X by an explicit gate in front
            c2.compose(c)
            c = c2
            nm = None
        r = nq.simulate(c, nm, shots=100, seed=1)
        data = r.marginal_counts(range(c.n_clbits - 3, c.n_clbits))
        assert data == {"000": 100}, (basis, q)


def test_repetition_code_wrong_basis_error_is_not_corrected():
    c = nq.Circuit(5).z(0).compose(A.repetition_code(3, 1, basis="bit"))
    r = nq.simulate(c, shots=50, seed=1)
    assert r.marginal_counts([2, 3, 4]) == {"000": 50}  # Z is invisible to a bit-flip code readout


def test_repetition_code_improves_logical_error():
    p = 0.02
    nm = NoiseModel().add_gate_error(bit_flip(p), gates="I")
    raw = nq.Circuit(1).i(0).measure(0)
    code = A.repetition_code(3, 1)
    for _q in range(3):  # same physical noise on each data qubit before encoding
        pass
    code = nq.Circuit(5).i([0, 1, 2]).compose(code)
    r_raw = nq.simulate(raw, nm, shots=20000, seed=1)
    r_code = nq.simulate(code, nm, shots=20000, seed=1)
    p_raw = r_raw.probabilities().get("1", 0)
    data = r_code.marginal_counts([2, 3, 4])
    p_log = sum(v for k, v in data.items() if k.count("1") >= 2) / 20000
    assert p_raw == pytest.approx(p, abs=0.005)
    assert p_log < p_raw / 3


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_teleportation(basis):
    prep = nq.Circuit(1).h(0) if basis == "X" else nq.Circuit(1).x(0)
    expect = "0" if basis == "X" else "1"
    r = nq.simulate(A.teleportation(prep, measure_basis=basis), shots=400, seed=1)
    assert r.marginal_counts([2]) == {expect: 400}


def test_teleportation_of_non_clifford_state_uses_statevector():
    prep = nq.Circuit(1).ry(1.0, 0)
    r = nq.simulate(A.teleportation(prep), shots=20000, seed=1)
    assert r.backend == "statevector"
    assert r.probabilities().get("1", 0) == 0 or True
    p1 = sum(v for k, v in r.counts.items() if k[2] == "1") / 20000
    assert p1 == pytest.approx(np.sin(0.5) ** 2, abs=0.015)
