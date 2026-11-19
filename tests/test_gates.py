import numpy as np
import pytest
import stim

from noisiq.ir import gates as G
from tests.conftest import random_unitary

ALL_FIXED = list(G.STANDARD_GATES.values())
PARAMETRIC = [G.rx, G.ry, G.rz, G.p, G.cp, G.crx, G.cry, G.crz]


@pytest.mark.parametrize("gate", ALL_FIXED, ids=lambda g: g.name)
def test_fixed_gates_are_unitary_and_sized(gate):
    d = 2**gate.num_qubits
    assert gate.matrix.shape == (d, d)
    assert np.allclose(gate.matrix.conj().T @ gate.matrix, np.eye(d))


@pytest.mark.parametrize("gate", ALL_FIXED, ids=lambda g: g.name)
def test_inverse_is_inverse(gate):
    assert np.allclose(gate.inverse().matrix @ gate.matrix, np.eye(gate.matrix.shape[0]))


@pytest.mark.parametrize("factory", PARAMETRIC, ids=lambda f: f.__name__)
@pytest.mark.parametrize("theta", [0.0, 0.123456789, np.pi / 3, -2.5])
def test_parametric_gates_keep_exact_angle(factory, theta):
    g = factory(theta)
    assert g.params == (theta,)  # exact, never rounded
    assert np.allclose(g.inverse().matrix @ g.matrix, np.eye(g.matrix.shape[0]))


def test_u_gate_inverse_and_special_cases():
    g = G.u(0.3, 1.1, -0.7)
    assert np.allclose(g.inverse().matrix @ g.matrix, np.eye(2))
    assert np.allclose(G.u(np.pi / 2, 0, np.pi).matrix, G.H.matrix)


def test_rotation_identities():
    assert np.allclose(G.p(np.pi / 2).matrix, G.S.matrix)
    assert np.allclose(G.p(np.pi / 4).matrix, G.T.matrix)
    assert np.allclose(G.rx(np.pi).matrix, -1j * G.X.matrix)
    assert np.allclose(G.SX.matrix @ G.SX.matrix, G.X.matrix)


@pytest.mark.parametrize(
    "gate,expected",
    [
        (G.H, True),
        (G.S, True),
        (G.SX, True),
        (G.CX, True),
        (G.CZ, True),
        (G.SWAP, True),
        (G.ISWAP, True),
        (G.T, False),
        (G.CCX, False),
        (G.CH, False),
        (G.rz(0.3), False),
        (G.rz(np.pi / 2), True),
        (G.rx(np.pi), True),
        (G.p(np.pi), True),
    ],
)
def test_clifford_detection_from_matrix(gate, expected):
    assert gate.is_clifford is expected


def test_custom_clifford_gate_is_recognized_regardless_of_name():
    m = stim.Tableau.random(2).to_unitary_matrix(endian="big")
    g = G.unitary(m, name="H")  # misleading name must not matter
    assert g.is_clifford
    assert g.symplectic.shape == (4, 4)


@pytest.mark.parametrize("gate", [G.X, G.Y, G.Z, G.I])
def test_pauli_detection(gate):
    assert gate.is_pauli
    assert not G.H.is_pauli


def test_symplectic_matches_conjugation(rng):
    # For every Clifford, M @ bits(P) must equal bits(U P U†) for all Paulis P.
    from noisiq.noise.channels import pauli_matrix

    for gate in [g for g in ALL_FIXED if g.is_clifford]:
        k = gate.num_qubits
        U = gate.matrix
        for label in ["".join(t) for t in np.array(np.meshgrid(*[list("IXYZ")] * k)).T.reshape(-1, k)]:
            P = pauli_matrix(label)
            out = U @ P @ U.conj().T
            bits = np.array([c in "XY" for c in label] + [c in "ZY" for c in label], dtype=np.uint8)
            new = (gate.symplectic @ bits) & 1
            lab = "".join(
                {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}[(int(new[j]), int(new[k + j]))] for j in range(k)
            )
            Q = pauli_matrix(lab)
            phase = np.trace(Q.conj().T @ out) / 2**k
            assert np.isclose(abs(phase), 1.0), (gate.name, label)


def test_controlled_gate_structure():
    g = G.controlled(G.H)
    assert g.num_controls == 1 and g.base == G.H
    assert np.allclose(g.matrix[:2, :2], np.eye(2))
    assert np.allclose(g.matrix[2:, 2:], G.H.matrix)
    gg = G.controlled(G.X, 3)
    assert gg.num_qubits == 4 and gg.num_controls == 3


def test_non_unitary_and_bad_shapes_rejected():
    with pytest.raises(ValueError):
        G.Gate("BAD", np.array([[1, 1], [0, 1]]))
    with pytest.raises(ValueError):
        G.Gate("BAD", np.eye(3))
    with pytest.raises(ValueError):
        G.Gate("BAD", np.eye(2, 3))


def test_gate_equality_and_hash():
    assert G.rz(0.5) == G.rz(0.5)
    assert G.rz(0.5) != G.rz(0.6)
    assert hash(G.rz(0.5)) == hash(G.rz(0.5))
    assert G.X != G.Z


def test_matrix_is_read_only():
    with pytest.raises(ValueError):
        G.H.matrix[0, 0] = 2


def test_gate_from_name_and_aliases():
    assert G.gate_from_name("cnot") == G.CX
    assert G.gate_from_name("u1", [0.3]) == G.p(0.3)
    with pytest.raises(KeyError):
        G.gate_from_name("nope")
    with pytest.raises(ValueError):
        G.gate_from_name("h", [1.0])


def test_display_labels():
    assert G.rz(np.pi / 4).display_label == "RZ(π/4)"
    assert G.SDG.display_label == "S†"
    assert G.p(-np.pi / 2).display_label == "P(-π/2)"


def test_random_custom_unitary_inverse(rng):
    g = G.unitary(random_unitary(4, rng), "U4")
    assert not g.is_clifford
    assert np.allclose(g.inverse().matrix @ g.matrix, np.eye(4))
    assert g.inverse().inverse() == g
