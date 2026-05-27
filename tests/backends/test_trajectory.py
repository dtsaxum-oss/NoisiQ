import pytest
import numpy as np
from noisiq.ir import Circuit
from noisiq.ir import gates as ir_gates
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.noise.coherent_errors import CoherentRotation
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.backends.trajectory_backend import TrajectoryBackend
from noisiq.results import SimulationResult
from noisiq.noise.kraus_channels import KrausChannel

def test_t1_decay_curve():
    T1 = 1.0
    times = [0.0, 0.5, 1.0, 2.0]
    shots = 500
    
    backend = TrajectoryBackend()
    
    for t in times:
        # Prepare |1> state
        c = Circuit(1)
        c.x(0)
        
        noise = AmplitudeDamping(T1=T1, t=t)
        res = backend.run(c, noise_model=noise, n_shots=shots, seed=42)
        
        # Expected probability of measuring 1 is exp(-t/T1)
        expected_p1 = np.exp(-t / T1)

        # Read P(|1⟩) directly from the density matrix via partial trace
        p1 = res.excited_state_probability(0)

        # Allow 5% margin of error due to Monte Carlo sampling
        assert abs(p1 - expected_p1) < 0.05

def test_trajectory_underflow():
    # Test that the zero-probability fallback works in Kraus sampling
    c = Circuit(1)
    c.h(0)

    # Create an unphysical Kraus channel with 0 matrices to force sum_p = 0
    k0 = np.zeros((2, 2), dtype=complex)
    k1 = np.zeros((2, 2), dtype=complex)

    class ZeroChannel(KrausChannel):
        def describe(self) -> dict:
            return {"channel": "zero"}

    import unittest.mock
    with unittest.mock.patch.object(KrausChannel, '_validate_operators'):
        noise = ZeroChannel(operators=[k0, k1])

    backend = TrajectoryBackend()
    res = backend.run(c, noise_model=noise, n_shots=10, seed=42)

    # Should not crash and should return a SimulationResult with a density matrix
    assert isinstance(res, SimulationResult)
    assert res.final_state is not None
    assert res.meta["n_shots"] == 10


# ---------------------------------------------------------------------------
# Multi-qubit Kraus: CoherentRotation (single Kraus op → deterministic)
# ---------------------------------------------------------------------------

def test_single_qubit_coherent_rotation_z_half_pi():
    """Z rotation by π/2 on |+⟩ should yield density matrix of |−⟩."""
    c = Circuit(1)
    c.add_gate(ir_gates.H, (0,), t=0)
    noise = {0: CoherentRotation(axis='Z', epsilon=np.pi / 2)}

    res = TrajectoryBackend().run(c, noise_model=noise, n_shots=10, seed=0)

    # exp(-iπ/2 * Z)|+⟩ = -i|−⟩ — same density matrix as |−⟩
    minus = np.array([1.0, -1.0]) / np.sqrt(2)
    expected_rho = np.outer(minus, minus.conj())
    np.testing.assert_allclose(res.final_state, expected_rho, atol=1e-10)


def test_two_qubit_coherent_ix_on_bell():
    """IX rotation by π/2 on Bell state Φ+ transforms it to Ψ+.

    Note: ZZ is trivial on Bell state Φ+ because both |00⟩ and |11⟩ are
    ZZ eigenstates with eigenvalue +1 → only a global phase. Use IX instead,
    which maps |00⟩→|01⟩ and |11⟩→|10⟩ (the Ψ+ Bell state).
    """
    c = Circuit(2)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=1)

    # exp(-iπ/2 * I⊗X) = -i(I⊗X); converts Φ+ → Ψ+ (up to global phase)
    noise = {1: CoherentRotation(axis='IX', epsilon=np.pi / 2)}

    res = TrajectoryBackend().run(c, noise_model=noise, n_shots=10, seed=0)

    # Φ+ = (|00⟩+|11⟩)/√2; Ψ+ = (|01⟩+|10⟩)/√2
    psi_plus = np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2)
    expected_rho = np.outer(psi_plus, psi_plus.conj())
    np.testing.assert_allclose(res.final_state, expected_rho, atol=1e-10)


def test_zero_epsilon_coherent_rotation_noiseless():
    """CoherentRotation with epsilon=0 is the identity channel."""
    c = Circuit(2)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=1)

    # Without noise
    ref = TrajectoryBackend().run(c, noise_model=None, n_shots=50, seed=1)

    # With identity CoherentRotation on the CNOT
    noise = {1: CoherentRotation(axis='ZZ', epsilon=0.0)}
    with_noise = TrajectoryBackend().run(c, noise_model=noise, n_shots=50, seed=1)

    np.testing.assert_allclose(ref.final_state, with_noise.final_state, atol=1e-12)


# ---------------------------------------------------------------------------
# CorrelatedPauliError in trajectory backend
# ---------------------------------------------------------------------------

def test_correlated_pauli_deterministic_zz():
    """CorrelatedPauliError with ZZ prob=1 on CNOT maps Φ+ → Φ+ (ZZ trivial on Bell).

    ZZ|00⟩ = |00⟩, ZZ|11⟩ = |11⟩ → Bell state unchanged. Confirms the
    CorrelatedPauliError path in the backend doesn't corrupt the state.
    """
    c = Circuit(2)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=1)

    noise = {1: CorrelatedPauliError({'ZZ': 1.0})}
    res = TrajectoryBackend().run(c, noise_model=noise, n_shots=100, seed=42)

    phi_plus = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    expected_rho = np.outer(phi_plus, phi_plus.conj())
    np.testing.assert_allclose(res.final_state, expected_rho, atol=1e-10)


def test_correlated_pauli_ix_deterministic():
    """CorrelatedPauliError with IX prob=1 on CNOT flips qubit 1: Φ+ → Ψ+."""
    c = Circuit(2)
    c.add_gate(ir_gates.H, (0,), t=0)
    c.add_gate(ir_gates.CNOT, (0, 1), t=1)

    noise = {1: CorrelatedPauliError({'IX': 1.0})}
    res = TrajectoryBackend().run(c, noise_model=noise, n_shots=50, seed=7)

    # IX on Φ+: X flips qubit 1 → (|01⟩+|10⟩)/√2 = Ψ+
    psi_plus = np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2)
    expected_rho = np.outer(psi_plus, psi_plus.conj())
    np.testing.assert_allclose(res.final_state, expected_rho, atol=1e-10)
