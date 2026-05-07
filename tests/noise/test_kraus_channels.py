import pytest
import numpy as np
from noisiq.noise.kraus_channels import KrausChannel
from noisiq.noise.amplitude_damping import AmplitudeDamping
from noisiq.noise.t2_dephasing import Dephasing

class InvalidChannel(KrausChannel):
    def __init__(self):
        K0 = np.array([[1, 0], [0, 0.5]], dtype=complex)
        K1 = np.array([[0, 0.5], [0, 0]], dtype=complex)
        super().__init__([K0, K1])

def test_kraus_validation():
    with pytest.raises(ValueError, match="not trace-preserving"):
        InvalidChannel()

def test_amplitude_damping():
    # t = 0 (identity)
    ad = AmplitudeDamping(T1=1.0, t=0.0)
    assert np.allclose(ad.operators[0], np.eye(2))
    assert np.allclose(ad.operators[1], np.zeros((2, 2)))
    
    # t -> infinity (complete decay)
    ad = AmplitudeDamping(T1=1.0, t=100.0)
    assert np.allclose(ad.operators[0], np.array([[1, 0], [0, 0]]))
    assert np.allclose(ad.operators[1], np.array([[0, 1], [0, 0]]))

def test_dephasing():
    # t = 0
    de = Dephasing(T2=1.0, t=0.0)
    assert np.allclose(de.operators[0], np.eye(2))
    assert np.allclose(de.operators[1], np.zeros((2, 2)))
    
    # t -> infinity (complete dephasing, p -> 0.5)
    de = Dephasing(T2=1.0, t=100.0)
    assert np.allclose(de.operators[0], np.sqrt(0.5) * np.eye(2))
    assert np.allclose(de.operators[1], np.sqrt(0.5) * np.array([[1, 0], [0, -1]]))
    
    # Test T1 and Tphi constructor
    de2 = Dephasing(t=1.0, T1=1.0, Tphi=1.0)
    rate = 1.0 / (2 * 1.0) + 1.0 / 1.0
    expected_p = 0.5 * (1.0 - np.exp(-1.0 * rate))
    assert np.allclose(de2.operators[0], np.sqrt(1 - expected_p) * np.eye(2))
