import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fatpy.data_parsing.stress_tensor import stress_tensor
import numpy as np

tensor = stress_tensor(sx=100, sy=50, sz=30, sxy=10, syz=5, sxz=3)

def test_s1()->None:
    expected= 102.13
    assert np.isclose(tensor.s1(), expected,1e-3)
def test_s2()->None:
    expected = 49.04941134618069
    assert np.isclose(tensor.s2(), expected,1e-3)
def test_s3()->None:
    expected = 28.81382332394616
    assert np.isclose(tensor.s3(), expected,1e-3)
def test_mises()->None:
    expected = 65.58963332722635
    assert np.isclose(tensor.von_mises(), expected,1e-3)
def test_tresca()->None:
    expected = 73.32294200592699
    assert np.isclose(tensor.tresca(), expected,1e-3)
def test_max_principal()->None:
    expected = tensor.s1()
    assert np.isclose(tensor.max_principal_stress(), expected,1e-3)
def test_signed_mises()->None:
    expected = tensor.sign()*tensor.von_mises()
    assert np.isclose(tensor.signed_von_mises(), expected,1e-3)