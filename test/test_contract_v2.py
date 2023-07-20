import numpy as np
from einsumpy.contract_v2 import *


def test_remove_floating_point_trailing_zeros():
    assert remove_floating_point_trailing_zeros("2.0a") == "2.0a"
    assert remove_floating_point_trailing_zeros("0.2") == "0.2"
    assert remove_floating_point_trailing_zeros("0.200") == "0.2"
    assert remove_floating_point_trailing_zeros("0.20100") == "0.201"
    assert remove_floating_point_trailing_zeros("0.20a0.2010") == "0.2a0.201"
    assert remove_floating_point_trailing_zeros("0.201+0.20200") == "0.201+0.202"


def test_identify_tensors():
    assert list(identify_tensors("x_i")) == ["x_i"]
    assert list(identify_tensors("x_iy_i")) == ["x_i", "y_i"]
    assert list(identify_tensors("x_i(y_i)")) == ["x_i", "y_i"]
    assert list(identify_tensors("x_{ij}")) == ["x_{ij}"]
    assert list(identify_tensors("x_{ij}y_i")) == ["x_{ij}", "y_i"]
    assert list(identify_tensors("3x_iy_j")) == ["x_i", "y_j"]
    assert list(identify_tensors("3x_i/2")) == ["x_i"]
    assert list(identify_tensors("\\mu_i")) == ["\\mu_i"]
    assert list(identify_tensors("\\mu_i + y_i")) == ["\\mu_i", "y_i"]
    assert list(identify_tensors("(3/2)\\mu_i(3) + 3y_i2")) == ["\\mu_i", "y_i"]
    assert list(identify_tensors("D_{aij}A_{aij}")) == ["D_{aij}", "A_{aij}"]
    assert list(identify_tensors("D_{aij}A_{aij} + D_{aij}B_{aijbnm}D_{bnm}")) == \
           ["D_{aij}", "A_{aij}", "D_{aij}", "B_{aijbnm}", "D_{bnm}"]


def test_to_contractions_and_coefficients():
    assert to_contractions_and_coefficients("x_i") == {"x_i": "1"}
    assert to_contractions_and_coefficients("x_ix_i") == {"x_ix_i": "1"}
    assert to_contractions_and_coefficients("x_iy_i") == {"x_iy_i": "1"}
    assert to_contractions_and_coefficients("x_iy_i + y_ix_i") == {"x_iy_i": "2"}
    assert to_contractions_and_coefficients("x_i(y_i + z_i)") == {"x_iy_i": "1", "x_iz_i": "1"}
    assert to_contractions_and_coefficients("D_{aij}B_{aijbnm}D_{bnm}") == {"D_{aij}B_{aijbnm}D_{bnm}": "1"}
    assert to_contractions_and_coefficients("2.0x_i") == {"x_i": "2.0"}


def test_tensor_to_kernel_indices():
    assert tensor_to_kernel_indices("x_i") == ("x", "i")
    assert tensor_to_kernel_indices("x_{ij}") == ("x", "ij")
    assert tensor_to_kernel_indices("x_{ijk}") == ("x", "ijk")


def test_contraction_name():
    assert str(Contraction("x_ix_i")) == "x_ix_i"
    assert str(Contraction("x_ix_i+y_iy_i")) == "x_ix_i+y_iy_i"
    assert str(Contraction("x_ix_i-y_iy_i")) == "x_ix_i-y_iy_i"
    assert str(Contraction("-x_ix_i-y_iy_i")) == "-x_ix_i-y_iy_i"
    assert str(Contraction("-3x_ix_i/2")) == "-(3/2)x_ix_i"
    assert str(Contraction("-3x_ix_i/2/3")) == "-(1/2)x_ix_i"


def test_dot_products():
    for n in range(10):
        c = Contraction("x_ix_i + x_ix_i - y_ix_i", x=[n])
        assert str(c) == "2x_ix_i-x_iy_i"
        x, y = np.random.random((2, n))
        assert np.allclose(c.evaluate(x=x, y=y), 2 * np.dot(x, x) - np.dot(x, y))


def test_matrix_mult():
    for n in range(10):
        c = Contraction("x_{in}y_{nj}", x=[n, n], y=[n, n])
        assert str(c) == "x_{in}y_{nj}"
        x, y = np.random.random((2, n, n))
        assert np.allclose(c.evaluate(x=x, y=y), x @ y)


def test_density_matrix_energy():
    for n in range(10):
        c = Contraction("D_{aij}A_{aij} + D_{aij}B_{aijbnm}D_{bnm}",
                        D=[2, n, n], A=[2, n, n], B=[2, n, n, 2, n, n])
        assert str(c) == "D_{aij}A_{aij}+D_{aij}B_{aijbnm}D_{bnm}"
        D = np.random.random((2, n, n))
        A = np.random.random((2, n, n))
        B = np.random.random((2, n, n, 2, n, n))
        assert np.allclose(c.evaluate(D=D, A=A, B=B),
                           np.einsum("aij,aij", D, A) +
                           np.einsum("aij,aijbnm,bnm", D, B, D))
