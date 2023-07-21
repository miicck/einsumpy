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
    assert list(identify_tensors("I_{jb}I_{ia}y_{nm}")) == ["I_{jb}", "I_{ia}", "y_{nm}"]
    assert list(identify_tensors("I(10)_{ij}")) == ["I(10)_{ij}"]
    assert list(identify_tensors("0.5I(10)_{ij}")) == ["I(10)_{ij}"]
    assert list(identify_tensors("3I(10)_{ij}/2")) == ["I(10)_{ij}"]
    assert list(identify_tensors("D_{aij}B_{aijbnm}D_{bnm}")) == ["D_{aij}", "B_{aijbnm}", "D_{bnm}"]


def test_to_contractions_and_coefficients():
    assert to_contractions_and_coefficients("+I_{jb}I_{ia}y_{nm}") == {"I_{jb}I_{ia}y_{nm}": "1"}
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
    assert str(Contraction("x_ix_i", x=[2])) == "x_ix_i"
    assert str(Contraction("x_ix_i+y_iy_i", x=[2], y=[2])) == "x_ix_i+y_iy_i"
    assert str(Contraction("x_ix_i-y_iy_i", x=[2], y=[2])) == "x_ix_i-y_iy_i"
    assert str(Contraction("-x_ix_i-y_iy_i", x=[2], y=[2])) == "-x_ix_i-y_iy_i"
    assert str(Contraction("-3x_ix_i/2", x=[2])) == "-(3/2)x_ix_i"
    assert str(Contraction("-3x_ix_i/2/3", x=[2])) == "-(1/2)x_ix_i"


def test_dot_products():
    for n in range(10):
        c = Contraction("x_ix_i + x_ix_i - y_ix_i", x=[n], y=[n])
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


def test_derivative_index_error():
    c = Contraction("x_iy_i", x=[2], y=[2])
    try:
        c.derivative("x_i")
        assert False
    except IndexClash as e:
        pass


def test_derivative_index_error_2():
    c = Contraction("x_{ij}y_{ij}", x=[2, 2], y=[2, 2])
    try:
        c.derivative("x_{in}")
        assert False
    except IndexClash as e:
        pass


def test_derivative_index_error_3():
    c = Contraction("x_{ij}y_{ij}", x=[2, 2], y=[2, 2])
    try:
        c.derivative("x_{ij}")
        assert False
    except IndexClash as e:
        pass


def test_derivative_1():
    for n in range(10):
        c = Contraction("x_iy_i", x=[n], y=[n])
        d = c.derivative("x_n")
        assert str(d) == "y_{n}"


def test_derivative_2():
    for n in range(10):
        c = Contraction("x_{ij}y_{ij}", x=[n, n], y=[n, n])
        d = c.derivative("y_{nm}")
        assert str(d) == "x_{nm}"


def test_derivative_3():
    for n in range(10):
        c = Contraction("x_iy_i + y_iy_i", x=[n], y=[n])
        d = c.derivative("x_j")
        assert str(d) == "y_{j}"


def test_derivative_4():
    for n in range(10):
        c = Contraction("x_ix_i", x=[n])
        d = c.derivative("x_j")
        assert str(d) == "2x_{j}"


def test_derivative_5():
    for n in range(10):
        c = Contraction("y_{ij}y_{ij}x_{ij}", x=[n, n], y=[n, n])
        d = c.derivative("x_{nm}")
        assert str(d) == "y_{nm}y_{nm}"


def test_derivative_6():
    for n in range(10):
        c = Contraction("x_{ij}y_{nm}", x=[n, n], y=[n, n])
        x, y = np.random.random((2, n, n))
        assert np.allclose(c.evaluate(x=x, y=y), np.einsum("ij,nm->ijnm", x, y))
        d = c.derivative("x_{ab}")
        assert str(d) == f"I({n})_{{ia}}I({n})_{{jb}}y_{{nm}}"


def test_derivative_7():
    for n in range(10):
        c = Contraction("d_{ij}(A_{ij} + (B_{ijnm} + B_{nmij})D_{nm}) + "
                        "L_{ij}(d_{in}D_{nj} + D_{in}d_{nj} - d_{ij})",
                        d=[n, n], D=[n, n], A=[n, n], L=[n, n], B=[n, n, n, n])

        d = c.derivative("d_{ab}")
        assert str(d) == "A_{ab}-L_{ab}+B_{abnm}D_{nm}+D_{nm}B_{nmab}+L_{aj}D_{bj}+L_{ib}D_{ia}"


def test_evaluate_derivative():
    for n in range(10):
        c = Contraction("x_iy_i", x=[n], y=[n])
        d = c.derivative("x_j")
        y = np.random.random(n)
        assert np.allclose(d.evaluate(y=y), y)


def test_evaluate_derivative_with_kronecker():
    for n in range(10):
        c = Contraction("x_{ij}y_{nm}", x=[n, n], y=[n, n])
        d = c.derivative("x_{ab}")
        assert str(d) == f"I({n})_{{ia}}I({n})_{{jb}}y_{{nm}}"
        y = np.random.random((n, n))
        I = np.identity(n)
        assert np.allclose(d.evaluate(y=y), np.einsum("ia,jb,nm->iajbnm", I, I, y))
