import numpy as np
from einsumpy.contract import *


def test_contraction_name():
    assert str(Contraction("x_ix_i", x=[2])) == "x_ix_i"
    assert str(Contraction("x_ix_i+y_iy_i", x=[2], y=[2])) == "x_ix_i+y_iy_i"
    assert str(Contraction("x_ix_i-y_iy_i", x=[2], y=[2])) == "x_ix_i-y_iy_i"
    assert str(Contraction("-x_ix_i-y_iy_i", x=[2], y=[2])) == "-x_ix_i-y_iy_i"
    assert str(Contraction("-3x_ix_i/2", x=[2])) == "-(3/2)x_ix_i"
    assert str(Contraction("-3x_ix_i/2/3", x=[2])) == "-(1/2)x_ix_i"


def test_indices():
    c = Contraction("x_{ijk}y_{ijl}D_{ijlm}", x=[2, 2, 2], y=[2, 2, 2], D=[2, 2, 2, 2])
    assert c.all_indices == {"i", "j", "k", "l", "m"}


def test_suggest_indices():
    c = Contraction("x_{ab}", x=[2, 2])
    assert c.suggest_new_indices(2) == "ij"


def test_suggest_indices_fail():
    c = Contraction("x_{ab}", x=[2, 2])
    try:
        c.suggest_new_indices(100000)
        assert False
    except TooManyIndices:
        pass


def test_free_indices():
    c = Contraction("x_{in}y_{nj}", x=[2, 2], y=[2, 2])
    assert c.free_indices == "ij"


def test_free_indices_2():
    c = Contraction("x_{in}y_{nj} + z_{in}y_{nm}x_{mj}", x=[2, 2], y=[2, 2], z=[2, 2])
    assert c.free_indices == "ij"


def test_free_indices_fail():
    c = Contraction("x_{in}y_{nj} + x_{an}y_{nb}", x=[2, 2], y=[2, 2])
    try:
        fi = c.free_indices
        assert False
    except IndexMismatch:
        pass


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


def test_free_index_simplification():
    c = Contraction("x_{i}y_{i} + x_{j}y_{j}", x=[2], y=[2])
    assert str(c) == "2x_{i}y_{i}"


def test_missing_shape():
    try:
        c = Contraction("x_i")
        assert False
    except MissingShape:
        pass


def test_missing_shape_2():
    try:
        c = Contraction("x_iy_i", x=[2])
        assert False
    except MissingShape:
        pass


def test_wrong_shape():
    try:
        c = Contraction("x_i", x=[2, 2])
        assert False
    except WrongShape as e:
        pass


def test_wrong_shape_2():
    try:
        c = Contraction("x_{ij}", x=[2])
        assert False
    except WrongShape as e:
        pass


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


def test_derivative_8():
    for n in range(10):
        c = Contraction("(d_{in}D_{nj} + D_{in}d_{nj} + d_{ij})"
                        "(d_{im}D_{mj} + D_{im}d_{mj} + d_{ij}) + 4d_{ij}G_{ij}",
                        d=[n, n], D=[n, n], G=[n, n])
        d = c.derivative("d_{ab}")
        # TODO: actually test result


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
