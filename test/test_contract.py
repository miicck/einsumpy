from einsumpy.contract import *


def test_build_contraction():
    c = Contraction("x_iy_i", x=[2], y=[2])
    assert str(c) == "x_{i}y_{i}"


def test_too_few_shapes():
    try:
        c = Contraction("x_iy_i")
        assert False
    except IndicesMismatch:
        pass


def test_derivative_tensor_not_found_error():
    try:
        x = Contraction("x_iy_i", x=[2], y=[2])
        list(x.derivative("z_j"))
        assert False
    except TensorNotFound:
        pass


def test_identity():
    for n in range(10):
        c = Contraction("x_i", x=[n])
        x = np.random.random(n)
        assert np.allclose(c.evaluate(x=x), x)


def test_dot_product():
    for n in range(10):
        c = Contraction("x_iy_i", x=[n], y=[n])
        assert c.free_indices == ""
        assert str(c) == "x_{i}y_{i}"
        assert str(c.derivative("x_j")) == "y_{j}"


def test_derivative_indices_in_use():
    c = Contraction("x_iy_i", x=[2], y=[2])
    try:
        list(c.derivative("x_i"))
        assert False
    except IndicesInUse:
        pass


def test_matrix_mult():
    for n in range(1, 10):
        c = Contraction("x_{in}y_{nj}", x=[n, n], y=[n, n])
        assert str(c) == "x_{in}y_{nj}"

        x = np.random.random((n, n))
        y = np.random.random((n, n))
        assert np.allclose(c.evaluate(x=x, y=y), x @ y)

        d = c.derivative("x_{ab}")
        assert str(d) == f"I{n}" + "_{ia}y_{bj}"
        d_eval = d.evaluate(y=y)
        assert d.free_indices == "iabj"
        assert np.allclose(d_eval[0, 0, :, :], y)


def test_dm_trace():
    for nbas in range(1, 10):
        c = Contraction("D_{aij}H_{aij}", D=[2, nbas, nbas], H=[2, nbas, nbas])
        assert str(c) == "D_{aij}H_{aij}"

        D = np.random.random((2, nbas, nbas))
        H = np.random.random((2, nbas, nbas))
        assert np.allclose(c.evaluate(D=D, H=H), np.einsum("aij,aij", D, H))

        d = c.derivative("D_{bnm}")
        assert str(d) == "H_{bnm}"


def test_composite_build():
    c = Contraction("-x_i-y_i", x=[2], y=[2])
    assert str(c) == "-x_{i}-y_{i}"

    c = Contraction("x_i+y_i", x=[2], y=[2])
    assert str(c) == "x_{i}+y_{i}"

    c = Contraction("+x_i-y_i", x=[2], y=[2])
    assert str(c) == "x_{i}-y_{i}"

    c = Contraction("2x_i/3 + y_i/3", x=[2], y=[2])
    assert str(c) == "(2/3)x_{i}+(1/3)y_{i}"

    c = Contraction("2x_i/3 + y_i/3/3", x=[2], y=[2])
    assert str(c) == "(2/3)x_{i}+(1/9)y_{i}"

    c = Contraction("2x_i/3 + (1/3)y_i/3/3", x=[2], y=[2])
    assert str(c) == "(2/3)x_{i}+(1/27)y_{i}"


def test_composite_evaluate():
    for n in range(10):
        c = Contraction("x_iy_i - x_ix_i", x=[n], y=[n])
        x = np.random.random(n)
        y = np.random.random(n)
        assert np.allclose(c.evaluate(x=x, y=y), np.dot(x, y) - np.dot(x, x))


def test_composite_derivative():
    for n in range(10):
        c = Contraction("x_iy_i - x_ix_i", x=[n], y=[n])
        x = np.random.random(n)
        y = np.random.random(n)
        assert np.allclose(c.evaluate(x=x, y=y), np.dot(x, y) - np.dot(x, x))

        print(c.derivative("x_j"))
