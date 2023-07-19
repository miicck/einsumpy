from einsumpy.contract import *


def test_explicit_subscript():
    assert LatexTools.make_subscripts_explicit("x_i") == "x_{i}"
    assert LatexTools.make_subscripts_explicit("x_ij") == "x_{i}j"
    assert LatexTools.make_subscripts_explicit("x_{ij}") == "x_{ij}"
    assert LatexTools.make_subscripts_explicit("x_iy_i") == "x_{i}y_{i}"
    assert LatexTools.make_subscripts_explicit("x_{ij}y_i") == "x_{ij}y_{i}"


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
    for n in range(1, 10):
        c = Contraction("x_iy_i", x=[1], y=[1])
        assert str(c) == "x_{i}y_{i}"

        derivative_terms = list(c.derivative("x_j"))
        assert len(derivative_terms) == 1
        assert str(derivative_terms[0]) == "y_{j}"


def test_derivative_indices_in_use():
    c = Contraction("x_iy_i", x=[2], y=[2])
    try:
        list(c.derivative("x_i"))
        assert False
    except IndicesInUse:
        pass


def test_output_indices():
    c = Contraction("x_iy_{ij}", x=[2], y=[2, 2])
    assert c.einsum_input_indices == "i,ij"
    assert c.einsum_output_indices == "j"
    assert c.einsum_string == "i,ij->j"


def test_matrix_mult():
    for n in range(1, 10):
        c = Contraction("x_{in}y_{nj}", x=[n, n], y=[n, n])
        assert str(c) == "x_{in}y_{nj}"

        x = np.random.random((n, n))
        y = np.random.random((n, n))
        assert np.allclose(c.evaluate(x=x, y=y), x @ y)

        d = list(c.derivative("x_{ab}"))
        assert len(d) == 1
        d = d[0]
        assert str(d) == f"I{n}" + "_{ia}y_{bj}"

        assert d.einsum_output_indices == "iabj"
        assert np.allclose(d.evaluate(y=y)[0, 0, :, :], y)


def test_dm_trace():
    for nbas in range(1, 10):
        c = Contraction("D_{aij}H_{aij}", D=[2, nbas, nbas], H=[2, nbas, nbas])
        assert str(c) == "D_{aij}H_{aij}"

        D = np.random.random((2, nbas, nbas))
        H = np.random.random((2, nbas, nbas))
        assert np.allclose(c.evaluate(D=D, H=H), np.einsum("aij,aij", D, H))

        d = list(c.derivative("D_{bnm}"))
        assert len(d) == 1
        d = d[0]
        assert str(d) == "H_{bnm}"


def test_latex_remove_whitespace():
    assert LatexTools.remove_unnecessary_whitespace("x_i + y_i") == "x_i+y_i"
    assert LatexTools.remove_unnecessary_whitespace(r"\mu x") == r"\mu x"
    assert LatexTools.remove_unnecessary_whitespace(r"\mu  x") == r"\mu x"
    assert LatexTools.remove_unnecessary_whitespace(r"\mu   x") == r"\mu x"
    assert LatexTools.remove_unnecessary_whitespace(r"\mu   \nu x") == r"\mu \nu x"
    assert LatexTools.remove_unnecessary_whitespace(r"x x") == r"xx"
    assert LatexTools.remove_unnecessary_whitespace(r"xy x") == r"xyx"
    assert LatexTools.remove_unnecessary_whitespace(r"\mu_ix") == r"\mu_ix"
    assert LatexTools.remove_unnecessary_whitespace(r"\mu_i x") == r"\mu_ix"


def test_split_terms():
    assert list(LatexTools.split_terms("x_i + y_i")) == ["x_{i}", "+y_{i}"]
    assert list(LatexTools.split_terms("2x_i + y_i/3 + z_{ij}")) == ["2x_{i}", "+y_{i}/3", "+z_{ij}"]
    assert list(LatexTools.split_terms("x_i  +  0.5y_i")) == ["x_{i}", "+0.5y_{i}"]
    assert list(LatexTools.split_terms(r"\mu_i  -\nu_i x")) == [r"\mu_{i}", r"-\nu_{i}x"]
    assert list(LatexTools.split_terms(r"+\mu_i - x")) == [r"+\mu_{i}", "-x"]
    assert list(LatexTools.split_terms(r"+\mu_ij - x")) == [r"+\mu_{i}j", "-x"]
    assert list(LatexTools.split_terms(r"+2\mu_ij/3 - x")) == [r"+2\mu_{i}j/3", "-x"]


def test_split_coefficient():
    assert LatexTools.split_coefficient("1x") == ("1", "x")
    assert LatexTools.split_coefficient("2x3") == ("6", "x")
    assert LatexTools.split_coefficient("3x2/3") == ("(6/3)", "x")
    assert LatexTools.split_coefficient("+x") == ("1", "x")
    assert LatexTools.split_coefficient("+x/2") == ("(1/2)", "x")
    assert LatexTools.split_coefficient("2x/3") == ("(2/3)", "x")
    assert LatexTools.split_coefficient("+1x") == ("1", "x")
    assert LatexTools.split_coefficient("-1x") == ("-1", "x")
    assert LatexTools.split_coefficient("2x/3") == ("(2/3)", "x")
    assert LatexTools.split_coefficient("2x^2/3") == ("(2/3)", "x^2")
    assert LatexTools.split_coefficient("-2x_2/3") == ("-(2/3)", "x_2")
    assert LatexTools.split_coefficient("+2x_2/3") == ("(2/3)", "x_2")
    assert LatexTools.split_coefficient(r"+2.0x_2\mu /3") == ("(2.0/3)", r"x_2\mu")


def test_composite_build():
    c = CompositeContraction("-x_i-y_i", x=[2], y=[2])
    assert str(c) == "-x_{i}-y_{i}"

    c = CompositeContraction("x_i+y_i", x=[2], y=[2])
    assert str(c) == "x_{i}+y_{i}"

    c = CompositeContraction("+x_i-y_i", x=[2], y=[2])
    assert str(c) == "x_{i}-y_{i}"

    c = CompositeContraction("2x_i/3 + y_i/3", x=[2], y=[2])
    assert str(c) == "(2/3)x_{i}+(1/3)y_{i}"


def test_composite_evaluate():
    for n in range(10):
        c = CompositeContraction("x_iy_i - x_ix_i", x=[n], y=[n])
        x = np.random.random(n)
        y = np.random.random(n)
        assert np.allclose(c.evaluate(x=x, y=y), np.dot(x, y) - np.dot(x, x))


def test_composite_derivative():
    for n in range(10):
        c = CompositeContraction("x_iy_i - x_ix_i", x=[n], y=[n])
        x = np.random.random(n)
        y = np.random.random(n)
        assert np.allclose(c.evaluate(x=x, y=y), np.dot(x, y) - np.dot(x, x))

        print(c.derivative("x_j"))
