from einsumpy.latex_tools import *


def test_explicit_subscript():
    assert make_subscripts_explicit("x_i") == "x_{i}"
    assert make_subscripts_explicit("x_ij") == "x_{i}j"
    assert make_subscripts_explicit("x_{ij}") == "x_{ij}"
    assert make_subscripts_explicit("x_iy_i") == "x_{i}y_{i}"
    assert make_subscripts_explicit("x_{ij}y_i") == "x_{ij}y_{i}"


def test_latex_remove_whitespace():
    assert remove_unnecessary_whitespace("x_i + y_i") == "x_i+y_i"
    assert remove_unnecessary_whitespace(r"\mu x") == r"\mu x"
    assert remove_unnecessary_whitespace(r"\mu  x") == r"\mu x"
    assert remove_unnecessary_whitespace(r"\mu   x") == r"\mu x"
    assert remove_unnecessary_whitespace(r"\mu   \nu x") == r"\mu \nu x"
    assert remove_unnecessary_whitespace(r"x x") == r"xx"
    assert remove_unnecessary_whitespace(r"xy x") == r"xyx"
    assert remove_unnecessary_whitespace(r"\mu_ix") == r"\mu_ix"
    assert remove_unnecessary_whitespace(r"\mu_i x") == r"\mu_ix"


def test_split_parenthetical_factors():
    assert list(split_parenthetical_factors("(a)(b)")) == ["a", "b"]
    assert list(split_parenthetical_factors("(a)(b)(c)")) == ["a", "b", "c"]
    assert list(split_parenthetical_factors("(a+b)(b+2)(c+2(3+4))")) == ["a+b", "b+2", "c+2(3+4)"]


def test_split_terms():
    assert list(split_terms("x_i + y_i")) == ["x_{i}", "+y_{i}"]
    assert list(split_terms("2x_i + y_i/3 + z_{ij}")) == ["2x_{i}", "+y_{i}/3", "+z_{ij}"]
    assert list(split_terms("x_i  +  0.5y_i")) == ["x_{i}", "+0.5y_{i}"]
    assert list(split_terms(r"\mu_i  -\nu_i x")) == [r"\mu_{i}", r"-\nu_{i}x"]
    assert list(split_terms(r"+\mu_i - x")) == [r"+\mu_{i}", "-x"]
    assert list(split_terms(r"+\mu_ij - x")) == [r"+\mu_{i}j", "-x"]
    assert list(split_terms(r"+2\mu_ij/3 - x")) == [r"+2\mu_{i}j/3", "-x"]


def test_split_coefficient():
    assert split_coefficient("1x") == ("1", "x")
    assert split_coefficient("2x3") == ("6", "x")
    assert split_coefficient("3x2/3") == ("(6/3)", "x")
    assert split_coefficient("+x") == ("1", "x")
    assert split_coefficient("+x/2") == ("(1/2)", "x")
    assert split_coefficient("2x/3") == ("(2/3)", "x")
    assert split_coefficient("+1x") == ("1", "x")
    assert split_coefficient("-1x") == ("-1", "x")
    assert split_coefficient("2x/3") == ("(2/3)", "x")
    assert split_coefficient("2x^2/3") == ("(2/3)", "x^2")
    assert split_coefficient("-2x_2/3") == ("-(2/3)", "x_2")
    assert split_coefficient("+2x_2/3") == ("(2/3)", "x_2")
    assert split_coefficient(r"+2.0x_2\mu /3") == ("(2.0/3)", r"x_2\mu")
