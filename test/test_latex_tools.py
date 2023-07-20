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
    assert list(split_parenthetical_factors("a")) == ["a"]
    assert list(split_parenthetical_factors("(a)")) == ["a"]
    assert list(split_parenthetical_factors("((a))")) == ["a"]
    assert list(split_parenthetical_factors("a(b)")) == ["a", "b"]
    assert list(split_parenthetical_factors("(a)b")) == ["a", "b"]
    assert list(split_parenthetical_factors("ab(c)")) == ["ab", "c"]
    assert list(split_parenthetical_factors("ab(c)d")) == ["ab", "c", "d"]
    assert list(split_parenthetical_factors("ab(c)d(e)")) == ["ab", "c", "d", "e"]
    assert list(split_parenthetical_factors("a(b+c)")) == ["a", "b+c"]
    assert list(split_parenthetical_factors("a(b+c(d+e))")) == ["a", "b+c(d+e)"]
    assert list(split_parenthetical_factors("(a)(b)")) == ["a", "b"]
    assert list(split_parenthetical_factors("(a)(b)(c)")) == ["a", "b", "c"]
    assert list(split_parenthetical_factors("(a+b)(b+2)(c+2(3+4))")) == ["a+b", "b+2", "c+2(3+4)"]


def test_simplify_constant():
    assert simplify_constant("2(3)") == "6"
    assert simplify_constant("2(3)2") == "12"
    assert simplify_constant("(2)3(2)") == "12"
    assert simplify_constant("232") == "232"
    assert simplify_constant("2(3)2(3)") == "36"
    assert simplify_constant("1/3") == "1/3"
    assert simplify_constant("2(1/3)") == "2/3"
    assert simplify_constant("(1/3)(1/3)") == "1/9"
    assert simplify_constant("(1/3/3)") == "1/9"
    assert simplify_constant("0.1(0.2)") == "0.02"
    assert simplify_constant("0.1/0.2") == "0.5"


def test_split_terms():
    assert list(split_terms("+x")) == ["x"]
    assert list(split_terms("-1")) == ["-1"]
    assert list(split_terms("x_i + y_i")) == ["x_{i}", "+y_{i}"]
    assert list(split_terms("2x_i + y_i/3 + z_{ij}")) == ["2x_{i}", "+y_{i}/3", "+z_{ij}"]
    assert list(split_terms("x_i  +  0.5y_i")) == ["x_{i}", "+0.5y_{i}"]
    assert list(split_terms(r"\mu_i  -\nu_i x")) == [r"\mu_{i}", r"-\nu_{i}x"]
    assert list(split_terms(r"+\mu_i - x")) == [r"\mu_{i}", "-x"]
    assert list(split_terms(r"+\mu_ij - x")) == [r"\mu_{i}j", "-x"]
    assert list(split_terms(r"+2\mu_ij/3 - x")) == [r"2\mu_{i}j/3", "-x"]
    assert list(split_terms("1y_{j}+(-1-1)x_{j}")) == ["1y_{j}", "(-1-1)x_{j}"]


def test_split_coefficient():
    assert split_coefficient("(1/3)x") == ("(1/3)", "x")
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
    assert split_coefficient("-2x_2/3") == ("-(2/3)", "x_{2}")
    assert split_coefficient("+2x_2/3") == ("(2/3)", "x_{2}")
    assert split_coefficient(r"+2x_2\mu /3") == ("(2/3)", r"x_{2}\mu")
    assert split_coefficient("(1/3)y_i/3/3") == ("(1/27)", "y_{i}")


def test_sum_cefficients():
    assert sum_coefficients(["x"]) == "x"
    assert sum_coefficients(["x", "y"]) == "(x+y)"
    assert sum_coefficients(["x", "+y"]) == "(x+y)"
    assert sum_coefficients(["x", "-y"]) == "(x-y)"
    assert sum_coefficients(["-x", "-y"]) == "(-x-y)"
