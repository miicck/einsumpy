import numpy as np
from einsumpy.optimize import *


def test_minimize():
    # Setup a quadratic in x
    c = Contraction("x_{i}h_{ij}x_{j}", x=[1], h=[1, 1])

    x0 = np.ones(1) * 2
    h = np.ones((1, 1))
    assert np.allclose(c.evaluate(x=x0, h=h), 4)

    x_min, c_min = minimize(c, "x", x=x0, h=h)

    assert abs(x_min) < 1e-10
    assert abs(c_min) < 1e-20


def test_minimize_2():
    def random_positive_hessian(n: int):
        m = np.random.rand(n, n)
        mx = np.sum(np.abs(m), axis=1)
        np.fill_diagonal(m, mx)
        return (m + m.T) / 2.0

    for n in range(1, 10):
        # Set up a random quadratic with a minimum
        c = Contraction("x_{i}g_{i} + x_{i}h_{ij}x_{j}", x=[n], g=[n], h=[n, n])
        x = np.random.random(n)
        g = np.random.random(n)
        h = random_positive_hessian(n)

        x_min, c_min = minimize(c, "x", x=x, g=g, h=h)

        x_expected = -np.linalg.inv(h + h.T) @ g
        assert np.allclose(x_min, x_expected, atol=1e-5, rtol=1e-5)
