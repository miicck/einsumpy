import numpy as np
from einsumpy.optimize import *


def test_minimize():
    # Setup a quadratic in x
    c = Contraction("x_{i}h_{ij}x_{j}", x=[1], h=[1, 1])

    x0 = np.ones(1) * 2
    h = np.ones((1, 1))
    assert np.allclose(c.evaluate(x=x0, h=h), 4)

    x_min, c_min = minimize(c, "x", x=x0, h=h)
    assert abs(x_min) < 1e-6
    assert abs(c_min) < 1e-12
