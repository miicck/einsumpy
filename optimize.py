import numpy as np
from einsumpy import parsing
from typing import Iterable, Tuple
from einsumpy.contract import Contraction
from scipy.optimize import minimize as scipy_minimize


def minimize(contraction: Contraction, tensor: str, **tensor_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gradient = contraction.derivative(tensor)
    shape = contraction.get_shape(tensor)
    kernel = parsing.strip_indices(tensor)

    def scipy_cost(x: np.ndarray):
        tensor_values[kernel] = np.reshape(x, shape)
        return contraction.evaluate(**tensor_values)

    def scipy_gradient(x: np.ndarray):
        tensor_values[kernel] = np.reshape(x, shape)
        return gradient.evaluate(**tensor_values).flat

    res = scipy_minimize(scipy_cost, jac=scipy_gradient, x0=tensor_values[kernel].flat)
    return np.reshape(res.x, shape), res.fun
