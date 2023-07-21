from typing import Iterable, Tuple, List, Dict, Set
from sympy.simplify import simplify
from sympy.parsing import parse_expr
from sympy import Expr
import numpy as np
from einsumpy import parsing


class IndexClash(Exception):

    @staticmethod
    def from_set_and_expressions(clash: Set[str], expr_1: str, expr_2: str):

        clash_format = ", ".join(str(j) for j in clash)
        clash_format = f"The indices {{{clash_format}}} appear"
        if len(clash) == 1:
            clash_format = f"The index {list(clash)[0]} appears"

        clash_hint = "please choose non-clashing indices"
        if len(clash) == 1:
            clash_hint = "please choose a non-clashing index"

        return IndexClash(f"{clash_format} in both '{expr_1}' and '{expr_2}', {clash_hint}.")


class UnkownTensor(Exception):
    pass


class MissingShape(Exception):
    pass


class WrongShape(Exception):
    pass


class Contraction:

    def __init__(self, latex: str, **shapes: Iterable[int]):
        self._terms = parsing.to_contractions_and_coefficients(latex)
        self._shapes = {s: list(shapes[s]) for s in shapes}

        for term in self._terms:
            for tensor in parsing.identify_tensors(term):
                kernel, indices = parsing.tensor_to_kernel_indices(tensor)
                if kernel not in self._shapes:
                    raise MissingShape(f"Shape not specified for the tensor '{kernel}'")

                shape = self._shapes[kernel]
                if len(shape) != len(indices):
                    raise WrongShape(f"Shape {shape} specified for tensor '{tensor}' "
                                     f"incompatible with indices '{indices}'")

    def evaluate(self, **tensors: np.ndarray):

        def get_tensor(kernel: str) -> np.ndarray:

            # Check if this was an input kernel
            if kernel in tensors:
                return tensors[kernel]

            # Check if this is a Kronecker delta
            if kernel.startswith("I"):
                assert kernel[-1] == ")" and kernel[1] == "(", f"Misformated identity: '{kernel}'"
                return np.identity(int(kernel[2:-1]))

            # Unkown tensor
            raise UnkownTensor(f"The tensor '{kernel}' was not specified in {list(tensors)}")

        result = None
        for term in self._terms:
            # Get the operands for this term, and their indices
            kernels_indices = [parsing.tensor_to_kernel_indices(t) for t in
                               parsing.identify_tensors(term)]
            einsum_indices = ",".join(t[1] for t in kernels_indices)
            tensor_values = [get_tensor(t[0]) for t in kernels_indices]

            # Work out output indices
            output_indices = einsum_indices.replace(",", "")
            index_count = {i: sum(j == i for j in output_indices) for i in output_indices}
            for i in index_count:
                if index_count[i] > 1:
                    # Remove dummy indices from output
                    output_indices = output_indices.replace(i, "")

            einsum_string = einsum_indices + "->" + output_indices

            # Evaluate this term
            t_res = eval(self._terms[term]) * np.einsum(einsum_string, *tensor_values)

            # Add this term to the overall result
            if result is None:
                result = t_res
            else:
                result += t_res

        return result

    def derivative(self, tensor: str) -> 'Contraction':

        # d/dX_{ij} => target_kernel = x, target_indices = ij
        target_kernel, target_indices = parsing.tensor_to_kernel_indices(tensor)

        # Will contain a latex string representing the result
        result_str = ""
        result_shapes = {}

        # Check for a clash between indicies in derivative
        # and indicies in this expression
        for term in self._terms:
            for t in parsing.identify_tensors(term):
                clash = set(target_indices).intersection(set(parsing.tensor_to_kernel_indices(t)[1]))
                if len(clash) > 0:
                    raise IndexClash.from_set_and_expressions(
                        clash, str(self), f"d/d{target_kernel}_{{{target_indices}}}")

        # Take derivative of each term, sum the result
        for term in self._terms:

            # Get kernels and indices of this term
            # A_{ij}B_{ij} kernels = A, B indices = ij, ij
            kernels_indices = [parsing.tensor_to_kernel_indices(t) for t in parsing.identify_tensors(term)]

            # Find occurances of the target kernel in the contraction
            for i, (kernel, indices) in enumerate(kernels_indices):
                if kernel != target_kernel:
                    continue

                # Map the indices from this kernel onto those in the target
                index_map = {index: index_t for index, index_t in zip(indices, target_indices)}

                # Get the other terms remaining after this term has been reduced by the derivative
                other_terms = [list(kernels_indices[j]) for j in range(len(kernels_indices)) if j != i]
                kronecker_additions = []

                for index in index_map:

                    # Check for occurance of the index we're replacing
                    # and replace it with the index of the differential
                    replaced = False
                    for j, (ok, oi) in enumerate(other_terms):
                        if index in oi:
                            # Replace the index
                            other_terms[j][1] = oi.replace(index, index_map[index])
                            replaced = True

                    # No replacement was possible, insert an explicit Kronecker delta
                    if not replaced:
                        dim = self._shapes[kernel][indices.index(index)]
                        I_kernel = f"I({dim})"
                        kronecker_additions.append([I_kernel, index + index_map[index]])
                        self._shapes[I_kernel] = [dim, dim]

                # Insert the kronecker additions where the x would have been
                # done after the above loop to preserve order of indices
                # e.g. dx_{ij}/dx_{ab} -> I_{ia}I_{jb} rather than I_{jb}I_{ia}
                other_terms[i:i] = kronecker_additions

                # Build contraction and save shapes
                contraction_string = ""
                for ok, oi in other_terms:
                    result_shapes[ok] = self._shapes[ok]
                    contraction_string += ok + "_{" + oi + "}"

                # Add this contribution to the result string
                result_str += parsing.coefficient_to_string(self._terms[term]) + contraction_string

        # Parse result string into a contraction object
        return Contraction(result_str, **result_shapes)

    def __str__(self):
        return "".join(parsing.coefficient_to_string(self._terms[t], first_in_expression=i == 0) + t
                       for i, t in enumerate(self._terms))
