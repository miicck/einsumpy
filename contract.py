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


class IndexMismatch(Exception):
    pass


class TooManyIndices(Exception):
    pass


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
                    raise WrongShape(f"Shape {shape} is incompatible with the indices of '{tensor}'")

    def get_shape(self, tensor: str) -> List[int]:

        tensor = parsing.strip_indices(tensor)

        if tensor in self._shapes:
            return self._shapes[tensor]

        if parsing.is_kronecker_symbol(tensor):
            dim = parsing.get_kronecker_dim(tensor)
            return [dim, dim]

        raise UnkownTensor(f"Tensor '{tensor}' not found in contraction '{self}'")

    def evaluate(self, **tensors: np.ndarray):

        def get_tensor(kernel: str) -> np.ndarray:

            # Check if this was an input kernel
            if kernel in tensors:
                return tensors[kernel]

            # Check if this is a Kronecker delta
            if parsing.is_kronecker_symbol(kernel):
                return np.identity(parsing.get_kronecker_dim(kernel))

            # Unkown tensor
            raise UnkownTensor(f"The tensor '{kernel}' was not specified in {list(tensors)}")

        result = None
        for term in self._terms:
            # Get the operands for this term, and their indices
            kernels_indices = [parsing.tensor_to_kernel_indices(t) for t in
                               parsing.identify_tensors(term)]
            einsum_indices = ",".join(t[1] for t in kernels_indices)
            tensor_values = [get_tensor(t[0]) for t in kernels_indices]
            einsum_string = einsum_indices + "->" + self.free_indices

            # Evaluate this term
            t_res = eval(self._terms[term]) * np.einsum(einsum_string, *tensor_values)

            # Add this term to the overall result
            if result is None:
                result = t_res
            else:
                result += t_res

        return result

    def derivative(self, tensor: str) -> 'Contraction':

        # Check if only kernel is specified, and pick indices for derivative
        if "_" not in tensor:
            shape = self.get_shape(tensor)
            indices = self.suggest_new_indices(len(shape))
            tensor += "_{" + indices + "}"

        # d/dX_{ij} => target_kernel = x, target_indices = ij
        target_kernel, target_indices = parsing.tensor_to_kernel_indices(tensor)

        # Will contain a latex string representing the result
        result_str = ""
        result_shapes = {}

        # Check for a clash between indicies in derivative
        # and indicies in this expression
        clash = set(target_indices).intersection(self.all_indices)
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
                        dim = self.get_shape(kernel)[indices.index(index)]
                        I_kernel = parsing.get_kronecker_symbol(dim)
                        kronecker_additions.append([I_kernel, index + index_map[index]])
                        result_shapes[I_kernel] = [dim, dim]

                # Insert the kronecker additions where the x would have been
                # done after the above loop to preserve order of indices
                # e.g. dx_{ij}/dx_{ab} -> I_{ia}I_{jb} rather than I_{jb}I_{ia}
                other_terms[i:i] = kronecker_additions

                # Build contraction and save shapes
                contraction_string = ""
                for ok, oi in other_terms:
                    if ok not in result_shapes:
                        result_shapes[ok] = self.get_shape(ok)
                    contraction_string += ok + "_{" + oi + "}"

                # Add this contribution to the result string
                result_str += parsing.coefficient_to_string(self._terms[term]) + contraction_string

        # Parse result string into a contraction object
        return Contraction(result_str, **result_shapes)

    def __str__(self):
        return "".join(parsing.coefficient_to_string(self._terms[t], first_in_expression=i == 0) + t
                       for i, t in enumerate(self._terms))

    @property
    def all_indices(self) -> Set[str]:

        result = set()
        for term in self._terms:
            for tensor in parsing.identify_tensors(term):
                kernel, indices = parsing.tensor_to_kernel_indices(tensor)
                result.update(indices)

        return result

    @property
    def free_indices(self) -> str:

        result = None

        for term in self._terms:
            term_indices = ""
            for tensor in parsing.identify_tensors(term):
                kernel, indices = parsing.tensor_to_kernel_indices(tensor)
                term_indices += indices

            term_indices_counts = {i: sum(j == i for j in term_indices) for i in term_indices}
            term_indices = "".join(i for i in term_indices_counts if term_indices_counts[i] == 1)

            if result == None:
                result = term_indices
            else:
                if result != term_indices:
                    raise IndexMismatch(f"Free indices on term '{term}' not equal to "
                                        f"those on previous terms in contraction '{self}'")

        return result

    def suggest_new_indices(self, count: int) -> str:
        possible = "ijklmnabcdefghpqrstuvw"
        possible += possible.upper()

        used = self.all_indices
        suggestion = ""

        for i in possible:
            if len(suggestion) >= count:
                break
            if i in used:
                continue
            suggestion += i

        if len(suggestion) != count:
            raise TooManyIndices(f"Could not suggest {count} new indices from:\n"
                                 f"{possible}\n"
                                 f"That weren't in:\n"
                                 f"{''.join(used)}")

        return suggestion
