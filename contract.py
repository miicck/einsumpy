from typing import Iterable, Tuple, List, Dict, Set
from sympy.simplify import simplify
from sympy.parsing import parse_expr
from sympy import Expr
import numpy as np


class LatexError(Exception):
    pass


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


def remove_floating_point_trailing_zeros(s: str) -> str:
    digits = set("0123456789")

    # Work backwards until we hit a "."
    for i in range(len(s) - 1, -1, -1):
        if s[i] != ".":
            continue

        # Work forwards until we run out of digits
        for j in range(i + 1, len(s) + 1):
            if j >= len(s) or s[j] not in digits:
                break

        # Strip trailing zeros from floating point
        new_fp = s[i:j].rstrip("0")

        # If they were all stripped, add one back in
        if new_fp[-1] == ".":
            new_fp += "0"

        s = s[:i] + new_fp + s[j:]

    return s


def identify_tensors(latex: str) -> Iterable[str]:
    # Characters that could name a variable in latex
    naming_chars = set("abcdefghijlkmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Identify individual tensors
    for i in range(len(latex)):
        if latex[i] != "_":
            continue

        if i == len(latex) - 1:
            raise LatexError(f"_ at end of expression: '{latex}'")

        # Identify end of indices
        i_end = i + 1
        if latex[i + 1] == "{":
            # Find matching }
            for j in range(i + 2, len(latex)):
                if latex[j] == "}":
                    i_end = j
                    break
            if i_end == i + 1:
                raise LatexError(f"Unmatched {{ in expression: '{latex}'")

        # Identify start of tensor
        i_start = i - 1

        # Check for bracketed kernel like I(30)
        if latex[i_start] == ")":
            for j in range(i_start, -1, -1):
                if latex[j] == "(":
                    i_start = j - 1
                    break

            if latex[i_start + 1] != "(":
                raise LatexError(f"Unmatched ( in '{latex}'")

        # Check for backslash-delimited names (e.g. \mu)
        for j in range(i_start, -1, -1):
            if latex[j] == "\\":
                i_start = j
                break
            if latex[j] not in naming_chars:
                break

        yield latex[i_start:i_end + 1]


def to_contractions_and_coefficients(latex: str) -> Dict[str, str]:
    from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication
    from sympy import Symbol
    from sympy import expand
    from sympy.printing import StrPrinter

    # Get a list of unique tensors in the expression,
    # in the same order as they appear in the expression
    tensors = list(identify_tensors(latex))
    seen = set()
    tensors = [t for t in tensors if not (t in seen or seen.add(t))]

    # Replace each tensor with a sympy symbol
    symbol_names = "abcdefghijklmnopqrstuvwxyz"
    tensor_to_placeholders = {tensors[i]: symbol_names[i] for i in range(len(tensors))}
    placeholder_to_tensors = {symbol_names[i]: tensors[i] for i in range(len(tensors))}
    for t in tensor_to_placeholders:
        # Whitespace is neccassary to pick up implicit multiplication
        latex = latex.replace(t, " " + tensor_to_placeholders[t] + " ")

    # Parse the result with sympy and simplify
    transformations = (standard_transformations + (implicit_multiplication,))
    symb_dict = {s: Symbol(s, commutative=True) for s in placeholder_to_tensors}
    expr = expand(simplify(parse_expr(latex, symb_dict, transformations=transformations)))

    # Return result as dictionary with original tensor symbols
    dict = expr.as_coefficients_dict()

    # Replace powers a**2 with aa
    # so that x_i**2 becomes x_ix_i
    class CustomPrinter(StrPrinter):
        def _print_Pow(self, expr):
            if expr.exp.is_integer:
                return f"{expr.base}" * expr.exp
            else:
                return super()._print_Pow(expr)

    # Define a method to convert a sympy expression
    # in terms of the placeholder symbols into a latex tensor string
    custom_print = CustomPrinter().doprint

    def to_tensor_str(s: Expr):
        s = custom_print(s)
        s = s.replace("*", "")

        # Replace placeholder variables with their
        # corresponding tensors (must be done in one
        # go to avoid replacing parts of tensors
        # already inserted)
        # e.g. "ab" with a -> T_{b} b -> T_{c}
        # should not replace the T_{b} with T_{T_{c}}
        return "".join(placeholder_to_tensors[c]
                       if c in placeholder_to_tensors
                       else c
                       for c in s)

    # Convert a sympy expression representing
    # a coefficient into a latex string
    def to_coeff_str(s: Expr):
        return remove_floating_point_trailing_zeros(str(s))

    return {to_tensor_str(s): to_coeff_str(dict[s]) for s in dict}


def coefficient_to_string(c: str, first_in_expression: bool = False) -> str:
    c = parse_expr(c)
    c_str = str(c)

    # Remove information about sign
    if c_str[0] in {"-", "+"}:
        c_str = c_str[1:]

    # A coefficient of 1 is just left blank
    if c_str == "1":
        c_str = ""

    # Put () around divisions
    if "/" in c_str:
        c_str = f"({c_str})"

    # First coefficient doesn't need a + in front of it
    if first_in_expression:
        return f"-{c_str}" if c < 0 else c_str

    # Re-sign the coefficient
    return f"-{c_str}" if c < 0 else f"+{c_str}"


def tensor_to_kernel_indices(tensor: str) -> Tuple[str, str]:
    i = tensor.find("_")
    if i < 0:
        raise LatexError(f"Tensor '{tensor}' does not have a subscript!")

    return tensor[:i], tensor[i + 1:].replace("{", "").replace("}", "")


class Contraction:

    def __init__(self, latex: str, **shapes: Iterable[int]):
        self._terms = to_contractions_and_coefficients(latex)
        self._shapes = {s: list(shapes[s]) for s in shapes}

        for term in self._terms:
            for tensor in identify_tensors(term):
                kernel, indices = tensor_to_kernel_indices(tensor)
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
            kernels_indices = [tensor_to_kernel_indices(t) for t in identify_tensors(term)]
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
        target_kernel, target_indices = tensor_to_kernel_indices(tensor)

        # Will contain a latex string representing the result
        result_str = ""
        result_shapes = {}

        # Check for a clash between indicies in derivative
        # and indicies in this expression
        for term in self._terms:
            for t in identify_tensors(term):
                clash = set(target_indices).intersection(set(tensor_to_kernel_indices(t)[1]))
                if len(clash) > 0:
                    raise IndexClash.from_set_and_expressions(
                        clash, str(self), f"d/d{target_kernel}_{{{target_indices}}}")

        # Take derivative of each term, sum the result
        for term in self._terms:

            # Get kernels and indices of this term
            # A_{ij}B_{ij} kernels = A, B indices = ij, ij
            kernels_indices = [tensor_to_kernel_indices(t) for t in identify_tensors(term)]

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
                result_str += coefficient_to_string(self._terms[term]) + contraction_string

        # Parse result string into a contraction object
        return Contraction(result_str, **result_shapes)

    def __str__(self):
        return "".join(coefficient_to_string(self._terms[t], first_in_expression=i == 0) + t
                       for i, t in enumerate(self._terms))
