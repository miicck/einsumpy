from typing import Iterable, Tuple, List, Dict
from sympy.simplify import simplify
from sympy.parsing import parse_expr
from sympy import Expr
import numpy as np


class LatexError(Exception):
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
        for t in placeholder_to_tensors:
            s = s.replace(t, placeholder_to_tensors[t])
        return s

    # Convert a sympy expression representing
    # a coefficient into a latex string
    def to_coeff_str(s: Expr):
        return remove_floating_point_trailing_zeros(str(s))

    return {to_tensor_str(s): to_coeff_str(dict[s]) for s in dict}


def tensor_to_kernel_indices(tensor: str) -> Tuple[str, str]:
    i = tensor.find("_")
    if i < 0:
        raise LatexError(f"Tensor '{tensor}' does not have a subscript!")

    return tensor[:i], tensor[i + 1:].replace("{", "").replace("}", "")


class Contraction:

    def __init__(self, latex: str, **shapes: Iterable[int]):
        self._terms = to_contractions_and_coefficients(latex)
        self._shapes = shapes

    def evaluate(self, **tensors: np.ndarray):
        result = None
        for t in self._terms:
            # Get the operands for this term, and their indices
            kernels_indices = [tensor_to_kernel_indices(t) for t in identify_tensors(t)]
            einsum_indices = ",".join(t[1] for t in kernels_indices)
            tensor_values = [tensors[t[0]] for t in kernels_indices]

            # Evaluate this term
            t_res = eval(self._terms[t]) * np.einsum(einsum_indices, *tensor_values)

            # Add this term to the overall result
            if result is None:
                result = t_res
            else:
                result += t_res

        return result

    def __str__(self):

        def print_coeff(c: str):
            c = parse_expr(c)
            c_str = str(c)

            if c_str[0] in {"-", "+"}:
                c_str = c_str[1:]

            if c_str == "1":
                c_str = ""

            if "/" in c_str:
                c_str = f"({c_str})"

            if print_coeff.first:
                print_coeff.first = False
                return f"-{c_str}" if c < 0 else c_str

            return f"-{c_str}" if c < 0 else f"+{c_str}"

        print_coeff.first = True

        return "".join(print_coeff(self._terms[t]) + t for t in self._terms)
