from typing import Iterable, Tuple, List, Dict, Set
from sympy.simplify import simplify
from sympy.parsing import parse_expr
from sympy import Expr
import numpy as np


class ParsingError(Exception):
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
            raise ParsingError(f"_ at end of expression: '{latex}'")

        # Identify end of indices
        i_end = i + 1
        if latex[i + 1] == "{":
            # Find matching }
            for j in range(i + 2, len(latex)):
                if latex[j] == "}":
                    i_end = j
                    break
            if i_end == i + 1:
                raise ParsingError(f"Unmatched {{ in expression: '{latex}'")

        # Identify start of tensor
        i_start = i - 1

        # Check for bracketed kernel like I(30)
        if latex[i_start] == ")":
            for j in range(i_start, -1, -1):
                if latex[j] == "(":
                    i_start = j - 1
                    break

            if latex[i_start + 1] != "(":
                raise ParsingError(f"Unmatched ( in '{latex}'")

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
        raise ParsingError(f"Tensor '{tensor}' does not have a subscript!")

    return tensor[:i], tensor[i + 1:].replace("{", "").replace("}", "")


def strip_indices(tensor: str) -> str:
    if "_" not in tensor:
        return tensor
    return tensor.split("_")[0]


def is_kronecker_symbol(kernel: str) -> bool:
    return kernel[0] == "I" and kernel[1] == "(" and kernel[-1] == ")"


def get_kronecker_symbol(dim: int) -> str:
    return f"I({dim})"


def get_kronecker_dim(symbol: str) -> int:
    if not symbol[0] == "I":
        raise ParsingError(f"Kronecker tensor '{symbol}' does not start with 'I'")
    if not symbol[1] == "(":
        raise ParsingError(f"Missing ( in Kronecker tensor '{symbol}'")
    if not symbol[-1] == ")":
        raise ParsingError(f"Missing ) in Kronecker tensor '{symbol}'")

    try:
        return int(symbol[2:-1])
    except ValueError:
        raise ParsingError(f"Could not parse dimension from Kronecker tensor '{symbol}'")
