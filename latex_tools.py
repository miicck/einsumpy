from typing import Iterable, Tuple


def standardize(latex: str) -> str:
    return make_subscripts_explicit(remove_unnecessary_whitespace(latex))


def make_subscripts_explicit(latex: str):
    for i in range(len(latex) - 2, -1, -1):
        if latex[i] == "_" and latex[i + 1] != "{":
            latex = latex[:i + 1] + "{" + latex[i + 1] + "}" + latex[i + 2:]
    return latex


def remove_unnecessary_whitespace(latex: str) -> str:
    result = ""
    allow = False
    for c in latex:
        if c == "\\":
            allow = True
        elif c == "_":
            allow = False
        elif c == " ":
            if not allow:
                continue
            allow = False
        result += c
    return result


def split_terms(latex: str) -> Iterable[str]:
    latex = standardize(latex)

    def remove_first_plus(term):
        if remove_first_plus.first:
            remove_first_plus.first = False
            if term[0] == "+":
                return term[1:]
        return term

    remove_first_plus.first = True

    i_start = 0
    for i in range(len(latex)):
        if latex[i] in {"-", "+"} and (i == 0 or latex[i - 1] != "("):
            if i > i_start:
                yield remove_first_plus(latex[i_start:i])
            i_start = i

    yield remove_first_plus(latex[i_start:])


def split_parenthetical_factors(latex: str) -> Iterable[str]:
    def split_single(l: str) -> Tuple[str, str]:

        if l[0] != "(":

            # Find first (
            i = l.find("(")
            if i == -1:
                # No ( => all one term
                return l, None

            # Ensure first term is bracketed
            l = "(" + l[:i] + ")" + l[i:]

        # Find first matching pair of ( )
        for i in range(len(l)):

            if l[i] in {"+", "-"}:
                return latex, None  # Seperate terms before ( => not a produce

            if l[i] == "(":
                depth = 1
                for j in range(i + 1, len(l)):
                    if l[j] == "(":
                        depth += 1
                    elif l[j] == ")":
                        depth -= 1
                        if depth == 0:
                            return l[i + 1: j], l[:i] + l[j + 1:]

        raise Exception(f"Could not split a parenthetical factor from {l}")

    remaining_latex = standardize(latex)
    while True:
        factor, remaining_latex = split_single(remaining_latex)

        while factor.startswith("(") and factor.endswith(")"):
            factor = factor[1:-1]

        yield factor
        if remaining_latex is None or len(remaining_latex) == 0:
            return


def simplify_constant(latex_constant: str, max_float_digits=10):
    terms = list(split_terms(latex_constant))
    if len(terms) > 1:
        raise NotImplementedError(terms)

    factors = list(split_parenthetical_factors(latex_constant))

    if "." in latex_constant:
        f = float(eval("*".join(factors)))
        return str(round(f, max_float_digits))

    numerators = []
    denominators = []
    for f in split_parenthetical_factors(latex_constant):
        s = f.split("/")
        numerators.append(s[0])
        denominators.extend(s[1:])

    numerator = str(eval("*".join(numerators)))
    if len(denominators) == 0:
        return numerator

    denominator = str(eval("*".join(denominators)))
    return f"{numerator}/{denominator}"


def split_coefficient(latex_term: str) -> Tuple[str, str]:
    # Ensure the input is considered a single term
    terms = list(split_terms(latex_term))
    if len(terms) > 1:
        raise Exception("Split coefficient only works with a single term!\n"
                        f"The expression: {latex_term}\n"
                        f"Yields the terms: {terms}")
    latex_term = terms[0]

    coeff_chars = set("+-0123456789./()")

    def is_main_bit(i):
        if latex_term[i] not in coeff_chars:
            return True
        if i > 0 and latex_term[i - 1] in {"_", "^"}:
            return True
        return False

    for i_pre in range(len(latex_term)):
        if is_main_bit(i_pre):
            break

    for i_post in range(len(latex_term) - 1, -1, -1):
        if is_main_bit(i_post):
            i_post += 1
            break

    # Identify coefficient prefactor
    pre = latex_term[:i_pre]
    if len(pre) == 0 or pre in {"+", "-"}:
        pre = pre + "1"

    # Identify coefficient postfactor
    post = latex_term[i_post:]
    if post.startswith("/") or len(post) == 0:
        post = "1" + post

    # Simplify the combined pre*post coeffficient
    coeff = simplify_constant(f"({pre})({post})")

    # Parenthesize division
    if "/" in coeff:
        if coeff[0] in {"-", "+"}:
            coeff = coeff[0] + "(" + coeff[1:] + ")"
        else:
            coeff = "(" + coeff + ")"

    main_bit = latex_term[i_pre:i_post]
    return coeff.strip(), main_bit.strip()


def sum_coefficients(coeffs: Iterable[str]) -> str:
    result = ""
    for i, c in enumerate(coeffs):
        c = standardize(c)
        if c[0] in {"+", "-"} or i == 0:
            result += c
        else:
            result += f"+{c}"

    if i > 0:
        result = "(" + result + ")"

    return result
