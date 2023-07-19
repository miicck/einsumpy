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

    i_start = 0
    for i in range(len(latex)):
        if latex[i] in {"-", "+"}:
            if i > i_start:
                yield latex[i_start:i]
            i_start = i
    yield latex[i_start:]


def split_parenthetical_factors(latex: str) -> Iterable[str]:
    def split_single(l: str) -> Tuple[str, str]:
        for i in range(len(l)):

            if l[i] in {"+", "-"}:
                return None

            if l[i] == "(":
                depth = 0
                for j in range(i + 1, len(l)):
                    if l[j] == "(":
                        depth += 1
                    if l[j] == ")":
                        if depth == 0:
                            return l[i + 1:j], l[:i] + l[j:]
                        depth -= 1
        return None

    while True:
        split = split_single(latex)
        if split is None:
            return

        yield split[0]
        latex = split[1]


def split_coefficient(latex_term: str) -> Tuple[str, str]:
    coeff_chars = set("+-0123456789./")

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

    # Evaluate product of pre and post apart from divisions
    to_eval = pre + "*" + post
    coeff = "/".join(str(eval(x)) for x in to_eval.split("/"))

    # Parenthesize division
    if "/" in coeff:
        if coeff[0] in {"-", "+"}:
            coeff = coeff[0] + "(" + coeff[1:] + ")"
        else:
            coeff = "(" + coeff + ")"

    main_bit = latex_term[i_pre:i_post]
    return coeff.strip(), main_bit.strip()
