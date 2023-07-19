import numpy as np
from typing import Iterable, Tuple, List, Union


class IndicesMismatch(Exception):
    pass


class TensorNotFound(Exception):
    pass


class TooManyIndices(Exception):
    pass


class IndicesInUse(Exception):
    pass


Tensor = Union[np.ndarray, Iterable, int, float]


class LatexTools:

    @staticmethod
    def standardize(latex: str) -> str:
        return LatexTools.make_subscripts_explicit(LatexTools.remove_unnecessary_whitespace(latex))

    @staticmethod
    def make_subscripts_explicit(latex: str):
        for i in range(len(latex) - 2, -1, -1):
            if latex[i] == "_" and latex[i + 1] != "{":
                latex = latex[:i + 1] + "{" + latex[i + 1] + "}" + latex[i + 2:]
        return latex

    @staticmethod
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

    @staticmethod
    def split_terms(latex: str) -> Iterable[str]:
        latex = LatexTools.standardize(latex)

        i_start = 0
        for i in range(len(latex)):
            if latex[i] in {"-", "+"}:
                if i > i_start:
                    yield latex[i_start:i]
                i_start = i
        yield latex[i_start:]

    @staticmethod
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


class Contraction:

    def __init__(self, latex: str, **shapes: Iterable[int]):
        self._tensors, self._indices = Contraction.latex_to_tensors_indices(latex)

        for t in self._tensors:
            if t not in shapes:
                raise IndicesMismatch(f"No shape provided for tensor: {t}")

        self._shapes = [list(shapes[t]) for t in self._tensors]
        self._path = np.einsum_path(self.einsum_string, *[np.ones(s) for s in self._shapes], optimize='optimal')

    def evaluate(self, **tensors: Tensor) -> np.ndarray:

        # Convert input tensors into numpy arrays with correct dimension
        einsum_tensors = []
        for i, t, s in zip(self._indices, self._tensors, self._shapes):
            if t in tensors:
                einsum_tensors.append(Contraction.to_numpy(tensors[t], len(i)))
            elif t.startswith("I"):
                einsum_tensors.append(np.identity(int(t[1:])))
            else:
                raise TensorNotFound(f"The tensor {t} was not found in the contraction {self}")

        # Perform contraction
        return np.einsum(self.einsum_string, *einsum_tensors, optimize=self._path[0])

    def derivative(self, tensor: str) -> Iterable['Contraction']:

        tensor, indices = tensor.split("_")
        indices = indices.replace("}", "").replace("{", "")

        if tensor not in self._tensors:
            raise TensorNotFound(f"The tensor {tensor} was not found in the contraction {self}")

        # Find indices already in use
        used_indices = "".join(self._indices)

        clash_indices = set(used_indices).intersection(set(indices))
        if indices is not None and len(clash_indices) > 0:
            raise IndicesInUse(f"The indices {{{', '.join(clash_indices)}}} are already in use!")

        # Find occurrences of the tensor that we're taking derivatives with respect to
        for i, (t_ind, t) in enumerate(zip(self._indices, self._tensors)):
            if t != tensor:
                continue

            # Get the remaining tensors after this one has disappeared due to derivative
            other_indices = [j for j in range(len(self._indices)) if j != i]
            term_indices = [self._indices[j] for j in other_indices]
            term_tensors = [self._tensors[j] for j in other_indices]
            term_shapes = [self._shapes[j] for j in other_indices]

            # Replace indices corresponding to this occurrence of the tensor with the new indices
            # (i.e. do the index replacement due to the Kronecker delta)
            for j, (a, b) in enumerate(zip(t_ind, indices)):

                if any(a in p for p in term_indices):
                    # If a appears in at least one other tensor, we can just replace it
                    term_indices = [p.replace(a, b) for p in term_indices]
                else:
                    # Add an explicit Kronecker delta (where the previous tensor was)
                    size = self._shapes[i][j]
                    term_tensors.insert(i, f"I{size}")
                    term_indices.insert(i, a + b)
                    term_shapes.insert(i, [size, size])

            yield Contraction(Contraction.to_latex(term_tensors, term_indices),
                              **{t: s for t, s in zip(term_tensors, term_shapes)})

    def __str__(self):
        return self.latex

    @property
    def latex(self) -> str:
        return Contraction.to_latex(self._tensors, self._indices)

    @property
    def einsum_input_indices(self) -> str:
        return ",".join(self._indices)

    @property
    def einsum_output_indices(self) -> str:
        ii = self.einsum_input_indices.replace(",", "")
        counts = {i: sum(x == i for x in ii) for i in ii}
        for i in counts:
            if counts[i] > 1:
                ii = ii.replace(i, "")
        return ii

    @property
    def einsum_string(self) -> str:
        oi = self.einsum_output_indices
        if len(oi) == 0:
            return self.einsum_input_indices
        return self.einsum_input_indices + "->" + oi

    @staticmethod
    def to_numpy(value: Tensor, dim: int):
        if isinstance(value, Iterable):
            result = np.asarray(value)
        else:
            for d in range(dim):
                value = [value]
            result = np.asarray(value)
        assert len(result.shape) == dim
        return result

    @staticmethod
    def to_latex(tensors: Iterable[str], indices: Iterable[str]):
        return LatexTools.standardize("".join(t + f"_{{{i}}}" for t, i in zip(tensors, indices)))

    @staticmethod
    def latex_to_tensors_indices(latex: str) -> Tuple[List[str], List[str]]:
        latex = LatexTools.standardize(latex)
        split = [t.replace("{", "").split("_") for t in latex.split("}") if len(t.strip()) > 0]
        return [s[0].strip() for s in split], [s[1].strip() for s in split]


class CompositeContraction:

    def __init__(self, latex: str, **shapes: Iterable[int]):
        split = [LatexTools.split_coefficient(t) for t in LatexTools.split_terms(latex)]
        terms = [c[1] for c in split]
        self._coefficients = [c[0] for c in split]
        self._shapes = {x: shapes[x] for x in shapes}
        self._contractions = [Contraction(t, **self._shapes) for t in terms]

    def evaluate(self, **tensors: Tensor) -> np.ndarray:
        return sum(eval(coeff) * contr.evaluate(**tensors) for
                   coeff, contr in zip(self._coefficients, self._contractions))

    def derivative(self, tensor: str) -> 'CompositeContraction':
        latex = ""
        for coeff, contr in zip(self._coefficients, self._contractions):
            for t in contr.derivative(tensor):
                latex += coeff + t.latex
        return CompositeContraction(latex, **self._shapes)

    def __str__(self):
        return self.latex

    @property
    def latex(self) -> str:
        latex = ""
        for i, (coeff, contr) in enumerate(zip(self._coefficients, self._contractions)):

            if coeff == "":
                coeff = "1"

            if coeff == "-1":
                coeff = "-"
            elif coeff == "+1":
                coeff = "+"
            elif coeff == "1":
                if i == 0:
                    coeff = ""
                else:
                    coeff = "+"

            if i > 0 and coeff[0] not in {"-", "+"}:
                coeff = "+" + coeff

            latex += coeff + contr.latex
        return latex
