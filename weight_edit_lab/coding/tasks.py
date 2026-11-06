from dataclasses import dataclass

from ..tasks import rng_for


@dataclass(frozen=True)
class Problem:
    name: str
    instruction: str
    solution: str
    cases: tuple
    family: str = "micro"
    instruction_budget: int = 20000
    completion_prefix: str = ""

    @property
    def prompt(self) -> str:
        return ("Write a Python function solve(xs), where xs is a list of integers. "
                + self.instruction + " Return only the function, no explanation. "
                "Do not use imports, annotations, recursion, or helper functions.")


def problems(split: str, seed: int, count: int = 4) -> list[Problem]:
    if split not in ("support", "score", "test", "retention_score", "retention_test", "rehearsal") or type(seed) is not int or seed < 0:
        raise ValueError("invalid coding split or seed")
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError("problem count must be in 1..12")
    rng = rng_for("coding-v2", split, seed)
    result = []
    for i in range(count):
        # Parameter-disjoint instances, NOT unseen task families. Retention
        # deliberately reuses old skills with independent executable cases.
        offset = {"support": 0, "score": 3, "test": 6}.get(split, 0)
        k = rng.randint(2 + offset, 4 + offset)
        threshold = rng.randint(-7 + 2 * offset, -5 + 2 * offset)
        if split in ("retention_score", "retention_test", "rehearsal"):
            family = i % 4
            instructions = ["Return the sum of all elements.", "Return the elements in reverse order.",
                            "Return the number of strictly positive elements.",
                            "Return the smallest element, or 0 if the list is empty."]
            codes = ["sum(xs)", "xs[::-1]", "sum(1 for x in xs if x > 0)", "min(xs) if xs else 0"]
            functions = [lambda xs: sum(xs), lambda xs: xs[::-1],
                         lambda xs: sum(1 for x in xs if x > 0), lambda xs: min(xs) if xs else 0]
            instruction, code, reference = instructions[family], codes[family], functions[family]
        else:
            family = i % 4
            if family == 0:
                instruction = f"Return the sum of elements strictly greater than {threshold} whose value is divisible by {k}."
                code = f"sum(x for x in xs if x > {threshold} and x % {k} == 0)"
                reference = lambda xs, t=threshold, k=k: sum(x for x in xs if x > t and x % k == 0)
            elif family == 1:
                instruction = f"Return a list of the zero-based indices of elements less than or equal to {threshold}, in order."
                code = f"[i for i, x in enumerate(xs) if x <= {threshold}]"
                reference = lambda xs, t=threshold: [i for i, x in enumerate(xs) if x <= t]
            elif family == 2:
                instruction = f"Return the number of adjacent pairs whose difference (second minus first) is at least {k}."
                code = f"sum(1 for i in range(1, len(xs)) if xs[i] - xs[i-1] >= {k})"
                reference = lambda xs, k=k: sum(1 for i in range(1, len(xs)) if xs[i]-xs[i-1] >= k)
            else:
                instruction = f"Return a list with each element clipped to the inclusive interval [-{k}, {k}], preserving order."
                code = f"[max(-{k}, min({k}, x)) for x in xs]"
                reference = lambda xs, k=k: [max(-k, min(k, x)) for x in xs]
        inputs = [[], [threshold], [k, -k, 0], [threshold-1, threshold, threshold+1]]
        inputs += [[rng.randint(-20, 20) for _ in range(rng.randint(1, 12))] for _ in range(8)]
        result.append(Problem(f"{split}-{seed}-{i}", instruction, f"def solve(xs):\n    return {code}\n",
                              tuple((xs, reference(xs)) for xs in inputs)))
    return result
