"""Exact mathematics and logic generators used by the pilot and full suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import random
from typing import Literal


Domain = Literal["mathematics", "logic"]
Tree = int | tuple["Tree", "Tree"]

TREE_SHAPES: tuple[Tree, ...] = (
    (((0, 1), 2), 3),
    ((0, (1, 2)), 3),
    ((0, 1), (2, 3)),
    (0, ((1, 2), 3)),
    (0, (1, (2, 3))),
)

MATH_WORDING_BANKS: dict[str, tuple[str, ...]] = {
    "pilot_training": (
        "Evaluate this fully parenthesized expression. Output one integer only: {expression}",
        "Compute the exact value of {expression}. Respond with exactly one integer.",
        "Find the integer value of the fully parenthesized expression {expression}. Output only the answer.",
        "Solve {expression}. Return only its signed base-10 integer value.",
    ),
    "pilot_test": (
        "What integer does {expression} evaluate to? Return just the numeral.",
        "Calculate {expression}; your entire response must be the resulting integer.",
        "Determine the exact result of {expression}. Give no text besides the integer.",
        "Reduce the fully parenthesized arithmetic expression {expression} and print one integer.",
    ),
}

LOGIC_WORDING_BANKS: dict[str, tuple[str, ...]] = {
    "pilot_training": (
        "Given {assignment}, evaluate {expression}. Output exactly true or false in lowercase.",
        "Under the assignment {assignment}, what is the value of {expression}? Answer only true or false.",
        "Evaluate the Boolean expression {expression} using {assignment}. Return one lowercase Boolean.",
        "With {assignment}, determine {expression}. Your whole response must be true or false.",
    ),
    "pilot_test": (
        "For {assignment}, is {expression} true or false? Print only the lowercase result.",
        "Compute the truth value of {expression} when {assignment}. Respond with just true or false.",
        "Using the variable values {assignment}, reduce {expression} to one lowercase Boolean.",
        "Decide {expression} under {assignment}; no output is allowed except true or false.",
    ),
}


@dataclass(frozen=True)
class BenignInstance:
    instance_id: str
    split: str
    template_id: str
    domain: Domain
    latent_signature: str
    prompt: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "BenignInstance":
        return cls(**raw)

    def score(self, response: str) -> int:
        return int(response.strip() == self.answer)


def _stable_seed(seed: int, *parts: object) -> int:
    digest = sha256("|".join([str(seed), *map(str, parts)]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _math_render(tree: Tree, leaves: list[int], operators: list[str], cursor: list[int]) -> tuple[str, int]:
    if isinstance(tree, int):
        return str(leaves[tree]), leaves[tree]
    left_text, left_value = _math_render(tree[0], leaves, operators, cursor)
    operator = operators[cursor[0]]
    cursor[0] += 1
    right_text, right_value = _math_render(tree[1], leaves, operators, cursor)
    if operator == "+":
        value = left_value + right_value
    elif operator == "-":
        value = left_value - right_value
    else:
        value = left_value * right_value
    return f"({left_text} {operator} {right_text})", value


def _logic_render(
    tree: Tree,
    variables: list[str],
    assignments: dict[str, bool],
    negated: list[bool],
    operators: list[str],
    cursor: list[int],
) -> tuple[str, bool]:
    if isinstance(tree, int):
        name = variables[tree]
        value = assignments[name]
        return (f"NOT {name}", not value) if negated[tree] else (name, value)
    left_text, left_value = _logic_render(tree[0], variables, assignments, negated, operators, cursor)
    operator = operators[cursor[0]]
    cursor[0] += 1
    right_text, right_value = _logic_render(tree[1], variables, assignments, negated, operators, cursor)
    if operator == "AND":
        value = left_value and right_value
    elif operator == "OR":
        value = left_value or right_value
    else:
        value = left_value != right_value
    return f"({left_text} {operator} {right_text})", value


def generate_benign_split(
    *, split: str, domain: Domain, count: int, seed: int, wording_bank: str = "pilot_training"
) -> list[BenignInstance]:
    rows: list[BenignInstance] = []
    for index in range(count):
        rng = random.Random(_stable_seed(seed, split, domain, index))
        shape_index = rng.randrange(len(TREE_SHAPES))
        tree = TREE_SHAPES[shape_index]
        if domain == "mathematics":
            templates = MATH_WORDING_BANKS.get(wording_bank)
            if templates is None:
                raise ValueError(f"Unknown mathematics wording bank: {wording_bank}")
            leaves = [rng.randint(1, 99) for _ in range(4)]
            operators = [rng.choice(["+", "-", "×"]) for _ in range(3)]
            expression, value = _math_render(tree, leaves, operators, [0])
            wording_index = rng.randrange(len(templates))
            prompt = templates[wording_index].format(expression=expression)
            answer = str(value)
            latent_signature = sha256(
                repr((domain, tree, tuple(leaves), tuple(operators))).encode("utf-8")
            ).hexdigest()
        else:
            templates = LOGIC_WORDING_BANKS.get(wording_bank)
            if templates is None:
                raise ValueError(f"Unknown logic wording bank: {wording_bank}")
            variables = rng.sample(["A", "B", "C", "D", "E", "F", "G", "H"], 4)
            assignments = {variable: bool(rng.getrandbits(1)) for variable in variables}
            negated = [rng.random() < 0.5 for _ in variables]
            operators = [rng.choice(["AND", "OR", "XOR"]) for _ in range(3)]
            expression, value = _logic_render(tree, variables, assignments, negated, operators, [0])
            assignment_text = ", ".join(f"{key}={'true' if value else 'false'}" for key, value in assignments.items())
            wording_index = rng.randrange(len(templates))
            prompt = templates[wording_index].format(assignment=assignment_text, expression=expression)
            answer = "true" if value else "false"
            latent_signature = sha256(
                repr(
                    (
                        domain,
                        tree,
                        tuple(variables),
                        tuple(assignments.items()),
                        tuple(negated),
                        tuple(operators),
                    )
                ).encode("utf-8")
            ).hexdigest()
        template_id = f"{wording_bank}-{domain}-wording-{wording_index}-shape-{shape_index}"
        rows.append(
            BenignInstance(
                f"{split}-{domain}-{index:05d}",
                split,
                template_id,
                domain,
                latent_signature,
                prompt,
                answer,
            )
        )
    return rows
