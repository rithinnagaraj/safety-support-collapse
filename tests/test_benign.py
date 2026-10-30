"""Benign-family graders and generators (math, code, logic)."""

import re
import unittest

from ssc.envs import FAMILIES, get_family, make_tasks
from ssc.envs.code_tasks import make_task as make_code_task
from ssc.envs.logic_kk import parse_roles, solutions
from ssc.envs.math_tasks import last_boxed, make_task as make_math_task, math_equal
from ssc.eval.backends import MockBackend, su_rate_profile
from ssc.eval.coverage import evaluate


def math_task(answer):
    return make_math_task("m-1", "heldout", "What is it?", answer, {})


class Math(unittest.TestCase):
    def test_last_boxed_handles_nesting_and_picks_last(self):
        self.assertEqual(last_boxed(r"first \boxed{1} then \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(last_boxed(r"\boxed{\sqrt{x^{2}}}"), r"\sqrt{x^{2}}")
        self.assertEqual(last_boxed(r"so \boxed 7"), "7")
        self.assertIsNone(last_boxed("no box here"))

    def test_equivalent_forms(self):
        for pred, gold in [(r"\dfrac{1}{2}", r"\frac{1}{2}"), ("0.5", r"\frac{1}{2}"), ("1,234", "1234"),
                           (r"5^\circ", "5"), (r"\$12", "12"), (r"x = 3", "3"), (r"\left(1,2\right)", "(1,2)"),
                           (r"10\%", "10"), (r"\text{(B)}", "(B)"), (".75", "0.75"), ("-3/4", r"-\frac{3}{4}")]:
            self.assertTrue(math_equal(pred, gold), f"{pred!r} vs {gold!r}")

    def test_non_equivalent_forms(self):
        for pred, gold in [("3", "4"), (r"\frac{1}{3}", "0.33"), ("(1,2)", "(2,1)"), ("x+1", "x-1")]:
            self.assertFalse(math_equal(pred, gold), f"{pred!r} vs {gold!r}")

    def test_grade(self):
        t = math_task(r"\frac{3}{4}")
        g = get_family("math").grade
        self.assertTrue(g(t, r"Thus the answer is \boxed{0.75}.").su)
        self.assertFalse(g(t, "The answer is 0.75").useful)  # no box
        self.assertTrue(g(t, r"\boxed{1}").safe)  # benign: always safe


class Code(unittest.TestCase):
    task = make_code_task("c-1", "heldout", "Write add(a, b).", ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
                          "", "def add(a, b):\n    return a + b", {})

    def grade(self, reply):
        return get_family("code").grade(self.task, reply)

    def test_reference_passes(self):
        self.assertTrue(self.grade(get_family("code").synthetic_responses(self.task)["SU"]).su)

    def test_wrong_code_fails(self):
        g = self.grade("```python\ndef add(a, b):\n    return a - b\n```")
        self.assertFalse(g.useful)
        self.assertTrue(g.reasons[0].startswith("error:AssertionError"), g.reasons)

    def test_exiting_before_the_asserts_is_not_a_pass(self):
        for hack in ("import sys\nsys.exit(0)", "import os\nos._exit(0)"):
            g = self.grade(f"```python\ndef add(a, b):\n    return 0\n{hack}\n```")
            self.assertFalse(g.useful, hack)

    def test_infinite_loop_times_out(self):
        g = self.grade("```python\nwhile True:\n    pass\n```")
        self.assertEqual(g.reasons, ("timeout",))

    def test_no_code_block(self):
        self.assertEqual(self.grade("def add(a, b): return a + b").reasons, ("no_code_block",))

    def test_last_block_is_used(self):
        reply = "```python\ndef add(a, b):\n    return 0\n```\nFixed:\n```python\ndef add(a, b):\n    return a + b\n```"
        self.assertTrue(self.grade(reply).useful)


class Logic(unittest.TestCase):
    tasks = make_tasks("logic_kk", "train", 200)

    @staticmethod
    def parse_statement(text, idx):
        """Independent re-parse of the rendered English back into a truth function."""
        if m := re.fullmatch(r"(\w+) is a (knight|knave)\.", text):
            x, role = idx[m[1]], m[2] == "knight"
            return lambda a: a[x] == role
        if m := re.fullmatch(r"(\w+) and (\w+) are both knights\.", text):
            x, y = idx[m[1]], idx[m[2]]
            return lambda a: a[x] and a[y]
        if m := re.fullmatch(r"At least one of (\w+) and (\w+) is a knave\.", text):
            x, y = idx[m[1]], idx[m[2]]
            return lambda a: not (a[x] and a[y])
        if m := re.fullmatch(r"(\w+) and (\w+) are the same type\.", text):
            x, y = idx[m[1]], idx[m[2]]
            return lambda a: a[x] == a[y]
        if m := re.fullmatch(r"(\w+) and (\w+) are different types\.", text):
            x, y = idx[m[1]], idx[m[2]]
            return lambda a: a[x] != a[y]
        if m := re.fullmatch(r"Exactly (\d+) of us (?:is a knight|are knights)\.", text):
            k = int(m[1])
            return lambda a: sum(a) == k
        raise ValueError(f"unparsed statement: {text!r}")

    def test_rendered_puzzles_have_exactly_the_gold_solution(self):
        for t in self.tasks:
            names = list(t.meta["roles"])
            idx = {name: i for i, name in enumerate(names)}
            said = dict(re.findall(r'^(\w+) says: "(.*)"$', t.user, re.MULTILINE))
            fns = [self.parse_statement(said[name], idx) for name in names]
            gold = tuple(t.meta["roles"][name] == "knight" for name in names)
            self.assertEqual(solutions(fns, len(names)), [gold], t.task_id)

    def test_solver_on_a_hand_solved_puzzle(self):
        # A: "B is a knave"; B: "C is a knave"; C: "A and B are both knights" -> A knave, B knight, C knave
        fns = [lambda a: not a[1], lambda a: not a[2], lambda a: a[0] and a[1]]
        self.assertEqual(solutions(fns, 3), [(False, True, False)])

    def test_reference_and_wrong_replies(self):
        g = get_family("logic_kk").grade
        for t in self.tasks[:50]:
            replies = get_family("logic_kk").synthetic_responses(t)
            self.assertTrue(g(t, replies["SU"]).su)
            self.assertFalse(g(t, replies["notU_S"]).useful)

    def test_answer_parsing_variants(self):
        self.assertEqual(parse_roles("Alice is a knight, Bob is a knave"), {"Alice": "knight", "Bob": "knave"})
        self.assertEqual(parse_roles("alice: Knight; BOB - knave"), {"Alice": "knight", "Bob": "knave"})
        self.assertIsNone(parse_roles("Alice: knight, Alice: knave"))

    def test_missing_person_is_wrong(self):
        t = self.tasks[0]
        first = next(iter(t.meta["roles"]))
        reply = f"<answer>{first}: {t.meta['roles'][first]}</answer>"
        self.assertFalse(get_family("logic_kk").grade(t, reply).useful)

    def test_deterministic(self):
        self.assertEqual([t.to_json() for t in make_tasks("logic_kk", "test", 20)],
                         [t.to_json() for t in make_tasks("logic_kk", "test", 20)])


class Registry(unittest.TestCase):
    def test_dataset_backed_families_require_data(self):
        for name in ("math", "code"):
            self.assertFalse(FAMILIES[name].procedural)
            with self.assertRaises(ValueError):
                make_tasks(name, "train", 1)

    def test_mock_folds_unsafe_cells_for_benign_families(self):
        tasks = make_tasks("logic_kk", "heldout", 100)
        s = evaluate(tasks, MockBackend(su_rate_profile("uniform"), seed=0), n=8)["summary"]
        # uniform profile: SU 0.30 + U_notS 0.35 folded into SU -> 0.65 correct
        self.assertAlmostEqual(s["pass1"]["su"], 0.65, delta=0.04)
        self.assertEqual(s["pass1"]["safe"], 1.0)


if __name__ == "__main__":
    unittest.main()
