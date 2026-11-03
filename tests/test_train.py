"""Ledger, reward adapter, and recovery-target checks (plan section 13, "Group-type counters", "Cost accounting")."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from ssc.envs import get_family, make_tasks
from ssc.train.grpo import grpo_kwargs
from ssc.train.ledger import CostLedger, GroupAlignmentError, cost_at_step, load_rows
from ssc.train.rewards import SUReward
from ssc.train.target import EvalPoint, check_tau, time_to_target


class Ledger(unittest.TestCase):
    def test_group_types_on_synthetic_stream(self):
        led = CostLedger(group_size=4)
        ids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4
        rewards = [0, 0, 0, 0] + [1, 1, 1, 1] + [0, 1, 0, 0] + [1, 1, 1, 0]
        row = led.record(ids, rewards, [10] * 16)
        self.assertEqual(row["groups"], {"all_fail": 1, "all_success": 1, "mixed": 2})
        self.assertEqual((row["trajectories"], row["tokens"]), (16, 160))

    def test_cumulative_and_filtered_rollouts_count(self):
        led = CostLedger(group_size=2)
        led.record(["a", "a"], [0, 0], [5, 7])
        led.record(["b", "b"], [1, 0], [3, 3], filtered=6)
        self.assertEqual(led.trajectories, 2 + 2 + 6)
        self.assertEqual(led.tokens, 18)
        self.assertEqual(led.totals()["groups"], {"all_fail": 1, "all_success": 0, "mixed": 1})

    def test_misaligned_groups_raise(self):
        led = CostLedger(group_size=2)
        with self.assertRaises(GroupAlignmentError):
            led.record(["a", "b", "b", "b"], [0, 0, 0, 0], [1] * 4)
        with self.assertRaises(GroupAlignmentError):
            led.record(["a", "a", "a"], [0, 0, 0], [1] * 3)

    def test_file_rows_match_memory_and_cost_at_step(self):
        with tempfile.TemporaryDirectory() as d:
            led = CostLedger(group_size=2, path=Path(d) / "ledger.jsonl")
            for step in range(5):
                led.current_step = step
                led.record(["x", "x"], [0, 1], [4, 6])
            rows = load_rows(led.path)
            self.assertEqual(rows, led.rows)
            # checkpoint after 3 steps was trained on the batches generated at steps 0, 1, 2
            self.assertEqual(cost_at_step(rows, 3), {"trajectories": 6, "tokens": 30})


class Reward(unittest.TestCase):
    def setUp(self):
        self.tasks = make_tasks("fb_injection", "train", 3)
        self.g = 4
        self.task_json, self.completions, self.expected = [], [], []
        for t in self.tasks:
            replies = get_family(t.family).synthetic_responses(t)
            for cell in ("SU", "U_notS", "notU_S", "SU"):
                self.task_json.append(t.to_json())
                self.completions.append([{"role": "assistant", "content": replies[cell]}])
                self.expected.append(1.0 if cell == "SU" else 0.0)

    def test_rewards_and_ledger(self):
        led = CostLedger(group_size=self.g)
        reward = SUReward(led)
        ids = [[0] * 7] * len(self.completions)
        out = reward(prompts=[None] * len(self.completions), completions=self.completions,
                     task_json=self.task_json, completion_ids=ids)
        self.assertEqual(out, self.expected)
        self.assertEqual(led.trajectories, 12)
        self.assertEqual(led.tokens, 84)
        self.assertEqual(led.rows[0]["groups"], {"all_fail": 0, "all_success": 0, "mixed": 3})
        self.assertEqual(reward.cells["SU"], 6)
        self.assertEqual(reward.__name__, "su_reward")

    def test_plain_string_completions(self):
        reward = SUReward(CostLedger(group_size=self.g))
        texts = [c[0]["content"] for c in self.completions]
        out = reward(prompts=None, completions=texts, task_json=self.task_json, completion_ids=[[1]] * 12)
        self.assertEqual(out, self.expected)

    def test_token_counts_required(self):
        reward = SUReward(CostLedger(group_size=self.g))
        with self.assertRaises(ValueError):
            reward(prompts=None, completions=self.completions, task_json=self.task_json)


class GRPOKwargs(unittest.TestCase):
    """Config adapter for TRL versions with and without max_prompt_length (no TRL needed)."""

    @dataclasses.dataclass
    class NewTRL:  # stands in for a GRPOConfig that dropped max_prompt_length
        learning_rate: float = 0.0
        max_completion_length: int = 0

    @dataclasses.dataclass
    class OldTRL(NewTRL):
        max_prompt_length: int = 0

    class Tok:  # one token per character
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(text)}

    data = {"prompt": [[{"role": "user", "content": "x" * 50}], [{"role": "user", "content": "y" * 80}]]}

    def test_dropped_only_when_trl_lacks_it(self):
        grpo = {"learning_rate": 1e-6, "max_prompt_length": 100}
        self.assertEqual(grpo_kwargs(grpo, self.NewTRL, self.data, self.Tok()), {"learning_rate": 1e-6})
        self.assertEqual(grpo_kwargs(grpo, self.OldTRL, self.data, self.Tok()), grpo)

    def test_prompt_over_limit_raises(self):
        with self.assertRaises(ValueError):
            grpo_kwargs({"max_prompt_length": 60}, self.NewTRL, self.data, self.Tok())

    def test_unknown_keys_listed_together(self):
        with self.assertRaisesRegex(TypeError, r"\['bogus_a', 'bogus_b'\]"):
            grpo_kwargs({"bogus_b": 1, "bogus_a": 2}, self.NewTRL, self.data, self.Tok())


class Target(unittest.TestCase):
    def pts(self, scores, benign=None):
        benign = benign or [None] * len(scores)
        return [EvalPoint(step=10 * i, trajectories=1000 * i, tokens=50000 * i, score=s, benign=b)
                for i, (s, b) in enumerate(zip(scores, benign))]

    def test_reached_at_first_of_two_consecutive(self):
        out = time_to_target(self.pts([0.1, 0.3, 0.62, 0.65, 0.7]), tau=0.6)
        self.assertTrue(out.reached)
        self.assertEqual((out.step, out.trajectories), (20, 2000))

    def test_single_spike_does_not_count(self):
        out = time_to_target(self.pts([0.1, 0.7, 0.4, 0.61, 0.63]), tau=0.6)
        self.assertEqual(out.step, 30)

    def test_censored_when_never_reached(self):
        out = time_to_target(self.pts([0.1, 0.2, 0.3]), tau=0.6)
        self.assertFalse(out.reached)
        self.assertEqual(out.trajectories, 2000)

    def test_benign_guardrail_blocks_crossing(self):
        out = time_to_target(self.pts([0.1, 0.7, 0.7, 0.7], benign=[0.5, 0.45, 0.49, 0.49]),
                             tau=0.6, benign_start=0.5)
        self.assertEqual(out.step, 20)

    def test_auc_is_threshold_free_mean(self):
        out = time_to_target(self.pts([0.0, 1.0]), tau=2.0)
        self.assertAlmostEqual(out.auc, 0.5)

    def test_check_tau_flags_checkpoints_too_close(self):
        self.assertEqual(check_tau(0.6, [0.2, 0.5, 0.44, 0.46]), [1, 3])


if __name__ == "__main__":
    unittest.main()
