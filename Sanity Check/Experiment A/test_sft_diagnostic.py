"""Small CPU-only checks for the exploratory SFT diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run_sft_diagnostic import audit_demonstration_tokens, preserve_training_rng  # noqa: E402
from hf_runtime import TokenSegment  # noqa: E402


@dataclass
class _Instance:
    instance_id: str
    oracle_calls: tuple[dict[str, object], ...]


class _Tokenizer:
    eos_token_id = 99

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        if not ids:
            return ""
        return {1: '{"tool":"finish","arguments":{"answer":"done"}}',
                2: '{"tool":"send","arguments":{"recipient":"wrong"}}'}[ids[0]]


class _FakeCuda:
    def __init__(self) -> None:
        self.state = [7]

    def is_available(self) -> bool:
        return True

    def get_rng_state_all(self) -> list[int]:
        return list(self.state)

    def set_rng_state_all(self, state: list[int]) -> None:
        self.state = list(state)


class _FakeTorch:
    def __init__(self) -> None:
        self.state = 3
        self.cuda = _FakeCuda()

    def get_rng_state(self) -> int:
        return self.state

    def set_rng_state(self, state: int) -> None:
        self.state = state


class SFTDiagnosticTests(unittest.TestCase):
    def test_valid_oracle_target_and_eos_are_recorded(self) -> None:
        instance = _Instance("example", ({"tool": "finish", "arguments": {"answer": "done"}},))
        demo = [[TokenSegment((10,), (1, 99), ())]]
        report = audit_demonstration_tokens(_Tokenizer(), [instance], demo)
        self.assertTrue(report["pass"])
        self.assertEqual(report["assistant_turns"], 1)
        self.assertEqual(report["targets_containing_one_eos"], 1)

    def test_wrong_decoded_target_is_flagged(self) -> None:
        instance = _Instance("example", ({"tool": "finish", "arguments": {"answer": "done"}},))
        demo = [[TokenSegment((10,), (2, 99), ())]]
        report = audit_demonstration_tokens(_Tokenizer(), [instance], demo)
        self.assertFalse(report["pass"])
        self.assertEqual(report["issues"][0]["instance_id"], "example")

    def test_rng_restored_even_when_evaluation_raises(self) -> None:
        torch = _FakeTorch()
        with self.assertRaisesRegex(RuntimeError, "evaluation"):
            with preserve_training_rng(torch):
                torch.state = 50
                torch.cuda.state = [60]
                raise RuntimeError("evaluation")
        self.assertEqual(torch.state, 3)
        self.assertEqual(torch.cuda.state, [7])


if __name__ == "__main__":
    unittest.main()
