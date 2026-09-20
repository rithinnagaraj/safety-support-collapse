"""Evaluation routines with fresh sampling streams and complete trace saving."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Iterable

SECTION_DIR = Path(__file__).resolve().parent
ROOT = SECTION_DIR.parent
for path in (ROOT, SECTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benign_tasks import BenignInstance  # noqa: E402
from protocol_core import read_jsonl, write_json, write_jsonl  # noqa: E402
from resource_tracking import ResourceMeter  # noqa: E402
from safety_harness import EpisodeRunner  # noqa: E402
from safety_tasks import SafetyInstance  # noqa: E402
from pilot_analysis import summarize_benign, summarize_safety  # noqa: E402


def load_safety(path: Path) -> list[SafetyInstance]:
    return [SafetyInstance.from_dict(row) for row in read_jsonl(path)]


def load_benign(path: Path) -> list[BenignInstance]:
    return [BenignInstance.from_dict(row) for row in read_jsonl(path)]


def evaluate_safety(
    policy: Any,
    instances: Iterable[SafetyInstance],
    *,
    rollouts: int,
    stream_seed: int,
    checkpoint: str,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_sampling_state = policy.sampling_state()
    policy.reset_sampling(stream_seed)
    meter = ResourceMeter(policy)
    meter.start()
    runner = EpisodeRunner()
    rows: list[dict[str, Any]] = []
    with policy.merged_evaluation():
        for instance in instances:
            for rollout in range(rollouts):
                episode = runner.run(instance, policy)
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "instance_id": instance.instance_id,
                        "latent_id": instance.latent_id,
                        "template_id": instance.template_id,
                        "family": instance.family,
                        "horizon": instance.horizon,
                        "rollout": rollout,
                        "stream_seed": stream_seed,
                        "utility": episode.score.utility,
                        "safety": episode.score.safety,
                        "joint": episode.score.joint,
                        "termination": episode.termination,
                        "generated_tokens": episode.generated_tokens,
                        "input_tokens": sum(
                            len(payload.context_ids) for payload in episode.training_payloads if payload is not None
                        ),
                        "actions": [action.to_dict() for action in episode.actions],
                        "deliveries": list(episode.deliveries),
                        "assistant_turns": list(episode.raw_assistant_turns),
                    }
                )
    write_jsonl(output_path, rows)
    write_json(
        output_path.with_suffix(".resources.json"),
        meter.stop(
            input_tokens=sum(int(row["input_tokens"]) for row in rows),
            generated_tokens=sum(int(row["generated_tokens"]) for row in rows),
            category="evaluation",
        ),
    )
    policy.restore_sampling_state(previous_sampling_state)
    return rows, summarize_safety(rows, rollouts)


def evaluate_benign(
    policy: Any,
    instances: Iterable[BenignInstance],
    *,
    checkpoint: str,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    previous_sampling_state = policy.sampling_state()
    rows: list[dict[str, Any]] = []
    meter = ResourceMeter(policy)
    meter.start()
    with policy.merged_evaluation():
        for instance in instances:
            turn = policy.complete(instance.prompt, greedy=True)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "instance_id": instance.instance_id,
                    "template_id": instance.template_id,
                    "domain": instance.domain,
                    "response": turn.text,
                    "answer": instance.answer,
                    "correct": instance.score(turn.text),
                    "generated_tokens": turn.generated_tokens,
                    "input_tokens": len(turn.training_payload.context_ids),
                }
            )
    write_jsonl(output_path, rows)
    write_json(
        output_path.with_suffix(".resources.json"),
        meter.stop(
            input_tokens=sum(int(row["input_tokens"]) for row in rows),
            generated_tokens=sum(int(row["generated_tokens"]) for row in rows),
            category="evaluation",
        ),
    )
    policy.restore_sampling_state(previous_sampling_state)
    return rows, summarize_benign(rows)
