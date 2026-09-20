"""Pilot benign, recovery, and oracle-demonstration training loops."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Callable, Sequence

SECTION_DIR = Path(__file__).resolve().parent
ROOT = SECTION_DIR.parent
for path in (ROOT, SECTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benign_tasks import BenignInstance  # noqa: E402
from hf_runtime import HFPolicy, TokenSegment  # noqa: E402
from policy_training import TrainTrajectory, group_relative_update, make_optimizer, supervised_train  # noqa: E402
from protocol_core import write_json, write_jsonl  # noqa: E402
from resource_tracking import ResourceMeter  # noqa: E402
from safety_harness import EpisodeRunner, GeneratedTurn, SYSTEM_PROMPT  # noqa: E402
from safety_tasks import SafetyInstance  # noqa: E402


def _cycling_order(length: int, seed: int) -> list[int]:
    order = list(range(length))
    random.Random(seed).shuffle(order)
    return order


def train_benign_stage(
    policy: HFPolicy,
    instances: Sequence[BenignInstance],
    *,
    updates: int,
    prompts_per_update: int,
    rollouts_per_prompt: int,
    seed: int,
    optimizer_settings: dict[str, Any],
    kl_coefficient: float,
    output_dir: Path,
    optimizer: Any | None = None,
) -> list[dict[str, Any]]:
    if len(instances) != updates * prompts_per_update:
        raise ValueError("Pilot benign stages must visit each frozen prompt exactly once")
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = optimizer or make_optimizer(policy, optimizer_settings)
    meter = ResourceMeter(policy)
    meter.start()
    order = _cycling_order(len(instances), seed)
    history: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for update in range(updates):
        batch = [instances[index] for index in order[update * prompts_per_update : (update + 1) * prompts_per_update]]
        groups: list[list[TrainTrajectory]] = []
        for instance in batch:
            group: list[TrainTrajectory] = []
            for rollout in range(rollouts_per_prompt):
                turn = policy.complete(instance.prompt, greedy=False)
                reward = float(instance.score(turn.text))
                record = {
                    "update": update + 1,
                    "instance_id": instance.instance_id,
                    "domain": instance.domain,
                    "rollout": rollout,
                    "response": turn.text,
                    "reward": reward,
                    "generated_tokens": turn.generated_tokens,
                    "input_tokens": len(turn.training_payload.context_ids),
                }
                group.append(TrainTrajectory((turn.training_payload,), reward, record))
                trajectory_rows.append(record)
            groups.append(group)
        summary = group_relative_update(
            policy,
            optimizer,
            groups,
            ratio_clip=float(optimizer_settings["ratio_clip"]),
            kl_coefficient=kl_coefficient,
            gradient_norm_clip=float(optimizer_settings["gradient_norm_clip"]),
        )
        history.append({"update": update + 1, **summary.to_dict()})
        write_json(output_dir / "training-state.json", {"completed_updates": update + 1, "history": history})
    write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
    write_json(
        output_dir / "resource-usage.json",
        meter.stop(
            input_tokens=sum(int(row["input_tokens"]) for row in trajectory_rows),
            generated_tokens=sum(int(row["generated_tokens"]) for row in trajectory_rows),
            category="training",
        ),
    )
    return history


class _OracleCapturePolicy:
    def __init__(self, policy: HFPolicy, calls: Sequence[dict[str, Any]]) -> None:
        self.policy = policy
        self.calls = iter(calls)
        self.segments: list[TokenSegment] = []

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> GeneratedTurn:
        call = next(self.calls)
        text = json.dumps(call, separators=(",", ":"))
        segment = self.policy.make_segment(messages, text)
        self.segments.append(segment)
        return GeneratedTurn(text, min(max_new_tokens, len(segment.output_ids)), segment)


def build_oracle_demonstrations(policy: HFPolicy, instances: Sequence[SafetyInstance]) -> list[list[TokenSegment]]:
    demonstrations: list[list[TokenSegment]] = []
    for instance in instances:
        capture = _OracleCapturePolicy(policy, instance.oracle_calls)
        episode = EpisodeRunner().run(instance, capture)
        if episode.score.joint != 1:
            raise RuntimeError(f"Oracle demonstration failed for {instance.instance_id}")
        demonstrations.append(capture.segments)
    return demonstrations


def train_oracle_sft(
    policy: HFPolicy,
    instances: Sequence[SafetyInstance],
    *,
    optimizer_settings: dict[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
) -> list[dict[str, float]]:
    demo_meter = ResourceMeter(policy)
    demo_meter.start()
    demonstrations = build_oracle_demonstrations(policy, instances)
    demo_usage = demo_meter.stop(
        input_tokens=sum(len(segment.context_ids) for demo in demonstrations for segment in demo),
        generated_tokens=sum(len(segment.output_ids) for demo in demonstrations for segment in demo),
        category="demonstration-acquisition",
    )
    training_meter = ResourceMeter(policy)
    training_meter.start()
    history = supervised_train(
        policy,
        demonstrations,
        optimizer_settings=optimizer_settings,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    write_json(output_dir / "sft-history.json", history)
    write_json(output_dir / "demonstration-resource-usage.json", demo_usage)
    write_json(
        output_dir / "supervised-training-resource-usage.json",
        training_meter.stop(
            input_tokens=epochs * sum(len(segment.context_ids) for demo in demonstrations for segment in demo),
            generated_tokens=epochs * sum(len(segment.output_ids) for demo in demonstrations for segment in demo),
            category="supervised-training",
        ),
    )
    return history


def _adapter_merge_check(
    policy: HFPolicy,
    prompt: str,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    torch = policy.torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    context_ids = policy._context_ids(messages)
    device = next(policy.model.parameters()).device
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    policy.model.eval()
    with torch.no_grad():
        unmerged = policy.model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    with policy.merged_evaluation():
        with torch.no_grad():
            merged = policy.model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    with torch.no_grad():
        restored = policy.model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    merged_close = bool(
        torch.allclose(unmerged, merged, atol=absolute_tolerance, rtol=relative_tolerance)
    )
    # The active model must be completely untouched by disposable merged
    # evaluation.  This is deliberately stricter than the BF16 merged-output
    # comparison, where quantization introduces bounded rounding differences.
    restored_close = bool(torch.equal(unmerged, restored))
    argmax_equal = (
        int(unmerged.argmax().cpu())
        == int(merged.argmax().cpu())
        == int(restored.argmax().cpu())
    )
    return {
        "pass": merged_close and restored_close and argmax_equal,
        "merged_close": merged_close,
        "restored_close": restored_close,
        "restored_exact": restored_close,
        "argmax_equal": argmax_equal,
        "unmerged_merged_max_abs_difference": float((unmerged - merged).abs().max().cpu()),
        "unmerged_restored_max_abs_difference": float((unmerged - restored).abs().max().cpu()),
        "unmerged_argmax": int(unmerged.argmax().cpu()),
        "merged_argmax": int(merged.argmax().cpu()),
        "restored_argmax": int(restored.argmax().cpu()),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def run_calibration_smoke(
    policy: HFPolicy,
    instances: Sequence[SafetyInstance],
    *,
    adapter_settings: dict[str, Any],
    optimizer_settings: dict[str, Any],
    smoke_settings: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Disposable smoke run for adapter targets, merging, and effective batch size."""
    prompt_count = int(smoke_settings["prompts_per_update"])
    rollout_count = int(smoke_settings["rollouts_per_prompt"])
    selected = list(instances[:prompt_count])
    if len(selected) != prompt_count:
        raise ValueError(f"Calibration smoke requires {prompt_count} prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    meter = ResourceMeter(policy)
    meter.start()

    demonstrations = build_oracle_demonstrations(policy, selected)
    supervised_train(
        policy,
        demonstrations,
        optimizer_settings=optimizer_settings,
        epochs=int(smoke_settings["sft_epochs"]),
        batch_size=int(smoke_settings["sft_batch_size"]),
        seed=seed,
    )
    policy.reset_sampling(seed + 1)
    optimizer = make_optimizer(policy, optimizer_settings)
    groups: list[list[TrainTrajectory]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for instance in selected:
        group: list[TrainTrajectory] = []
        for rollout in range(rollout_count):
            episode = EpisodeRunner().run(instance, policy)
            segments = tuple(payload for payload in episode.training_payloads if payload is not None)
            record = {
                "instance_id": instance.instance_id,
                "rollout": rollout,
                "utility": episode.score.utility,
                "safety": episode.score.safety,
                "reward": episode.score.joint,
                "termination": episode.termination,
                "input_tokens": sum(len(segment.context_ids) for segment in segments),
                "generated_tokens": episode.generated_tokens,
            }
            group.append(TrainTrajectory(segments, float(episode.score.joint), record))
            trajectory_rows.append(record)
        groups.append(group)
    update_summary = group_relative_update(
        policy,
        optimizer,
        groups,
        ratio_clip=float(optimizer_settings["ratio_clip"]),
        kl_coefficient=0.01,
        gradient_norm_clip=float(optimizer_settings["gradient_norm_clip"]),
    )
    trainable_names = [name for name, parameter in policy.model.named_parameters() if parameter.requires_grad]
    target_checks = {
        target: any(f".{target}." in name and "lora_" in name for name in trainable_names)
        for target in adapter_settings["target_modules"]
    }
    merge = _adapter_merge_check(
        policy,
        selected[0].prompt,
        absolute_tolerance=float(smoke_settings["merge_absolute_tolerance"]),
        relative_tolerance=float(smoke_settings["merge_relative_tolerance"]),
    )
    checks = {
        "one_update": int(smoke_settings["updates"]) == 1,
        "prompt_batch_is_16": len(groups) == 16,
        "eight_rollouts_per_prompt": all(len(group) == 8 for group in groups),
        "effective_trajectory_batch_is_128": update_summary.trajectories == 128,
        "only_lora_parameters_trainable": bool(trainable_names)
        and all("lora_" in name for name in trainable_names),
        "all_target_modules_present": all(target_checks.values()),
        "adapter_merge_equivalent": bool(merge["pass"]),
    }
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "target_modules": target_checks,
        "trainable_parameter_tensors": len(trainable_names),
        "update": update_summary.to_dict(),
        "merge": merge,
    }
    write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
    write_json(output_dir / "decision.json", report)
    write_json(
        output_dir / "resource-usage.json",
        meter.stop(
            input_tokens=(
                (1 + int(smoke_settings["sft_epochs"]))
                * sum(len(segment.context_ids) for demo in demonstrations for segment in demo)
                + sum(int(row["input_tokens"]) for row in trajectory_rows)
            ),
            generated_tokens=(
                (1 + int(smoke_settings["sft_epochs"]))
                * sum(len(segment.output_ids) for demo in demonstrations for segment in demo)
                + sum(int(row["generated_tokens"]) for row in trajectory_rows)
            ),
            category="calibration-smoke",
        ),
    )
    return report


def train_recovery(
    policy: HFPolicy,
    instances: Sequence[SafetyInstance],
    *,
    updates: int,
    prompts_per_update: int,
    rollouts_per_prompt: int,
    seed: int,
    optimizer_settings: dict[str, Any],
    output_dir: Path,
    validation_callback: Callable[[int, HFPolicy], dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(instances) != 512:
        raise ValueError("Pilot recovery requires the frozen 512-prompt training split")
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(policy, optimizer_settings)
    meter = ResourceMeter(policy)
    meter.start()
    order = _cycling_order(len(instances), seed)
    history: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    validation_points = {0, 2048, 4096, 6144, 8192}
    meter.pause()
    validations = [validation_callback(0, policy)]
    meter.resume()
    for update in range(updates):
        start = (update * prompts_per_update) % len(order)
        indices = [order[(start + offset) % len(order)] for offset in range(prompts_per_update)]
        groups: list[list[TrainTrajectory]] = []
        for index in indices:
            instance = instances[index]
            group: list[TrainTrajectory] = []
            for rollout in range(rollouts_per_prompt):
                episode = EpisodeRunner().run(instance, policy)
                segments = tuple(payload for payload in episode.training_payloads if payload is not None)
                record = {
                    "update": update + 1,
                    "instance_id": instance.instance_id,
                    "rollout": rollout,
                    "reward": episode.score.joint,
                    "utility": episode.score.utility,
                    "safety": episode.score.safety,
                    "termination": episode.termination,
                    "generated_tokens": episode.generated_tokens,
                    "input_tokens": sum(len(segment.context_ids) for segment in segments),
                    "actions": [action.to_dict() for action in episode.actions],
                    "assistant_turns": list(episode.raw_assistant_turns),
                }
                group.append(TrainTrajectory(segments, float(episode.score.joint), record))
                trajectory_rows.append(record)
            groups.append(group)
        summary = group_relative_update(
            policy,
            optimizer,
            groups,
            ratio_clip=float(optimizer_settings["ratio_clip"]),
            kl_coefficient=0.0,
            gradient_norm_clip=float(optimizer_settings["gradient_norm_clip"]),
        )
        trajectories_seen = (update + 1) * prompts_per_update * rollouts_per_prompt
        history.append({"update": update + 1, "trajectories": trajectories_seen, **summary.to_dict()})
        if trajectories_seen in validation_points:
            meter.pause()
            policy.save_adapter(output_dir / "checkpoints" / f"trajectories-{trajectories_seen:05d}")
            validations.append(validation_callback(trajectories_seen, policy))
            meter.resume()
        write_json(
            output_dir / "training-state.json",
            {"completed_updates": update + 1, "history": history, "validations": validations},
        )
    write_jsonl(output_dir / "trajectories.jsonl", trajectory_rows)
    write_json(
        output_dir / "resource-usage.json",
        meter.stop(
            input_tokens=sum(int(row["input_tokens"]) for row in trajectory_rows),
            generated_tokens=sum(int(row["generated_tokens"]) for row in trajectory_rows),
            category="training-excluding-validation",
        ),
    )
    return validations
