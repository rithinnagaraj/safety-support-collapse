"""Experiment A's development-selected supervised-learning check.

Everything here is specific to the Experiment A gate. Shared model, trainer,
evaluation, and resource helpers remain in their section or repository modules.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hf_runtime import HFPolicy, TokenSegment
from policy_training import supervised_train
from protocol_core import write_json
from resource_tracking import ResourceMeter
from safety_harness import parse_call
from safety_tasks import SafetyInstance
from pilot_evaluation import evaluate_safety
from pilot_training import build_oracle_demonstrations


@contextmanager
def preserve_training_rng(torch: Any) -> Any:
    """Prevent development sampling from changing later training randomness."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def validate_training_diversity(
    instances: Sequence[SafetyInstance], *, expected_prompts: int, expected_latents: int
) -> dict[str, Any]:
    """Require paired variants and balanced family-by-horizon strata."""
    if len(instances) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} calibration training prompts; found {len(instances)}")
    by_latent: dict[str, list[SafetyInstance]] = {}
    for instance in instances:
        by_latent.setdefault(instance.latent_id, []).append(instance)
    if len(by_latent) != expected_latents:
        raise ValueError(f"Expected {expected_latents} unique calibration tasks; found {len(by_latent)}")
    bad_variants = {
        latent_id: sorted(row.surface_variant_id for row in rows)
        for latent_id, rows in by_latent.items()
        if {row.surface_variant_id for row in rows} != {0, 1} or len(rows) != 2
    }
    if bad_variants:
        raise ValueError(f"Calibration training variants are not paired: {list(bad_variants.items())[:3]}")
    strata = Counter((rows[0].family, rows[0].horizon) for rows in by_latent.values())
    expected_per_stratum = expected_latents // 4
    expected_keys = {
        ("file_access", "current"),
        ("file_access", "future"),
        ("recipient_authorization", "current"),
        ("recipient_authorization", "future"),
    }
    if expected_latents % 4 or set(strata) != expected_keys or any(
        count != expected_per_stratum for count in strata.values()
    ):
        raise ValueError(f"Calibration training latent strata are not balanced: {dict(strata)}")
    return {
        "prompts": len(instances),
        "unique_latent_tasks": len(by_latent),
        "surface_variants_per_task": 2,
        "latent_tasks_by_family_and_horizon": {
            f"{family}/{horizon}": strata[(family, horizon)] for family, horizon in sorted(strata)
        },
    }


def audit_demonstration_tokens(
    tokenizer: Any,
    instances: Sequence[SafetyInstance],
    demonstrations: Sequence[Sequence[TokenSegment]],
) -> dict[str, Any]:
    """Verify decoded targets, assistant boundaries, and end-of-message tokens."""
    if len(instances) != len(demonstrations):
        raise ValueError("Demonstration count differs from the training instance count")
    issues: list[dict[str, Any]] = []
    target_lengths: list[int] = []
    last_tokens: Counter[str] = Counter()
    eos_count = 0
    turn_count = 0
    for instance, segments in zip(instances, demonstrations):
        if len(segments) != len(instance.oracle_calls):
            issues.append({"instance_id": instance.instance_id, "problem": "oracle turn count mismatch"})
        for turn, (segment, call) in enumerate(zip(segments, instance.oracle_calls)):
            turn_count += 1
            target = list(segment.output_ids)
            target_lengths.append(len(target))
            if not segment.context_ids or not target:
                issues.append({"instance_id": instance.instance_id, "turn": turn, "problem": "empty context or target"})
                continue
            last_tokens[str(target[-1])] += 1
            decoded = tokenizer.decode(target, skip_special_tokens=True).strip()
            try:
                observed = parse_call(decoded)
                expected = parse_call(json.dumps(call, separators=(",", ":")))
                if observed != expected:
                    raise ValueError("decoded assistant target differs from oracle action")
            except (TypeError, ValueError) as exc:
                issues.append({
                    "instance_id": instance.instance_id,
                    "turn": turn,
                    "problem": str(exc),
                    "decoded_target": decoded,
                })
            eos_positions = [index for index, token in enumerate(target) if token == tokenizer.eos_token_id]
            if len(eos_positions) != 1:
                issues.append({
                    "instance_id": instance.instance_id,
                    "turn": turn,
                    "problem": f"expected one end-of-message token; found {len(eos_positions)}",
                })
            else:
                eos_count += 1
                suffix = tokenizer.decode(target[eos_positions[0] + 1 :], skip_special_tokens=True)
                if suffix.strip():
                    issues.append({
                        "instance_id": instance.instance_id,
                        "turn": turn,
                        "problem": "non-whitespace content follows the end-of-message token",
                        "decoded_suffix": suffix,
                    })
    return {
        "instances": len(instances),
        "assistant_turns": turn_count,
        "pass": not issues,
        "all_decoded_targets_match_oracle_and_end_cleanly": not issues,
        "issues": issues,
        "target_last_token_ids": dict(sorted(last_tokens.items())),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "targets_containing_one_eos": eos_count,
        "minimum_target_tokens": min(target_lengths) if target_lengths else 0,
        "maximum_target_tokens": max(target_lengths) if target_lengths else 0,
        "mean_target_tokens": sum(target_lengths) / len(target_lengths) if target_lengths else 0.0,
    }


def select_development_checkpoint(
    summaries: Mapping[int, Mapping[str, Any]],
    *,
    maximum_malformed_rate_increase: float,
    minimum_joint_improvement: float,
) -> dict[str, Any]:
    """Choose the best trained epoch without consulting final validation."""
    if 0 not in summaries:
        raise ValueError("Development summaries must include the untrained epoch-0 baseline")
    baseline = summaries[0]["metrics"]
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for epoch in sorted(value for value in summaries if value > 0):
        metrics = summaries[epoch]["metrics"]
        joint_improvement = float(metrics["joint"]) - float(baseline["joint"])
        malformed_increase = float(metrics["invalid_rate"]) - float(baseline["invalid_rate"])
        row = {
            "epoch": epoch,
            "joint": float(metrics["joint"]),
            "joint_improvement": joint_improvement,
            "invalid_rate": float(metrics["invalid_rate"]),
            "malformed_rate_increase": malformed_increase,
        }
        if (
            joint_improvement > minimum_joint_improvement
            and malformed_increase <= maximum_malformed_rate_increase
        ):
            candidates.append(row)
        else:
            rejected.append(row)
    # Highest joint wins; lower malformed rate and then earlier epoch are fixed tie-breakers.
    candidates.sort(key=lambda row: (-row["joint"], row["invalid_rate"], row["epoch"]))
    selected = candidates[0] if candidates else None
    return {
        "status": "selected" if selected else "no_eligible_checkpoint",
        "selected_epoch": selected["epoch"] if selected else None,
        "selected": selected,
        "eligible_candidates": candidates,
        "rejected_candidates": rejected,
        "criteria": {
            "joint_improvement_strictly_greater_than": minimum_joint_improvement,
            "maximum_malformed_rate_increase": maximum_malformed_rate_increase,
            "tie_break_order": ["highest_joint", "lowest_invalid_rate", "earliest_epoch"],
        },
    }


def train_with_development_selection(
    policy: HFPolicy,
    training: Sequence[SafetyInstance],
    development: Sequence[SafetyInstance],
    *,
    optimizer_settings: dict[str, Any],
    epochs: int,
    batch_size: int,
    training_seed: int,
    development_rollouts: int,
    development_seed: int,
    maximum_malformed_rate_increase: float,
    minimum_joint_improvement: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Train once, inspect development only, and retain all epoch adapters."""
    diversity = validate_training_diversity(training, expected_prompts=128, expected_latents=64)
    write_json(output_dir / "training-diversity.json", diversity)

    demo_meter = ResourceMeter(policy)
    demo_meter.start()
    demonstrations = build_oracle_demonstrations(policy, training)
    audit = audit_demonstration_tokens(policy.tokenizer, training, demonstrations)
    write_json(output_dir / "demonstration-token-audit.json", audit)
    write_json(
        output_dir / "demonstration-resource-usage.json",
        demo_meter.stop(
            input_tokens=sum(len(segment.context_ids) for demo in demonstrations for segment in demo),
            generated_tokens=sum(len(segment.output_ids) for demo in demonstrations for segment in demo),
            category="demonstration-acquisition",
        ),
    )
    if not audit["pass"]:
        raise RuntimeError("Oracle demonstration token audit failed; training is blocked")
    print(
        f"Audited {audit['assistant_turns']} supervised assistant targets across "
        f"{diversity['unique_latent_tasks']} unique tasks.",
        flush=True,
    )

    summaries: dict[int, dict[str, Any]] = {}
    training_meter = ResourceMeter(policy)

    def evaluate_epoch(epoch: int, history: list[dict[str, float]]) -> None:
        if epoch > 0:
            training_meter.pause()
            policy.save_adapter(output_dir / "checkpoints" / f"epoch-{epoch}")
        print(f"Evaluating calibration development at epoch {epoch}/{epochs}...", flush=True)
        with preserve_training_rng(policy.torch):
            _, metrics = evaluate_safety(
                policy,
                development,
                rollouts=development_rollouts,
                stream_seed=development_seed,
                checkpoint=f"calibration-development-epoch-{epoch}",
                output_path=output_dir / "development" / f"epoch-{epoch}.jsonl",
            )
        losses = [float(row["loss"]) for row in history if int(row["epoch"]) == epoch]
        summaries[epoch] = {
            "epoch": epoch,
            "metrics": metrics,
            "mean_training_loss": sum(losses) / len(losses) if losses else None,
        }
        write_json(output_dir / "development-summaries.json", summaries)
        write_json(output_dir / "sft-history.json", history)
        print(
            f"Development epoch {epoch}: joint={metrics['joint']:.4f}, "
            f"malformed={metrics['invalid_rate']:.4f}",
            flush=True,
        )
        if epoch > 0:
            training_meter.resume()

    evaluate_epoch(0, [])
    training_meter.start()
    history = supervised_train(
        policy,
        demonstrations,
        optimizer_settings=optimizer_settings,
        epochs=epochs,
        batch_size=batch_size,
        seed=training_seed,
        on_epoch_end=evaluate_epoch,
    )
    write_json(
        output_dir / "supervised-training-resource-usage.json",
        training_meter.stop(
            input_tokens=epochs * sum(len(segment.context_ids) for demo in demonstrations for segment in demo),
            generated_tokens=epochs * sum(len(segment.output_ids) for demo in demonstrations for segment in demo),
            category="supervised-training-excluding-development-evaluation",
        ),
    )
    selection = select_development_checkpoint(
        summaries,
        maximum_malformed_rate_increase=maximum_malformed_rate_increase,
        minimum_joint_improvement=minimum_joint_improvement,
    )
    report = {
        "training_diversity": diversity,
        "demonstration_token_audit_pass": audit["pass"],
        "development_summaries": summaries,
        "selection": selection,
        "history_entries": len(history),
    }
    write_json(output_dir / "development-selection.json", report)
    return report
