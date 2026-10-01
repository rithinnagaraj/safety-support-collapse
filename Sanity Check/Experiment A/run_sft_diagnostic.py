"""Exploratory calibration-only SFT token audit and epoch-by-epoch evaluation.

This does not rerun or replace Experiment A, and its output is not a GO gate.
Run only after a completed Experiment A failed its learning check.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXPERIMENT_DIR.parent
ROOT = SECTION_DIR.parent
for directory in (ROOT, SECTION_DIR, EXPERIMENT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hf_runtime import HFPolicy  # noqa: E402
from policy_training import supervised_train  # noqa: E402
from protocol_core import content_hash, read_json, read_jsonl, write_json  # noqa: E402
from resource_tracking import ResourceMeter  # noqa: E402
from safety_harness import PARSER_SCHEMA_VERSION, SYSTEM_PROMPT  # noqa: E402
from safety_tasks import TOOL_SCHEMA  # noqa: E402
from diagnose_sft import analyze  # noqa: E402
from pilot_evaluation import evaluate_safety, load_safety  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    component_hashes,
    dependency_versions,
    device_inventory,
    git_revision,
    require_matching_run,
    tokenizer_fingerprint,
)
from pilot_training import build_oracle_demonstrations  # noqa: E402
from learning_check import audit_demonstration_tokens, preserve_training_rng  # noqa: E402


def verify_diagnostic_inputs(manifest_path: Path, experiment_a_dir: Path, device: str) -> dict[str, Any]:
    """Check frozen inputs while intentionally allowing new diagnostic code."""
    manifest = read_json(manifest_path)
    original = read_json(experiment_a_dir / "decision.json")
    if original.get("decision", {}).get("checks", {}).get("learning_check") is not False:
        raise ValueError("A completed Experiment A with a failed learning check is required")
    require_matching_run(experiment_a_dir, manifest, "Original Experiment A")
    if device != manifest["device"]["execution_device"]:
        raise ValueError("Diagnostic device differs from the frozen Experiment A device")
    if dependency_versions() != manifest["dependencies"]:
        raise ValueError("Dependency versions differ from the frozen Experiment A environment")
    if device_inventory() != manifest["device"]["detected"]:
        raise ValueError("Detected GPU configuration differs from the frozen Experiment A environment")
    if tokenizer_fingerprint(manifest["model"]["id"], manifest["model"]["revision"]) != manifest["tokenizer"]:
        raise ValueError("Tokenizer or chat template differs from frozen Experiment A")
    schema_hash = content_hash({
        "system_prompt": SYSTEM_PROMPT,
        "tool_schema": TOOL_SCHEMA,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
    })
    if schema_hash != manifest["parser_schema_hash"]:
        raise ValueError("Tool/parser schema changed since Experiment A")
    for name, digest in component_hashes().items():
        if manifest[name] != digest:
            raise ValueError(f"Frozen task component changed since Experiment A: {name}")
    for split_name in ("calibration_train", "calibration_validation"):
        entry = manifest["splits"][split_name]
        rows = list(read_jsonl(manifest_path.parent / entry["file"]))
        if len(rows) != entry["count"] or content_hash(rows) != entry["sha256"]:
            raise ValueError(f"Frozen calibration split changed: {split_name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument("--experiment-a-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-a")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-a-diagnostic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollouts", type=int, default=8, choices=[8], help="Fixed exploratory draws per held-out prompt")
    parser.add_argument("--stream-seed", type=int, default=13_001, choices=[13_001])
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    experiment_a_dir = args.experiment_a_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Diagnostic output already exists; choose a fresh --output-dir: {output_dir}")
    manifest = verify_diagnostic_inputs(manifest_path, experiment_a_dir, args.device)
    if output_dir == experiment_a_dir or experiment_a_dir in output_dir.parents:
        raise SystemExit("Diagnostic output must not be inside the original Experiment A directory")
    config = manifest["frozen_config"]
    training = load_safety(manifest_path.parent / manifest["splits"]["calibration_train"]["file"])
    validation = load_safety(manifest_path.parent / manifest["splits"]["calibration_validation"]["file"])
    selected = training[:64]
    if len(selected) != 64 or len(validation) != 128:
        raise ValueError("Diagnostic requires the original 64 training and 128 held-out calibration prompts")

    output_dir.mkdir(parents=True)
    write_json(output_dir / "provenance.json", {
        "kind": "exploratory-calibration-diagnostic-v1",
        "not_a_go_gate": True,
        "original_manifest_hash": content_hash(manifest),
        "original_source_revision": manifest["code_revision"],
        "diagnostic_source_revision": git_revision(),
        "model": manifest["model"],
        "device": manifest["device"],
        "training_instance_ids": [row.instance_id for row in selected],
        "validation_split_hash": manifest["splits"]["calibration_validation"]["sha256"],
        "rollouts_per_prompt": args.rollouts,
        "same_stream_seed_at_each_epoch": args.stream_seed,
        "epochs": [0, 1, 2, 3, 4],
    })
    model = manifest["model"]
    policy = HFPolicy(
        model_id=model["id"], revision=model["revision"], adapter=config["adapter"],
        device=args.device, seed=12_001,
    )
    demo_meter = ResourceMeter(policy)
    demo_meter.start()
    demonstrations = build_oracle_demonstrations(policy, selected)
    token_audit = audit_demonstration_tokens(policy.tokenizer, selected, demonstrations)
    write_json(output_dir / "demonstration-token-audit.json", token_audit)
    write_json(output_dir / "demonstration-resource-usage.json", demo_meter.stop(
        input_tokens=sum(len(segment.context_ids) for demo in demonstrations for segment in demo),
        generated_tokens=sum(len(segment.output_ids) for demo in demonstrations for segment in demo),
        category="diagnostic-demonstration-acquisition",
    ))
    if not token_audit["pass"]:
        raise SystemExit("Oracle assistant targets failed the tokenizer audit; see demonstration-token-audit.json")
    print(
        f"Audited {token_audit['assistant_turns']} oracle assistant turns; "
        "all decoded targets match their intended actions.",
        flush=True,
    )

    epoch_rows: dict[int, list[dict[str, Any]]] = {}
    epoch_summaries: dict[str, Any] = {}
    training_meter = ResourceMeter(policy)

    def evaluate_epoch(epoch: int, history: list[dict[str, float]]) -> None:
        if epoch > 0:
            training_meter.pause()
        print(f"Evaluating held-out calibration prompts after epoch {epoch}/4...", flush=True)
        with preserve_training_rng(policy.torch):
            rows, metrics = evaluate_safety(
                policy, validation, rollouts=args.rollouts,
                stream_seed=args.stream_seed,
                checkpoint=f"diagnostic-epoch-{epoch}",
                output_path=output_dir / f"epoch-{epoch}" / "calibration-validation.jsonl",
            )
        epoch_rows[epoch] = rows
        epoch_losses = [float(item["loss"]) for item in history if int(item["epoch"]) == epoch]
        summary = {
            "epoch": epoch,
            "metrics": metrics,
            "mean_training_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else None,
        }
        if epoch > 0:
            comparison = analyze(epoch_rows[0], rows, history, max_examples=3)
            summary["malformed_parser_categories"] = comparison["after_sft_malformed_parser_categories"]
            summary["by_family"] = comparison["by"]["family"]
            summary["by_horizon"] = comparison["by"]["horizon"]
            summary["newly_malformed_examples"] = comparison["newly_malformed_examples"]
        epoch_summaries[str(epoch)] = summary
        write_json(output_dir / "epoch-summary.json", epoch_summaries)
        write_json(output_dir / "sft-history.json", history)
        print(
            f"Epoch {epoch}: joint={metrics['joint']:.4f}, "
            f"malformed={metrics['invalid_rate']:.4f}, "
            f"loss={summary['mean_training_loss']}",
            flush=True,
        )
        if epoch > 0:
            training_meter.resume()

    evaluate_epoch(0, [])
    training_meter.start()
    history = supervised_train(
        policy, demonstrations,
        optimizer_settings=config["optimizer"],
        epochs=4, batch_size=16, seed=12_001,
        on_epoch_end=evaluate_epoch,
    )
    write_json(output_dir / "sft-history.json", history)
    write_json(output_dir / "supervised-training-resource-usage.json", training_meter.stop(
        input_tokens=4 * sum(len(segment.context_ids) for demo in demonstrations for segment in demo),
        generated_tokens=4 * sum(len(segment.output_ids) for demo in demonstrations for segment in demo),
        category="diagnostic-supervised-training",
    ))
    print(f"Exploratory diagnosis complete: {output_dir / 'epoch-summary.json'}")


if __name__ == "__main__":
    main()
