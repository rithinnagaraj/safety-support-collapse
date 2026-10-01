"""Run Section 1A: harness acceptance and disposable supervised learning check."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import unittest

EXPERIMENT_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXPERIMENT_DIR.parent
ROOT = SECTION_DIR.parent
for path in (ROOT, SECTION_DIR, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hf_runtime import HFPolicy  # noqa: E402
from protocol_core import content_hash, read_jsonl, write_json  # noqa: E402
from safety_harness import run_oracle  # noqa: E402
from safety_tasks import SafetyInstance  # noqa: E402
from manual_review import create_review_pack, validate_review  # noqa: E402
from pilot_analysis import (  # noqa: E402
    experiment_a_decision,
    paired_joint_interval,
    paired_malformed_interval,
    repeated_c32_interval,
)
from pilot_evaluation import evaluate_safety, load_safety  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    mark_run_complete,
    record_run_metadata,
    validate_manifest,
)
from pilot_training import run_calibration_smoke  # noqa: E402
from learning_check import train_with_development_selection  # noqa: E402


def _unit_tests_pass() -> bool:
    suite = unittest.defaultTestLoader.discover(str(EXPERIMENT_DIR), pattern="test_*.py")
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def _all_safety_instances(manifest_path: Path, manifest: dict) -> list[SafetyInstance]:
    result: list[SafetyInstance] = []
    for metadata in manifest["splits"].values():
        if metadata["kind"] == "safety":
            result.extend(load_safety(manifest_path.parent / metadata["file"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-a")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000, choices=[10_000])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    manifest = validate_manifest(manifest_path, require_ml=True, execution_device=args.device)
    record_run_metadata(args.output_dir, manifest, "experiment-a")
    config = manifest["frozen_config"]
    seeds = config["seeds"]
    gate_settings = config["gates"]["experiment_a"]

    all_instances = _all_safety_instances(manifest_path, manifest)
    oracle_failures = []
    for instance in all_instances:
        episode = run_oracle(instance)
        if episode.score.joint != 1:
            oracle_failures.append({"instance_id": instance.instance_id, "episode": episode.to_dict()})
    write_json(args.output_dir / "oracle-check.json", {"passed": not oracle_failures, "failures": oracle_failures})
    unit_pass = _unit_tests_pass()

    calibration_train = load_safety(manifest_path.parent / manifest["splits"]["calibration_train"]["file"])
    calibration_development = load_safety(
        manifest_path.parent / manifest["splits"]["calibration_development"]["file"]
    )
    calibration_validation = load_safety(manifest_path.parent / manifest["splits"]["calibration_validation"]["file"])
    review_path = args.review_file or (args.output_dir / "manual-review.jsonl")
    review_binding = content_hash(manifest)
    if not review_path.exists():
        create_review_pack(calibration_validation, review_path, review_binding)
        raise SystemExit(
            f"Created {review_path}. Inspect all 50 rows, set reviewed=true and observed labels, then rerun with --review-file."
        )
    manual_pass = validate_review(review_path, review_binding)
    if not manual_pass:
        raise SystemExit("Manual review is incomplete or disagrees with expected labels; Section 1A is blocked.")

    model = config["model"]
    smoke_policy = HFPolicy(
        model_id=model["id"],
        revision=model["revision"],
        adapter=config["adapter"],
        device=args.device,
        seed=int(seeds["experiment_a_smoke"]),
    )
    smoke_report = run_calibration_smoke(
        smoke_policy,
        calibration_train,
        adapter_settings=config["adapter"],
        optimizer_settings=config["optimizer"],
        smoke_settings=config["calibration_smoke"],
        seed=int(seeds["experiment_a_smoke"]),
        output_dir=args.output_dir / "calibration-smoke",
    )
    del smoke_policy
    if not smoke_report["pass"]:
        raise SystemExit("Calibration smoke failed; training is blocked until adapter/batch checks pass.")

    initial = HFPolicy(
        model_id=model["id"], revision=model["revision"], adapter=None, device=args.device,
        seed=int(seeds["experiment_a_initial_evaluation"][0]),
    )
    first_rows, first_metrics = evaluate_safety(
        initial,
        calibration_validation,
        rollouts=32,
        stream_seed=int(seeds["experiment_a_initial_evaluation"][0]),
        checkpoint="initial",
        output_path=args.output_dir / "initial-evaluation-1.jsonl",
    )
    second_rows, second_metrics = evaluate_safety(
        initial,
        calibration_validation,
        rollouts=32,
        stream_seed=int(seeds["experiment_a_initial_evaluation"][1]),
        checkpoint="initial",
        output_path=args.output_dir / "initial-evaluation-2.jsonl",
    )
    repeated_interval = repeated_c32_interval(first_rows, second_rows, args.bootstrap_replicates)
    torch = initial.torch
    del initial
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    disposable = HFPolicy(
        model_id=model["id"],
        revision=model["revision"],
        adapter=config["adapter"],
        device=args.device,
        seed=int(seeds["experiment_a_sft"]),
    )
    learning_report = train_with_development_selection(
        disposable,
        calibration_train,
        calibration_development,
        optimizer_settings=config["optimizer"],
        epochs=int(config["training"]["sft_epochs"]),
        batch_size=int(config["training"]["sft_batch_size"]),
        training_seed=int(seeds["experiment_a_sft"]),
        development_rollouts=int(config["training"]["calibration_development_rollouts"]),
        development_seed=int(seeds["experiment_a_development_evaluation"]),
        maximum_malformed_rate_increase=float(gate_settings["maximum_malformed_rate_increase"]),
        minimum_joint_improvement=float(gate_settings["development_min_joint_improvement"]),
        output_dir=args.output_dir,
    )
    selected_epoch = learning_report["selection"]["selected_epoch"]
    trained_metrics = None
    sft_interval = None
    malformed_interval = None
    if selected_epoch is not None:
        torch = disposable.torch
        del disposable
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        selected = HFPolicy(
            model_id=model["id"],
            revision=model["revision"],
            adapter=None,
            adapter_checkpoint=args.output_dir / "checkpoints" / f"epoch-{selected_epoch}",
            device=args.device,
            seed=int(seeds["experiment_a_sft"]),
        )
        trained_rows, trained_metrics = evaluate_safety(
            selected,
            calibration_validation,
            rollouts=32,
            stream_seed=int(seeds["experiment_a_final_evaluation"]),
            checkpoint=f"calibration-sft-selected-epoch-{selected_epoch}",
            output_path=args.output_dir / "selected-sft-validation.jsonl",
        )
        sft_interval = paired_joint_interval(first_rows, trained_rows, args.bootstrap_replicates)
        malformed_interval = paired_malformed_interval(first_rows, trained_rows, args.bootstrap_replicates)
        del selected
    decision = experiment_a_decision(
        oracle_pass=not oracle_failures,
        unit_pass=unit_pass,
        manual_review_pass=manual_pass,
        calibration_smoke_pass=bool(smoke_report["pass"]),
        first_metrics=first_metrics,
        second_metrics=second_metrics,
        repeated_interval=repeated_interval,
        sft_interval=sft_interval,
        malformed_interval=malformed_interval,
        development_checkpoint_selected=selected_epoch is not None,
        gate_settings=gate_settings,
    )
    report = {
        "decision": decision,
        "initial_evaluation_1": first_metrics,
        "initial_evaluation_2": second_metrics,
        "repeatability_c32_difference": repeated_interval,
        "calibration_smoke": smoke_report,
        "development_selection": learning_report,
        "selected_epoch": selected_epoch,
        "selected_sft_validation": trained_metrics,
        "sft_joint_improvement": sft_interval,
        "sft_malformed_rate_change": malformed_interval,
        "disposable_model_discarded": True,
    }
    write_json(args.output_dir / "decision.json", report)
    mark_run_complete(args.output_dir, manifest)
    print("PASS" if decision["pass"] else "FAIL")


if __name__ == "__main__":
    main()
