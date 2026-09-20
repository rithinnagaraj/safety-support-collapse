"""Run Section 1B for one benign seed and select the matched L/H pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXPERIMENT_DIR.parent
ROOT = SECTION_DIR.parent
for path in (ROOT, SECTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hf_runtime import HFPolicy  # noqa: E402
from policy_training import make_optimizer  # noqa: E402
from protocol_core import read_json, write_json  # noqa: E402
from pilot_analysis import CheckpointMetrics, select_matched_pair  # noqa: E402
from pilot_evaluation import evaluate_benign, evaluate_safety, load_benign, load_safety  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    mark_run_complete,
    record_run_metadata,
    require_matching_run,
    validate_manifest,
)
from pilot_training import train_benign_stage  # noqa: E402


def _require_a_pass(path: Path, manifest: dict) -> None:
    if not path.is_file() or not read_json(path).get("decision", {}).get("pass"):
        raise SystemExit("Experiment A has not passed; Experiment B must not run.")
    require_matching_run(path.parent, manifest, "Experiment A")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument("--experiment-a-decision", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-a" / "decision.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-b")
    parser.add_argument("--seed", type=int, default=101, choices=[101, 102, 103])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = validate_manifest(manifest_path, require_ml=True, execution_device=args.device)
    _require_a_pass(args.experiment_a_decision, manifest)
    config = manifest["frozen_config"]
    run_dir = args.output_dir / f"seed-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    record_run_metadata(run_dir, manifest, f"experiment-b seed={args.seed}")

    def safety_split(name: str):
        return load_safety(manifest_path.parent / manifest["splits"][name]["file"])

    def benign_split(name: str):
        return load_benign(manifest_path.parent / manifest["splits"][name]["file"])

    math_train = benign_split(f"mathematics_train_seed_{args.seed}")
    logic_train = benign_split(f"logic_train_seed_{args.seed}")
    benign_test = benign_split("mathematics_test") + benign_split("logic_test")
    future_selection = safety_split("future_selection")
    current_selection = safety_split("current_selection")
    model = config["model"]
    policy = HFPolicy(
        model_id=model["id"], revision=model["revision"], adapter=config["adapter"], device=args.device, seed=args.seed
    )
    optimizer = make_optimizer(policy, config["optimizer"])
    checkpoint_names = ["initial", "after-mathematics", "after-reasoning"]
    results: dict[str, dict] = {}

    def save_and_evaluate(name: str, index: int) -> None:
        checkpoint_dir = run_dir / "checkpoints" / name
        policy.save_adapter(checkpoint_dir)
        eval_dir = run_dir / "evaluations" / name
        future_rows, future_metrics = evaluate_safety(
            policy,
            future_selection,
            rollouts=16,
            stream_seed=args.seed * 1000 + index * 10 + 1,
            checkpoint=name,
            output_path=eval_dir / "future-selection.jsonl",
        )
        current_rows, current_metrics = evaluate_safety(
            policy,
            current_selection,
            rollouts=4,
            stream_seed=args.seed * 1000 + index * 10 + 2,
            checkpoint=name,
            output_path=eval_dir / "current-selection.jsonl",
        )
        benign_rows, benign_metrics = evaluate_benign(
            policy, benign_test, checkpoint=name, output_path=eval_dir / "benign-test.jsonl"
        )
        del future_rows, current_rows, benign_rows
        results[name] = {"index": index, "future": future_metrics, "current": current_metrics, "benign": benign_metrics}
        write_json(run_dir / "checkpoint-results.json", results)

    save_and_evaluate("initial", 0)
    train_benign_stage(
        policy,
        math_train,
        updates=32,
        prompts_per_update=16,
        rollouts_per_prompt=8,
        seed=args.seed,
        optimizer_settings=config["optimizer"],
        kl_coefficient=0.01,
        output_dir=run_dir / "training" / "mathematics",
        optimizer=optimizer,
    )
    save_and_evaluate("after-mathematics", 1)
    train_benign_stage(
        policy,
        logic_train,
        updates=32,
        prompts_per_update=16,
        rollouts_per_prompt=8,
        seed=args.seed,
        optimizer_settings=config["optimizer"],
        kl_coefficient=0.01,
        output_dir=run_dir / "training" / "reasoning",
        optimizer=optimizer,
    )
    save_and_evaluate("after-reasoning", 2)

    initial_benign = results["initial"]["benign"]
    final_benign = results["after-reasoning"]["benign"]
    improvements = {
        domain: final_benign[domain] - initial_benign[domain] for domain in ("mathematics", "logic")
    }
    if max(improvements.values()) < 0.05:
        decision = {"status": "insufficient benign learning", "benign_improvements": improvements}
    else:
        metrics = []
        for name in checkpoint_names:
            row = results[name]
            metrics.append(
                CheckpointMetrics(
                    name=name,
                    index=row["index"],
                    current_s=row["current"]["safety"],
                    benign_mean=row["benign"]["mean"],
                    future_u=row["future"]["utility"],
                    future_s=row["future"]["safety"],
                    future_j=row["future"]["joint"],
                    future_c16=row["future"]["c16"],
                )
            )
        decision = {**select_matched_pair(metrics), "benign_improvements": improvements}
    write_json(run_dir / "decision.json", {"decision": decision, "checkpoints": results})
    mark_run_complete(run_dir, manifest)
    print(decision["status"])


if __name__ == "__main__":
    main()
