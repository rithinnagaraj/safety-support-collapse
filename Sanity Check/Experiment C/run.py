"""Run Section 1C: independent confirmation of the selected coverage gap."""

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

from coverage_analysis import paired_coverage_bootstrap  # noqa: E402
from hf_runtime import HFPolicy  # noqa: E402
from protocol_core import read_json, write_json  # noqa: E402
from pilot_analysis import confirmation_decision, pair_prompt_counts  # noqa: E402
from pilot_evaluation import evaluate_safety, load_safety  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    mark_run_complete,
    record_run_metadata,
    require_matching_run,
    validate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument("--experiment-b-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-b")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "experiment-c")
    parser.add_argument("--seed", type=int, default=101, choices=[101, 102, 103])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000, choices=[10_000])
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = validate_manifest(manifest_path, require_ml=True, execution_device=args.device)
    config = manifest["frozen_config"]
    b_dir = args.experiment_b_dir / f"seed-{args.seed}"
    require_matching_run(b_dir, manifest, "Experiment B")
    b_report_path = b_dir / "decision.json"
    if not b_report_path.is_file():
        raise SystemExit("Experiment B result is missing.")
    b_report = read_json(b_report_path)
    selection = b_report["decision"]
    if selection.get("status") != "selected":
        raise SystemExit(f"Experiment B did not select a pair: {selection.get('status')}")

    run_dir = args.output_dir / f"seed-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    record_run_metadata(run_dir, manifest, f"experiment-c seed={args.seed}")
    confirmation = load_safety(manifest_path.parent / manifest["splits"]["future_confirmation"]["file"])
    rows_by_label: dict[str, list[dict]] = {}
    metrics_by_label: dict[str, dict] = {}
    model = config["model"]
    for label, name, offset in (("low", selection["low"], 1), ("high", selection["high"], 2)):
        policy = HFPolicy(
            model_id=model["id"],
            revision=model["revision"],
            adapter=None,
            adapter_checkpoint=b_dir / "checkpoints" / name,
            device=args.device,
            seed=args.seed * 10_000 + offset,
        )
        rows, metrics = evaluate_safety(
            policy,
            confirmation,
            rollouts=32,
            stream_seed=args.seed * 10_000 + offset,
            checkpoint=name,
            output_path=run_dir / f"{label}-future-confirmation.jsonl",
        )
        rows_by_label[label] = rows
        metrics_by_label[label] = metrics
        del policy
    paired = pair_prompt_counts(rows_by_label["low"], rows_by_label["high"])
    interval = paired_coverage_bootstrap(
        paired,
        count_a_key="count_a",
        count_b_key="count_b",
        n=32,
        k=32,
        replicates=args.bootstrap_replicates,
    ).to_dict()
    decision = confirmation_decision(metrics_by_label["low"], metrics_by_label["high"], interval)
    report = {
        "decision": decision,
        "selection": {"low": selection["low"], "high": selection["high"]},
        "low_metrics": metrics_by_label["low"],
        "high_metrics": metrics_by_label["high"],
        "coverage_difference_interval": interval,
        "retained_b_checks": {
            label: {
                "current": b_report["checkpoints"][name]["current"],
                "benign": b_report["checkpoints"][name]["benign"],
            }
            for label, name in (("low", selection["low"]), ("high", selection["high"]))
        },
    }
    write_json(run_dir / "decision.json", report)
    mark_run_complete(run_dir, manifest)
    print("PASS" if decision["pass"] else "FAIL")


if __name__ == "__main__":
    main()
