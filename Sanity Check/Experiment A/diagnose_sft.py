"""Read-only diagnosis of a failed Experiment A supervised-learning check.

Uses only the frozen calibration split and Experiment A artifacts. It does not
validate the current source fingerprint, retrain, or alter the original run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXPERIMENT_DIR.parent
ROOT = SECTION_DIR.parent
for directory in (ROOT, SECTION_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from protocol_core import read_json, read_jsonl  # noqa: E402
from safety_harness import MalformedCall, parse_call  # noqa: E402


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "joint": _mean(rows, "joint"),
        "utility": _mean(rows, "utility"),
        "safety": _mean(rows, "safety"),
        "malformed": sum(row["termination"] == "malformed" for row in rows) / len(rows) if rows else 0.0,
        "token_limit": sum(row["termination"] == "token_limit" for row in rows) / len(rows) if rows else 0.0,
        "action_limit": sum(row["termination"] == "action_limit" for row in rows) / len(rows) if rows else 0.0,
        "mean_generated_tokens": _mean(rows, "generated_tokens"),
    }


def _first_bad_turn(row: dict[str, Any]) -> tuple[str, str] | None:
    for turn in row["assistant_turns"]:
        try:
            parse_call(turn)
        except MalformedCall as exc:
            detail = str(exc)
            if detail.startswith("Invalid JSON"):
                kind = "invalid_json"
            elif detail.startswith("Unknown tool"):
                kind = "unknown_tool"
            elif "arguments" in detail or "fields" in detail:
                kind = "schema_or_arguments"
            else:
                kind = "other_parser_error"
            return kind, detail
    return None


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["instance_id"]), int(row["rollout"])


def analyze(
    initial: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    history: list[dict[str, Any]],
    *,
    max_examples: int = 12,
    initial_repeat: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len({_key(row) for row in initial}) != len(initial) or len({_key(row) for row in trained}) != len(trained):
        raise ValueError("Duplicate instance/rollout keys in calibration traces")
    old = {_key(row): row for row in initial}
    new = {_key(row): row for row in trained}
    if not old or set(old) != set(new):
        raise ValueError("Initial and trained traces must cover the same instance/rollout keys")
    for key in old:
        if any(old[key][field] != new[key][field] for field in ("family", "horizon", "template_id")):
            raise ValueError(f"Calibration metadata changed for {key}")

    strata: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("family", "horizon", "template_id"):
        values = sorted({str(row[field]) for row in initial})
        strata[field] = {}
        for value in values:
            before = [row for row in initial if str(row[field]) == value]
            after = [row for row in trained if str(row[field]) == value]
            left, right = _metrics(before), _metrics(after)
            strata[field][value] = {
                "initial": left,
                "after_sft": right,
                "joint_change": right["joint"] - left["joint"],
                "malformed_change": right["malformed"] - left["malformed"],
            }

    transitions: Counter[str] = Counter()
    newly_malformed: list[dict[str, Any]] = []
    malformed_details: Counter[str] = Counter()
    for key in sorted(old):
        before, after = old[key], new[key]
        transitions[f"{before['termination']} -> {after['termination']}"] += 1
        if after["termination"] == "malformed":
            bad = _first_bad_turn(after)
            malformed_details[bad[0] if bad else "no_invalid_turn_recorded"] += 1
        if before["termination"] != "malformed" and after["termination"] == "malformed":
            bad = _first_bad_turn(after)
            newly_malformed.append({
                "instance_id": key[0],
                "rollout": key[1],
                "family": after["family"],
                "horizon": after["horizon"],
                "initial_joint": before["joint"],
                "after_sft_joint": after["joint"],
                "initial_termination": before["termination"],
                "parser_error": bad[1] if bad else None,
                "initial_turns": before["assistant_turns"],
                "after_sft_turns": after["assistant_turns"],
            })

    epoch_losses: dict[int, list[float]] = defaultdict(list)
    for entry in history:
        epoch_losses[int(entry["epoch"])].append(float(entry["loss"]))
    loss_summary = {
        str(epoch): {
            "batches": len(losses),
            "mean": sum(losses) / len(losses),
            "first": losses[0],
            "last": losses[-1],
        }
        for epoch, losses in sorted(epoch_losses.items())
    }

    before, after = _metrics(initial), _metrics(trained)
    report = {
        "scope": "Experiment A frozen calibration validation only; descriptive, not a new gate",
        "initial": before,
        "after_sft": after,
        "changes": {field: after[field] - before[field] for field in ("joint", "utility", "safety", "malformed", "token_limit", "action_limit", "mean_generated_tokens")},
        "by": strata,
        "termination_transitions": dict(sorted(transitions.items())),
        "after_sft_malformed_parser_categories": dict(sorted(malformed_details.items())),
        "newly_malformed_count": len(newly_malformed),
        "newly_malformed_examples": newly_malformed[:max_examples],
        "sft_loss_by_epoch": loss_summary,
        "interpretation_limits": [
            "Rollouts share prompt/rollout indices but use different random streams; transitions are descriptive, not counterfactual pairs.",
            "A falling supervised loss does not by itself prove correct assistant-token masking or held-out generalization.",
            "Raw traces can identify failure modes, but not establish a unique cause without a controlled calibration-only follow-up.",
        ],
    }
    if initial_repeat is not None:
        if {_key(row) for row in initial_repeat} != set(old):
            raise ValueError("Second initial draw must cover the same calibration keys")
        repeat = _metrics(initial_repeat)
        report["initial_repeat"] = repeat
        report["initial_draw_difference"] = {
            field: repeat[field] - before[field]
            for field in ("joint", "utility", "safety", "malformed", "token_limit", "action_limit")
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-a-dir", type=Path, default=ROOT / "artifacts" / "sanity-check" / "experiment-a")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; otherwise print to stdout")
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    run_dir = args.experiment_a_dir.resolve()
    decision = read_json(run_dir / "decision.json")
    if decision.get("decision", {}).get("checks", {}).get("learning_check") is not False:
        parser.error("This diagnostic requires a completed Experiment A with a failed learning check")
    report = analyze(
        list(read_jsonl(run_dir / "initial-evaluation-1.jsonl")),
        list(read_jsonl(run_dir / "sft-evaluation.jsonl")),
        read_json(run_dir / "sft-history.json"),
        max_examples=args.max_examples,
        initial_repeat=list(read_jsonl(run_dir / "initial-evaluation-2.jsonl")),
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote read-only diagnosis to {args.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
