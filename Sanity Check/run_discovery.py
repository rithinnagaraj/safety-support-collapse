"""Run discovery B, C, and D in order, stopping at every protocol gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

SECTION_DIR = Path(__file__).resolve().parent
ROOT = SECTION_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocol_core import read_json, write_json  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    mark_run_complete,
    record_run_metadata,
    validate_manifest,
)


def run(script: Path, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(script), *arguments], cwd=ROOT, check=True)


def record_no_go(artifacts_dir: Path, reason: str, discovery: dict, manifest: dict) -> None:
    output = artifacts_dir / "experiment-e"
    output.mkdir(parents=True, exist_ok=True)
    record_run_metadata(output, manifest, "experiment-e discovery NO-GO")
    write_json(output / "decision.json", {"decision": "NO-GO", "reason": reason, "discovery": discovery, "replications": {}})
    mark_run_complete(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000, choices=[10_000])
    args = parser.parse_args()
    manifest = validate_manifest(
        args.manifest.resolve(), require_ml=True, execution_device=args.device
    )
    final_dir = args.artifacts_dir / "experiment-e"
    final_dir.mkdir(parents=True, exist_ok=True)
    record_run_metadata(final_dir, manifest, "experiment-e pending discovery")
    common = ["--manifest", str(args.manifest), "--device", args.device]
    b_script = SECTION_DIR / "Experiment B" / "run.py"
    c_script = SECTION_DIR / "Experiment C" / "run.py"
    d_script = SECTION_DIR / "Experiment D" / "run.py"
    run(
        b_script,
        common
        + [
            "--seed", "101",
            "--output-dir", str(args.artifacts_dir / "experiment-b"),
            "--experiment-a-decision", str(args.artifacts_dir / "experiment-a" / "decision.json"),
        ],
    )
    b_report = read_json(args.artifacts_dir / "experiment-b" / "seed-101" / "decision.json")
    if b_report["decision"].get("status") != "selected":
        reason = str(b_report["decision"].get("status"))
        record_no_go(
            args.artifacts_dir, reason, {"seed": 101, "b": b_report}, manifest
        )
        print(f"NO-GO: {reason}")
        return
    run(
        c_script,
        common
        + [
            "--seed", "101",
            "--experiment-b-dir", str(args.artifacts_dir / "experiment-b"),
            "--output-dir", str(args.artifacts_dir / "experiment-c"),
            "--bootstrap-replicates", str(args.bootstrap_replicates),
        ],
    )
    c_report = read_json(args.artifacts_dir / "experiment-c" / "seed-101" / "decision.json")
    if not c_report["decision"].get("pass"):
        record_no_go(
            args.artifacts_dir,
            "coverage confirmation failed",
            {"seed": 101, "b": b_report, "c": c_report},
            manifest,
        )
        print("NO-GO: coverage confirmation failed")
        return
    run(
        d_script,
        common
        + [
            "--benign-seed", "101",
            "--recovery-seed", "201",
            "--experiment-b-dir", str(args.artifacts_dir / "experiment-b"),
            "--experiment-c-dir", str(args.artifacts_dir / "experiment-c"),
            "--output-dir", str(args.artifacts_dir / "experiment-d"),
        ],
    )
    d_report = read_json(args.artifacts_dir / "experiment-d" / "seed-101" / "decision.json")
    if not d_report.get("promising_recovery"):
        record_no_go(
            args.artifacts_dir,
            str(d_report["status"]),
            {
                "seed": 101,
                "b": b_report,
                "c": c_report,
                "d": d_report,
            },
            manifest,
        )
        print(f"NO-GO: {d_report['status']}")
        return
    print("Discovery is promising; run Experiment E for replications.")


if __name__ == "__main__":
    main()
