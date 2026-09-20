"""Run Section 1E replications and apply the fixed GO/NO-GO rule."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

EXPERIMENT_DIR = Path(__file__).resolve().parent
SECTION_DIR = EXPERIMENT_DIR.parent
ROOT = SECTION_DIR.parent
for path in (ROOT, SECTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from protocol_core import read_json, write_json  # noqa: E402
from pilot_io import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    mark_run_complete,
    record_run_metadata,
    require_matching_run,
    validate_manifest,
)


def _run(script: Path, arguments: list[str]) -> None:
    command = [sys.executable, str(script), *arguments]
    subprocess.run(command, cwd=ROOT, check=True)


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
    output_dir = args.artifacts_dir / "experiment-e"
    output_dir.mkdir(parents=True, exist_ok=True)
    record_run_metadata(output_dir, manifest, "experiment-e")
    discovery_c = args.artifacts_dir / "experiment-c" / "seed-101" / "decision.json"
    discovery_d = args.artifacts_dir / "experiment-d" / "seed-101" / "decision.json"
    discovery_b = args.artifacts_dir / "experiment-b" / "seed-101" / "decision.json"
    if not discovery_b.is_file():
        raise SystemExit("Discovery Experiment B result is missing.")
    require_matching_run(discovery_b.parent, manifest, "Discovery Experiment B")
    require_matching_run(discovery_c.parent, manifest, "Discovery Experiment C")
    require_matching_run(discovery_d.parent, manifest, "Discovery Experiment D")
    if not discovery_c.is_file() or not read_json(discovery_c).get("decision", {}).get("pass"):
        raise SystemExit("Discovery Experiment C did not pass; replications must not run.")
    if not discovery_d.is_file() or not read_json(discovery_d).get("promising_recovery"):
        raise SystemExit("Discovery recovery was not promising; replications must not run.")
    discovery_b_report = read_json(discovery_b)
    discovery_c_report = read_json(discovery_c)
    discovery_d_report = read_json(discovery_d)

    b_script = SECTION_DIR / "Experiment B" / "run.py"
    c_script = SECTION_DIR / "Experiment C" / "run.py"
    d_script = SECTION_DIR / "Experiment D" / "run.py"
    mapping = ((102, 202), (103, 203))
    reports: dict[str, dict] = {}
    for benign_seed, recovery_seed in mapping:
        common = ["--manifest", str(args.manifest), "--device", args.device]
        _run(
            b_script,
            common + [
                "--seed", str(benign_seed),
                "--output-dir", str(args.artifacts_dir / "experiment-b"),
                "--experiment-a-decision", str(args.artifacts_dir / "experiment-a" / "decision.json"),
            ],
        )
        b_report = read_json(args.artifacts_dir / "experiment-b" / f"seed-{benign_seed}" / "decision.json")
        if b_report["decision"].get("status") != "selected":
            reports[str(benign_seed)] = {"b": b_report, "passes_c_and_d": False}
            continue
        _run(
            c_script,
            common + [
                "--seed", str(benign_seed),
                "--experiment-b-dir", str(args.artifacts_dir / "experiment-b"),
                "--output-dir", str(args.artifacts_dir / "experiment-c"),
                "--bootstrap-replicates", str(args.bootstrap_replicates),
            ],
        )
        c_report = read_json(args.artifacts_dir / "experiment-c" / f"seed-{benign_seed}" / "decision.json")
        if not c_report["decision"].get("pass"):
            reports[str(benign_seed)] = {"b": b_report, "c": c_report, "passes_c_and_d": False}
            continue
        _run(
            d_script,
            common + [
                "--benign-seed", str(benign_seed),
                "--recovery-seed", str(recovery_seed),
                "--experiment-b-dir", str(args.artifacts_dir / "experiment-b"),
                "--experiment-c-dir", str(args.artifacts_dir / "experiment-c"),
                "--output-dir", str(args.artifacts_dir / "experiment-d"),
            ],
        )
        d_report = read_json(args.artifacts_dir / "experiment-d" / f"seed-{benign_seed}" / "decision.json")
        reports[str(benign_seed)] = {
            "b": b_report,
            "c": c_report,
            "d": d_report,
            "passes_c_and_d": bool(d_report.get("promising_recovery")),
        }
    go = any(report.get("passes_c_and_d") for report in reports.values())
    final = {
        "decision": "GO" if go else "NO-GO",
        "discovery": {
            "seed": 101,
            "b": discovery_b_report,
            "c": discovery_c_report,
            "d": discovery_d_report,
            "passes_c_and_d": True,
        },
        "replications": reports,
    }
    write_json(output_dir / "decision.json", final)
    mark_run_complete(output_dir, manifest)
    print(final["decision"])


if __name__ == "__main__":
    main()
