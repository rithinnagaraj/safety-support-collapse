"""Run several evaluations against one checkpoint, loading the model once.

    python -m ssc.eval.suite --backend vllm --model runs/pilot/stage0/final --spec spec.json

The spec is a JSON list. Each item:
  {"name": "fb", "family": "fb_injection", "data": "data/fb_injection/heldout.jsonl",
   "num_tasks": 500, "n": 32, "seed": 1, "max_tokens": 512, "out": "results/.../fb_d1.json",
   "meta": {"step": 0}, "save_samples": null}
Optional: "attach_benign": ["math", "code", ...] names of other items in the
same spec; their mean SU pass@1 is written to this item's meta["benign"]
(used by the recovery-target guardrail). Items whose `out` already exists
are skipped, so an interrupted suite resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ssc.envs import make_tasks
from ssc.envs.generate import load_tasks
from ssc.eval.backends import make_backend
from ssc.eval.coverage import evaluate


def _tasks(item: dict):
    n = item.get("num_tasks")
    if item.get("data"):
        tasks = load_tasks(item["data"])
        bad = {t.family for t in tasks} - {item["family"]}
        if bad:
            raise ValueError(f"{item['data']} has tasks from {bad}, expected {item['family']}")
        return tasks[:n] if n else tasks
    return make_tasks(item["family"], item.get("split", "heldout"), n or 500, item.get("data_seed", 0))


def run_suite(spec: list[dict], backend, workers: int = 16) -> dict[str, dict]:
    names = [it["name"] for it in spec]
    if len(set(names)) != len(names):
        raise ValueError("spec item names must be unique")
    results: dict[str, dict] = {}
    for item in spec:
        out = Path(item["out"])
        if out.exists():
            results[item["name"]] = json.loads(out.read_text())
            continue
        t0 = time.time()
        res = evaluate(_tasks(item), backend, item["n"], item.get("level", "eval"), item.get("group_size", 8),
                       Path(item["save_samples"]) if item.get("save_samples") else None,
                       seed=item.get("seed"), max_tokens=item.get("max_tokens"), workers=workers)
        res["meta"] = dict(item.get("meta", {}))
        res["config"] = {k: v for k, v in item.items() if k != "meta"}
        res["wall_seconds"] = time.time() - t0
        results[item["name"]] = res
        if not item.get("attach_benign"):
            _save(out, res)
    for item in spec:  # attach benign accuracy once every item has run
        if item.get("attach_benign"):
            res = results[item["name"]]
            res["meta"]["benign"] = sum(results[b]["summary"]["pass1"]["su"] for b in item["attach_benign"]) / len(
                item["attach_benign"])
            _save(Path(item["out"]), res)
    return results


def _save(path: Path, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=2))
    tmp.replace(path)  # atomic, so a half-written file never counts as done


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["vllm", "hf", "mock"], required=True)
    p.add_argument("--model")
    p.add_argument("--mock-profile", default="uniform")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024, help="default when a spec item sets none")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--workers", type=int, default=16, help="parallel graders (code runs in subprocesses)")
    args = p.parse_args(argv)

    spec = json.loads(args.spec.read_text())
    if all(Path(it["out"]).exists() for it in spec):
        print("all outputs exist; nothing to do")
        return
    kw = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=0)
    if args.backend == "vllm":
        kw["gpu_memory_utilization"] = args.gpu_memory_utilization
    backend = make_backend(args.backend, args.model, args.mock_profile, **kw)
    results = run_suite(spec, backend, args.workers)
    print(json.dumps({name: r["summary"]["pass1"] for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
