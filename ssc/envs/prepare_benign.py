"""Download and convert the dataset-backed benign families (GPU machine; needs `datasets`).

    python -m ssc.envs.prepare_benign --out data

Writes, each with train/heldout/test .jsonl and a manifest:
  data/math_easy   S1: GSM8K + MATH levels 1-3
  data/math_hard   S4: MATH levels 4-5
  data/code        S2: MBPP (train = train+validation+prompt splits; heldout/test from MBPP test)

Every kept task is checked against its own grader: a math task must have an
extractable gold answer, and an MBPP task's reference solution must pass its
tests in our harness. Tasks that fail are dropped and counted in the manifest.

Hugging Face dataset IDs move occasionally; override them with the flags if a
load fails.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ssc.envs import code_tasks, math_tasks
from ssc.types import Task

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def _write(out: Path, splits: dict[str, list[Task]], extra: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for split, tasks in splits.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(t.to_json() + "\n" for t in tasks)
    manifest = {"counts": {s: len(t) for s, t in splits.items()}, **extra}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _split_eval(pool: list, n_heldout: int, n_test: int, rng: random.Random) -> tuple[list, list]:
    pool = pool[:]
    rng.shuffle(pool)
    return pool[:n_heldout], pool[n_heldout : n_heldout + n_test]


def _math_rows(args) -> dict[str, list[dict]]:
    from datasets import load_dataset

    rows: dict[str, list[dict]] = {"train": [], "test": []}
    gsm = load_dataset(args.gsm8k, "main")
    for split in ("train", "test"):
        for i, ex in enumerate(gsm[split]):
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            rows[split].append({"id": f"gsm8k-{split}-{i}", "problem": ex["question"], "answer": gold,
                                "level": 0, "source": "gsm8k"})
    for subj in MATH_SUBJECTS:
        ds = load_dataset(args.math_dataset, subj)
        for split in ("train", "test"):
            for i, ex in enumerate(ds[split]):
                gold = math_tasks.last_boxed(ex["solution"])
                level = int(ex["level"].split()[-1]) if ex["level"].split()[-1].isdigit() else -1
                rows[split].append({"id": f"math-{subj}-{split}-{i}", "problem": ex["problem"], "answer": gold,
                                    "level": level, "source": f"math/{subj}"})
    return rows


def prepare_math(args, rng: random.Random) -> dict:
    rows = _math_rows(args)
    manifests = {}
    for name, keep in (("math_easy", lambda r: r["source"] == "gsm8k" or 1 <= r["level"] <= 3),
                       ("math_hard", lambda r: r["source"] != "gsm8k" and r["level"] >= 4)):
        dropped = 0

        def tasks(split_rows, split):
            nonlocal dropped
            out = []
            for r in split_rows:
                if not r["answer"]:
                    dropped += 1
                    continue
                out.append(math_tasks.make_task(f"{name}-{r['id']}", split, r["problem"], r["answer"],
                                                {"source": r["source"], "level": r["level"]}))
            return out

        train = [r for r in rows["train"] if keep(r)]
        heldout, test = _split_eval([r for r in rows["test"] if keep(r)], args.n_heldout, args.n_test, rng)
        manifests[name] = _write(args.out / name, {"train": tasks(train, "train"), "heldout": tasks(heldout, "heldout"),
                                                   "test": tasks(test, "test")}, {"dropped_no_gold": dropped})
    return manifests


def prepare_code(args, rng: random.Random) -> dict:
    from datasets import load_dataset

    ds = load_dataset(args.mbpp, "full")
    failed = []

    def convert(ex, split):
        tests, setup = list(ex["test_list"]), ex.get("test_setup_code") or ""
        t = code_tasks.make_task(f"code-mbpp-{ex['task_id']}", split, ex["text"], tests, setup, ex["code"],
                                 {"source": "mbpp", "mbpp_id": ex["task_id"]})
        if not code_tasks.grade(t, code_tasks.synthetic_responses(t)["SU"]).useful:
            failed.append(ex["task_id"])
            return None
        return t

    train = [t for s in ("train", "validation", "prompt") for ex in ds[s] if (t := convert(ex, "train"))]
    heldout_ex, test_ex = _split_eval(list(ds["test"]), args.n_code_heldout, args.n_code_test, rng)
    heldout = [t for ex in heldout_ex if (t := convert(ex, "heldout"))]
    test = [t for ex in test_ex if (t := convert(ex, "test"))]
    return {"code": _write(args.out / "code", {"train": train, "heldout": heldout, "test": test},
                           {"dropped_reference_fails": failed})}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--only", choices=["math", "code"])
    p.add_argument("--gsm8k", default="openai/gsm8k")
    p.add_argument("--math-dataset", default="EleutherAI/hendrycks_math")
    p.add_argument("--mbpp", default="google-research-datasets/mbpp")
    p.add_argument("--n-heldout", type=int, default=300)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-code-heldout", type=int, default=250)
    p.add_argument("--n-code-test", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    manifests = {}
    if args.only in (None, "math"):
        manifests.update(prepare_math(args, random.Random(args.seed)))
    if args.only in (None, "code"):
        manifests.update(prepare_code(args, random.Random(args.seed + 1)))
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
