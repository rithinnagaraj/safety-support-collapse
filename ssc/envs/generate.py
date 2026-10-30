"""Write task splits to JSONL.

    py -m ssc.envs.generate --family fb_injection --out data/fb_injection
    py -m ssc.envs.generate --family fa_secret --show 1   # print one example
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ssc.envs import FAMILIES, SPLITS, get_family, make_tasks
from ssc.types import Task

DEFAULT_SIZES = {"train": 4000, "heldout": 500, "test": 500}


def load_tasks(path: str | Path) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        return [Task.from_json(line) for line in f if line.strip()]


def _prompt_hash(t: Task) -> str:
    return hashlib.sha256((t.system + "\x00" + t.user).encode()).hexdigest()


def build_splits(family: str, sizes: dict[str, int], seed: int) -> tuple[dict[str, list[Task]], int]:
    """Generate every split with no prompt repeated within or across splits.

    Small procedural spaces (e.g. 3-person logic puzzles) do repeat, so a
    duplicate is skipped and the next index is drawn. Deterministic for a seed.
    """
    fam = get_family(family)
    seen: set[str] = set()
    splits: dict[str, list[Task]] = {}
    skipped = 0
    for split in SPLITS:
        tasks: list[Task] = []
        index = 0
        while len(tasks) < sizes[split]:
            if index > 20 * sizes[split] + 1000:
                raise RuntimeError(f"{family}/{split}: too many duplicate prompts; the generator space is too small")
            t = fam.generate(split, index, seed)
            index += 1
            h = _prompt_hash(t)
            if h in seen:
                skipped += 1
                continue
            seen.add(h)
            # Every prompt must admit a safe-and-useful reply.
            if not fam.grade(t, fam.synthetic_responses(t)["SU"], "eval").su:
                raise RuntimeError(f"reference SU reply fails its own grader: {t.task_id}")
            tasks.append(t)
        splits[split] = tasks
    return splits, skipped


def write_splits(family: str, out: Path, sizes: dict[str, int], seed: int) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    splits, skipped = build_splits(family, sizes, seed)
    for split, tasks in splits.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(t.to_json() + "\n" for t in tasks)
    counts = {s: len(t) for s, t in splits.items()}
    manifest = {"family": family, "seed": seed, "counts": counts, "duplicates_skipped": skipped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=sorted(n for n, f in FAMILIES.items() if f.procedural),
                   help="procedural families only; math/code come from ssc.envs.prepare_benign")
    p.add_argument("--out", type=Path, help="output directory (writes train/heldout/test .jsonl)")
    p.add_argument("--seed", type=int, default=0)
    for split, n in DEFAULT_SIZES.items():
        p.add_argument(f"--n-{split}", type=int, default=n)
    p.add_argument("--show", type=int, default=0, help="print N example prompts and exit")
    args = p.parse_args(argv)

    if args.show:
        for t in make_tasks(args.family, "train", args.show, args.seed):
            print(f"=== {t.task_id}  meta={ {k: v for k, v in t.meta.items() if k not in ('sensitive', 'payload')} }")
            print(f"[system]\n{t.system}\n\n[user]\n{t.user}\n\n[gold] {t.answer}\n")
        return
    if args.out is None:
        p.error("--out is required unless --show is given")
    sizes = {s: getattr(args, f"n_{s}") for s in SPLITS}
    print(json.dumps(write_splits(args.family, args.out, sizes, args.seed)))


if __name__ == "__main__":
    main()
