# Safety-support collapse in continual RLVR

Design: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). Implemented: **M0** (measurement tools) and **M1** (pilot pipeline).
Nothing here has been run on a GPU yet.

| Module | What it does |
|---|---|
| `ssc/envs/` | Task families and graders. Safety: `fa_secret`, `fa_secret_v2` (QA + redaction), `fb_injection` (summarization). Benign: `math` (GSM8K + MATH), `code` (MBPP, executes code), `logic_kk` (knights and knaves) |
| `ssc/envs/prepare_benign.py` | Downloads and converts GSM8K / MATH / MBPP (GPU machine, needs `datasets`) |
| `ssc/eval/coverage.py` | n samples per prompt -> pass@1, C32, pass@k curve, predicted group signal |
| `ssc/eval/suite.py` | Several evaluations against one checkpoint, model loaded once, resumable |
| `ssc/eval/retest.py` | Test-retest reliability between two independent draws |
| `ssc/eval/breakdown.py` | Outcome cells, reply length and leak kinds of saved samples, split by task attributes (mode, pressure, injection) |
| `ssc/train/grpo.py` | GRPO run with trajectory budget, cost ledger, saved checkpoints and `final/` |
| `ssc/train/ledger.py`, `target.py` | Cost and group-type logging; time-to-target with censoring and guardrail |
| `ssc/audit/` | Blind stratified export for hand labeling; grader precision/recall |
| `ssc/pilot/run.py` | M1 driver: data -> stage0 -> benign stages -> coverage -> recovery -> ceiling -> eval -> analyze |
| `ssc/pilot/analyze.py` | Pilot report: H1 preview, tau, recovery cost, mechanism, noise, power, budget, GPU-hours, prereg draft |

## On any machine (standard library only)

```bash
py -m unittest discover -s tests -t .
```

```bash
py -m ssc.pilot.run --config configs/pilot.json --dry-run
```

On Linux, use `python` in place of `py`.

## On the GPU machine

```bash
pip install -e ".[gpu]"
```

### Finish M0: base-model coverage, grader audit, training smoke test

```bash
python -m ssc.envs.generate --family fb_injection --out data/fb_injection
```

```bash
python -m ssc.eval.coverage --backend vllm --model Qwen/Qwen2.5-0.5B-Instruct --family fb_injection --data data/fb_injection/heldout.jsonl --n 32 --seed 1 --out results/base/fb_draw1.json --save-samples results/base/fb_samples.jsonl
```

```bash
python -m ssc.audit.export --samples results/base/fb_samples.jsonl --per-cell 50 --out audit/fb_injection
```

Fill in `human_useful` / `human_safe` (y/n) in `audit/fb_injection.csv`, then:

```bash
python -m ssc.audit.score --labels audit/fb_injection.csv --key audit/fb_injection.key.json
```

Repeat the coverage, export and score commands for `fa_secret`, then do a short training smoke test (stop it after a few steps and check `ledger.jsonl`):

```bash
python -m ssc.train.grpo --config configs/recovery_fb.json --out runs/smoke/fb
```

### Run the pilot (M1)

```bash
python -m ssc.pilot.run --config configs/pilot.json
```

Every step skips work whose outputs exist, so rerun the same command after an interruption. A run that died mid-training must have its directory deleted first; the driver says which one. Run one step at a time with `--steps`, e.g. `--steps coverage,analyze`. Results land in `results/pilot/report.md`, `report.json` and `prereg_draft.json`.

Notes:
- `configs/pilot.json` batch sizes fit a 24 GB GPU (L4). On an A100, raise `per_device_train_batch_size` to 64.
- `keep_checkpoints: false` deletes each recovery checkpoint after it has been scored (full-precision 0.5B checkpoints are about 2 GB each).
- Code grading executes model-written code. It caps time and memory but is not a sandbox; run training inside a container.

## Not yet built

- Entropy-bonus training for config K2: TRL has no entropy bonus, so it needs a small trainer subclass. The pilot uses only K0.
- Benign-recovery control task (H2 double dissociation), H3 demonstration injection, H4 replay arms.
- Expansion families and the MiniOffice agentic environment (Phase 4).
