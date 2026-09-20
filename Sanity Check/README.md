# Section 1 — sanity check

## Layout

- `pilot_config.json` contains all fixed pilot settings and thresholds.
- `generate_data.py` creates all safety and benign splits before training.
- `pilot_*.py` contains code shared by multiple Section 1 experiments.
- `Experiment A` through `Experiment E` contain only gate-specific entry points and tests.
- Reusable task, harness, analysis, model, and trainer modules remain at repository root.

## Prerequisites

Use Python 3.11 or newer and install the pinned root requirements into an isolated
environment. A CUDA device with enough memory for Qwen2.5-3B BF16 training is required
for the actual experiment. The verifier and unit tests do not require the ML packages.

Run the three test files before freezing data:

```powershell
python "Sanity Check/Experiment A/test_harness.py"
python "Sanity Check/Experiment B/test_selection.py"
python "Sanity Check/Experiment D/test_recovery.py"
```

## Frozen data and manifest

Resolve the immutable Hugging Face revision first, then generate all 18 splits in one
operation (including seed-specific benign training lists for 101–103). Do this only
after the source tree, environment, and hardware are final.

```powershell
python "Sanity Check/generate_data.py" `
  --model-revision "<40-character-model-commit-sha>" `
  --device-description "<GPU model, count, CUDA and driver>"
```

Generation writes `Sanity Check/data/manifest.json`. Every training entry point checks
all split hashes, the parser schema, source-tree hash, dependency versions, and the
tokenizer/native-chat-template fingerprint at the frozen model commit.

## Required order

1. Start Experiment A. On the first call it writes a 50-row manual-review JSONL and
   exits. Inspect every row; set `reviewed` to `true` and copy the manually observed
   binary labels into `observed_utility` and `observed_safety`. Rerun the same command.
   The rerun first executes and gates on a disposable calibration smoke update that
   verifies all LoRA targets, the 16×8 effective batch, and merge/unmerge equivalence.

   ```powershell
   python "Sanity Check/Experiment A/run.py"
   ```

2. Only if A writes `decision.pass=true`, run discovery B–D. The orchestrator stops at
   an insufficient-learning, unmatched-pair, or failed-confirmation gate.

   ```powershell
   python "Sanity Check/run_discovery.py"
   ```

3. Only if discovery D records `promising_recovery=true`, run the two unchanged
   replications and the GO/NO-GO rule.

   ```powershell
   python "Sanity Check/Experiment E/run.py"
   ```

For a non-default accelerator selector, add `--device cuda:1` consistently. Run
artifacts include complete trajectories/action logs, labels, invalid/truncation rates,
checkpoint and selection failures, token counts, and separate resource ledgers for
training, evaluation, monitoring, and demonstrations.
