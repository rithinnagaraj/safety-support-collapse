# Safety-Support Collapse experiments

This repository implements the protocol in `Safety-Support-Collapse-Execution-Plan.md`.
Only the Section 1 sanity check is implemented. Section 2 must not be added or run
until Section 1E records a `GO` decision.

Shared code that is expected to be reused by later sections lives at the repository
root: task generators, the exact tool harness, coverage/bootstrap analysis, the
Hugging Face runtime, training rules, and resource accounting. Section-wide pilot
configuration and orchestration live in `Sanity Check/`. Files unique to a gate live
under `Sanity Check/Experiment A` through `Experiment E`.

The generated `data/`, model `checkpoints/`, and run `artifacts/` are intentionally
not source-controlled. The generated manifest hashes every split and freezes the
model revision, source tree, dependency versions, device description, parser/schema,
configuration, and seed map. Training refuses to start if these have drifted.

See `Sanity Check/README.md` for the exact run order.

