"""Pilot analysis: turns pilot outputs into the decisions M1 exists to make.

    python -m ssc.pilot.analyze --config configs/pilot.json

Writes to results/pilot/:
  report.md          human-readable summary
  report.json        every number in the report
  prereg_draft.json  tau, budget and margins to review and then freeze

Sections (each is skipped, with a note, if its inputs do not exist yet):
  1. Stage table      safety, coverage, group signal and benign accuracy by stage (mean, SD over seeds)
  2. H1 preview       does fb SU pass@1 stay within the margin while fb C32 drops?
  3. Target tau       from the ceiling runs' plateaus; checkpoints too close to tau are listed
  4. Recovery         time-to-target per run, stage-0 vs stage-4 cost, mechanism (group types)
  5. Noise and power  seed SDs, measurement SD, and the sample sizes they imply
  6. Budget and time  recommended budget B, measured GPU time, projected Phase-1 GPU-hours

This is exploratory: tau, B and sample sizes are set from these numbers, so
none of them is evidence for the hypotheses.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, median, stdev

from ssc.eval.retest import reliability
from ssc.pilot.run import Pilot
from ssc.train.ledger import load_rows
from ssc.train.target import check_tau, load_points, time_to_target

# Phase-1 run counts from the plan (sections 4-5), used to project GPU time.
PHASE1 = {"lineages": 9, "stages": 4, "checkpoints": 37, "recovery_runs": 111 + 48 + 32}


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _sd(xs: list[float]) -> float | None:
    return stdev(xs) if len(xs) >= 2 else None


def _mean_sd(xs: list[float]) -> dict:
    return {"mean": fmean(xs) if xs else None, "sd": _sd(xs), "n": len(xs)}


class Analysis:
    def __init__(self, cfg: dict, config_path: Path):
        self.cfg = cfg
        self.a = cfg["analysis"]
        self.pilot = Pilot(cfg, config_path, dry_run=True)
        self.res = self.pilot.res
        self.notes: list[str] = []

    # ---------------------------------------------------------------- per checkpoint

    def metrics(self, ck: str) -> dict | None:
        cov = lambda name, d: _read(self.pilot.coverage_out(ck, name, d))
        fb = [s for d in (1, 2) if (s := cov("fb", d))]
        fa = [s for d in (1, 2) if (s := cov("fa", d))]
        benign = {b: cov(b, 1) for b in self.cfg["benign_evals"]}
        if not fb or not fa or not all(benign.values()):
            return None
        n = fb[0]["summary"]["n"]
        c = f"C{n}"
        m = {
            "fb_su_pass1": fmean(s["summary"]["pass1"]["su"] for s in fb),
            "fb_useful_pass1": fmean(s["summary"]["pass1"]["useful"] for s in fb),
            "fb_C": fmean(s["summary"][c]["su"] for s in fb),
            "fb_C_useful": fmean(s["summary"][c]["useful"] for s in fb),
            "fb_C_safe": fmean(s["summary"][c]["safe"] for s in fb),
            "fb_all_fail": fb[0]["summary"]["group_signal"]["all_fail"],
            "fb_all_success": fb[0]["summary"]["group_signal"]["all_success"],
            "fb_mixed": fb[0]["summary"]["group_signal"]["mixed"],
            "fb_truncation": fb[0]["summary"]["truncation_rate"],
            "fa_safe_pass1": fmean(s["summary"]["pass1"]["safe"] for s in fa),
            "fa_su_pass1": fmean(s["summary"]["pass1"]["su"] for s in fa),
            "fa_C": fmean(s["summary"][c]["su"] for s in fa),
            "benign_acc": fmean(s["summary"]["pass1"]["su"] for s in benign.values()),
            "benign_C": fmean(s["summary"][f"C{s['summary']['n']}"]["su"] for s in benign.values()),
            "benign_by_eval": {b: s["summary"]["pass1"]["su"] for b, s in benign.items()},
        }
        if len(fb) == 2:
            m["fb_C_draw_diff"] = fb[0]["summary"][c]["su"] - fb[1]["summary"][c]["su"]
            m["fb_retest"] = reliability(fb[0], fb[1])
        v2 = cov("fa_v2", 1)
        if v2:
            m["fa_v2_su_pass1"] = v2["summary"]["pass1"]["su"]
        return m

    # ---------------------------------------------------------------- 1-2. stages and H1

    KEYS = ["fb_su_pass1", "fb_C", "fb_C_useful", "fb_all_fail", "fb_all_success", "fb_mixed",
            "fa_safe_pass1", "fa_su_pass1", "benign_acc", "benign_C"]

    def stage_table(self, ms: dict[str, dict]) -> list[dict] | None:
        if "stage0" not in ms:
            self.notes.append("Stage table: stage-0 coverage missing.")
            return None
        rows = [{"stage": "stage0", **{k: _mean_sd([ms["stage0"][k]]) for k in self.KEYS}}]
        for st in self.cfg["benign_stages"]:
            per_seed = [ms[f"s{s}_{st['name']}"] for s in self.cfg["benign_seeds"] if f"s{s}_{st['name']}" in ms]
            if per_seed:
                row = {"stage": st["name"], **{k: _mean_sd([m[k] for m in per_seed]) for k in self.KEYS}}
                row["delta_vs_stage0"] = {k: _mean_sd([m[k] - ms["stage0"][k] for m in per_seed]) for k in self.KEYS}
                rows.append(row)
        return rows

    def h1_preview(self, rows: list[dict] | None) -> dict | None:
        if not rows or len(rows) < 2:
            return None
        last = rows[-1]
        d = {k: last["delta_vs_stage0"][k]["mean"] for k in ("fb_su_pass1", "fa_safe_pass1", "fb_C", "fb_C_useful",
                                                              "benign_acc")}
        eq, drop = self.a["equivalence_margin"], self.a["min_coverage_drop"]
        stable = abs(d["fb_su_pass1"]) <= eq and abs(d["fa_safe_pass1"]) <= eq
        dropped = d["fb_C"] <= -drop
        if stable and dropped:
            verdict = "PATTERN PRESENT: safety pass@1 stable, coverage dropped."
        elif dropped:
            verdict = "Coverage dropped, but safety pass@1 also moved beyond the margin (not hidden)."
        elif stable:
            verdict = "No coverage drop: safety pass@1 and coverage both stable."
        else:
            verdict = "Safety pass@1 moved and coverage did not drop."
        return {"final_stage": last["stage"], "mean_deltas": d, "equivalence_margin": eq,
                "min_coverage_drop": drop, "safety_stable": stable, "coverage_dropped": dropped,
                "safety_specific": dropped and d["fb_C_useful"] > -drop, "verdict": verdict,
                "caveat": f"Point estimates from {last['fb_C']['n']} seeds; not an equivalence test."}

    # ---------------------------------------------------------------- 3-4. tau and recovery

    def run_points(self, name: str):
        run = self.pilot.runs / "recovery" / name
        if not (run / "ledger.jsonl").exists() or not list((run / "evals").glob("step*.json")):
            return None
        return load_points(str(run / "ledger.jsonl"), str(run / "evals" / "step*.json"))

    def tau(self, ms: dict[str, dict]) -> dict | None:
        plateaus = {}
        for name in self.pilot.ceiling_runs():
            pts = self.run_points(name)
            if pts and len(pts) >= self.a["plateau_last_k"]:
                plateaus[name] = fmean(p.score for p in pts[-self.a["plateau_last_k"]:])
        if len(plateaus) < 2:
            self.notes.append("Target tau: needs both ceiling runs evaluated.")
            return None
        tau = round(min(plateaus.values()) - self.a["tau_margin"], 3)
        names = sorted(ms)
        too_close = [names[i] for i in check_tau(tau, [ms[n]["fb_su_pass1"] for n in names], self.a["min_tau_gap"])]
        return {"tau": tau, "plateaus": plateaus, "excluded_checkpoints": too_close,
                "rule": f"tau = lowest plateau - {self.a['tau_margin']}; must be >= start + {self.a['min_tau_gap']}"}

    def recovery(self, ms: dict[str, dict], tau: float | None) -> dict | None:
        if tau is None:
            return None
        budget = self.cfg["recovery"]["budget_trajectories"]
        runs = []
        for name, (start, seed) in self.pilot.recovery_runs().items():
            pts = self.run_points(name)
            if not pts:
                continue
            benign0 = next((p.benign for p in pts if p.step == 0), None)
            out = time_to_target(pts, tau, self.a["consecutive"], benign0, self.a["max_benign_drop"])
            ledger = load_rows(self.pilot.runs / "recovery" / name / "ledger.jsonl")
            early = ledger[: max(1, len(ledger) // 5)]  # first fifth of the run
            g = {k: sum(r["groups"][k] for r in early) for k in ("all_fail", "all_success", "mixed")}
            total = sum(g.values()) or 1
            runs.append({
                "run": name, "start": start, "group": "stage0" if start == "stage0" else "stage4", "seed": seed,
                "reached": out.reached, "trajectories": out.trajectories if out.reached else budget,
                "tokens": out.tokens, "auc": out.auc,
                "start_fb_C": ms.get(start, {}).get("fb_C"), "start_fb_su_pass1": ms.get(start, {}).get("fb_su_pass1"),
                "predicted_mixed": ms.get(start, {}).get("fb_mixed"),
                "observed_early_groups": {k: v / total for k, v in g.items()},
            })
        if not runs:
            self.notes.append("Recovery: no evaluated recovery runs yet.")
            return None

        def summary(group: str) -> dict:
            rs = [r for r in runs if r["group"] == group]
            reached = [r["trajectories"] for r in rs if r["reached"]]
            return {"runs": len(rs), "reached": len(reached),
                    "median_trajectories_reached": median(reached) if reached else None,
                    "mean_log_trajectories": fmean(math.log(r["trajectories"]) for r in rs) if rs else None}

        s0, s4 = summary("stage0"), summary("stage4")
        ratio = None
        if s0["mean_log_trajectories"] is not None and s4["mean_log_trajectories"] is not None:
            ratio = math.exp(s4["mean_log_trajectories"] - s0["mean_log_trajectories"])
        censored = any(not r["reached"] for r in runs)
        return {"tau": tau, "budget": budget, "runs": runs, "stage0": s0, "stage4": s4,
                "cost_ratio_stage4_over_stage0": ratio,
                "ratio_note": "censored runs counted at the budget, so the ratio is a lower bound" if censored else ""}

    # ---------------------------------------------------------------- 5. noise and power

    def noise_and_power(self, ms: dict[str, dict], rows: list[dict] | None, rec: dict | None) -> dict:
        out: dict = {}
        z = NormalDist().inv_cdf
        za, zb = z(1 - self.a["alpha"] / 2), z(self.a["power"])

        diffs = [m["fb_C_draw_diff"] for m in ms.values() if "fb_C_draw_diff" in m]
        if len(diffs) >= 2:
            out["fb_C_measurement_sd"] = stdev(diffs) / math.sqrt(2)  # SD of one 32-sample draw
            out["fb_retest_p_correlation_mean"] = fmean(m["fb_retest"]["p_correlation"] for m in ms.values()
                                                       if "fb_retest" in m)
        if rows and len(rows) > 1:
            seed_sds = [r["fb_C"]["sd"] for r in rows[1:] if r["fb_C"]["sd"] is not None]
            if seed_sds:
                out["fb_C_seed_sd"] = math.sqrt(fmean(s * s for s in seed_sds))
            delta_sd = rows[-1]["delta_vs_stage0"]["fb_C"]["sd"]
            if delta_sd is not None:
                drop = self.a["min_coverage_drop"]
                # At least 2 lineages, since an SD cannot be estimated from one.
                n = max(2, math.ceil(((za + zb) * delta_sd / drop) ** 2))
                out["lineages_to_detect_coverage_drop"] = {"drop": drop, "sd_of_delta": delta_sd, "n": n}
        if rec:
            by_start: dict[str, list[float]] = {}
            for r in rec["runs"]:
                by_start.setdefault(r["start"], []).append(math.log(r["trajectories"]))
            groups = [v for v in by_start.values() if len(v) >= 2]
            dof = sum(len(v) - 1 for v in groups)
            if dof:
                pooled = math.sqrt(sum((len(v) - 1) * stdev(v) ** 2 for v in groups) / dof)
                out["log_cost_seed_sd"] = pooled
                out["recovery_runs_per_group"] = {
                    f"cost_ratio_{r}": max(2, math.ceil(2 * ((za + zb) * pooled / math.log(r)) ** 2))
                    for r in (1.25, 1.5, 2.0)}
        out["assumptions"] = f"two-sided alpha={self.a['alpha']}, power={self.a['power']}, normal approximation"
        return out

    # ---------------------------------------------------------------- 6. budget and time

    def budget_and_time(self, rec: dict | None) -> dict:
        out: dict = {}
        if rec:
            reached = [r["trajectories"] for r in rec["runs"] if r["group"] == "stage4" and r["reached"]]
            if reached:
                out["recommended_budget_trajectories"] = int(self.a["budget_multiple_of_median"] * median(reached))
            else:
                out["recommended_budget_trajectories"] = None
                self.notes.append(f"Budget: no stage-4 run reached tau within {rec['budget']} trajectories; "
                                  "raise the recovery budget or lower the target before Phase 1.")

        def wall(path: Path) -> float | None:
            meta = _read(path / "run_meta.json")
            return meta["wall_seconds"] / 3600 if meta else None

        stage_h = [h for s in self.cfg["benign_seeds"] for st in self.cfg["benign_stages"]
                   if (h := wall(self.pilot.lineage_path(s, st["name"])))]
        rec_h = [h for name in self.pilot.recovery_runs() if (h := wall(self.pilot.runs / "recovery" / name))]
        cov_h = []
        for ck in self.pilot.checkpoints():
            files = list((self.res / "coverage" / ck).glob("*_d*.json"))
            if files:
                cov_h.append(sum(json.loads(f.read_text()).get("wall_seconds", 0) for f in files) / 3600)
        eval_h = []
        for name in self.pilot.recovery_runs():
            evals = self.pilot.runs / "recovery" / name / "evals"
            files = [*evals.glob("step*.json"), *(evals / "benign").glob("step*.json")]
            if files:
                eval_h.append(sum(json.loads(f.read_text()).get("wall_seconds", 0) for f in files) / 3600)
        hours = {"stage0": wall(self.pilot.stage0_path()),
                 "benign_stage_mean": fmean(stage_h) if stage_h else None,
                 "recovery_run_mean": fmean(rec_h) if rec_h else None,
                 "coverage_per_checkpoint_mean": fmean(cov_h) if cov_h else None,
                 "recovery_eval_per_run_mean": fmean(eval_h) if eval_h else None}
        out["measured_gpu_hours"] = hours
        if all(hours[k] is not None for k in ("benign_stage_mean", "recovery_run_mean", "coverage_per_checkpoint_mean")):
            per_rec = hours["recovery_run_mean"] + (hours["recovery_eval_per_run_mean"] or 0)
            out["projected_phase1_gpu_hours"] = {
                "benign_stages": PHASE1["lineages"] * PHASE1["stages"] * hours["benign_stage_mean"],
                "coverage": PHASE1["checkpoints"] * hours["coverage_per_checkpoint_mean"],
                "recovery_incl_eval": PHASE1["recovery_runs"] * per_rec,
                "note": "at the pilot's model size, GPU and budgets; scale for a larger model or budget B",
            }
            p = out["projected_phase1_gpu_hours"]
            p["total"] = p["benign_stages"] + p["coverage"] + p["recovery_incl_eval"]
        return out

    # ---------------------------------------------------------------- report

    def run(self) -> dict:
        ms = {ck: m for ck in self.pilot.checkpoints() if (m := self.metrics(ck))}
        if not ms:
            self.notes.append("No checkpoint has complete coverage results yet.")
        rows = self.stage_table(ms) if ms else None
        tau = self.tau(ms) if ms else None
        rec = self.recovery(ms, tau["tau"] if tau else None)
        report = {
            "checkpoint_metrics": ms,
            "stage_table": rows,
            "h1_preview": self.h1_preview(rows),
            "tau": tau,
            "recovery": rec,
            "noise_and_power": self.noise_and_power(ms, rows, rec),
            "budget_and_time": None,
            "notes": self.notes,
        }
        report["budget_and_time"] = self.budget_and_time(rec)
        report["prereg_draft"] = self.prereg(report)
        return report

    def prereg(self, report: dict) -> dict:
        n = next(e["n"] for e in self.cfg["coverage_evals"] if e["name"] == "fb")
        return {
            "status": "DRAFT from the pilot: review, then freeze (dated commit or OSF) before Phase 1",
            "primary_family": "fb_injection",
            "coverage_metric": f"C{n} (>= 1 SU reply among {n} samples), two independent draws",
            "group_size": self.cfg["group_size"],
            "tau": report["tau"]["tau"] if report["tau"] else None,
            "min_tau_gap": self.a["min_tau_gap"],
            "consecutive_evals": self.a["consecutive"],
            "max_benign_drop": self.a["max_benign_drop"],
            "budget_trajectories": (report["budget_and_time"] or {}).get("recommended_budget_trajectories"),
            "equivalence_margin": self.a["equivalence_margin"],
            "min_coverage_drop": self.a["min_coverage_drop"],
            "h2_covariates": ["target SU pass@1", "target useful pass@1", "benign accuracy",
                              "fa safety pass@1", "entropy", "response length"],
            "excluded_checkpoints": report["tau"]["excluded_checkpoints"] if report["tau"] else None,
        }


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        m, sd = x.get("mean"), x.get("sd")
        if m is None:
            return "-"
        return _fmt(m, pct) + (f" +/- {_fmt(sd, pct)}" if sd is not None else "")
    return f"{100 * x:.1f}%" if pct else f"{x:.3g}"


def to_markdown(r: dict) -> str:
    out = ["# Pilot report (M1)", "", "_Exploratory: tau, budget and sample sizes are set from these numbers._", ""]
    if r["notes"]:
        out += ["## Not yet available", *[f"- {n}" for n in r["notes"]], ""]
    if r["stage_table"]:
        cols = [("fb_su_pass1", "FB SU pass@1"), ("fb_C", "FB C32"), ("fb_C_useful", "FB useful C32"),
                ("fb_all_fail", "all-fail"), ("fb_all_success", "all-success"), ("fa_safe_pass1", "FA safe pass@1"),
                ("benign_acc", "benign acc")]
        out += ["## 1. By stage (mean +/- SD over seeds)", "",
                "| stage | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for row in r["stage_table"]:
            out.append(f"| {row['stage']} | " + " | ".join(_fmt(row[k], pct=True) for k, _ in cols) + " |")
        out.append("")
    if r["h1_preview"]:
        h = r["h1_preview"]
        out += ["## 2. H1 preview", "", f"**{h['verdict']}**", "",
                *[f"- change in {k} ({h['final_stage']} minus stage0): {_fmt(v, pct=True)}" for k, v in h["mean_deltas"].items()],
                f"- Safety-specific (useful coverage held): {h['safety_specific']}", f"- {h['caveat']}", ""]
    if r["tau"]:
        t = r["tau"]
        out += ["## 3. Recovery target", "", f"- tau = **{t['tau']}** ({t['rule']})",
                *[f"- plateau {k}: {_fmt(v, pct=True)}" for k, v in t["plateaus"].items()],
                f"- checkpoints too close to tau (exclude from comparisons): {t['excluded_checkpoints'] or 'none'}", ""]
    if r["recovery"]:
        rec = r["recovery"]
        out += ["## 4. Recovery cost and mechanism", "",
                "Predicted mixed = share of 8-sample groups expected to carry learning signal, from the start "
                "checkpoint's coverage samples. Observed = share actually seen in the first fifth of the run.", "",
                "| run | reached | trajectories | start FB C32 | predicted mixed | observed early mixed | early all-fail |",
                "|---|---|---|---|---|---|---|"]
        for x in rec["runs"]:
            g = x["observed_early_groups"]
            out.append(f"| {x['run']} | {x['reached']} | {x['trajectories']} | {_fmt(x['start_fb_C'], True)} | "
                       f"{_fmt(x['predicted_mixed'], True)} | {_fmt(g['mixed'], True)} | {_fmt(g['all_fail'], True)} |")
        out += ["", f"- stage-4 / stage-0 cost ratio (geometric mean): {_fmt(rec['cost_ratio_stage4_over_stage0'])} "
                    f"{rec['ratio_note']}", ""]
    npw = r["noise_and_power"]
    out += ["## 5. Noise and power", "", *[f"- {k}: {json.dumps(v) if isinstance(v, dict) else _fmt(v)}"
                                         for k, v in npw.items()], ""]
    bt = r["budget_and_time"]
    out += ["## 6. Budget and time", "", *[f"- {k}: {json.dumps(v, default=str)}" for k, v in bt.items()], ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    args = p.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    report = Analysis(cfg, args.config).run()
    res = Path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    (res / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (res / "report.md").write_text(to_markdown(report), encoding="utf-8")
    (res / "prereg_draft.json").write_text(json.dumps(report["prereg_draft"], indent=2))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
