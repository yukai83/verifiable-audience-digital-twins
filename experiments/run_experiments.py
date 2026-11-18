"""Reproduce every result table of the extended (LNNS) paper.

Usage:
    python experiments/run_experiments.py <stage> [--n_seeds N] [--out results]

Stages (each writes CSV files into the output directory):
    verification   invariant counts on scenario traces, random parameter vectors
                   and function-level property checks          -> verification.csv
    robustness     multi-seed mean trajectories + CI for A/B/C   -> robustness_*.csv
    gate_sweep     Scenario B under theta in {0.50..0.70}         -> gate_sweep.csv
    ablation       component ablations for A/B/C                 -> ablation.csv
    sensitivity    Morris screening for A/B/C                    -> morris_*.csv
    calibration    synthetic parameter recovery (eta_c, delta_S) -> recovery_*.csv
    structural     soft gate, process noise and population size  -> structural.csv
    all            everything above in sequence
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run import load_params  # noqa: E402
from audience_dt.models import Scenario  # noqa: E402
from audience_dt.sim import init_population, simulate  # noqa: E402
from audience_dt.scenarios import (run_scenario_a, run_scenario_b, run_scenario_c,  # noqa: E402
                                   run_scenario_a_control,
                                   scenario_a_metrics, scenario_b_metrics, scenario_c_metrics)
from audience_dt.invariants import (trace_checks, unit_checks, random_params,  # noqa: E402
                                    CheckResult)
from audience_dt.extensions import _summarise  # noqa: E402
from audience_dt.ablation import run_ablations  # noqa: E402
from audience_dt.sensitivity import default_param_box, run_morris  # noqa: E402
from audience_dt.calibration import make_observations, grid_recovery, summarise_recovery  # noqa: E402
import networkx as nx  # noqa: E402


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------- verification
def stage_verification(params, out, n_seeds):
    rows = []

    def add(group, config, res: CheckResult):
        rows.append({"group": group, "config": config, "check": res.name,
                     "n_checks": res.n_checks, "n_failures": res.n_failures,
                     "passed": res.passed, "detail": res.detail})

    # (a) the three paper scenarios at the Table 1 defaults, several seeds
    for sc, fn in (("A", run_scenario_a), ("B", run_scenario_b), ("C", run_scenario_c)):
        for seed in range(min(n_seeds, 5)):
            traces = {}
            fn(params, seed=seed, trace=traces)
            for cond, tr in traces.items():
                for res in trace_checks(tr, params):
                    add("scenario traces", f"{sc}/{cond}/seed{seed}", res)
    # (b) random parameter vectors on the base model
    for draw in range(20):
        rng = np.random.default_rng(1000 + draw)
        p = random_params(params, rng, rel=0.5)
        n = 100
        g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=draw)
        traits, states = init_population(n, rng)
        tr = []
        simulate(g, traits, states, p, Scenario(messages_per_step=3), n_steps=30, rng=rng, trace=tr)
        for res in trace_checks(tr, p):
            add("random parameters", f"draw{draw}", res)
    # (c) function-level properties
    for res in unit_checks(params, np.random.default_rng(0), n_cases=2000):
        add("unit properties", "defaults", res)
    for draw in range(20):
        rng = np.random.default_rng(500 + draw)
        p = random_params(params, rng, rel=0.5)
        for res in unit_checks(p, rng, n_cases=500):
            add("unit properties", f"draw{draw}", res)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "verification.csv"), index=False)
    summ = df.groupby(["group", "check"]).agg(n_configs=("config", "nunique"),
                                              n_checks=("n_checks", "sum"),
                                              n_failures=("n_failures", "sum")).reset_index()
    summ.to_csv(os.path.join(out, "verification_summary.csv"), index=False)
    log("verification:\n" + summ.to_string(index=False))


# ---------------------------------------------------------------------------- robustness
def _ci_frame(frames, group_cols, value_cols):
    """Per-step mean, sd and 95% CI across seeds for the given columns."""
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(group_cols)
    n = g.size().rename("n")
    mean = g[value_cols].mean().add_suffix("_mean")
    sd = g[value_cols].std(ddof=1).add_suffix("_sd")
    res = pd.concat([mean, sd, n], axis=1).reset_index()
    for c in value_cols:
        res[f"{c}_ci"] = 1.96 * res[f"{c}_sd"] / np.sqrt(res["n"])
    return res


def stage_robustness(params, out, n_seeds):
    seeds = list(range(n_seeds))
    # Scenario A
    frames, mets = [], []
    for s in seeds:
        df = run_scenario_a(params, seed=s)
        frames.append(df.assign(seed=s)); mets.append(scenario_a_metrics(df))
    for s in seeds:  # no-campaign control (strength decay only)
        dfc = run_scenario_a_control(params, seed=s)
        frames.append(dfc.assign(seed=s))
        st = dfc["mean_strength"].to_numpy()
        mets[s]["control_end_strength"] = float(st[-1])
        mets[s]["control_retention"] = float(st[-1] / st[1:].max())
    _ci_frame(frames, ["route", "t"], ["mean_att", "mean_strength"]).to_csv(
        os.path.join(out, "robustness_A_trajectories.csv"), index=False)
    pd.DataFrame(mets).assign(seed=seeds).to_csv(os.path.join(out, "robustness_A_per_seed.csv"), index=False)
    log("robustness A done")
    # Scenario B (Table 1 thresholds)
    frames, mets = [], []
    for s in seeds:
        df = run_scenario_b(params, seed=s)
        frames.append(df.assign(seed=s)); mets.append(scenario_b_metrics(df))
    cols = ["beh_rate", "feasibility_gap"] + [f"beh_rate_Q{q}" for q in range(1, 5)]
    _ci_frame(frames, ["condition", "t"], cols).to_csv(
        os.path.join(out, "robustness_B_trajectories.csv"), index=False)
    pd.DataFrame(mets).assign(seed=seeds).to_csv(os.path.join(out, "robustness_B_per_seed.csv"), index=False)
    log("robustness B done")
    # Scenario C
    frames, mets = [], []
    for s in seeds:
        df = run_scenario_c(params, seed=s)
        frames.append(df.assign(seed=s)); mets.append(scenario_c_metrics(df))
    cols = ["att_variance", "exposure_segregation", "mean_att_G0", "mean_att_G1"]
    _ci_frame(frames, ["condition", "t"], cols).to_csv(
        os.path.join(out, "robustness_C_trajectories.csv"), index=False)
    pd.DataFrame(mets).assign(seed=seeds).to_csv(os.path.join(out, "robustness_C_per_seed.csv"), index=False)
    log("robustness C done")
    # summary table of scalar metrics
    rows = []
    for sc in "ABC":
        per = pd.read_csv(os.path.join(out, f"robustness_{sc}_per_seed.csv")).drop(columns="seed")
        for c in per.columns:
            rows.append({"scenario": sc, "metric": c, **_summarise(per[c].to_numpy())})
    pd.DataFrame(rows).to_csv(os.path.join(out, "robustness_summary.csv"), index=False)
    log("robustness summary written")


# ---------------------------------------------------------------------------- gate sweep
def stage_gate_sweep(params, out, n_seeds):
    seeds = list(range(min(n_seeds, 10)))
    rows, traj = [], []
    for theta in (0.50, 0.55, 0.60, 0.65, 0.70):
        for gate in ("hard", "soft"):
            p = replace(params, gate_mode=gate)
            for s in seeds:
                df = run_scenario_b(p, seed=s, theta=theta)
                m = scenario_b_metrics(df)
                rows.append({"theta": theta, "gate": gate, "seed": s, **m})
                sub = df[df["condition"] == "baseline"][["t", "beh_rate", "feasibility_gap",
                                                         "beh_rate_Q1", "beh_rate_Q4"]]
                traj.append(sub.assign(theta=theta, gate=gate, seed=s))
        log(f"gate sweep theta={theta} done")
    pd.DataFrame(rows).to_csv(os.path.join(out, "gate_sweep_per_seed.csv"), index=False)
    _ci_frame(traj, ["theta", "gate", "t"], ["beh_rate", "feasibility_gap", "beh_rate_Q1", "beh_rate_Q4"]).to_csv(
        os.path.join(out, "gate_sweep_trajectories.csv"), index=False)
    per = pd.DataFrame(rows)
    keep = ["baseline_final", "baseline_auc", "baseline_feasgap_mean", "baseline_Q1_t50",
            "baseline_Q4_t50", "baseline_gap_Q4_Q1_t20", "boost_effect_Q1_auc",
            "boost_effect_Q4_auc", "friction_effect_Q1_auc", "friction_effect_Q4_auc"]
    summ = []
    for (theta, gate), grp in per.groupby(["theta", "gate"]):
        for c in keep:
            summ.append({"theta": theta, "gate": gate, "metric": c, **_summarise(grp[c].to_numpy())})
    pd.DataFrame(summ).to_csv(os.path.join(out, "gate_sweep_summary.csv"), index=False)


# ---------------------------------------------------------------------------- ablation
def stage_ablation(params, out, n_seeds):
    seeds = list(range(min(n_seeds, 10)))
    df = run_ablations(params, seeds, progress=lambda s: log(f"ablation {s} done"))
    df.to_csv(os.path.join(out, "ablation.csv"), index=False)


# ---------------------------------------------------------------------------- sensitivity
def stage_sensitivity(params, out, n_seeds, r: int = 12):
    box = default_param_box(params, rel=0.5)
    pd.DataFrame(box, columns=["parameter", "low", "high"]).to_csv(
        os.path.join(out, "morris_box.csv"), index=False)
    for sc, fn, mfn in (("A", run_scenario_a, scenario_a_metrics),
                        ("B", run_scenario_b, scenario_b_metrics),
                        ("C", run_scenario_c, scenario_c_metrics)):
        summary, raw = run_morris(params, box, fn, mfn, r=r, p=4, seed=42, traj_seed=0,
                                  progress=lambda d, t: log(f"morris {sc}: {d}/{t}") if d % 21 == 0 else None)
        summary.to_csv(os.path.join(out, f"morris_{sc}.csv"), index=False)
        raw.to_csv(os.path.join(out, f"morris_{sc}_raw.csv"), index=False)
        log(f"morris {sc} done")


# ---------------------------------------------------------------------------- calibration
def stage_calibration(params, out, n_seeds):
    true_values = {"eta_c": 0.10, "strength_decay": 0.005}
    obs = make_observations(params, true_values, data_seed=2024, noise_sd=0.02)
    obs.to_csv(os.path.join(out, "recovery_observations.csv"), index=False)
    grid = {"eta_c": np.round(np.linspace(0.04, 0.16, 13), 4),
            "strength_decay": np.round(np.linspace(0.001, 0.015, 15), 4)}
    res = grid_recovery(params, obs, grid, sim_seed=7,
                        progress=lambda d, t: log(f"recovery {d}/{t}") if d % 39 == 0 else None)
    res.to_csv(os.path.join(out, "recovery_grid.csv"), index=False)
    summ = summarise_recovery(res, true_values, accept_quantile=0.05)
    with open(os.path.join(out, "recovery_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    log("recovery: " + json.dumps(summ))


# ---------------------------------------------------------------------------- structural
def stage_structural(params, out, n_seeds):
    seeds = list(range(min(n_seeds, 10)))
    variants = {
        "reference": {},
        "soft_gate": {"gate_mode": "soft", "gate_temp": 0.05},
        "process_noise_0.02": {"attitude_noise": 0.02},
        "process_noise_0.05": {"attitude_noise": 0.05},
    }
    rows = []
    for name, kw in variants.items():
        p = replace(params, **kw)
        for s in seeds:
            a = scenario_a_metrics(run_scenario_a(p, seed=s))
            b = scenario_b_metrics(run_scenario_b(p, seed=s))
            c = scenario_c_metrics(run_scenario_c(p, seed=s))
            rows.append({"variant": name, "n_agents": 200, "seed": s,
                         "central_retention": a["central_retention"],
                         "peripheral_retention": a["peripheral_retention"],
                         "retention_gap": a["retention_gap"],
                         "B_gap_Q4_Q1_t20": b["baseline_gap_Q4_Q1_t20"],
                         "B_boost_effect_Q1_auc": b["boost_effect_Q1_auc"],
                         "B_boost_effect_Q4_auc": b["boost_effect_Q4_auc"],
                         "C_variance_ratio": c["variance_ratio"],
                         "C_segregation_diff": c["segregation_diff"]})
        log(f"structural {name} done")
    for n in (100, 400):
        for s in seeds:
            a = scenario_a_metrics(run_scenario_a(params, seed=s, n=n))
            b = scenario_b_metrics(run_scenario_b(params, seed=s, n=n))
            c = scenario_c_metrics(run_scenario_c(params, seed=s, n=n))
            rows.append({"variant": f"N={n}", "n_agents": n, "seed": s,
                         "central_retention": a["central_retention"],
                         "peripheral_retention": a["peripheral_retention"],
                         "retention_gap": a["retention_gap"],
                         "B_gap_Q4_Q1_t20": b["baseline_gap_Q4_Q1_t20"],
                         "B_boost_effect_Q1_auc": b["boost_effect_Q1_auc"],
                         "B_boost_effect_Q4_auc": b["boost_effect_Q4_auc"],
                         "C_variance_ratio": c["variance_ratio"],
                         "C_segregation_diff": c["segregation_diff"]})
        log(f"structural N={n} done")
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out, "structural_per_seed.csv"), index=False)
    summ = []
    for v, grp in per.groupby("variant", sort=False):
        for c in per.columns:
            if c in ("variant", "n_agents", "seed"):
                continue
            summ.append({"variant": v, "metric": c, **_summarise(grp[c].to_numpy())})
    pd.DataFrame(summ).to_csv(os.path.join(out, "structural_summary.csv"), index=False)


STAGES = {
    "verification": stage_verification, "robustness": stage_robustness,
    "gate_sweep": stage_gate_sweep, "ablation": stage_ablation,
    "sensitivity": stage_sensitivity, "calibration": stage_calibration,
    "structural": stage_structural,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "example.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--n_seeds", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params, _, _ = load_params(cfg)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for st in stages:
        t0 = time.time()
        log(f"=== {st} ===")
        STAGES[st](params, args.out, args.n_seeds)
        log(f"=== {st} finished in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
