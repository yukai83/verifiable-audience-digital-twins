"""The three demonstration scenarios (A, B, C) as reusable functions.

These were previously defined inside ``run.py``; they now live in the package
so that the verification, robustness, ablation and sensitivity experiments can
import them without pulling in the plotting code. ``run.py`` re-exports them,
so the command-line interface is unchanged.

Every runner has the signature ``fn(params, seed=42, **options)`` and returns
a long-format DataFrame with a baseline row at t=0 followed by one row per
simulated step.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import networkx as nx
import pandas as pd

from .models import Params, Scenario
from .sim import init_population, init_population_identity_groups, simulate
from .metrics import outputs_to_frame, outputs_to_frame_with_groups

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def baseline_row_from_states(states: dict, groups: dict | None = None) -> dict:
    agent_ids = list(states.keys())
    att_arr = np.array([states[i].att for i in agent_ids], dtype=float)
    strength_arr = np.array([states[i].strength for i in agent_ids], dtype=float)
    intent_arr = np.array([states[i].intent for i in agent_ids], dtype=float)
    beh_arr = np.array([states[i].beh for i in agent_ids], dtype=float)

    row = {
        "t": 0,
        "mean_att": float(att_arr.mean()),
        "mean_strength": float(strength_arr.mean()),
        "mean_intent": float(intent_arr.mean()),
        "beh_rate": float(beh_arr.mean()),
        "feasibility_gap": 0.0,
        "att_variance": float(att_arr.var()),
        "exposure_segregation": float("nan"),
    }
    if groups:
        for lbl in sorted(set(groups.values())):
            idx = [i for i in agent_ids if groups[i] == lbl]
            if idx:
                row[f"beh_rate_{lbl}"] = float(np.mean([states[i].beh for i in idx]))
                row[f"mean_att_{lbl}"] = float(np.mean([states[i].att for i in idx]))
    return row


def prepend_baseline(df: pd.DataFrame, baseline_row: dict) -> pd.DataFrame:
    df_shifted = df.copy()
    df_shifted["t"] = df_shifted["t"] + 1
    return pd.concat([pd.DataFrame([baseline_row]), df_shifted], ignore_index=True, sort=False)


# Scenario A -- route-dependent durability ---------------------------------------
SCENARIO_A_CENTRAL = dict(aq_range=(0.7, 0.9), sc_range=(0.2, 0.4), sp_range=(0.2, 0.4),
                          ea_range=(0.2, 0.4), load0=0.1)
SCENARIO_A_PERIPHERAL = dict(aq_range=(0.3, 0.5), sc_range=(0.7, 0.9), sp_range=(0.7, 0.9),
                             ea_range=(0.6, 0.8), load0=0.6)


def run_scenario_a_control(params: Params, seed: int = 42, n: int = 200, n_steps: int = 50) -> pd.DataFrame:
    """No-campaign control for Scenario A: the same population and network with
    no messages at all, so strength only decays (Eq. 3). Its end/peak ratio is the
    floor against which the two campaigns' retention should be read."""
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    traits, states = init_population(n, np.random.default_rng(seed))
    baseline_row = baseline_row_from_states(states)
    scen = Scenario(messages_per_step=3, campaign_end_step=0)
    out, _ = simulate(g, traits, states, params, scen, n_steps=n_steps, rng=np.random.default_rng(seed))
    return prepend_baseline(outputs_to_frame(out), baseline_row).assign(route="control")


def run_scenario_a(params: Params, seed: int = 42, n: int = 200, n_steps: int = 50,
                   campaign_end_step: int = 20, trace: Optional[dict] = None) -> pd.DataFrame:
    """Central-favouring vs peripheral-favouring campaign, X_m = +1, then a
    no-message observation phase. ``trace`` (dict) receives per-route traces
    when supplied, for the verification suite."""
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    traits_base, states_base = init_population(n, np.random.default_rng(seed))
    baseline_row = baseline_row_from_states(states_base)

    frames = []
    for route, spec, rng_seed in (("central", SCENARIO_A_CENTRAL, seed),
                                  ("peripheral", SCENARIO_A_PERIPHERAL, seed + 100)):
        traits = copy.deepcopy(traits_base)
        states = copy.deepcopy(states_base)
        for s in states.values():
            s.load = spec["load0"]
        scen = Scenario(messages_per_step=3, aq_range=spec["aq_range"], sc_range=spec["sc_range"],
                        sp_range=spec["sp_range"], ea_range=spec["ea_range"],
                        xm_range=(1.0, 1.0), campaign_end_step=campaign_end_step)
        tr = [] if trace is not None else None
        out, _ = simulate(g, traits, states, params, scen, n_steps=n_steps,
                          rng=np.random.default_rng(rng_seed), trace=tr)
        if trace is not None:
            trace[route] = tr
        frames.append(prepend_baseline(outputs_to_frame(out), baseline_row).assign(route=route))
    return pd.concat(frames, ignore_index=True)


# Scenario B -- feasibility-constrained behaviour --------------------------------
def capability_quartiles(states: dict) -> Dict[int, str]:
    ids = list(states.keys())
    cap_vals = np.array([states[i].cap for i in ids])
    qthr = np.percentile(cap_vals, [25, 50, 75])
    groups = {}
    for i in ids:
        c = states[i].cap
        groups[i] = "Q1" if c <= qthr[0] else "Q2" if c <= qthr[1] else "Q3" if c <= qthr[2] else "Q4"
    return groups


def run_scenario_b(params: Params, seed: int = 42, n: int = 200, n_steps: int = 60,
                   theta: Optional[float] = None, cap_boost: float = 0.20,
                   friction_cut: float = 0.20, trace: Optional[dict] = None) -> pd.DataFrame:
    """Baseline vs targeted capability boost (Q1) vs population-wide friction cut.

    ``theta`` sets both feasibility thresholds theta_C and theta_O for this
    scenario. The default (``None``) leaves them at the values carried by
    ``params`` (0.50 in Table 1), which is the configuration behind Fig. 3 of
    the conference paper. The extended analysis sweeps theta in {0.5, 0.6, 0.7}
    to show when the gate binds (Sect. 7.2 of the extended paper).
    """
    p_b = params if theta is None else replace(params, theta_cap=theta, theta_opp=theta)
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    traits_base, states_base = init_population(n, np.random.default_rng(seed))
    groups = capability_quartiles(states_base)
    baseline_row = baseline_row_from_states(states_base, groups=groups)
    std_scen = Scenario(messages_per_step=3)

    conditions = []
    states_1 = copy.deepcopy(states_base)
    conditions.append(("baseline", states_1, seed + 1))

    states_2 = copy.deepcopy(states_base)
    for i, grp in groups.items():
        if grp == "Q1":
            states_2[i].cap = min(1.0, states_2[i].cap + cap_boost)
    conditions.append(("cap_boost_Q1", states_2, seed + 2))

    states_3 = copy.deepcopy(states_base)
    for i in states_3:
        states_3[i].fr = float(np.clip(states_3[i].fr - friction_cut, 0.0, 1.0))
    conditions.append(("friction_reduce", states_3, seed + 3))

    frames = []
    for cond, states, rng_seed in conditions:
        tr = [] if trace is not None else None
        out, _ = simulate(g, copy.deepcopy(traits_base), states, p_b, std_scen,
                          n_steps=n_steps, rng=np.random.default_rng(rng_seed),
                          groups=groups, trace=tr)
        if trace is not None:
            trace[cond] = tr
        frames.append(prepend_baseline(outputs_to_frame_with_groups(out), baseline_row)
                      .assign(condition=cond))
    return pd.concat(frames, ignore_index=True)


# Scenario C -- amplification and clustering -------------------------------------
def run_scenario_c(params: Params, seed: int = 42, n: int = 200, n_steps: int = 60,
                   gamma1_on: float = 2.0, trace: Optional[dict] = None) -> pd.DataFrame:
    """Two identity groups; no amplification (gamma1=0) vs amplification."""
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    pol_scen = Scenario(messages_per_step=3, xm_range=(-1.0, 1.0))
    traits_base, states_base, grps = init_population_identity_groups(n, np.random.default_rng(seed))
    baseline_row = baseline_row_from_states(states_base, groups=grps)

    frames = []
    for cond, gamma1_val in [("no_amplification", 0.0), ("amplification", gamma1_on)]:
        p_cond = replace(params, gamma1=gamma1_val)
        tr = [] if trace is not None else None
        out, _ = simulate(g, copy.deepcopy(traits_base), copy.deepcopy(states_base), p_cond,
                          pol_scen, n_steps=n_steps, rng=np.random.default_rng(seed),
                          groups=grps, track_segregation=True, trace=tr)
        if trace is not None:
            trace[cond] = tr
        frames.append(prepend_baseline(outputs_to_frame_with_groups(out), baseline_row)
                      .assign(condition=cond))
    return pd.concat(frames, ignore_index=True)


# Scalar diagnostics per scenario -------------------------------------------------
def scenario_a_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Peak/end strength and retention per route, plus the retention gap."""
    out: Dict[str, float] = {}
    for route in ("central", "peripheral"):
        sub = df[df["route"] == route]
        s = sub["mean_strength"].to_numpy()
        post = s[1:] if len(s) > 1 else s
        peak = float(post.max()) if len(post) else float("nan")
        end = float(s[-1]) if len(s) else float("nan")
        out[f"{route}_peak_strength"] = peak
        out[f"{route}_end_strength"] = end
        out[f"{route}_retention"] = (end / peak) if peak > 0 else float("nan")
        out[f"{route}_end_att"] = float(sub["mean_att"].iloc[-1])
    out["retention_gap"] = out["central_retention"] - out["peripheral_retention"]
    out["end_att_gap"] = out["central_end_att"] - out["peripheral_end_att"]
    return out


def _time_to_rate(t: np.ndarray, rate: np.ndarray, level: float) -> float:
    idx = np.where(rate >= level)[0]
    return float(t[idx[0]]) if len(idx) else float("nan")


def scenario_b_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Final and mid-run behaviour rates by quartile and condition, the Q4-Q1
    uptake gap, the time for Q1 to reach 50% uptake, and the area under the
    behaviour-rate curve (cumulative uptake) per condition."""
    out: Dict[str, float] = {}
    for cond in ("baseline", "cap_boost_Q1", "friction_reduce"):
        sub = df[df["condition"] == cond].sort_values("t")
        t = sub["t"].to_numpy()
        for q in ("Q1", "Q2", "Q3", "Q4"):
            col = f"beh_rate_{q}"
            r = sub[col].to_numpy()
            out[f"{cond}_{q}_final"] = float(r[-1])
            out[f"{cond}_{q}_t50"] = _time_to_rate(t, r, 0.5)
            out[f"{cond}_{q}_auc"] = float(_trapz(r, t) / (t[-1] - t[0]))
        out[f"{cond}_gap_Q4_Q1_t20"] = float(sub["beh_rate_Q4"].iloc[20] - sub["beh_rate_Q1"].iloc[20])
        out[f"{cond}_auc"] = float(_trapz(sub["beh_rate"].to_numpy(), t) / (t[-1] - t[0]))
        out[f"{cond}_final"] = float(sub["beh_rate"].iloc[-1])
        out[f"{cond}_feasgap_mean"] = float(sub["feasibility_gap"].iloc[1:].mean())
    out["boost_effect_Q1_auc"] = out["cap_boost_Q1_Q1_auc"] - out["baseline_Q1_auc"]
    out["boost_effect_Q4_auc"] = out["cap_boost_Q1_Q4_auc"] - out["baseline_Q4_auc"]
    out["friction_effect_Q1_auc"] = out["friction_reduce_Q1_auc"] - out["baseline_Q1_auc"]
    out["friction_effect_Q4_auc"] = out["friction_reduce_Q4_auc"] - out["baseline_Q4_auc"]
    return out


def scenario_c_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """End-of-run variance, mean exposure segregation, and group divergence."""
    out: Dict[str, float] = {}
    for cond in ("no_amplification", "amplification"):
        sub = df[df["condition"] == cond].sort_values("t")
        out[f"{cond}_var_end"] = float(sub["att_variance"].iloc[-1])
        out[f"{cond}_seg_mean"] = float(sub["exposure_segregation"].iloc[1:].mean())
        out[f"{cond}_G0_end"] = float(sub["mean_att_G0"].iloc[-1])
        out[f"{cond}_G1_end"] = float(sub["mean_att_G1"].iloc[-1])
        out[f"{cond}_divergence_end"] = out[f"{cond}_G1_end"] - out[f"{cond}_G0_end"]
    out["variance_ratio"] = out["amplification_var_end"] / out["no_amplification_var_end"]
    out["segregation_diff"] = out["amplification_seg_mean"] - out["no_amplification_seg_mean"]
    out["divergence_diff"] = out["amplification_divergence_end"] - out["no_amplification_divergence_end"]
    return out
