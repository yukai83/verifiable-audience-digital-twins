"""Command-line runner for the audience model and paper scenarios."""
import argparse
import copy
import yaml
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace

from audience_dt.models import Params, Scenario
from audience_dt.sim import init_population, init_population_identity_groups, simulate
from audience_dt.metrics import outputs_to_frame, outputs_to_frame_with_groups, durability_half_life
from audience_dt.verify import verify_route_monotonicity, verify_ic_formula, ablation_visibility
from audience_dt.scenarios import (run_scenario_a, run_scenario_b, run_scenario_c,
                                   baseline_row_from_states, prepend_baseline,
                                   scenario_a_metrics)
from audience_dt.extensions import run_multiseed, local_sensitivity, run_ablation_baseline


def load_params(cfg: dict) -> tuple:
    exp = cfg["exposure"]
    route = cfg["route"]
    att = cfg["attitude"]
    st = cfg["strength"]
    intent = cfg["intention"]
    thr = cfg["thresholds"]
    dyn = cfg["dynamics"]
    timing = cfg["timing"]

    p = Params(
        alpha=exp["alpha"], gamma0=exp["gamma0"], gamma1=exp["gamma1"],
        rho_sp=exp["rho"]["sp"], rho_sc=exp["rho"]["sc"],
        rho_ic=exp["rho"]["ic"], rho_ea=exp["rho"]["ea"], rho_load=exp["rho"]["load"],
        w0=route["w0"], w_nfc=route["w_nfc"], w_cap=route["w_cap"],
        w_mr=route["w_mr"], w_load=route["w_load"], w_involve=route["w_involve"],
        kappa_c=att["kappa_c"], kappa_p=att["kappa_p"],
        a_sc=att["a_sc"], a_sp=att["a_sp"], a_ic=att["a_ic"],
        a_ev=att["a_ev"], a_ea=att["a_ea"],
        eta_c=st["eta_c"], eta_p=st["eta_p"], b_sp=st["b_sp"], lambda_s=st["lambda_s"],
        beta0=intent["beta0"], beta_as=intent["beta_as"], beta_mr=intent["beta_mr"],
        beta_ma=intent["beta_ma"], beta_norm=intent["beta_norm"],
        beta_opp=intent["beta_opp"], beta_fric=intent["beta_fric"],
        theta_intent=thr["theta_intent"], theta_cap=thr["theta_cap"], theta_opp=thr["theta_opp"],
        load_decay=dyn["load_decay"], load_from_exposure=dyn["load_from_exposure"],
        norm_mu=dyn["norm_mu"], mr_lr=dyn["mr_lr"], ma_lr=dyn["ma_lr"],
        cap_lr=dyn["cap_lr"], opp_lr=dyn["opp_lr"],
        strength_decay=dyn.get("strength_decay", 0.005),
        delta_r=dyn.get("delta_r", 0.005),
        delta_a=dyn.get("delta_a", 0.005),
    )
    s = Scenario(messages_per_step=timing["messages_per_step"])
    return p, s, cfg


def build_graph(cfg: dict, seed: int) -> nx.Graph:
    n = cfg["population"]["n_agents"]
    net = cfg["population"]["network"]
    if net["type"] == "watts_strogatz":
        return nx.watts_strogatz_graph(n=n, k=net["k"], p=net["p_rewire"], seed=seed)
    raise ValueError(f"Unsupported network type: {net['type']}")


def plot_scenario_a(df: pd.DataFrame, save_path: str = "scenario_a.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Scenario A: Route-Dependent Durability (Xm = +1, campaign ends step 20)",
                 fontweight="bold")
    for route, color in [("central", "steelblue"), ("peripheral", "darkorange")]:
        sub = df[df["route"] == route]
        axes[0].plot(sub["t"], sub["mean_att"],      label=route, color=color)
        axes[1].plot(sub["t"], sub["mean_strength"],  label=route, color=color)
    for ax, title in zip(axes, ["Mean Attitude", "Mean Strength (durability)"]):
        ax.axvline(20, color="grey", ls="--", lw=1, label="campaign end")
        ax.set_xlabel("Step"); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scenario_b(df: pd.DataFrame, save_path: str = "scenario_b.png"):
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    conditions = ["baseline", "cap_boost_Q1", "friction_reduce"]
    colors = {"baseline": "steelblue", "cap_boost_Q1": "darkorange", "friction_reduce": "green"}
    fig, axes  = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Scenario B: Feasibility-Constrained Behaviour\n"
                 "Behaviour rate per capability quartile, three conditions",
                 fontsize=12, fontweight="bold")
    for ax, q in zip(axes.flat, quartiles):
        col = f"beh_rate_{q}"
        for cond in conditions:
            sub = df[df["condition"] == cond]
            if col in sub.columns:
                ax.plot(sub["t"], sub[col], label=cond, color=colors[cond])
        ax.set_title(f"Quartile {q}"); ax.set_xlabel("Step"); ax.set_ylabel("Behaviour rate")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_scenario_c(df: pd.DataFrame, save_path: str = "scenario_c.png"):
    colors_cond = {"no_amplification": "steelblue", "amplification": "crimson"}
    colors_grp = {"G0": "#1f77b4", "G1": "#ff7f0e"}
    fig, axes   = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Scenario C: Amplification and Emergent Clustering\n"
                 "G0 (Pi=−1) vs G1 (Pi=+1)",
                 fontsize=12, fontweight="bold")
    for ax, cond in zip(axes[0], ["amplification", "no_amplification"]):
        sub = df[df["condition"] == cond]
        for grp, color in colors_grp.items():
            col = f"mean_att_{grp}"
            if col in sub.columns:
                ax.plot(sub["t"], sub[col], label=grp, color=color)
        lbl = "amplification on" if cond == "amplification" else "no amplification"
        ax.set_title(f"Group attitudes ({lbl})")
        ax.axhline(0, color="grey", lw=0.8, ls="--"); ax.legend()
    ax = axes[1, 0]
    for cond, color in colors_cond.items():
        sub = df[df["condition"] == cond]
        ax.plot(sub["t"], sub["att_variance"], label=cond, color=color)
    ax.set_title("Attitude variance (polarisation)"); ax.legend()
    ax = axes[1, 1]
    for cond, color in colors_cond.items():
        sub = df[df["condition"] == cond]
        ax.plot(sub["t"], sub["exposure_segregation"], label=cond, color=color)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="baseline (0.5)")
    ax.set_title("Exposure segregation\n(ICi,m ≥ 0.75; 0.5 = random)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default="configs/example.yaml")
    ap.add_argument("--out_csv",   default="outputs.csv")
    ap.add_argument("--plot",      default="outputs.png")
    ap.add_argument("--verify",    action="store_true")
    ap.add_argument("--scenario_a", action="store_true")
    ap.add_argument("--scenario_b", action="store_true")
    ap.add_argument("--scenario_c", action="store_true")
    ap.add_argument("--robustness", action="store_true",
                    help="Multi-seed robustness summary for Scenario A (mean + 95% CI).")
    ap.add_argument("--sensitivity", action="store_true",
                    help="One-at-a-time local sensitivity of Scenario A durability.")
    ap.add_argument("--ablation_baseline", action="store_true",
                    help="Compare the full model against a diffusion+sentiment baseline.")
    ap.add_argument("--n_seeds", type=int, default=20,
                    help="Number of seeds for --robustness (default 20).")
    ap.add_argument("--theta_b", type=float, default=None,
                    help="Feasibility thresholds (theta_C = theta_O) for Scenario B. "
                         "Default: the values in the config (0.50 in Table 1).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    params, scenario, cfg = load_params(cfg)
    g = build_graph(cfg, seed)
    rng = np.random.default_rng(seed)
    traits, states = init_population(cfg["population"]["n_agents"], rng)
    outputs, _ = simulate(g=g, traits=traits, states=states, params=params,
                          scenario=scenario, n_steps=cfg["timing"]["n_steps"], rng=rng)
    df = outputs_to_frame(outputs)
    df.to_csv(args.out_csv, index=False)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("COM-B + ELM ABM: 200 Agents, 60 Steps", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    panels = [
        (gs[0, 0], "mean_att", "Mean Attitude", "steelblue", True),
        (gs[0, 1], "mean_strength", "Mean Strength", "darkorange", False),
        (gs[1, 0], "beh_rate", "Behaviour Rate", "green", False),
        (gs[1, 1], "feasibility_gap", "Feasibility Gap", "crimson", False),
    ]
    for spec, col, title, color, zero_line in panels:
        ax = fig.add_subplot(spec)
        ax.plot(df["t"], df[col], color=color)
        ax.set_title(title)
        ax.set_xlabel("Step")
        if zero_line:
            ax.axhline(0, color="grey", ls="--", lw=0.8)

    plt.savefig(args.plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out_csv}  {args.plot}")
    print(f"Durability half-life: {durability_half_life(df['mean_strength'].to_numpy())}")
    print(f"Final beh_rate={df['beh_rate'].iloc[-1]:.3f}  feasibility_gap={df['feasibility_gap'].iloc[-1]:.3f}  mean_att={df['mean_att'].iloc[-1]:.3f}")

    if args.scenario_a:
        print("\n=== Scenario A ===")
        df_a = run_scenario_a(params, seed=seed)
        df_a.to_csv("scenario_a.csv", index=False)
        plot_scenario_a(df_a, "scenario_a.png")
        for route in ["central", "peripheral"]:
            sub = df_a[df_a["route"] == route]["mean_strength"].to_numpy()
            post_baseline = sub[1:] if len(sub) > 1 else sub
            peak = post_baseline.max()
            peak_t = int(np.argmax(post_baseline)) + 1 if len(post_baseline) else 0
            end = sub[-1]
            ret = end / peak if peak > 0 else float("nan")
            hl = durability_half_life(post_baseline)
            print(f"  {route}: peak={peak:.3f} at step={peak_t}  end={end:.3f}  retention={ret:.1%}  half_life={hl}")

    if args.scenario_b:
        print("\n=== Scenario B ===")
        df_b = run_scenario_b(params, seed=seed, theta=args.theta_b)
        df_b.to_csv("scenario_b.csv", index=False)
        plot_scenario_b(df_b, "scenario_b.png")
        final_b = df_b[df_b["t"] == df_b["t"].max()]
        for cond in ["baseline", "cap_boost_Q1", "friction_reduce"]:
            row = final_b[final_b["condition"] == cond].iloc[0]
            rates = {q: f"{row.get(f'beh_rate_{q}', float('nan')):.2f}"
                     for q in ["Q1", "Q2", "Q3", "Q4"]}
            print(f"  {cond:20s}: {rates}")

    if args.scenario_c:
        print("\n=== Scenario C ===")
        df_c = run_scenario_c(params, seed=seed)
        df_c.to_csv("scenario_c.csv", index=False)
        plot_scenario_c(df_c, "scenario_c.png")
        for cond in ["no_amplification", "amplification"]:
            sub = df_c[df_c["condition"] == cond]
            var_end = sub["att_variance"].iloc[-1]
            mean_seg = sub["exposure_segregation"].mean()
            g0_att = sub["mean_att_G0"].iloc[-1] if "mean_att_G0" in sub.columns else float("nan")
            g1_att = sub["mean_att_G1"].iloc[-1] if "mean_att_G1" in sub.columns else float("nan")
            print(f"  {cond:22s}: att_var={var_end:.4f}  mean_seg={mean_seg:.3f}  G0={g0_att:.3f}  G1={g1_att:.3f}")

    if args.robustness:
        print("\n=== Multi-seed robustness (Scenario A) ===")
        seeds = list(range(args.n_seeds))
        summ = run_multiseed(run_scenario_a, scenario_a_metrics, params, seeds)
        summ.to_csv("robustness_scenario_a.csv")
        with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
            print(summ.loc[["central_retention", "peripheral_retention", "retention_gap"]])
        print("  Saved: robustness_scenario_a.csv")

    if args.sensitivity:
        print("\n=== Local sensitivity (Scenario A retention gap) ===")
        param_names = ["w_load", "kappa_c", "kappa_p", "eta_c", "eta_p",
                       "strength_decay", "lambda_s", "theta_intent"]
        sens = local_sensitivity(run_scenario_a, scenario_a_metrics, "retention_gap",
                                 params, param_names, rel_delta=0.10, seed=seed)
        sens.to_csv("sensitivity_scenario_a.csv")
        with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
            print(sens[["base_value", "elasticity"]])
        print("  Saved: sensitivity_scenario_a.csv")

    if args.ablation_baseline:
        print("\n=== Ablation: full model vs diffusion+sentiment ===")
        abl = run_ablation_baseline(run_scenario_a, params, seed=seed)
        abl.to_csv("ablation_baseline.csv")
        with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
            print(abl)
        print("  Saved: ablation_baseline.csv")

    if args.verify:
        print("\n=== Verification ===")
        print(verify_route_monotonicity(params, seed=seed))
        print(verify_ic_formula(params, seed=seed))
        print(ablation_visibility(params, seed=seed))


if __name__ == "__main__":
    main()
