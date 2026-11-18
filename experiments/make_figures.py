"""Build the figures of the extended paper from the CSV files in results/.

Usage:  python experiments/make_figures.py [--results results] [--out figures]

Every figure is written as PDF (for LaTeX) and PNG (for quick viewing).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated colour-vision-safe), with line styles as a
# second encoding so that the figures also read in greyscale print.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "grey": "#6b6b6b"}
STYLE = {"central": (C["blue"], "-"), "peripheral": (C["orange"], "--"), "control": (C["grey"], ":"),
         "baseline": (C["blue"], "-"), "cap_boost_Q1": (C["orange"], "--"),
         "friction_reduce": (C["aqua"], ":"),
         "no_amplification": (C["blue"], "-"), "amplification": (C["orange"], "--"),
         "G0": (C["blue"], "-"), "G1": (C["orange"], "--")}
LABEL = {"central": "central-favouring", "peripheral": "peripheral-favouring", "control": "no campaign (control)",
         "baseline": "baseline", "cap_boost_Q1": "capability boost (Q1)",
         "friction_reduce": "friction reduction (all)",
         "no_amplification": r"no amplification ($\gamma_1=0$)",
         "amplification": r"amplification ($\gamma_1=2$)",
         "G0": r"G0 ($P_i=-1$)", "G1": r"G1 ($P_i=+1$)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "lines.linewidth": 1.6, "figure.dpi": 150,
    "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "dejavuserif",
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", name)


def ribbon(ax, t, mean, ci, key, label=None):
    color, ls = STYLE[key]
    ax.fill_between(t, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)
    ax.plot(t, mean, color=color, ls=ls, label=label or LABEL[key])


def panel_label(ax, s):
    """Prefix the panel letter to the axis title (left-aligned)."""
    ttl = ax.get_title()
    ax.set_title("")
    ax.set_title(r"$\bf{" + s.strip("()") + "}$  " + ttl, loc="left", fontsize=9)


# --------------------------------------------------------------- Scenario A
def fig_scenario_a(res, out):
    df = pd.read_csv(os.path.join(res, "robustness_A_trajectories.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for route in ("central", "peripheral"):
        sub = df[df["route"] == route].sort_values("t")
        ribbon(axes[0], sub["t"], sub["mean_att_mean"], sub["mean_att_ci"], route)
        ribbon(axes[1], sub["t"], sub["mean_strength_mean"], sub["mean_strength_ci"], route)
    if (df["route"] == "control").any():
        sub = df[df["route"] == "control"].sort_values("t")
        ribbon(axes[1], sub["t"], sub["mean_strength_mean"], sub["mean_strength_ci"], "control")
    for ax, ttl, yl in zip(axes, ("Mean attitude direction", "Mean attitude strength"),
                           (r"$\bar{A}(t)$", r"$\bar{S}(t)$")):
        ax.axvline(20, color=C["grey"], ls="--", lw=0.9)
        ax.set_xlabel("step"); ax.set_ylabel(yl); ax.set_title(ttl)
    axes[0].text(20.8, 0.03, "campaign ends", fontsize=7, color=C["grey"])
    axes[0].set_ylim(-0.05, 1.0); axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc="center right", frameon=False)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    save(fig, out, "fig_scenario_a")


# --------------------------------------------------------------- Scenario B
def fig_scenario_b(res, out):
    df = pd.read_csv(os.path.join(res, "robustness_B_trajectories.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.8), sharex=True, sharey=True)
    for ax, q, lab in zip(axes.flat, ("Q1", "Q2", "Q3", "Q4"), ("(a)", "(b)", "(c)", "(d)")):
        for cond in ("baseline", "cap_boost_Q1", "friction_reduce"):
            sub = df[df["condition"] == cond].sort_values("t")
            ribbon(ax, sub["t"], sub[f"beh_rate_{q}_mean"], sub[f"beh_rate_{q}_ci"], cond)
        ax.set_title(f"capability quartile {q}" + (" (lowest)" if q == "Q1" else " (highest)" if q == "Q4" else ""))
        ax.set_ylim(0, 1.03)
        panel_label(ax, lab)
    for ax in axes[1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("behaviour rate")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "fig_scenario_b")


# --------------------------------------------------------------- Scenario C
def fig_scenario_c(res, out):
    df = pd.read_csv(os.path.join(res, "robustness_C_trajectories.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.8))
    for ax, cond, lab in zip(axes[0], ("amplification", "no_amplification"), ("(a)", "(b)")):
        sub = df[df["condition"] == cond].sort_values("t")
        for grp in ("G0", "G1"):
            ribbon(ax, sub["t"], sub[f"mean_att_{grp}_mean"], sub[f"mean_att_{grp}_ci"], grp)
        ax.axhline(0, color=C["grey"], lw=0.8, ls="--")
        ax.set_title("group attitudes, " + ("amplification" if cond == "amplification" else "no amplification"))
        ax.set_ylabel(r"mean $A_i$"); ax.set_ylim(-0.35, 0.45)
        panel_label(ax, lab)
    axes[0, 0].legend(loc="upper left", frameon=False)
    ax = axes[1, 0]
    for cond in ("no_amplification", "amplification"):
        sub = df[df["condition"] == cond].sort_values("t")
        ribbon(ax, sub["t"], sub["att_variance_mean"], sub["att_variance_ci"], cond)
    ax.set_title("attitude variance (polarisation)"); ax.set_ylabel(r"Var$(A_i)$"); ax.set_xlabel("step")
    ax.legend(loc="upper left", frameon=False); panel_label(ax, "(c)")
    ax = axes[1, 1]
    for cond in ("no_amplification", "amplification"):
        sub = df[df["condition"] == cond].sort_values("t")
        sub = sub[sub["t"] > 0]
        ribbon(ax, sub["t"], sub["exposure_segregation_mean"], sub["exposure_segregation_ci"], cond)
    ax.axhline(0.5, color=C["grey"], lw=0.8, ls="--")
    ax.text(1, 0.515, "random mixing (0.5)", fontsize=7, color=C["grey"])
    ax.set_title("exposure segregation")
    ax.set_ylabel("segregation"); ax.set_xlabel("step"); ax.set_ylim(0.1, 0.6)
    panel_label(ax, "(d)")
    fig.tight_layout()
    save(fig, out, "fig_scenario_c")


# --------------------------------------------------------------- gate sweep
def fig_gate_sweep(res, out):
    df = pd.read_csv(os.path.join(res, "gate_sweep_trajectories.csv"))
    df = df[df["gate"] == "hard"]
    thetas = sorted(df["theta"].unique())
    # sequential single-hue ramp (theta is a magnitude)
    ramp = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(thetas)))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.15))
    for th, col in zip(thetas, ramp):
        sub = df[df["theta"] == th].sort_values("t")
        axes[0].plot(sub["t"], sub["beh_rate_mean"], color=col, label=rf"$\theta$={th:.2f}")
        axes[0].fill_between(sub["t"], sub["beh_rate_mean"] - sub["beh_rate_ci"],
                             sub["beh_rate_mean"] + sub["beh_rate_ci"], color=col, alpha=0.15, lw=0)
        s1 = sub[sub["t"] > 0]
        axes[1].plot(s1["t"], s1["feasibility_gap_mean"], color=col)
        axes[2].plot(sub["t"], sub["beh_rate_Q4_mean"] - sub["beh_rate_Q1_mean"], color=col)
    axes[0].set_title("behaviour rate, baseline"); axes[0].set_ylabel(r"$\bar{B}(t)$")
    axes[1].set_title("feasibility gap"); axes[1].set_ylabel("blocked share of high-intent agents", fontsize=8)
    axes[2].set_title("capability gap (Q4 $-$ Q1)"); axes[2].set_ylabel("behaviour-rate difference", fontsize=8)
    for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_xlabel("step"); panel_label(ax, lab)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig_gate_sweep")


# --------------------------------------------------------------- ablation
def fig_ablation(res, out):
    df = pd.read_csv(os.path.join(res, "ablation.csv"))
    order = ["full", "no_route_split", "no_strength", "no_gate", "no_visibility"]
    names = {"full": "full model", "no_route_split": "no route split",
             "no_strength": "no strength layer", "no_gate": "no feasibility gate",
             "no_visibility": "no visibility proxy"}
    panels = [("A", "retention_gap", "A: retention gap\n(central $-$ peripheral)"),
              ("B", "baseline_gap_Q4_Q1_t20", "B: capability gap at step 20\n(Q4 $-$ Q1 behaviour rate)"),
              ("C", "variance_ratio", "C: variance ratio\n(amplification / none)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (sc, metric, title) in zip(axes, panels):
        sub = df[(df["scenario"] == sc) & (df["metric"] == metric)].set_index("ablation").reindex(order)
        ax.errorbar(sub["mean"], y, xerr=sub["ci95_half"], fmt="o", color=C["blue"],
                    ecolor=C["blue"], capsize=2, ms=4)
        ref = sub.loc["full", "mean"]
        ax.axvline(ref, color=C["grey"], ls=":", lw=0.9)
        if metric == "variance_ratio":
            ax.axvline(1.0, color=C["orange"], ls="--", lw=0.9)
        if metric != "variance_ratio":
            ax.axvline(0.0, color=C["orange"], ls="--", lw=0.9)
        ax.set_title(title, fontsize=8.5)
        ax.set_yticks(y); ax.set_yticklabels([names[o] for o in order])
    axes[0].set_xlim(-0.05, 1.0)
    fig.tight_layout()
    save(fig, out, "fig_ablation")


# --------------------------------------------------------------- Morris
PRETTY = {"alpha": r"$\alpha$", "gamma1": r"$\gamma_1$", "rho_load": r"$\rho_L$", "w_load": r"$w_L$",
          "w_nfc": r"$w_{NFC}$", "kappa_c": r"$\kappa_c$", "kappa_p": r"$\kappa_p$", "eta_c": r"$\eta_c$",
          "eta_p": r"$\eta_p$", "b_sp": r"$b_{SP}$", "lambda_s": r"$\lambda_s$",
          "strength_decay": r"$\delta_S$", "theta_intent": r"$\theta_I$", "theta_feas": r"$\theta_{C,O}$",
          "beta_as": r"$\beta_{AS}$", "beta_fric": r"$\beta_F$", "cap_lr": r"$\lambda_C$",
          "opp_lr": r"$\lambda_O$", "norm_mu": r"$\mu_N$", "load_from_exposure": r"$\lambda_L$"}


def fig_morris(res, out):
    targets = [("A", "retention_gap", "A: retention gap"),
               ("B", "baseline_gap_Q4_Q1_t20", "B: capability gap (step 20)"),
               ("C", "variance_ratio", "C: variance ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25))
    for ax, (sc, metric, title) in zip(axes, targets):
        df = pd.read_csv(os.path.join(res, f"morris_{sc}.csv"))
        sub = df[df["metric"] == metric].sort_values("mu_star", ascending=False)
        ax.scatter(sub["mu_star"], sub["sigma"], s=16, color=C["blue"], zorder=3)
        lim = max(sub["mu_star"].max(), sub["sigma"].max()) * 1.1
        ax.plot([0, lim], [0, lim], color=C["grey"], lw=0.7, ls=":")
        for _, r in sub.head(6).iterrows():
            ax.annotate(PRETTY.get(r["parameter"], r["parameter"]), (r["mu_star"], r["sigma"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set_xlabel(r"$\mu^*$ (overall influence)"); ax.set_title(title)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    axes[0].set_ylabel(r"$\sigma$ (interaction / non-linearity)")
    fig.tight_layout()
    save(fig, out, "fig_morris")


# --------------------------------------------------------------- recovery
def fig_recovery(res, out):
    grid = pd.read_csv(os.path.join(res, "recovery_grid.csv"))
    summ = json.load(open(os.path.join(res, "recovery_summary.json")))
    obs = pd.read_csv(os.path.join(res, "recovery_observations.csv"))
    piv = grid.pivot(index="strength_decay", columns="eta_c", values="distance")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    im = ax.pcolormesh(piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy(),
                       cmap="Blues_r", shading="nearest")
    thr = summ["accept_threshold"]
    ax.contour(piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy(), levels=[thr],
               colors=[C["orange"]], linewidths=1.2)
    ax.plot(summ["true_eta_c"], summ["true_strength_decay"], marker="*", ms=11, color=C["yellow"],
            mec="black", mew=0.6, ls="none", label="true value")
    ax.plot(summ["best_eta_c"], summ["best_strength_decay"], marker="o", ms=6, color=C["orange"],
            mec="black", mew=0.6, ls="none", label="best fit")
    ax.set_xlabel(r"central strength gain $\eta_c$"); ax.set_ylabel(r"strength decay $\delta_S$")
    ax.set_title("RMS distance to observed trajectories")
    ax.legend(loc="upper right", frameon=True, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("RMS distance", fontsize=8)
    ax.grid(False)
    ax = axes[1]
    ax.plot(obs["t"], obs["mean_strength_obs"], ".", color=C["grey"], ms=4, label="observed (noisy)")
    ax.plot(obs["t"], obs["mean_strength"], color=C["blue"], lw=1.2, label="generating model (data seed)")
    best = grid.loc[grid["distance"].idxmin()]
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import yaml
        from dataclasses import replace
        from run import load_params
        from audience_dt.calibration import central_route_trajectory
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "example.yaml")))
        params, _, _ = load_params(cfg)
        fit = central_route_trajectory(replace(params, eta_c=float(best["eta_c"]), strength_decay=float(best["strength_decay"])), seed=7)
        ax.plot(fit["t"], fit["mean_strength"], color=C["orange"], lw=1.2, ls="--", label="best fit (simulation seed)")
    except Exception as e:  # pragma: no cover
        print("best-fit overlay skipped:", e)
    ax.set_title(f"strength trajectory; best fit $\\eta_c$={best['eta_c']:.3f}, $\\delta_S$={best['strength_decay']:.4f}",
                 fontsize=8)
    ax.set_xlabel("step"); ax.set_ylabel(r"$\bar{S}(t)$"); ax.legend(frameon=False, fontsize=7)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    fig.tight_layout()
    save(fig, out, "fig_recovery")


FIGS = {"a": fig_scenario_a, "b": fig_scenario_b, "c": fig_scenario_c, "gate": fig_gate_sweep,
        "ablation": fig_ablation, "morris": fig_morris, "recovery": fig_recovery}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for k, fn in FIGS.items():
        if args.only and k not in args.only:
            continue
        try:
            fn(args.results, args.out)
        except FileNotFoundError as e:
            print(f"skip {k}: {e}")


if __name__ == "__main__":
    main()
