"""Print the Morris top-k rows used in the extended paper's Table (LaTeX cells)."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETTY = {"alpha": r"$\alpha$", "gamma1": r"$\gamma_1$", "rho_load": r"$\rho_L$", "w_load": r"$w_L$",
          "w_nfc": r"$w_{NFC}$", "kappa_c": r"$\kappa_c$", "kappa_p": r"$\kappa_p$", "eta_c": r"$\eta_c$",
          "eta_p": r"$\eta_p$", "b_sp": r"$b_{SP}$", "lambda_s": r"$\lambda_s$",
          "strength_decay": r"$\delta_S$", "theta_intent": r"$\theta_I$", "theta_feas": r"$\theta_{C,O}$",
          "beta_as": r"$\beta_{AS}$", "beta_fric": r"$\beta_F$", "cap_lr": r"$\lambda_C$",
          "opp_lr": r"$\lambda_O$", "norm_mu": r"$\mu_N$", "load_from_exposure": r"$\lambda_L$"}
TARGETS = {"A": "retention_gap", "B": "baseline_gap_Q4_Q1_t20", "C": "variance_ratio"}


def main(k=5, results=os.path.join(ROOT, "results")):
    for sc, metric in TARGETS.items():
        df = pd.read_csv(os.path.join(results, f"morris_{sc}.csv"))
        sub = df[df["metric"] == metric].sort_values("mu_star", ascending=False)
        cells = []
        for _, r in sub.head(k).iterrows():
            sign = "+" if r["mu"] > 0 else "$-$"
            cells.append(f"{PRETTY.get(r['parameter'], r['parameter'])} {r['mu_star']:.2f} ({r['sigma']:.2f}, {sign})")
        print(f"{sc}: " + "; ".join(cells))
        print(sub[["parameter", "mu", "mu_star", "sigma"]].head(10).round(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
