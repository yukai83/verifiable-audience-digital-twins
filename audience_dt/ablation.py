"""Component ablations of the full COM-B + ELM model.

Each ablation switches off one architectural component while leaving the
rest of the model and the scenario inputs unchanged, so that the contribution
of that component to each diagnostic can be read off directly:

  full            the reference model (Table 1 defaults)
  no_route_split  route_mode="fixed": p^c is a constant 0.5 for every exposure
                  (ELM route selection, Eq. 12, removed)
  no_strength     strength_mode="frozen": attitude strength keeps its initial
                  value (ELM strength layer, Eqs. 18-21 and Eq. 3, removed)
  no_gate         gate_mode="none": behaviour fires on intention alone
                  (COM-B feasibility gate, Eq. 27, removed)
  no_visibility   gamma1=0: exposure no longer depends on engagement
                  (visibility proxy, Eqs. 9-10, removed)

The diffusion-plus-sentiment baseline from the conference version is kept as
an external reference point for Scenario A (``extensions.run_diffusion_sentiment``).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from .models import Params
from .extensions import _summarise, run_diffusion_sentiment
from .scenarios import (run_scenario_a, run_scenario_b, run_scenario_c,
                        scenario_a_metrics, scenario_b_metrics, scenario_c_metrics)

ABLATIONS: Dict[str, dict] = {
    "full": {},
    "no_route_split": {"route_mode": "fixed", "pc_fixed": 0.5},
    "no_strength": {"strength_mode": "frozen"},
    "no_gate": {"gate_mode": "none"},
    "no_visibility": {"gamma1": 0.0},
}

# metric names reported per scenario in the paper table
REPORTED = {
    "A": ["central_retention", "peripheral_retention", "retention_gap", "end_att_gap"],
    "B": ["baseline_gap_Q4_Q1_t20", "baseline_Q1_t50", "baseline_Q4_t50",
          "boost_effect_Q1_auc", "boost_effect_Q4_auc", "friction_effect_Q1_auc",
          "friction_effect_Q4_auc", "baseline_feasgap_mean"],
    "C": ["no_amplification_var_end", "amplification_var_end", "variance_ratio",
          "segregation_diff", "divergence_diff"],
}

SCENARIOS = {
    "A": (run_scenario_a, scenario_a_metrics),
    "B": (run_scenario_b, scenario_b_metrics),
    "C": (run_scenario_c, scenario_c_metrics),
}


def run_ablations(params: Params, seeds: Sequence[int],
                  scenarios: Sequence[str] = ("A", "B", "C"),
                  ablations: Dict[str, dict] | None = None,
                  progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Run every (scenario, ablation, seed) and return a long table of metrics
    together with across-seed mean, sd and 95% CI."""
    ablations = ablations or ABLATIONS
    rows = []
    for sc in scenarios:
        fn, mfn = SCENARIOS[sc]
        for abl, kw in ablations.items():
            p = replace(params, **kw)
            # In Scenario C the "no_visibility" condition already exists inside the
            # scenario (gamma1 = 0 vs 2); the ablation sets the "on" level to 0 too.
            per_seed: Dict[str, List[float]] = {}
            for s in seeds:
                if sc == "C" and abl == "no_visibility":
                    df = fn(p, seed=int(s), gamma1_on=0.0)
                else:
                    df = fn(p, seed=int(s))
                for k, v in mfn(df).items():
                    per_seed.setdefault(k, []).append(float(v))
            for k, vals in per_seed.items():
                summ = _summarise(vals)
                rows.append({"scenario": sc, "ablation": abl, "metric": k, **summ})
            if progress:
                progress(f"{sc}/{abl}")
    out = pd.DataFrame(rows)

    # external baseline for Scenario A retention
    if "A" in scenarios:
        rets = []
        for s in seeds:
            b = run_diffusion_sentiment(params, seed=int(s))
            st = b["mean_strength"].to_numpy()
            rets.append(float(st[-1] / st[1:].max()))
        summ = _summarise(rets)
        out = pd.concat([out, pd.DataFrame([{"scenario": "A", "ablation": "diffusion_sentiment",
                                             "metric": "retention", **summ}])], ignore_index=True)
    return out
