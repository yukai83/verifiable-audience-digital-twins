"""Synthetic parameter recovery (identifiability check) for the calibration roadmap.

The conference paper proposes calibrating the model against aggregate
attitude-strength retention curves. Before real data are used, a model should
show that such curves actually carry information about the parameters that
are supposed to be calibrated. This module runs a synthetic experiment:

 1. generate "observed" Scenario A central-route trajectories (mean attitude
    and mean strength) from known parameter values with one seed, and add
    Gaussian observation noise;
 2. evaluate a grid over two target parameters -- the central strength gain
    eta_c and the strength decay delta_S -- with a *different* seed;
 3. score each grid point by the root-mean-square distance between simulated
    and observed trajectories, and report the minimiser and the set of grid
    points inside an ABC-style acceptance quantile.

A compact acceptance region around the true values indicates that the two
parameters are jointly identifiable from the proposed observables; a ridge
would indicate a trade-off that calibration data could not resolve.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Callable, Dict, Tuple

import numpy as np
import networkx as nx
import pandas as pd

from .models import Params, Scenario
from .sim import init_population, simulate
from .metrics import outputs_to_frame
from .scenarios import SCENARIO_A_CENTRAL


def central_route_trajectory(params: Params, seed: int, n: int = 200, n_steps: int = 50,
                             campaign_end_step: int = 20) -> pd.DataFrame:
    """Mean attitude and strength over time for the Scenario A central campaign."""
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=seed)
    traits, states = init_population(n, np.random.default_rng(seed))
    for s in states.values():
        s.load = SCENARIO_A_CENTRAL["load0"]
    spec = SCENARIO_A_CENTRAL
    scen = Scenario(messages_per_step=3, aq_range=spec["aq_range"], sc_range=spec["sc_range"],
                    sp_range=spec["sp_range"], ea_range=spec["ea_range"],
                    xm_range=(1.0, 1.0), campaign_end_step=campaign_end_step)
    out, _ = simulate(g, traits, states, params, scen, n_steps=n_steps,
                      rng=np.random.default_rng(seed))
    return outputs_to_frame(out)[["t", "mean_att", "mean_strength"]]


def make_observations(params: Params, true_values: Dict[str, float], data_seed: int,
                      noise_sd: float = 0.02) -> pd.DataFrame:
    p_true = replace(params, **true_values)
    obs = central_route_trajectory(p_true, seed=data_seed)
    rng = np.random.default_rng(data_seed + 1)
    obs["mean_att_obs"] = obs["mean_att"] + rng.normal(0, noise_sd, len(obs))
    obs["mean_strength_obs"] = obs["mean_strength"] + rng.normal(0, noise_sd, len(obs))
    return obs


def distance(sim: pd.DataFrame, obs: pd.DataFrame) -> float:
    """RMS distance on the mean-strength series only.

    Mean attitude direction does not depend on eta_c or delta_S (Eqs. 14-18
    contain neither), so including it would only add a noise floor to the
    objective; it is kept in the observation file for completeness.
    """
    d_str = np.mean((sim["mean_strength"].to_numpy() - obs["mean_strength_obs"].to_numpy()) ** 2)
    return float(np.sqrt(d_str))


def grid_recovery(params: Params, obs: pd.DataFrame,
                  grid: Dict[str, np.ndarray], sim_seed: int,
                  progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Evaluate the distance on a full factorial grid over two parameters."""
    (n1, g1), (n2, g2) = list(grid.items())
    rows = []
    total = len(g1) * len(g2)
    done = 0
    for v1 in g1:
        for v2 in g2:
            p = replace(params, **{n1: float(v1), n2: float(v2)})
            sim = central_route_trajectory(p, seed=sim_seed)
            rows.append({n1: float(v1), n2: float(v2), "distance": distance(sim, obs),
                         "end_strength": float(sim["mean_strength"].iloc[-1]),
                         "peak_strength": float(sim["mean_strength"].max())})
            done += 1
            if progress:
                progress(done, total)
    return pd.DataFrame(rows)


def summarise_recovery(res: pd.DataFrame, true_values: Dict[str, float],
                       accept_quantile: float = 0.05) -> Dict[str, float]:
    (n1, n2) = [c for c in res.columns if c not in ("distance", "end_strength", "peak_strength")]
    best = res.loc[res["distance"].idxmin()]
    thr = res["distance"].quantile(accept_quantile)
    acc = res[res["distance"] <= thr]
    out = {
        "best_" + n1: float(best[n1]), "best_" + n2: float(best[n2]),
        "true_" + n1: float(true_values[n1]), "true_" + n2: float(true_values[n2]),
        "best_distance": float(best["distance"]),
        "accept_threshold": float(thr), "n_accepted": int(len(acc)),
        "accepted_" + n1 + "_min": float(acc[n1].min()), "accepted_" + n1 + "_max": float(acc[n1].max()),
        "accepted_" + n2 + "_min": float(acc[n2].min()), "accepted_" + n2 + "_max": float(acc[n2].max()),
        "accepted_" + n1 + "_mean": float(acc[n1].mean()), "accepted_" + n2 + "_mean": float(acc[n2].mean()),
    }
    return out
