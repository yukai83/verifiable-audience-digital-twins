"""Global sensitivity screening with the Morris elementary-effects method.

The conference version reported a one-at-a-time local sensitivity sweep
(``extensions.local_sensitivity``). This module adds the Morris method
(Morris 1991; Campolongo et al. 2007), which screens the whole parameter box
rather than a neighbourhood of the defaults. For each trajectory a random
point on a p-level grid is drawn and each parameter is moved once by +/-Delta
in random order; the elementary effect of parameter j is the change in the
output divided by Delta. Reported per parameter and output:

  mu*    mean absolute elementary effect  (overall importance)
  mu     mean elementary effect           (sign of the effect)
  sigma  std of elementary effects        (interactions / non-linearity)

The implementation is self-contained (no SALib dependency) so that the whole
pipeline stays inspectable. All runs within a trajectory use the same random
seed (common random numbers), so the elementary effects measure parameter
influence rather than Monte-Carlo noise; seed-to-seed variability is reported
separately by the multi-seed robustness analysis.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import Params

ParamBox = List[Tuple[str, float, float]]


def default_param_box(params: Params, rel: float = 0.5) -> ParamBox:
    """A +/-rel box around the Table 1 defaults for the parameters that the
    conference paper flags as governing the three diagnostics, plus the
    exposure and slow-dynamics parameters that interact with them.

    ``theta_feas`` is a pseudo-parameter that moves theta_C and theta_O
    together so that the feasibility gate is screened as one factor.
    """
    names = ["alpha", "gamma1", "rho_load", "w_load", "w_nfc", "kappa_c", "kappa_p",
             "eta_c", "eta_p", "b_sp", "lambda_s", "strength_decay", "theta_intent",
             "theta_feas", "beta_as", "beta_fric", "cap_lr", "opp_lr", "norm_mu",
             "load_from_exposure"]
    box: ParamBox = []
    for n in names:
        base = params.theta_cap if n == "theta_feas" else getattr(params, n)
        lo, hi = sorted((base * (1 - rel), base * (1 + rel)))
        if n in ("alpha", "theta_intent", "theta_feas"):
            lo, hi = max(0.0, lo), min(1.0, hi)
        box.append((n, float(lo), float(hi)))
    return box


def apply_point(params: Params, box: ParamBox, x_unit: np.ndarray) -> Params:
    """Map a point in the unit hypercube to a Params instance."""
    kw = {}
    for (name, lo, hi), u in zip(box, x_unit):
        val = lo + u * (hi - lo)
        if name == "theta_feas":
            kw["theta_cap"] = float(val)
            kw["theta_opp"] = float(val)
        else:
            kw[name] = float(val)
    return replace(params, **kw)


def morris_trajectories(k: int, r: int, p: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Return ``r`` trajectories, each an array of shape (k+1, k) of unit-cube
    points where consecutive rows differ in exactly one coordinate by Delta."""
    delta = p / (2.0 * (p - 1))
    grid = np.arange(p) / (p - 1)
    trajs = []
    for _ in range(r):
        # base point restricted so that +Delta stays inside [0, 1]
        base = rng.choice(grid[grid <= 1 - delta + 1e-12], size=k)
        order = rng.permutation(k)
        pts = [base.copy()]
        x = base.copy()
        for j in order:
            x = x.copy()
            # move up if possible, otherwise down (keeps the point inside the cube)
            x[j] = x[j] + delta if x[j] + delta <= 1 + 1e-12 else x[j] - delta
            pts.append(x)
        trajs.append(np.vstack(pts))
    return trajs


def run_morris(params: Params,
               box: ParamBox,
               scenario_fn: Callable[[Params, int], pd.DataFrame],
               metric_fn: Callable[[pd.DataFrame], Dict[str, float]],
               r: int = 12, p: int = 4, seed: int = 42, traj_seed: int = 0,
               progress: Callable[[int, int], None] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Morris design and return (summary, raw_runs).

    ``summary`` has one row per (parameter, metric) with mu, mu_star, sigma and
    the number of elementary effects; ``raw_runs`` stores every evaluated point
    and metric so that other outputs can be analysed afterwards.
    """
    k = len(box)
    delta = p / (2.0 * (p - 1))
    rng = np.random.default_rng(traj_seed)
    trajs = morris_trajectories(k, r, p, rng)
    total = r * (k + 1)
    done = 0
    raw_rows = []
    ee: Dict[str, Dict[str, List[float]]] = {}   # metric -> param -> list of EE

    for ti, traj in enumerate(trajs):
        ys = []
        for row in traj:
            m = metric_fn(scenario_fn(apply_point(params, box, row), seed))
            ys.append(m)
            rec = {"trajectory": ti, **{f"x_{n}": float(v) for (n, _, _), v in zip(box, row)}}
            rec.update({f"y_{kk}": float(vv) for kk, vv in m.items()})
            raw_rows.append(rec)
            done += 1
            if progress:
                progress(done, total)
        for step in range(k):
            diff = traj[step + 1] - traj[step]
            j = int(np.argmax(np.abs(diff)))
            sign = np.sign(diff[j])
            for metric in ys[0].keys():
                y0, y1 = ys[step][metric], ys[step + 1][metric]
                if np.isnan(y0) or np.isnan(y1):
                    continue
                ee.setdefault(metric, {}).setdefault(box[j][0], []).append(sign * (y1 - y0) / delta)

    rows = []
    for metric, per_param in ee.items():
        for name, vals in per_param.items():
            arr = np.asarray(vals, dtype=float)
            rows.append({"metric": metric, "parameter": name, "mu": float(arr.mean()),
                         "mu_star": float(np.abs(arr).mean()),
                         "sigma": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                         "n_ee": int(arr.size)})
    summary = pd.DataFrame(rows)
    return summary, pd.DataFrame(raw_rows)
