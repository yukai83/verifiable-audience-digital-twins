"""Invariant and property checks for the verification suite.

Section 6 of the conference paper states three classes of verification
artefact that follow from the specification:

  I1  range invariants      -- every bounded state stays inside its range;
  I2  feasibility gate      -- B_i(t) = 1 only if intention, capability and
                               opportunity all clear their thresholds (Eq. 27);
  I3  exposure monotonicity -- E_{i,m} is non-decreasing in Peer_{i,m} with the
                               other terms fixed, and Peer_{i,m} = 0 whenever
                               no neighbour shared m at t-1 (Eqs. 8, 11).

This module turns those statements into executable checks, and adds the
bookkeeping properties that the specification implies (I4-I7). Every check
returns a :class:`CheckResult` with the number of individual comparisons made
and the number that failed, so the same functions serve both the pytest suite
(``tests/``) and the results table reported in the paper
(``experiments/run_verification.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence

import numpy as np
import networkx as nx

from .models import AgentState, AgentTraits, Message, Params
from .sim import (exposure_prob, identity_congruence, p_central, peer_component,
                  delta_strength, visibility)


@dataclass
class CheckResult:
    name: str
    n_checks: int
    n_failures: int
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.n_failures == 0

    def __add__(self, other: "CheckResult") -> "CheckResult":
        assert self.name == other.name
        return CheckResult(self.name, self.n_checks + other.n_checks,
                           self.n_failures + other.n_failures, self.detail)


# I1 -- range invariants -------------------------------------------------------
RANGES = {
    "cap": (0.0, 1.0), "opp": (0.0, 1.0), "mr": (0.0, 1.0), "ma": (0.0, 1.0),
    "load": (0.0, 1.0), "att": (-1.0, 1.0), "strength": (0.0, 1.0),
    "norm": (0.0, 1.0), "intent": (0.0, 1.0), "beh": (0.0, 1.0), "fr": (0.0, 1.0),
}


def check_ranges(trace: Sequence[dict], tol: float = 1e-12) -> CheckResult:
    """I1: every recorded state lies within its declared range at every step."""
    n = fails = 0
    for rec in trace:
        for field, (lo, hi) in RANGES.items():
            arr = rec["states"][field]
            n += arr.size
            fails += int(np.sum((arr < lo - tol) | (arr > hi + tol) | ~np.isfinite(arr)))
        for key in ("exposure_probs", "pc_values"):
            arr = rec[key]
            n += arr.size
            fails += int(np.sum((arr < -tol) | (arr > 1 + tol)))
        beh = rec["states"]["beh"]
        n += beh.size
        fails += int(np.sum(~np.isin(beh, (0.0, 1.0))))
    return CheckResult("I1 range invariants", n, fails)


# I2 -- feasibility gate -------------------------------------------------------
def check_feasibility_gate(trace: Sequence[dict], params: Params) -> CheckResult:
    """I2: B=1 implies intent>=theta_I, cap>=theta_C and opp>=theta_O.

    Under the default hard gate the converse also holds (the gate is exactly
    the AND of the three threshold tests), so both directions are checked.
    Under the soft gate only the forward implication is meaningful and the
    check is restricted to it.
    """
    n = fails = 0
    hard = getattr(params, "gate_mode", "hard") == "hard"
    for rec in trace:
        s = rec["states"]
        ok = ((s["intent"] >= params.theta_intent) & (s["cap"] >= params.theta_cap)
              & (s["opp"] >= params.theta_opp))
        beh = s["beh"] > 0.5
        n += beh.size
        fails += int(np.sum(beh & ~ok))
        if hard:
            n += beh.size
            fails += int(np.sum(ok & ~beh))
    return CheckResult("I2 feasibility gate", n, fails)


# I3 -- exposure monotonicity ----------------------------------------------------
def check_exposure_monotone(params: Params, rng: np.random.Generator,
                            n_cases: int = 500) -> CheckResult:
    """I3a: E_{i,m} is non-decreasing in the peer-share fraction, other terms fixed.

    Random agents, messages and neighbourhoods are drawn; for each case the
    neighbour share set is grown one neighbour at a time and the exposure
    probability must never fall.
    """
    n = fails = 0
    for _ in range(n_cases):
        k = int(rng.integers(2, 9))
        g = nx.complete_graph(k + 1)         # agent 0 with k neighbours
        tr = _random_traits(rng)
        st = _random_state(rng)
        msg = _random_message(rng, msg_id=0)
        prev = -1.0
        order = rng.permutation(np.arange(1, k + 1))
        for j in range(k + 1):
            shares = {0: list(order[:j])}
            e = exposure_prob(g, shares, 0, st, tr, msg, params)
            n += 1
            if e < prev - 1e-12:
                fails += 1
            prev = e
    return CheckResult("I3a exposure monotone in Peer", n, fails)


def check_peer_zero_without_shares(trace: Sequence[dict]) -> CheckResult:
    """I3b: Peer_{i,m}=0 whenever no neighbour shared m at t-1 (trace-level).

    At t=0 no message has been shared, so every recorded peer fraction must be
    exactly zero; at later steps the peer fraction can only be positive when
    the previous step recorded at least one share.
    """
    n = fails = 0
    prev_shares = 0
    for rec in trace:
        pf = rec["peer_fracs"]
        if prev_shares == 0:           # only steps where the property applies are counted
            n += pf.size
            fails += int(np.sum(pf != 0.0))
        prev_shares = rec["n_shares"]
    return CheckResult("I3b Peer=0 without prior shares", n, fails)


# I4 -- route-probability monotonicity (Eq. 12) ---------------------------------
def check_route_monotone(params: Params, rng: np.random.Generator,
                         n_cases: int = 500) -> CheckResult:
    """I4: p^c increases in NFC, capability and reflective motivation and
    decreases in load, holding everything else fixed (requires the Table 1
    sign pattern w_NFC, w_C, w_R > 0 and w_L > 0)."""
    n = fails = 0
    for _ in range(n_cases):
        tr = _random_traits(rng)
        st = _random_state(rng)
        msg = _random_message(rng, 0)
        base = p_central(st, tr, msg, params)
        eps = 0.1
        # NFC up -> p^c up
        tr_up = replace(tr, nfc=min(1.0, tr.nfc + eps))
        n += 1; fails += int(p_central(st, tr_up, msg, params) < base - 1e-12)
        # cap up -> p^c up
        st_up = replace_state(st, cap=min(1.0, st.cap + eps))
        n += 1; fails += int(p_central(st_up, tr, msg, params) < base - 1e-12)
        # mr up -> p^c up
        n += 1; fails += int(p_central(st, tr, msg, params, mr_star=min(1.0, st.mr + eps)) < base - 1e-12)
        # load up -> p^c down
        st_ld = replace_state(st, load=min(1.0, st.load + eps))
        n += 1; fails += int(p_central(st_ld, tr, msg, params) > base + 1e-12)
    return CheckResult("I4 route monotonicity (Eq. 12)", n, fails)


# I5 -- identity congruence formula (Eq. in Sect. 5.3) ---------------------------
def check_identity_congruence(rng: np.random.Generator, n_cases: int = 500) -> CheckResult:
    """I5: IC = 1 - |P_i - X_m|/2 lies in [0,1], equals 1 iff P_i = X_m and 0
    iff they are at opposite poles."""
    n = fails = 0
    for _ in range(n_cases):
        tr = _random_traits(rng)
        msg = _random_message(rng, 0)
        ic = identity_congruence(tr, msg)
        n += 1; fails += int(not (0.0 - 1e-12 <= ic <= 1.0 + 1e-12))
        n += 1; fails += int(abs(ic - (1 - abs(tr.pi - msg.xm) / 2)) > 1e-12)
    for pi, xm, expect in ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)):
        tr = AgentTraits(0.5, 0.5, 0.5, 0.5, pi)
        msg = Message(0, 0.5, 0.5, 0.5, 0.0, 0.5, xm)
        n += 1; fails += int(abs(identity_congruence(tr, msg) - expect) > 1e-12)
    return CheckResult("I5 identity congruence formula", n, fails)


# I6 -- strength-gain ordering (Eqs. 18-20) --------------------------------------
def check_strength_route_ordering(params: Params, rng: np.random.Generator,
                                  n_cases: int = 500) -> CheckResult:
    """I6: for a first exposure (rep=0) to a message with AQ >= SP, the central
    strength gain is at least the peripheral gain whenever eta_c >= eta_p*b_SP,
    so dS is non-decreasing in p^c. This is the mechanism behind Scenario A."""
    n = fails = 0
    if not (params.eta_c >= params.eta_p * params.b_sp):
        return CheckResult("I6 strength gain ordering", 0, 0, "precondition not met")
    for _ in range(n_cases):
        st = _random_state(rng)
        tr = _random_traits(rng)
        msg = _random_message(rng, 0)
        if msg.aq < msg.sp:
            continue
        d_lo = delta_strength(st, tr, msg, 0.0, params)
        d_hi = delta_strength(st, tr, msg, 1.0, params)
        n += 1; fails += int(d_hi < d_lo - 1e-12)
    return CheckResult("I6 strength gain ordering", n, fails)


# I7 -- bookkeeping: shares subset of processed, rep counts consistent -----------
def check_bookkeeping(trace: Sequence[dict]) -> CheckResult:
    """I7: shares never exceed processed exposures in a step, and the total
    repetition count equals the cumulative number of processed exposures."""
    n = fails = 0
    cum = 0
    for rec in trace:
        cum += rec["n_processed"]
        n += 1; fails += int(rec["n_shares"] > rec["n_processed"])
        n += 1; fails += int(rec["rep_total"] != cum)
    return CheckResult("I7 share/repetition bookkeeping", n, fails)


# I8 -- visibility monotone in engagement when gamma1 > 0 ----------------------
def check_visibility_monotone(params: Params, rng: np.random.Generator,
                              n_cases: int = 500) -> CheckResult:
    """I8: with gamma1 > 0, Vis is non-decreasing in SP, SC and EA and
    non-increasing in load (Eqs. 9-10 with the Table 1 sign pattern)."""
    n = fails = 0
    if params.gamma1 <= 0:
        return CheckResult("I8 visibility monotone in cues", 0, 0, "gamma1 <= 0")
    for _ in range(n_cases):
        st = _random_state(rng); tr = _random_traits(rng); msg = _random_message(rng, 0)
        base = visibility(st, tr, msg, params)
        for fld in ("sp", "sc", "ea"):
            m_up = replace(msg, **{fld: min(1.0, getattr(msg, fld) + 0.1)})
            n += 1; fails += int(visibility(st, tr, m_up, params) < base - 1e-12)
        st_ld = replace_state(st, load=min(1.0, st.load + 0.1))
        n += 1; fails += int(visibility(st_ld, tr, msg, params) > base + 1e-12)
    return CheckResult("I8 visibility monotone in cues", n, fails)


# helpers ------------------------------------------------------------------------
def replace_state(st: AgentState, **kw) -> AgentState:
    d = dict(cap=st.cap, opp=st.opp, mr=st.mr, ma=st.ma, load=st.load, att=st.att,
             strength=st.strength, norm=st.norm, intent=st.intent, beh=st.beh,
             fr=st.fr, rep=dict(st.rep))
    d.update(kw)
    return AgentState(**d)


def _random_traits(rng) -> AgentTraits:
    return AgentTraits(nfc=float(rng.uniform(0, 1)), trust_inst=float(rng.uniform(0, 1)),
                       trust_peer=float(rng.uniform(0, 1)),
                       identity_salience=float(rng.uniform(0, 1)),
                       pi=float(rng.uniform(-1, 1)))


def _random_state(rng) -> AgentState:
    return AgentState(cap=float(rng.uniform(0, 1)), opp=float(rng.uniform(0, 1)),
                      mr=float(rng.uniform(0, 1)), ma=float(rng.uniform(0, 1)),
                      load=float(rng.uniform(0, 1)), att=float(rng.uniform(-1, 1)),
                      strength=float(rng.uniform(0, 1)), norm=float(rng.uniform(0, 1)),
                      intent=0.0, beh=0, fr=float(rng.uniform(0, 1)), rep={})


def _random_message(rng, msg_id: int) -> Message:
    return Message(msg_id=msg_id, aq=float(rng.uniform(0, 1)), sc=float(rng.uniform(0, 1)),
                   sp=float(rng.uniform(0, 1)), ev=float(rng.uniform(-1, 1)),
                   ea=float(rng.uniform(0, 1)), xm=float(rng.uniform(-1, 1)),
                   involve=float(rng.uniform(0, 1)))


def random_params(base: Params, rng: np.random.Generator, rel: float = 0.5) -> Params:
    """Perturb every numeric parameter of ``base`` by a uniform relative factor
    in [1-rel, 1+rel] (sign preserved). Used to check that the invariants hold
    away from the Table 1 defaults, not only at them."""
    kw = {}
    for name, val in vars(base).items():
        if isinstance(val, (int, float)) and not isinstance(val, bool) and name not in ("pc_fixed", "gate_temp"):
            if val == 0.0:
                kw[name] = float(rng.uniform(-rel, rel))
            else:
                kw[name] = float(val * rng.uniform(1 - rel, 1 + rel))
    # thresholds and tau_share must stay probabilities
    for name in ("theta_intent", "theta_cap", "theta_opp", "alpha"):
        kw[name] = float(np.clip(kw[name], 0.0, 1.0))
    return replace(base, **kw)


def trace_checks(trace: Sequence[dict], params: Params) -> List[CheckResult]:
    """All trace-level checks for one simulated run."""
    return [check_ranges(trace), check_feasibility_gate(trace, params),
            check_peer_zero_without_shares(trace), check_bookkeeping(trace)]


def unit_checks(params: Params, rng: np.random.Generator, n_cases: int = 500) -> List[CheckResult]:
    """All function-level (unit) checks for one parameter vector."""
    return [check_exposure_monotone(params, rng, n_cases), check_route_monotone(params, rng, n_cases),
            check_identity_congruence(rng, n_cases), check_strength_route_ordering(params, rng, n_cases),
            check_visibility_monotone(params, rng, n_cases)]
