"""I1-I3b, I7: trace-level invariants on real simulation runs.

The checks run on the three paper scenarios at the Table 1 defaults and on
short base-model runs with randomly perturbed parameter vectors (+/-50%), so
that the invariants are shown to hold by construction rather than only at the
reference configuration.
"""
import copy

import numpy as np
import networkx as nx
import pytest

from audience_dt.invariants import (check_bookkeeping, check_feasibility_gate,
                                    check_peer_zero_without_shares, check_ranges,
                                    random_params)
from audience_dt.models import Scenario
from audience_dt.scenarios import run_scenario_a, run_scenario_b, run_scenario_c
from audience_dt.sim import init_population, simulate


def _assert_ok(res):
    assert res.n_checks > 0, res.name
    assert res.n_failures == 0, f"{res.name}: {res.n_failures}/{res.n_checks} failed"


@pytest.mark.parametrize("runner", [run_scenario_a, run_scenario_b, run_scenario_c])
def test_scenario_traces_satisfy_invariants(params, runner):
    traces = {}
    runner(params, seed=42, trace=traces)
    for name, tr in traces.items():
        assert len(tr) > 0, name
        _assert_ok(check_ranges(tr))
        _assert_ok(check_feasibility_gate(tr, params))
        _assert_ok(check_peer_zero_without_shares(tr))
        _assert_ok(check_bookkeeping(tr))


@pytest.mark.parametrize("draw", range(10))
def test_random_parameter_vectors_satisfy_invariants(params, draw):
    rng = np.random.default_rng(1000 + draw)
    p = random_params(params, rng, rel=0.5)
    n = 60
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=draw)
    traits, states = init_population(n, rng)
    tr = []
    simulate(g, traits, states, p, Scenario(messages_per_step=3), n_steps=20,
             rng=rng, trace=tr)
    _assert_ok(check_ranges(tr))
    _assert_ok(check_feasibility_gate(tr, p))
    _assert_ok(check_peer_zero_without_shares(tr))
    _assert_ok(check_bookkeeping(tr))


def test_soft_gate_forward_implication_only(params):
    """Under the soft gate B=1 still requires nothing beyond intent, cap, opp
    being probabilistically favourable; the forward implication is not exact,
    so the check must restrict itself to range and bookkeeping invariants."""
    from dataclasses import replace
    p = replace(params, gate_mode="soft", gate_temp=0.05)
    rng = np.random.default_rng(3)
    n = 60
    g = nx.watts_strogatz_graph(n=n, k=6, p=0.05, seed=3)
    traits, states = init_population(n, rng)
    tr = []
    simulate(g, traits, states, p, Scenario(), n_steps=15, rng=rng, trace=tr)
    _assert_ok(check_ranges(tr))
    _assert_ok(check_bookkeeping(tr))
