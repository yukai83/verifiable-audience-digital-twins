"""Regression tests for the qualitative claims of the paper at the defaults.

These guard the headline behavioural signatures: if a future change to the
model breaks one of them, the test names the claim that no longer holds.
"""
from dataclasses import replace

from audience_dt.scenarios import (run_scenario_a, run_scenario_b, run_scenario_c,
                                   scenario_a_metrics, scenario_b_metrics, scenario_c_metrics)


def test_claim_route_dependent_durability(params):
    m = scenario_a_metrics(run_scenario_a(params, seed=42))
    assert m["central_retention"] > 0.5
    assert m["peripheral_retention"] < 0.2
    assert m["central_end_att"] > m["peripheral_end_att"]


def test_claim_feasibility_gating(params):
    m = scenario_b_metrics(run_scenario_b(params, seed=42))
    # lowest-capability quartile lags the highest at step 20 in the baseline
    assert m["baseline_gap_Q4_Q1_t20"] > 0
    # the targeted boost helps Q1 more than Q4
    assert m["boost_effect_Q1_auc"] > m["boost_effect_Q4_auc"]


def test_claim_amplification_increases_variance_and_segregation(params):
    m = scenario_c_metrics(run_scenario_c(params, seed=42))
    assert m["variance_ratio"] > 1.0
    assert m["segregation_diff"] > 0


def test_ablation_switches_change_behaviour(params):
    frozen = scenario_a_metrics(run_scenario_a(replace(params, strength_mode="frozen"), seed=42))
    assert abs(frozen["retention_gap"]) < 1e-9     # no strength layer -> no durability gap
    m_gate = scenario_b_metrics(run_scenario_b(replace(params, gate_mode="none"), seed=42))
    base = scenario_b_metrics(run_scenario_b(params, seed=42))
    # Without the gate, behaviour is intention-driven: cumulative uptake can only
    # be higher, and the Q4-Q1 capability gap at step 20 should shrink.
    assert m_gate["baseline_auc"] >= base["baseline_auc"]
    assert m_gate["baseline_gap_Q4_Q1_t20"] <= base["baseline_gap_Q4_Q1_t20"]
