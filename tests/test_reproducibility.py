"""Determinism: identical seeds give identical outputs; different seeds differ."""
import numpy as np
import pandas as pd
import pytest

from audience_dt.scenarios import run_scenario_a, run_scenario_b, run_scenario_c


@pytest.mark.parametrize("runner", [run_scenario_a, run_scenario_b, run_scenario_c])
def test_same_seed_is_bitwise_reproducible(params, runner):
    a = runner(params, seed=11)
    b = runner(params, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ(params):
    a = run_scenario_a(params, seed=1)
    b = run_scenario_a(params, seed=2)
    assert not np.allclose(a["mean_att"].to_numpy(), b["mean_att"].to_numpy())
