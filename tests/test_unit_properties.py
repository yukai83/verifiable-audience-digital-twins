"""I3a, I4, I5, I6, I8: function-level properties of the update rules."""
import numpy as np
import pytest

from audience_dt.invariants import (check_exposure_monotone, check_identity_congruence,
                                    check_route_monotone, check_strength_route_ordering,
                                    check_visibility_monotone, random_params)


CHECKS = [check_exposure_monotone, check_route_monotone, check_strength_route_ordering,
          check_visibility_monotone]


@pytest.mark.parametrize("check", CHECKS)
def test_property_at_defaults(params, check):
    res = check(params, np.random.default_rng(0), n_cases=500)
    assert res.n_checks > 0
    assert res.n_failures == 0, res


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("draw", range(5))
def test_property_under_perturbed_params(params, check, draw):
    rng = np.random.default_rng(500 + draw)
    p = random_params(params, rng, rel=0.5)
    res = check(p, rng, n_cases=200)
    assert res.n_failures == 0, res


def test_identity_congruence_formula():
    res = check_identity_congruence(np.random.default_rng(0), n_cases=500)
    assert res.n_failures == 0, res
