"""Shared fixtures: Table 1 default parameters loaded from configs/example.yaml."""
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from audience_dt.models import Params  # noqa: E402


def load_default_params() -> Params:
    with open(os.path.join(ROOT, "configs", "example.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from run import load_params
    p, _, _ = load_params(cfg)
    return p


@pytest.fixture(scope="session")
def params() -> Params:
    return load_default_params()
