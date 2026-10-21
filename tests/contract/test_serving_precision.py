"""Serving carries the explicit input-normalization rounding policy into sessions."""

from pathlib import Path

import pytest

from monolith.backends.metal import load_configs
from monolith.serving.setup import ServingAssets


@pytest.mark.parametrize("commute", [False, True])
def test_serving_precision_is_an_explicit_session_option(commute):
    profile = load_configs()["apple-m3-pro-18c"]
    assets = ServingAssets(
        Path("checkpoint"), Path("pack"), 1024, 1024, profile, commute_norm=commute
    )
    _, options = assets.options(128)
    assert options["commute_norm"] is commute
