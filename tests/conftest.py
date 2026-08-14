"""Shared fixtures: a synthetic score matrix with known, plantable defects."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldbench import BenchmarkSpec  # noqa: E402
from goldbench.config import Thresholds  # noqa: E402


@pytest.fixture
def scores() -> pd.DataFrame:
    """190 models x 9 clean tasks + 1 redundant + 1 weak + a human-preference column.

    ``p_dup`` is a near-copy of ``p1`` (high VIF); ``r_weak`` is mostly noise (low
    loading). Everything else clears both thresholds comfortably.
    """
    rng = np.random.default_rng(20260814)
    n = 190
    general = rng.normal(size=n)
    latent = {
        name: 0.55 * general + 0.84 * rng.normal(size=n)
        for name in ("Perception", "Memory", "Reasoning")
    }

    columns = {}
    for prefix, construct in (("p", "Perception"), ("m", "Memory"), ("r", "Reasoning")):
        for index in (1, 2, 3):
            signal = latent[construct] + 0.6 * rng.normal(size=n)
            columns[f"{prefix}{index}"] = 70 + 12 * signal

    columns["p_dup"] = 0.97 * columns["p1"] + 0.03 * rng.normal(size=n) * 12
    columns["r_weak"] = 70 + 12 * (0.2 * latent["Reasoning"] + 0.98 * rng.normal(size=n))
    columns["arena_elo"] = 1100 + 90 * np.mean(list(latent.values()), axis=0)

    frame = pd.DataFrame(columns, index=[f"model_{i:03d}" for i in range(n)])
    frame.index.name = "model"
    return frame


@pytest.fixture
def spec() -> BenchmarkSpec:
    """Clean three-construct spec anchored on the human-preference column."""
    return BenchmarkSpec(
        constructs={
            "Perception": ["p1", "p2", "p3"],
            "Memory": ["m1", "m2", "m3"],
            "Reasoning": ["r1", "r2", "r3"],
        },
        human_column="arena_elo",
        structure="anchor",
        top_name="Gold",
        name="test-gold",
    )


@pytest.fixture
def flawed_spec(spec) -> BenchmarkSpec:
    """Same spec plus the two planted defects the pipeline is expected to remove."""
    from dataclasses import replace

    return replace(
        spec,
        constructs={
            "Perception": ["p1", "p2", "p3", "p_dup"],
            "Memory": ["m1", "m2", "m3"],
            "Reasoning": ["r1", "r2", "r3", "r_weak"],
        },
        thresholds=Thresholds(vif_max=5.0, loading_min=0.75, min_indicators=2),
        name="test-flawed",
    )
