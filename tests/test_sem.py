"""PLS-SEM estimation: structures, outputs and failure modes."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from goldbench import BenchmarkSpec, fit
from goldbench.metrics import max_htmt


def test_fit_recovers_strong_loadings_for_clean_tasks(scores, spec):
    result = fit(scores, spec)
    for task in spec.indicators:
        assert result.loadings[task] > 0.75, task
        assert result.vifs[task] < 5.0, task


def test_fit_flags_the_planted_defects(scores, flawed_spec):
    result = fit(scores, flawed_spec)
    assert result.vifs["p_dup"] > 5.0
    assert result.vifs["p1"] > 5.0  # its donor is equally inflated
    assert abs(result.loadings["r_weak"]) < 0.75


def test_loading_table_reports_threshold_verdicts(scores, flawed_spec):
    table = fit(scores, flawed_spec).loading_table().set_index("indicator")
    assert table.loc["r_weak", "loading_ok"] is np.False_ or not table.loc["r_weak", "loading_ok"]
    assert not table.loc["p_dup", "vif_ok"]
    assert table.loc["m1", "vif_ok"] and table.loc["m1", "loading_ok"]
    assert set(table.columns) == {
        "construct",
        "outer_loading",
        "outer_weight",
        "VIF",
        "loading_ok",
        "vif_ok",
    }


def test_diagnostics_are_in_range(scores, spec):
    diagnostics = fit(scores, spec).diagnostics
    assert 0.0 <= diagnostics["TC"] <= 1.0
    assert 0.0 <= diagnostics["D_div"] <= 1.0
    assert 0.0 < diagnostics["D_valid"] <= 1.0
    assert diagnostics["task_num"] == len(spec.indicators)


def test_htmt_matrix_is_square_and_symmetric(scores, spec):
    htmt = fit(scores, spec).htmt
    assert list(htmt.index) == spec.construct_names
    assert list(htmt.columns) == spec.construct_names
    values = htmt.to_numpy()
    assert np.allclose(values, values.T, equal_nan=True)
    assert 0.0 < max_htmt(htmt) < 1.0


def test_reliability_table_covers_every_construct(scores, spec):
    reliability = fit(scores, spec).reliability
    assert list(reliability.index) == spec.construct_names
    for column in ("cronbach_alpha", "rho_A", "composite_reliability", "AVE"):
        assert reliability[column].notna().all(), column
    assert (reliability["AVE"] > 0.5).all()
    assert (reliability["composite_reliability"] > 0.8).all()


def test_task_loadings_exclude_the_human_anchor(scores, spec):
    result = fit(scores, spec)
    assert "arena_elo" in result.loadings
    assert "arena_elo" not in result.task_loadings
    assert set(result.task_loadings) == set(spec.indicators)


@pytest.mark.parametrize("scheme", ["path", "centroid", "factorial"])
def test_all_inner_schemes_run(scores, spec, scheme):
    result = fit(scores, spec, scheme=scheme)
    assert result.diagnostics["TC"] > 0.75


def test_unknown_scheme_is_rejected(scores, spec):
    with pytest.raises(ValueError, match="unknown scheme"):
        fit(scores, spec, scheme="quantum")


def test_anchor_structure_regresses_human_preference_on_the_constructs(scores, spec):
    result = fit(scores, spec)
    assert result.r_squared["HumanPref"] > 0.5
    assert "HumanPref" in result.scores.columns


def test_chain_structure_orders_constructs_developmentally(scores, spec):
    chained = replace(spec, structure="chain", human_column=None)
    result = fit(scores, chained)
    # Perception is exogenous, Reasoning is predicted by both earlier layers.
    assert result.r_squared["Perception"] == pytest.approx(0.0)
    assert result.r_squared["Reasoning"] > 0.0
    assert "HumanPref" not in result.scores.columns


def test_two_stage_hoc_produces_a_second_order_score(scores, spec):
    hoc = replace(spec, structure="hoc_two_stage")
    result = fit(scores, hoc)
    assert "Gold" in result.scores.columns
    # First-order constructs are reported as indicators of the second-order construct.
    for construct in spec.construct_names:
        assert result.loadings[construct] > 0.5


def test_two_stage_hoc_works_without_a_human_column(scores, spec):
    hoc = replace(spec, structure="hoc_two_stage", human_column=None)
    result = fit(scores, hoc)
    assert "Gold" in result.scores.columns
    assert np.isfinite(result.scores["Gold"]).all()


def test_missing_columns_are_reported(scores, spec):
    broken = replace(
        spec, constructs={**spec.constructs, "Perception": ["p1", "absent_task"]}
    )
    with pytest.raises(ValueError, match="missing required columns"):
        fit(scores, broken)


def test_constant_column_is_rejected_with_guidance(scores, spec):
    saturated = scores.copy()
    saturated["p1"] = 100.0
    with pytest.raises(ValueError, match="zero variance"):
        fit(saturated, saturated_spec := spec)
    assert saturated_spec is spec  # fit must not mutate the spec


def test_benchmark_scope_vif_exceeds_construct_scope(scores, spec):
    within = fit(scores, spec, vif_scope="construct")
    across = fit(scores, spec, vif_scope="benchmark")
    assert across.vifs["p1"] > within.vifs["p1"]
    # Loadings come from the same measurement model either way.
    assert within.loadings["p1"] == pytest.approx(across.loadings["p1"])


def test_srmr_is_small_for_a_well_specified_model(scores, spec):
    assert 0.0 <= fit(scores, spec).srmr < 0.12


def test_mixed_score_scales_do_not_distort_loadings(scores, spec):
    """Indicators are standardised internally, so rescaling one task is a no-op."""
    baseline = fit(scores, spec)
    rescaled = scores.copy()
    rescaled["p1"] = rescaled["p1"] * 100.0 + 5000.0
    shifted = fit(rescaled, spec)
    assert shifted.loadings["p1"] == pytest.approx(baseline.loadings["p1"], abs=1e-6)
    assert shifted.diagnostics["TC"] == pytest.approx(baseline.diagnostics["TC"], abs=1e-6)


def test_latent_scores_are_standardised(scores, spec):
    result = fit(scores, spec)
    for construct in spec.construct_names:
        column = result.scores[construct]
        assert column.mean() == pytest.approx(0.0, abs=1e-6)
        assert column.std(ddof=1) == pytest.approx(1.0, abs=0.02)
