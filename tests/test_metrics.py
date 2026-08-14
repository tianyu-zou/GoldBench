"""Metric correctness: each statistic is checked against a closed form or a reference."""

import numpy as np
import pandas as pd
import pytest

from goldbench.metrics import (
    average_variance_extracted,
    composite_reliability,
    cronbach_alpha,
    dimensional_diversity,
    htmt_matrix,
    indicator_validity,
    max_htmt,
    overall_score,
    srmr,
    summarise,
    task_contribution,
    vif_scores,
)


@pytest.fixture
def correlated_block():
    rng = np.random.default_rng(7)
    n = 300
    factor = rng.normal(size=n)
    return pd.DataFrame(
        {f"x{i}": factor * 0.8 + rng.normal(size=n) * 0.6 for i in range(4)}
    )


def test_vif_matches_statsmodels(correlated_block):
    statsmodels = pytest.importorskip("statsmodels.stats.outliers_influence")
    design = np.column_stack([np.ones(len(correlated_block)), correlated_block.to_numpy()])
    reference = [
        statsmodels.variance_inflation_factor(design, i)
        for i in range(1, correlated_block.shape[1] + 1)
    ]
    computed = vif_scores(correlated_block, {"A": list(correlated_block.columns)})
    assert np.allclose([computed[c] for c in correlated_block.columns], reference)


def test_vif_of_single_indicator_is_one():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    assert vif_scores(frame, {"A": ["a"]}) == {"a": 1.0}


def test_vif_of_duplicate_column_is_infinite():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.5], "b": [1.0, 2.0, 3.0, 4.5]})
    assert vif_scores(frame, {"A": ["a", "b"]})["a"] == float("inf")


def test_vif_construct_scope_ignores_other_constructs(correlated_block):
    constructs = {"A": ["x0", "x1"], "B": ["x2", "x3"]}
    within = vif_scores(correlated_block, constructs, scope="construct")
    across = vif_scores(correlated_block, constructs, scope="benchmark")
    # x0's block-scoped VIF only sees x1, so it equals the two-variable case exactly.
    pair = vif_scores(correlated_block.loc[:, ["x0", "x1"]], {"A": ["x0", "x1"]})
    assert within["x0"] == pytest.approx(pair["x0"])
    assert across["x0"] > within["x0"]


def test_vif_scope_rejects_unknown_value(correlated_block):
    with pytest.raises(ValueError, match="scope"):
        vif_scores(correlated_block, {"A": ["x0", "x1"]}, scope="galaxy")


def test_htmt_matches_hand_computation():
    rng = np.random.default_rng(11)
    n = 250
    frame = pd.DataFrame({name: rng.normal(size=n) for name in ["a1", "a2", "b1", "b2"]})
    corr = frame.corr()
    hetero = np.mean(
        np.abs([corr.loc["a1", "b1"], corr.loc["a1", "b2"], corr.loc["a2", "b1"], corr.loc["a2", "b2"]])
    )
    mono = np.sqrt(abs(corr.loc["a1", "a2"]) * abs(corr.loc["b1", "b2"]))
    matrix = htmt_matrix(frame, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    assert matrix.loc["A", "B"] == pytest.approx(hetero / mono)
    assert matrix.loc["A", "B"] == pytest.approx(matrix.loc["B", "A"])
    assert np.isnan(matrix.loc["A", "A"])


def test_htmt_is_nan_for_single_indicator_construct():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({c: rng.normal(size=60) for c in ["a1", "a2", "b1"]})
    matrix = htmt_matrix(frame, {"A": ["a1", "a2"], "B": ["b1"]})
    assert np.isnan(matrix.loc["A", "B"])
    assert np.isnan(max_htmt(matrix))


def test_dimensional_diversity_clips_and_inverts():
    def matrix(value):
        return pd.DataFrame(
            [[np.nan, value], [value, np.nan]], index=["A", "B"], columns=["A", "B"]
        )

    assert dimensional_diversity(matrix(0.9)) == pytest.approx(1 / 1.8)
    assert dimensional_diversity(matrix(0.5)) == pytest.approx(1.0)
    # Below 0.5 the raw ratio exceeds 1 and the min(.,1) clip engages.
    assert dimensional_diversity(matrix(0.2)) == pytest.approx(1.0)


def test_task_contribution_uses_absolute_loadings():
    assert task_contribution({"a": 0.9, "b": -0.7}) == pytest.approx(0.8)
    assert np.isnan(task_contribution({}))


def test_indicator_validity_is_inverse_geometric_mean():
    assert indicator_validity({"a": 2.0, "b": 8.0}) == pytest.approx(0.25)
    assert indicator_validity({"a": 1.0, "b": 1.0}) == pytest.approx(1.0)
    assert np.isnan(indicator_validity({}))


def test_indicator_validity_ignores_non_finite_entries():
    assert indicator_validity({"a": 4.0, "b": float("inf")}) == pytest.approx(0.25)


def test_cronbach_alpha_matches_formula(correlated_block):
    columns = list(correlated_block.columns)
    k = len(columns)
    item_variance = correlated_block.var(ddof=1).sum()
    total_variance = correlated_block.sum(axis=1).var(ddof=1)
    expected = k / (k - 1) * (1 - item_variance / total_variance)
    assert cronbach_alpha(correlated_block, columns) == pytest.approx(expected)


def test_cronbach_alpha_undefined_for_single_item(correlated_block):
    assert np.isnan(cronbach_alpha(correlated_block, ["x0"]))


def test_composite_reliability_and_ave_match_closed_forms():
    loadings = [0.9, 0.8, 0.85]
    squared_sum = sum(loadings) ** 2
    error = sum(1 - value ** 2 for value in loadings)
    assert composite_reliability(loadings) == pytest.approx(squared_sum / (squared_sum + error))
    assert average_variance_extracted(loadings) == pytest.approx(np.mean(np.square(loadings)))


def test_perfect_loadings_give_unit_reliability():
    assert composite_reliability([1.0, 1.0]) == pytest.approx(1.0)
    assert average_variance_extracted([1.0, 1.0]) == pytest.approx(1.0)


def test_srmr_is_zero_for_identical_matrices(correlated_block):
    corr = correlated_block.corr()
    assert srmr(corr, corr) == pytest.approx(0.0)


def test_srmr_scores_only_off_diagonal():
    observed = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    implied = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=["a", "b"], columns=["a", "b"])
    assert srmr(observed, implied) == pytest.approx(0.2)


def test_overall_score_weights_by_contribution():
    scores = {"a": 100.0, "b": 0.0}
    assert overall_score(scores) == pytest.approx(50.0)
    assert overall_score(scores, {"a": 3.0, "b": 1.0}) == pytest.approx(75.0)


def test_overall_score_ignores_tasks_without_weights():
    assert overall_score({"a": 10.0, "b": 90.0}, {"a": 1.0}) == pytest.approx(10.0)


def test_summarise_bundles_all_three_diagnostics():
    htmt = pd.DataFrame([[np.nan, 0.5], [0.5, np.nan]], index=["A", "B"], columns=["A", "B"])
    summary = summarise({"a": 0.8, "b": 0.8}, {"a": 1.0, "b": 1.0}, htmt)
    assert summary["task_num"] == 2
    assert summary["TC"] == pytest.approx(0.8)
    assert summary["D_valid"] == pytest.approx(1.0)
    assert summary["overall"] == pytest.approx(summary["TC"] + summary["D_div"] + summary["D_valid"])
