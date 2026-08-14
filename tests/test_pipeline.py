"""Task selection (Algorithm 1): removal order, stopping rules and the floor guarantee."""

from dataclasses import replace

import numpy as np
import pytest

from goldbench import BenchmarkSpec, Thresholds, ablate_thresholds, fit, run_pipeline


def test_pipeline_removes_exactly_the_planted_defects(scores, flawed_spec):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    assert set(outcome.removed_tasks) == {"p_dup", "r_weak"}
    assert set(outcome.retained_tasks) == {"p1", "p2", "p3", "m1", "m2", "m3", "r1", "r2", "r3"}


def test_pipeline_leaves_a_clean_spec_untouched(scores, spec):
    outcome = run_pipeline(scores, spec, verbose=False)
    assert outcome.removed_tasks == []
    assert outcome.retained_tasks == spec.indicators
    assert len(outcome.history) == 1


def test_final_task_set_satisfies_both_thresholds(scores, flawed_spec):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    result = outcome.final_result
    for task in outcome.retained_tasks:
        assert result.vifs[task] <= flawed_spec.thresholds.vif_max, task
        assert abs(result.loadings[task]) >= flawed_spec.thresholds.loading_min, task


def test_redundancy_is_removed_before_weak_contribution(scores, flawed_spec):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    reasons = [step.reason for step in outcome.history if step.removed]
    assert reasons.index("vif") < reasons.index("loading")


def test_pruning_improves_indicator_validity(scores, flawed_spec):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    before = outcome.initial_result.diagnostics
    after = outcome.final_result.diagnostics
    assert after["D_valid"] > before["D_valid"]
    assert after["TC"] > before["TC"]


def test_history_records_one_row_per_iteration_plus_the_baseline(scores, flawed_spec):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    table = outcome.history_table()
    assert len(table) == len(outcome.removed_tasks) + 1
    assert table.iloc[0]["removed"] == ""
    assert table.iloc[0]["n_tasks"] == len(flawed_spec.indicators)
    assert table.iloc[-1]["n_tasks"] == len(outcome.retained_tasks)
    assert set(["TC", "D_div", "D_valid", "overall"]).issubset(table.columns)


def test_comparison_table_holds_before_and_after_columns(scores, flawed_spec):
    table = run_pipeline(scores, flawed_spec, verbose=False).comparison_table()
    assert list(table.columns) == ["initial", "final"]
    assert table.loc["task_num", "final"] < table.loc["task_num", "initial"]
    assert np.isfinite(table.loc["SRMR", "final"])


def test_min_indicators_floor_is_never_breached(scores, spec):
    impossible = replace(
        spec, thresholds=Thresholds(vif_max=1.05, loading_min=0.999, min_indicators=2)
    )
    outcome = run_pipeline(scores, impossible, verbose=False)
    for construct, indicators in outcome.final_spec.constructs.items():
        assert len(indicators) >= 2, construct
    assert len(outcome.final_spec.constructs) == len(spec.constructs)


def test_floor_of_one_allows_single_indicator_constructs(scores, spec):
    impossible = replace(
        spec, thresholds=Thresholds(vif_max=1.05, loading_min=0.999, min_indicators=1)
    )
    outcome = run_pipeline(scores, impossible, verbose=False)
    for indicators in outcome.final_spec.constructs.values():
        assert len(indicators) >= 1


def test_protected_tasks_are_reported(scores, spec):
    impossible = replace(
        spec, thresholds=Thresholds(vif_max=1.05, loading_min=0.999, min_indicators=2)
    )
    outcome = run_pipeline(scores, impossible, verbose=False)
    assert outcome.protected
    assert set(outcome.protected).issubset(set(outcome.initial_spec.indicators))


def test_iteration_cap_raises_rather_than_looping_forever(scores, flawed_spec):
    with pytest.raises(RuntimeError, match="did not stabilise"):
        run_pipeline(scores, flawed_spec, max_iterations=1, verbose=False)


def test_verbose_mode_traces_each_removal(scores, flawed_spec, capsys):
    run_pipeline(scores, flawed_spec, verbose=True)
    output = capsys.readouterr().out
    assert "drop 'p_dup'" in output
    assert "drop 'r_weak'" in output
    assert "converged" in output


def test_verbose_mode_explains_a_floor_limited_stop(scores, spec, capsys):
    impossible = replace(
        spec, thresholds=Thresholds(vif_max=1.05, loading_min=0.999, min_indicators=2)
    )
    run_pipeline(scores, impossible, verbose=True)
    assert "keep every construct measurable" in capsys.readouterr().out


def test_stricter_loading_threshold_never_keeps_more_tasks(scores, flawed_spec):
    lenient = run_pipeline(
        scores,
        replace(flawed_spec, thresholds=Thresholds(vif_max=5.0, loading_min=0.70)),
        verbose=False,
    )
    strict = run_pipeline(
        scores,
        replace(flawed_spec, thresholds=Thresholds(vif_max=5.0, loading_min=0.85)),
        verbose=False,
    )
    assert len(strict.retained_tasks) <= len(lenient.retained_tasks)


def test_ablation_grid_returns_one_row_per_threshold_pair(scores, flawed_spec):
    grid = [(5.0, 0.75), (5.0, 0.85), (3.0, 0.75)]
    table = ablate_thresholds(scores, flawed_spec, grid)
    assert len(table) == len(grid)
    assert list(table["delta_VIF"]) == [5.0, 5.0, 3.0]
    assert (table["task_num"] > 0).all()
    for column in ("TC", "D_div", "D_valid", "overall"):
        assert table[column].notna().all(), column
    assert table["retained"].str.len().gt(0).all()


def test_ablation_is_deterministic(scores, flawed_spec):
    first = ablate_thresholds(scores, flawed_spec, [(5.0, 0.75)])
    second = ablate_thresholds(scores, flawed_spec, [(5.0, 0.75)])
    assert first["retained"].iloc[0] == second["retained"].iloc[0]
    assert first["overall"].iloc[0] == pytest.approx(second["overall"].iloc[0])


def test_final_result_matches_a_direct_refit(scores, flawed_spec):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    refit = fit(scores, outcome.final_spec)
    assert refit.diagnostics["TC"] == pytest.approx(outcome.final_result.diagnostics["TC"])
    assert refit.diagnostics["D_valid"] == pytest.approx(
        outcome.final_result.diagnostics["D_valid"]
    )
