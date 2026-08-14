"""Score loading, aggregation and output writing."""

import json

import numpy as np
import pandas as pd
import pytest

from goldbench import (
    fit,
    format_pruning,
    format_summary,
    load_scores,
    model_scores,
    rank_correlation,
    run_pipeline,
    write_outputs,
)


def test_load_scores_from_csv_uses_the_first_column_as_the_index(scores, tmp_path):
    path = tmp_path / "scores.csv"
    scores.to_csv(path)
    loaded = load_scores(path)
    assert list(loaded.index) == list(scores.index)
    assert list(loaded.columns) == list(scores.columns)
    assert np.allclose(loaded["p1"], scores["p1"])


def test_load_scores_honours_an_explicit_index_column(scores, tmp_path):
    path = tmp_path / "scores.csv"
    scores.reset_index().to_csv(path, index=False)
    loaded = load_scores(path, index_column="model")
    assert loaded.index.name == "model"
    assert "model" not in loaded.columns


def test_load_scores_rejects_a_missing_index_column(scores, tmp_path):
    path = tmp_path / "scores.csv"
    scores.to_csv(path)
    with pytest.raises(ValueError, match="index column"):
        load_scores(path, index_column="absent")


def test_load_scores_reads_tsv(scores, tmp_path):
    path = tmp_path / "scores.tsv"
    scores.to_csv(path, sep="\t")
    assert list(load_scores(path).columns) == list(scores.columns)


def test_load_scores_rejects_unknown_extensions(tmp_path):
    path = tmp_path / "scores.parquet"
    path.write_bytes(b"not really a parquet file")
    with pytest.raises(ValueError, match="unsupported score file type"):
        load_scores(path)


def test_model_scores_produce_one_row_per_model(scores, spec):
    table = model_scores(scores, spec)
    assert len(table) == len(scores)
    assert list(table.columns) == spec.construct_names + ["overall"]
    assert table.notna().all().all()


def test_unweighted_layer_score_is_the_plain_mean(scores, spec):
    table = model_scores(scores, spec)
    expected = scores.loc["model_000", ["p1", "p2", "p3"]].mean()
    assert table.loc["model_000", "Perception"] == pytest.approx(expected)


def test_weighting_by_loadings_changes_the_aggregate(scores, spec):
    result = fit(scores, spec)
    weighted = model_scores(scores, spec, result.task_loadings)
    unweighted = model_scores(scores, spec)
    assert not np.allclose(weighted["overall"], unweighted["overall"])
    # Both remain inside the range of the underlying task scores.
    low, high = scores[spec.indicators].min().min(), scores[spec.indicators].max().max()
    assert weighted["overall"].between(low, high).all()


def test_overall_score_tracks_human_preference(scores, spec):
    table = model_scores(scores, spec, fit(scores, spec).task_loadings)
    assert rank_correlation(table["overall"], scores["arena_elo"]) > 0.6
    assert rank_correlation(table["overall"], scores["arena_elo"], "pearson") > 0.6


def test_rank_correlation_requires_enough_shared_models():
    left = pd.Series([1.0, 2.0], index=["a", "b"])
    right = pd.Series([1.0, 2.0], index=["a", "b"])
    assert np.isnan(rank_correlation(left, right))


def test_rank_correlation_rejects_unknown_methods():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="spearman"):
        rank_correlation(series, series, method="kendall")


def test_rank_correlation_uses_only_shared_models():
    left = pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"])
    right = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    assert rank_correlation(left, right) == pytest.approx(1.0)


def test_format_summary_reports_every_diagnostic(scores, spec):
    text = format_summary(fit(scores, spec))
    for fragment in ("TC", "D_div", "D_valid", "SRMR", "HTMT", "Reliability"):
        assert fragment in text
    for construct in spec.construct_names:
        assert construct in text


def test_format_pruning_lists_removals(scores, flawed_spec):
    text = format_pruning(run_pipeline(scores, flawed_spec, verbose=False))
    assert "p_dup" in text
    assert "r_weak" in text
    assert "Candidate pool" in text
    assert "Retained task set" in text


def test_write_outputs_creates_every_expected_file(scores, flawed_spec, tmp_path):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    written = write_outputs(tmp_path / "run", outcome.final_result, outcome, scores)
    names = {path.name for path in written}
    assert {
        "measurement_model.csv",
        "reliability.csv",
        "htmt.csv",
        "latent_scores.csv",
        "path_coefficients.csv",
        "pruning_history.csv",
        "diagnostics_before_after.csv",
        "summary.json",
        "report.txt",
        "refined_spec.json",
    }.issubset(names)
    assert all(path.exists() for path in written)


def test_summary_json_is_parseable_and_json_safe(scores, flawed_spec, tmp_path):
    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    write_outputs(tmp_path / "run", outcome.final_result, outcome, scores)
    payload = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert payload["n_tasks"] == len(outcome.retained_tasks)
    assert set(payload["pruning"]["removed_tasks"]) == {"p_dup", "r_weak"}
    assert payload["diagnostics"]["TC"] > 0.75
    assert payload["constructs"]["Perception"] == outcome.final_spec.constructs["Perception"]


def test_written_spec_reloads_and_reproduces_the_fit(scores, flawed_spec, tmp_path):
    from goldbench import BenchmarkSpec

    outcome = run_pipeline(scores, flawed_spec, verbose=False)
    write_outputs(tmp_path / "run", outcome.final_result, outcome, scores)
    reloaded = BenchmarkSpec.from_file(tmp_path / "run" / "refined_spec.json")
    assert reloaded.constructs == outcome.final_spec.constructs
    refit = fit(scores, reloaded)
    assert refit.diagnostics["TC"] == pytest.approx(outcome.final_result.diagnostics["TC"])


def test_write_outputs_without_pruning_omits_history(scores, spec, tmp_path):
    written = write_outputs(tmp_path / "eval", fit(scores, spec))
    names = {path.name for path in written}
    assert "measurement_model.csv" in names
    assert "pruning_history.csv" not in names
    assert "model_scores.csv" not in names


def test_model_scores_csv_is_written_when_scores_are_supplied(scores, spec, tmp_path):
    write_outputs(tmp_path / "eval", fit(scores, spec), scores=scores)
    table = pd.read_csv(tmp_path / "eval" / "model_scores.csv", index_col=0)
    assert len(table) == len(scores)
    assert "overall" in table.columns
