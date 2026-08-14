"""Specification validation and serialisation."""

import json

import pytest

from goldbench import BenchmarkSpec, Thresholds


def test_thresholds_reject_impossible_values():
    with pytest.raises(ValueError, match="vif_max"):
        Thresholds(vif_max=0.5)
    with pytest.raises(ValueError, match="loading_min"):
        Thresholds(loading_min=1.5)
    with pytest.raises(ValueError, match="min_indicators"):
        Thresholds(min_indicators=0)


def test_indicator_cannot_load_on_two_constructs():
    with pytest.raises(ValueError, match="exactly one construct"):
        BenchmarkSpec({"A": ["x", "y"], "B": ["y", "z"]}, structure="chain")


def test_construct_must_have_indicators():
    with pytest.raises(ValueError, match="no indicators"):
        BenchmarkSpec({"A": []}, structure="chain")


def test_duplicate_indicator_within_construct_is_rejected():
    with pytest.raises(ValueError, match="duplicate indicator"):
        BenchmarkSpec({"A": ["x", "x"], "B": ["y", "z"]}, structure="chain")


def test_indicator_may_not_share_a_name_with_a_latent_variable():
    with pytest.raises(ValueError, match="collide"):
        BenchmarkSpec({"A": ["A", "y"], "B": ["p", "q"]}, structure="chain")


def test_anchor_structure_requires_a_human_column():
    with pytest.raises(ValueError, match="human_column"):
        BenchmarkSpec({"A": ["x", "y"], "B": ["p", "q"]}, structure="anchor")


def test_unknown_structure_is_rejected():
    with pytest.raises(ValueError, match="unknown structure"):
        BenchmarkSpec({"A": ["x", "y"]}, structure="orbital")


def test_chain_structure_needs_two_constructs():
    with pytest.raises(ValueError, match="at least two"):
        BenchmarkSpec({"A": ["x", "y"]}, structure="chain")


def test_human_column_may_not_double_as_a_task():
    with pytest.raises(ValueError, match="also listed as a task"):
        BenchmarkSpec({"A": ["x", "elo"]}, human_column="elo", structure="anchor")


def test_indicators_preserve_declaration_order(spec):
    assert spec.indicators == ["p1", "p2", "p3", "m1", "m2", "m3", "r1", "r2", "r3"]
    assert spec.construct_names == ["Perception", "Memory", "Reasoning"]


def test_construct_of_resolves_and_raises(spec):
    assert spec.construct_of("m2") == "Memory"
    with pytest.raises(KeyError):
        spec.construct_of("nope")


def test_required_columns_include_the_human_column(spec):
    assert spec.required_columns() == spec.indicators + ["arena_elo"]


def test_without_removes_indicators_and_leaves_original_untouched(spec):
    pruned = spec.without(["p1", "r3"])
    assert pruned.constructs["Perception"] == ["p2", "p3"]
    assert pruned.constructs["Reasoning"] == ["r1", "r2"]
    assert spec.constructs["Perception"] == ["p1", "p2", "p3"]


def test_without_drops_emptied_constructs(spec):
    pruned = spec.without(["m1", "m2", "m3"])
    assert "Memory" not in pruned.constructs
    assert set(pruned.construct_names) == {"Perception", "Reasoning"}


def test_without_refuses_to_empty_the_model(spec):
    with pytest.raises(ValueError, match="every construct"):
        spec.without(spec.indicators)


def test_spec_round_trips_through_json(spec, tmp_path):
    path = tmp_path / "spec.json"
    spec.to_file(path)
    reloaded = BenchmarkSpec.from_file(path)
    assert reloaded.constructs == spec.constructs
    assert reloaded.thresholds == spec.thresholds
    assert reloaded.structure == spec.structure
    assert reloaded.human_column == spec.human_column


def test_spec_round_trips_through_yaml(spec, tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "spec.yaml"
    spec.to_file(path)
    reloaded = BenchmarkSpec.from_file(path)
    assert reloaded.constructs == spec.constructs
    assert reloaded.thresholds == spec.thresholds


def test_from_dict_applies_threshold_overrides():
    payload = {
        "constructs": {"A": ["x", "y"], "B": ["p", "q"]},
        "structure": "chain",
        "thresholds": {"vif_max": 3.0, "loading_min": 0.8},
    }
    loaded = BenchmarkSpec.from_dict(payload)
    assert loaded.thresholds.vif_max == 3.0
    assert loaded.thresholds.loading_min == 0.8


def test_name_defaults_to_the_filename_stem(tmp_path):
    path = tmp_path / "my_bench.json"
    path.write_text(
        json.dumps({"constructs": {"A": ["x", "y"], "B": ["p", "q"]}, "structure": "chain"}),
        encoding="utf-8",
    )
    assert BenchmarkSpec.from_file(path).name == "my_bench"


def test_shipped_specs_are_valid():
    from pathlib import Path

    spec_dir = Path(__file__).resolve().parents[1] / "specs"
    files = sorted(spec_dir.glob("*.json"))
    assert files, "no specs found to validate"
    for path in files:
        loaded = BenchmarkSpec.from_file(path)
        assert loaded.indicators
        assert loaded.construct_names
