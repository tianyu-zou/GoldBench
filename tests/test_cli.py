"""CLI argument handling and exit codes."""

import json

import pytest

from goldbench.cli import _parse_grid, main


@pytest.fixture
def paths(scores, flawed_spec, tmp_path):
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path)
    spec_path = tmp_path / "spec.json"
    flawed_spec.to_file(spec_path)
    return str(score_path), str(spec_path)


def test_parse_grid_accepts_pairs():
    assert _parse_grid("5:0.75, 7:0.7") == [(5.0, 0.75), (7.0, 0.7)]


def test_parse_grid_rejects_malformed_entries():
    with pytest.raises(Exception, match="DELTA_VIF"):
        _parse_grid("5-0.75")
    with pytest.raises(Exception, match="not numeric"):
        _parse_grid("five:0.75")
    with pytest.raises(Exception, match="empty"):
        _parse_grid("  ")


def test_evaluate_prints_diagnostics(paths, capsys):
    score_path, spec_path = paths
    assert main(["evaluate", "--scores", score_path, "--spec", spec_path]) == 0
    output = capsys.readouterr().out
    assert "Task contribution" in output
    assert "Measurement model" in output


def test_select_reports_removals_and_writes_outputs(paths, tmp_path, capsys):
    score_path, spec_path = paths
    outdir = tmp_path / "out"
    code = main(
        ["select", "--scores", score_path, "--spec", spec_path, "--outdir", str(outdir)]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "p_dup" in output and "r_weak" in output
    payload = json.loads((outdir / "summary.json").read_text())
    assert payload["n_tasks"] == 9


def test_select_quiet_suppresses_the_trace(paths, capsys):
    score_path, spec_path = paths
    assert main(["select", "--scores", score_path, "--spec", spec_path, "--quiet"]) == 0
    output = capsys.readouterr().out
    assert "[stage 2] iter" not in output
    assert "Benchmark evolution" in output


def test_ablate_prints_one_row_per_pair(paths, tmp_path, capsys):
    score_path, spec_path = paths
    outdir = tmp_path / "abl"
    code = main(
        [
            "ablate",
            "--scores",
            score_path,
            "--spec",
            spec_path,
            "--grid",
            "5:0.75,5:0.85",
            "--outdir",
            str(outdir),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "delta_VIF" in output
    assert (outdir / "ablation.csv").exists()


def test_threshold_flags_override_the_spec(paths, capsys):
    score_path, spec_path = paths
    main(["evaluate", "--scores", score_path, "--spec", spec_path, "--vif-max", "3"])
    assert "VIF <= 3.0" in capsys.readouterr().out


def test_missing_score_file_exits_with_code_two(tmp_path, flawed_spec, capsys):
    spec_path = tmp_path / "spec.json"
    flawed_spec.to_file(spec_path)
    code = main(["evaluate", "--scores", str(tmp_path / "absent.csv"), "--spec", str(spec_path)])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_spec_exits_with_code_two(scores, tmp_path, capsys):
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path)
    spec_path = tmp_path / "bad.json"
    spec_path.write_text(json.dumps({"constructs": {"A": ["p1", "p1"]}}), encoding="utf-8")
    assert main(["evaluate", "--scores", str(score_path), "--spec", str(spec_path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_task_in_spec_exits_with_code_one(scores, tmp_path, capsys):
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "constructs": {"A": ["p1", "ghost"], "B": ["m1", "m2"]},
                "structure": "chain",
            }
        ),
        encoding="utf-8",
    )
    assert main(["evaluate", "--scores", str(score_path), "--spec", str(spec_path)]) == 1
    assert "missing required columns" in capsys.readouterr().err


def test_command_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])
