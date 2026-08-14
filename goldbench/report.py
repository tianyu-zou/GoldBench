"""Report writing and score-matrix loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BenchmarkSpec
from .metrics import max_htmt, overall_score
from .pipeline import PruningResult
from .sem import SEMResult


def load_scores(path: str | Path, index_column: Optional[str] = None) -> pd.DataFrame:
    """Load a model-by-task score matrix from CSV, TSV, JSON or Excel.

    Rows are models, columns are tasks. The first column is used as the model index
    when ``index_column`` is not given and it holds non-numeric values.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"unsupported score file type: {path.suffix}")

    if index_column is not None:
        if index_column not in frame.columns:
            raise ValueError(f"index column {index_column!r} not found in {path.name}")
        frame = frame.set_index(index_column)
    elif len(frame.columns) and not pd.api.types.is_numeric_dtype(frame.iloc[:, 0]):
        frame = frame.set_index(frame.columns[0])

    frame.index = frame.index.map(str)
    return frame


def model_scores(
    data: pd.DataFrame,
    spec: BenchmarkSpec,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Per-model layer scores and overall score, weighted by task contribution.

    Implements the supplement's aggregation: layer score = sum(w_i s_i) / sum(w_i) over
    the tasks of that layer, and the overall score over all retained tasks. Pass the
    fitted outer loadings as ``weights``; ``None`` gives the unweighted mean.
    """
    rows = []
    for model_name, row in data.iterrows():
        record: Dict[str, object] = {"model": model_name}
        for lv, indicators in spec.constructs.items():
            present = {i: float(row[i]) for i in indicators if i in row.index}
            record[lv] = overall_score(present, weights)
        all_tasks = {i: float(row[i]) for i in spec.indicators if i in row.index}
        record["overall"] = overall_score(all_tasks, weights)
        rows.append(record)
    return pd.DataFrame(rows).set_index("model")


def rank_correlation(
    left: pd.Series,
    right: pd.Series,
    method: str = "spearman",
) -> float:
    """Rank correlation between two model rankings, over their shared models.

    ``method`` is ``"spearman"`` (rank, as used for ordinal human consensus) or
    ``"pearson"`` (linear, as used for continuous Chatbot Arena Elo).
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    shared = left.dropna().index.intersection(right.dropna().index)
    if len(shared) < 3:
        return float("nan")
    return float(left.loc[shared].corr(right.loc[shared], method=method))


def format_summary(result: SEMResult) -> str:
    """Human-readable summary of one fit."""
    diagnostics = result.diagnostics
    lines = [
        f"Benchmark: {result.spec.name}",
        f"Structure: {result.spec.structure}   "
        f"Constructs: {len(result.spec.constructs)}   "
        f"Tasks: {len(result.spec.indicators)}",
        f"Thresholds: VIF <= {result.spec.thresholds.vif_max}, "
        f"|loading| >= {result.spec.thresholds.loading_min}",
        "",
        "Benchmark-level diagnostics",
        f"  Task contribution      TC      = {diagnostics['TC']:.4f}",
        f"  Dimensional diversity  D_div   = {diagnostics['D_div']:.4f}",
        f"  Indicator validity     D_valid = {diagnostics['D_valid']:.4f}",
        f"  Composite (sum)        overall = {diagnostics['overall']:.4f}",
        f"  Max off-diagonal HTMT          = {max_htmt(result.htmt):.4f}",
        f"  SRMR                           = {result.srmr:.4f}",
        "",
        "Measurement model",
        result.loading_table().to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "Reliability and validity",
        result.reliability.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
        "HTMT matrix",
        result.htmt.to_string(float_format=lambda v: f"{v:.3f}"),
    ]
    if result.r_squared:
        endogenous = {
            lv: value for lv, value in result.r_squared.items() if abs(value) > 1e-12
        }
        if endogenous:
            lines += [
                "",
                "R^2 (endogenous constructs)",
                *[f"  {lv}: {value:.4f}" for lv, value in sorted(endogenous.items())],
            ]
    return "\n".join(lines)


def format_pruning(outcome: PruningResult) -> str:
    """Human-readable summary of a pruning run."""
    removed = outcome.removed_tasks
    lines = [
        "Benchmark evolution (Algorithm 1)",
        f"  Candidate pool : {len(outcome.initial_spec.indicators)} tasks",
        f"  Retained       : {len(outcome.retained_tasks)} tasks",
        f"  Removed        : {len(removed)} tasks",
        "",
        "Retained task set",
    ]
    for lv, indicators in outcome.final_spec.constructs.items():
        lines.append(f"  {lv}: {', '.join(indicators)}")
    if removed:
        lines += ["", "Removed tasks (in removal order)"]
        for step in outcome.history:
            if step.removed:
                value = "inf" if not np.isfinite(step.value or np.nan) else f"{step.value:.3f}"
                lines.append(f"  {step.removed} ({step.reason}={value})")
    lines += [
        "",
        "Diagnostics before vs. after",
        outcome.comparison_table().to_string(float_format=lambda v: f"{v:.4f}"),
    ]
    return "\n".join(lines)


def write_outputs(
    outdir: str | Path,
    result: SEMResult,
    outcome: Optional[PruningResult] = None,
    scores: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """Persist every table of a run as CSV plus a JSON/text summary.

    Returns the list of written paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _csv(frame: pd.DataFrame, name: str, index: bool = True) -> None:
        target = outdir / name
        frame.to_csv(target, index=index)
        written.append(target)

    _csv(result.loading_table(), "measurement_model.csv", index=False)
    _csv(result.reliability, "reliability.csv")
    _csv(result.htmt, "htmt.csv")
    _csv(result.scores, "latent_scores.csv")
    _csv(result.path_coefficients, "path_coefficients.csv")

    summary = {
        "benchmark": result.spec.name,
        "structure": result.spec.structure,
        "thresholds": result.spec.thresholds.as_dict(),
        "n_tasks": len(result.spec.indicators),
        "diagnostics": {k: _jsonable(v) for k, v in result.diagnostics.items()},
        "max_htmt": _jsonable(max_htmt(result.htmt)),
        "srmr": _jsonable(result.srmr),
        "r_squared": {k: _jsonable(v) for k, v in result.r_squared.items()},
        "constructs": result.spec.constructs,
    }
    if outcome is not None:
        summary["pruning"] = {
            "initial_tasks": len(outcome.initial_spec.indicators),
            "retained_tasks": outcome.retained_tasks,
            "removed_tasks": outcome.removed_tasks,
            "protected_tasks": outcome.protected,
        }
        _csv(outcome.history_table(), "pruning_history.csv", index=False)
        _csv(outcome.comparison_table(), "diagnostics_before_after.csv")

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(summary_path)

    report_path = outdir / "report.txt"
    text = format_summary(result)
    if outcome is not None:
        text = format_pruning(outcome) + "\n\n" + "=" * 72 + "\n\n" + text
    report_path.write_text(text + "\n", encoding="utf-8")
    written.append(report_path)

    spec_path = outdir / "refined_spec.json"
    result.spec.to_file(spec_path)
    written.append(spec_path)

    if scores is not None:
        weights = result.task_loadings
        _csv(model_scores(scores, result.spec, weights), "model_scores.csv")

    return written


def _jsonable(value):
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value
