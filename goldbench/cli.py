"""Command-line interface.

    python -m goldbench evaluate --scores S.csv --spec spec.json
    python -m goldbench select   --scores S.csv --spec spec.json --outdir runs/gold
    python -m goldbench ablate   --scores S.csv --spec spec.json --grid 5:0.75,5:0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from dataclasses import replace

from .config import BenchmarkSpec, Thresholds
from .pipeline import ablate_thresholds, run_pipeline
from .report import format_pruning, format_summary, load_scores, write_outputs
from .sem import fit

DEFAULT_GRID = "3:0.75,5:0.70,5:0.75,5:0.80,7:0.75"


def _parse_grid(text: str) -> List[Tuple[float, float]]:
    """Parse ``"5:0.75,7:0.70"`` into ``[(5.0, 0.75), (7.0, 0.70)]``."""
    pairs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(
                f"grid entry {chunk!r} must look like DELTA_VIF:LAMBDA_MIN, e.g. 5:0.75"
            )
        vif_text, loading_text = chunk.split(":", 1)
        try:
            pairs.append((float(vif_text), float(loading_text)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"grid entry {chunk!r} is not numeric") from exc
    if not pairs:
        raise argparse.ArgumentTypeError("grid is empty")
    return pairs


def _load(args: argparse.Namespace) -> Tuple:
    scores = load_scores(args.scores, index_column=args.index_column)
    spec = BenchmarkSpec.from_file(args.spec)
    overrides = {}
    if args.vif_max is not None:
        overrides["vif_max"] = args.vif_max
    if args.loading_min is not None:
        overrides["loading_min"] = args.loading_min
    if args.min_indicators is not None:
        overrides["min_indicators"] = args.min_indicators
    if overrides:
        spec = replace(spec, thresholds=Thresholds(**{**spec.thresholds.as_dict(), **overrides}))
    return scores, spec


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scores", required=True, help="model-by-task score matrix (CSV/TSV/JSON/XLSX)")
    parser.add_argument("--spec", required=True, help="benchmark specification (JSON/YAML)")
    parser.add_argument("--index-column", default=None, help="column holding model names")
    parser.add_argument("--outdir", default=None, help="directory for CSV/JSON outputs")
    parser.add_argument(
        "--scheme",
        default="path",
        choices=["path", "centroid", "factorial"],
        help="PLS inner weighting scheme (default: path)",
    )
    parser.add_argument(
        "--vif-scope",
        default="construct",
        choices=["construct", "benchmark"],
        help="reference set for VIF (default: construct, the paper convention)",
    )
    parser.add_argument("--vif-max", type=float, default=None, help="override delta_VIF")
    parser.add_argument("--loading-min", type=float, default=None, help="override lambda_min")
    parser.add_argument(
        "--min-indicators", type=int, default=None, help="override the per-construct floor"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m goldbench",
        description="SEM-based benchmark evaluation and task selection for MLLM benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="fit the measurement model once and report diagnostics (no pruning)"
    )
    _add_common(evaluate)

    select = subparsers.add_parser(
        "select", help="run the full task-selection pipeline (Algorithm 1)"
    )
    _add_common(select)
    select.add_argument("--max-iterations", type=int, default=100, help="Stage 2 safety cap")
    select.add_argument("--quiet", action="store_true", help="suppress the per-removal trace")

    ablate = subparsers.add_parser("ablate", help="sweep (delta_VIF, lambda_min) pairs")
    _add_common(ablate)
    ablate.add_argument(
        "--grid",
        default=DEFAULT_GRID,
        help=f"comma-separated DELTA_VIF:LAMBDA_MIN pairs (default: {DEFAULT_GRID})",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        scores, spec = _load(args)
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "evaluate":
            result = fit(scores, spec, scheme=args.scheme, vif_scope=args.vif_scope)
            print(format_summary(result))
            if args.outdir:
                written = write_outputs(args.outdir, result, scores=scores)
                print(f"\nWrote {len(written)} file(s) to {Path(args.outdir).resolve()}")

        elif args.command == "select":
            outcome = run_pipeline(
                scores,
                spec,
                max_iterations=args.max_iterations,
                scheme=args.scheme,
                vif_scope=args.vif_scope,
                verbose=not args.quiet,
            )
            print()
            print(format_pruning(outcome))
            print()
            print("=" * 72)
            print()
            print(format_summary(outcome.final_result))
            if args.outdir:
                written = write_outputs(
                    args.outdir, outcome.final_result, outcome=outcome, scores=scores
                )
                print(f"\nWrote {len(written)} file(s) to {Path(args.outdir).resolve()}")

        elif args.command == "ablate":
            grid = _parse_grid(args.grid)
            table = ablate_thresholds(
                scores, spec, grid, scheme=args.scheme, vif_scope=args.vif_scope
            )
            columns = ["delta_VIF", "lambda_min", "task_num", "TC", "D_div", "D_valid", "overall"]
            print(table.loc[:, columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            if args.outdir:
                outdir = Path(args.outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                target = outdir / "ablation.csv"
                table.to_csv(target, index=False)
                print(f"\nWrote {target.resolve()}")

    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
