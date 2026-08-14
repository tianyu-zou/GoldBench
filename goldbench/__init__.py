"""GoldBench: SEM-based benchmark evaluation and task selection for MLLM benchmarks.

Reference implementation of "Aligning MLLM Benchmark With Human Preferences via
Structural Equation Modeling". PLS-SEM estimation is delegated to the ``plspm``
library; this package adds the benchmark-level diagnostics (D_div, TC, D_valid) and the
iterative task-selection pipeline of Algorithm 1.

Typical use::

    from goldbench import BenchmarkSpec, load_scores, run_pipeline, format_pruning

    scores = load_scores("data/scores.csv")
    spec = BenchmarkSpec.from_file("specs/gold.json")
    outcome = run_pipeline(scores, spec)
    print(format_pruning(outcome))
"""

from .config import STRUCTURES, BenchmarkSpec, Thresholds
from .metrics import (
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
from .pipeline import PruningResult, PruningStep, ablate_thresholds, run_pipeline
from .report import (
    format_pruning,
    format_summary,
    load_scores,
    model_scores,
    rank_correlation,
    write_outputs,
)
from .sem import SEMResult, fit

__version__ = "1.0.0"

__all__ = [
    "BenchmarkSpec",
    "PruningResult",
    "PruningStep",
    "SEMResult",
    "STRUCTURES",
    "Thresholds",
    "__version__",
    "ablate_thresholds",
    "average_variance_extracted",
    "composite_reliability",
    "cronbach_alpha",
    "dimensional_diversity",
    "fit",
    "format_pruning",
    "format_summary",
    "htmt_matrix",
    "indicator_validity",
    "load_scores",
    "max_htmt",
    "model_scores",
    "overall_score",
    "rank_correlation",
    "run_pipeline",
    "srmr",
    "summarise",
    "task_contribution",
    "vif_scores",
    "write_outputs",
]
