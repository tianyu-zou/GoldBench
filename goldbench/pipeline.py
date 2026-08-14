"""Benchmark evolution pipeline -- Algorithm 1 of the paper.

Stage 1  Initialise: assign every candidate task to a Piaget-inspired latent construct.
Stage 2  Iteratively refine: re-estimate the PLS-SEM measurement model, drop the worst
         offending indicator by VIF > delta_VIF or |outer loading| < lambda_min, repeat
         until no indicator violates either threshold.
Stage 3  Validate: report Cronbach's alpha, composite reliability, AVE and HTMT for the
         surviving task set.

One indicator is removed per iteration rather than all offenders at once: dropping a
collinear task changes the VIFs and loadings of everything else in its block, so batch
removal routinely discards tasks that would have passed after the first deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BenchmarkSpec
from .sem import SEMResult, fit

# Two tasks whose VIFs are within this ratio are treated as equally collinear; see
# _most_collinear for why the raw maximum is not a usable ordering.
_TIE_RATIO = 0.95


@dataclass
class PruningStep:
    """One iteration of Stage 2."""

    iteration: int
    n_tasks: int
    removed: Optional[str]
    reason: Optional[str]
    value: Optional[float]
    diagnostics: Dict[str, float]

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "iteration": self.iteration,
            "n_tasks": self.n_tasks,
            "removed": self.removed or "",
            "reason": self.reason or "",
            "value": self.value if self.value is not None else np.nan,
        }
        row.update(self.diagnostics)
        return row


@dataclass
class PruningResult:
    """Outcome of the full pipeline."""

    initial_spec: BenchmarkSpec
    final_spec: BenchmarkSpec
    initial_result: SEMResult
    final_result: SEMResult
    history: List[PruningStep] = field(default_factory=list)
    protected: List[str] = field(default_factory=list)

    @property
    def removed_tasks(self) -> List[str]:
        """Deleted tasks, earliest removal first.

        Removal order is the informative one -- it says which violation was severe enough
        to be caught first -- and it matches the order printed in report.txt. Anything
        missing from the history (which should not happen) is appended in pool order so the
        list stays complete.
        """
        kept = set(self.final_spec.indicators)
        ordered = [step.removed for step in self.history if step.removed and step.removed not in kept]
        seen = set(ordered)
        ordered += [t for t in self.initial_spec.indicators if t not in kept and t not in seen]
        return ordered

    @property
    def retained_tasks(self) -> List[str]:
        return list(self.final_spec.indicators)

    def history_table(self) -> pd.DataFrame:
        return pd.DataFrame([step.as_row() for step in self.history])

    def comparison_table(self) -> pd.DataFrame:
        """Before/after diagnostics for the pruning run."""
        before = dict(self.initial_result.diagnostics)
        after = dict(self.final_result.diagnostics)
        before["SRMR"] = self.initial_result.srmr
        after["SRMR"] = self.final_result.srmr
        return pd.DataFrame({"initial": before, "final": after})


def _worst_violation(
    result: SEMResult,
    protected: set,
) -> Optional[Tuple[str, str, float]]:
    """Pick the single indicator to drop next, or None when the set is clean.

    Redundancy is treated first: a high-VIF task inflates its block-mates' loadings, so
    removing collinearity before judging contribution avoids discarding a task that only
    looked weak because a near-duplicate was absorbing its variance. Among candidates of
    the same kind, the most extreme violation goes first.

    Mutual near-duplicates are a genuine tie: a pair of tasks measuring the same thing
    report the *same* VIF, so the larger value cannot distinguish them. There the weaker
    outer loading breaks the tie, keeping whichever member contributes more to the
    construct. See ``_most_collinear``.
    """
    thresholds = result.spec.thresholds
    tasks = [t for t in result.spec.indicators if t not in protected]

    redundant = [
        (t, result.vifs[t])
        for t in tasks
        if np.isfinite(result.vifs.get(t, np.nan)) and result.vifs[t] > thresholds.vif_max
    ]
    infinite = [t for t in tasks if not np.isfinite(result.vifs.get(t, 1.0))]
    if infinite:
        # Perfectly collinear tasks share one VIF of inf; drop the least informative.
        task = min(infinite, key=lambda t: abs(result.loadings.get(t, 0.0)))
        return task, "vif", float("inf")
    if redundant:
        task, value = _most_collinear(redundant, result)
        return task, "vif", float(value)

    weak = [
        (t, abs(result.loadings[t]))
        for t in tasks
        if t in result.loadings and abs(result.loadings[t]) < thresholds.loading_min
    ]
    if weak:
        task, value = min(weak, key=lambda kv: kv[1])
        return task, "loading", float(value)
    return None


def _most_collinear(
    redundant: List[Tuple[str, float]],
    result: SEMResult,
) -> Tuple[str, float]:
    """Choose which of several over-collinear tasks to drop.

    Ranking by VIF alone is ill-defined for the case the threshold exists to catch. When
    two tasks are near-duplicates of each other, each explains the other, so both carry an
    almost identical (and very large) VIF -- the maximum picks between them arbitrarily,
    and the choice can flip on floating-point noise or column order.

    Tasks within ``_TIE_RATIO`` of the largest VIF are therefore treated as tied and
    resolved by outer loading, dropping the one that contributes least to its construct.
    That is the same criterion the loading stage would apply, so the two stages cannot
    disagree about which member of a duplicate pair is worth keeping.
    """
    worst = max(value for _, value in redundant)
    tied = [(t, v) for t, v in redundant if v >= worst * _TIE_RATIO]
    if len(tied) == 1:
        return tied[0]
    task, value = min(tied, key=lambda kv: (abs(result.loadings.get(kv[0], 0.0)), -kv[1]))
    return task, value


def _protected_tasks(result: SEMResult) -> set:
    """Tasks that must survive to keep every construct measurable.

    Stage 2 of the algorithm guarantees each construct retains at least
    ``min_indicators`` tasks. When a construct is already at that floor, its remaining
    tasks are protected -- the paper keeps the closest-to-threshold tasks rather than
    letting a cognitive dimension disappear from the model.
    """
    floor = result.spec.thresholds.min_indicators
    protected = set()
    for lv, indicators in result.spec.constructs.items():
        if len(indicators) <= floor:
            protected.update(indicators)
            continue
        # Rank by how comfortably each task clears the thresholds; the best `floor`
        # tasks in a construct are never removed.
        ranked = sorted(
            indicators,
            key=lambda t: (
                abs(result.loadings.get(t, 0.0))
                - max(0.0, result.vifs.get(t, 1.0) - result.spec.thresholds.vif_max)
            ),
            reverse=True,
        )
        protected.update(ranked[:floor])
    return protected


def run_pipeline(
    data: pd.DataFrame,
    spec: BenchmarkSpec,
    max_iterations: int = 100,
    scheme: str = "path",
    vif_scope: str = "construct",
    verbose: bool = True,
) -> PruningResult:
    """Run Algorithm 1 end to end and return the refined benchmark.

    Args:
        data: models (rows) x task scores (columns).
        spec: initial candidate pool with construct assignments and thresholds.
        max_iterations: safety cap on Stage 2.
        scheme: PLS inner weighting scheme.
        vif_scope: ``"construct"`` (paper convention) or ``"benchmark"``.
        verbose: print one line per removal.

    Returns:
        A :class:`PruningResult` holding the initial fit, the final fit and the trace.
    """
    current = spec
    initial_result = fit(data, current, scheme=scheme, vif_scope=vif_scope)
    result = initial_result
    history = [
        PruningStep(
            iteration=0,
            n_tasks=len(current.indicators),
            removed=None,
            reason=None,
            value=None,
            diagnostics=dict(result.diagnostics),
        )
    ]
    protected_final: List[str] = []

    for iteration in range(1, max_iterations + 1):
        protected = _protected_tasks(result)
        protected_final = sorted(protected)
        violation = _worst_violation(result, protected)
        if violation is None:
            if verbose:
                blocked = _blocked_violations(result, protected)
                if blocked:
                    print(
                        f"[stage 2] stopping: {len(blocked)} task(s) still violate a "
                        f"threshold but are retained to keep every construct measurable "
                        f"({', '.join(blocked)})"
                    )
                else:
                    print(f"[stage 2] converged after {iteration - 1} removal(s)")
            break

        task, reason, value = violation
        current = current.without([task])
        result = fit(data, current, scheme=scheme, vif_scope=vif_scope)
        history.append(
            PruningStep(
                iteration=iteration,
                n_tasks=len(current.indicators),
                removed=task,
                reason=reason,
                value=value,
                diagnostics=dict(result.diagnostics),
            )
        )
        if verbose:
            label = "VIF" if reason == "vif" else "loading"
            print(
                f"[stage 2] iter {iteration:>3}: drop {task!r} ({label}={value:.3f}) "
                f"-> {len(current.indicators)} tasks, "
                f"TC={result.diagnostics['TC']:.3f} "
                f"D_div={result.diagnostics['D_div']:.3f} "
                f"D_valid={result.diagnostics['D_valid']:.3f}"
            )
    else:
        raise RuntimeError(
            f"Stage 2 did not stabilise within {max_iterations} iterations; "
            "raise max_iterations or relax the thresholds."
        )

    return PruningResult(
        initial_spec=spec,
        final_spec=current,
        initial_result=initial_result,
        final_result=result,
        history=history,
        protected=protected_final,
    )


def _blocked_violations(result: SEMResult, protected: set) -> List[str]:
    """Protected tasks that still breach a threshold, for an honest stop message."""
    thresholds = result.spec.thresholds
    blocked = []
    for task in result.spec.indicators:
        if task not in protected:
            continue
        vif = result.vifs.get(task, np.nan)
        loading = abs(result.loadings.get(task, np.nan))
        if (np.isfinite(vif) and vif > thresholds.vif_max) or (
            np.isfinite(loading) and loading < thresholds.loading_min
        ):
            blocked.append(task)
    return blocked


def ablate_thresholds(
    data: pd.DataFrame,
    spec: BenchmarkSpec,
    grid: List[Tuple[float, float]],
    scheme: str = "path",
    vif_scope: str = "construct",
) -> pd.DataFrame:
    """Sweep (delta_VIF, lambda_min) pairs and tabulate the resulting benchmarks.

    Reproduces the ablation table: one row per threshold pair with the retained task
    count and the three diagnostics.
    """
    from dataclasses import replace

    from .config import Thresholds

    rows = []
    for vif_max, loading_min in grid:
        candidate = replace(
            spec,
            thresholds=Thresholds(
                vif_max=vif_max,
                loading_min=loading_min,
                min_indicators=spec.thresholds.min_indicators,
            ),
        )
        outcome = run_pipeline(
            data, candidate, scheme=scheme, vif_scope=vif_scope, verbose=False
        )
        diagnostics = outcome.final_result.diagnostics
        rows.append(
            {
                "delta_VIF": vif_max,
                "lambda_min": loading_min,
                "task_num": diagnostics["task_num"],
                "TC": diagnostics["TC"],
                "D_div": diagnostics["D_div"],
                "D_valid": diagnostics["D_valid"],
                "overall": diagnostics["overall"],
                "retained": ", ".join(outcome.retained_tasks),
            }
        )
    return pd.DataFrame(rows)
