"""Diagnostic metrics for benchmark quality.

Implements the three benchmark-level scores of Section III-C plus the indicator-level
statistics (HTMT, VIF) the pruning loop consumes:

    D_div   = min(1, 1 / (2 * max_{i != j} HTMT_ij))       dimensional diversity
    TC      = mean(|outer loading|) over all indicators      task contribution
    D_valid = (prod_j VIF_j) ^ (-1/n)                        indicator validity
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _pairwise_mean_abs_corr(corr: pd.DataFrame, rows: Sequence[str], cols: Sequence[str]) -> float:
    """Mean absolute correlation over the block ``rows`` x ``cols``."""
    block = corr.loc[rows, cols].to_numpy(dtype=float)
    return float(np.nanmean(np.abs(block)))


def _mean_abs_within(corr: pd.DataFrame, items: Sequence[str]) -> float:
    """Mean absolute correlation over the strictly-lower triangle of a block.

    This is the monotrait-heteromethod average. Single-indicator constructs have no
    within-construct pair, so HTMT is undefined for them (NaN).
    """
    if len(items) < 2:
        return float("nan")
    block = corr.loc[items, items].to_numpy(dtype=float)
    lower = np.tril_indices(len(items), k=-1)
    return float(np.nanmean(np.abs(block[lower])))


def htmt_matrix(data: pd.DataFrame, constructs: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Heterotrait-monotrait ratio of correlations between every construct pair.

    Args:
        data: observations x indicators.
        constructs: construct name -> indicator columns.

    Returns:
        Symmetric matrix indexed by construct name, with NaN on the diagonal and for
        any pair involving a single-indicator construct.
    """
    names = list(constructs)
    corr = data.loc[:, [c for inds in constructs.values() for c in inds]].corr()
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    within = {lv: _mean_abs_within(corr, list(constructs[lv])) for lv in names}
    for a_idx, a in enumerate(names):
        for b in names[a_idx + 1:]:
            denom = within[a] * within[b]
            if not np.isfinite(denom) or denom <= 0:
                value = float("nan")
            else:
                hetero = _pairwise_mean_abs_corr(corr, list(constructs[a]), list(constructs[b]))
                value = hetero / np.sqrt(denom)
            out.loc[a, b] = value
            out.loc[b, a] = value
    return out


def max_htmt(matrix: pd.DataFrame) -> float:
    """Largest off-diagonal HTMT value; NaN when no valid pair exists."""
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmax(values))


def dimensional_diversity(matrix: pd.DataFrame) -> float:
    """D_div = min(1, 1 / (2 * max HTMT)) -- Eq. (2) of the paper."""
    worst = max_htmt(matrix)
    if not np.isfinite(worst) or worst <= 0:
        return float("nan")
    return float(min(1.0, 1.0 / (2.0 * worst)))


def task_contribution(loadings: Mapping[str, float]) -> float:
    """TC = mean absolute outer loading across all retained indicators -- Eq. (3).

    The paper writes TC as a double sum normalised by the total indicator count, which
    reduces to the mean of |lambda| over the benchmark.
    """
    values = np.asarray([abs(float(v)) for v in loadings.values()], dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def indicator_validity(vifs: Mapping[str, float]) -> float:
    """D_valid = inverse geometric mean of indicator VIFs.

    Single-indicator constructs contribute VIF = 1 exactly (no collinearity possible),
    which is the neutral element of a geometric mean, so they neither help nor hurt.
    """
    values = np.asarray([float(v) for v in vifs.values()], dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    return float(np.exp(-np.mean(np.log(values))))


def vif_scores(
    data: pd.DataFrame,
    constructs: Mapping[str, Sequence[str]],
    scope: str = "construct",
) -> Dict[str, float]:
    """Variance inflation factor per indicator -- Eq. (4).

    VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing indicator j on the
    remaining indicators in its reference set.

    Args:
        data: observations x indicators.
        constructs: construct name -> indicator columns.
        scope: ``"construct"`` regresses each indicator on the others *within its own
            construct* (the convention behind the paper's tables, where two-indicator
            constructs share one VIF value); ``"benchmark"`` regresses on all other
            retained indicators across the whole benchmark.

    Returns:
        Indicator name -> VIF. Indicators with no reference peers get 1.0. Perfectly
        collinear indicators get ``inf``.
    """
    if scope not in {"construct", "benchmark"}:
        raise ValueError("scope must be 'construct' or 'benchmark'")

    all_indicators = [c for inds in constructs.values() for c in inds]
    out: Dict[str, float] = {}
    for lv, indicators in constructs.items():
        peers_pool = list(indicators) if scope == "construct" else all_indicators
        for target in indicators:
            others = [c for c in peers_pool if c != target]
            out[target] = _vif_single(data, target, others)
    return out


def _vif_single(data: pd.DataFrame, target: str, others: Sequence[str]) -> float:
    """VIF of ``target`` regressed on ``others`` via least squares with an intercept."""
    if not others:
        return 1.0
    y = data[target].to_numpy(dtype=float)
    x = data.loc[:, list(others)].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(y)), x])

    centred = y - y.mean()
    total_ss = float(centred @ centred)
    if total_ss <= 0:  # constant indicator carries no variance to inflate
        return 1.0

    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    r_squared = 1.0 - float(residual @ residual) / total_ss
    r_squared = min(max(r_squared, 0.0), 1.0)
    if r_squared >= 1.0 - 1e-12:
        return float("inf")
    return float(1.0 / (1.0 - r_squared))


def cronbach_alpha(data: pd.DataFrame, indicators: Sequence[str]) -> float:
    """Cronbach's alpha for one construct's indicator block -- Eq. (6)."""
    if len(indicators) < 2:
        return float("nan")
    block = data.loc[:, list(indicators)].astype(float)
    k = block.shape[1]
    item_variance = block.var(ddof=1).sum()
    total_variance = block.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return float("nan")
    return float(k / (k - 1) * (1.0 - item_variance / total_variance))


def composite_reliability(loadings: Sequence[float]) -> float:
    """CR = (sum lambda)^2 / ((sum lambda)^2 + sum theta), theta_i = 1 - lambda_i^2 -- Eq. (7)."""
    values = np.asarray([abs(float(v)) for v in loadings], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    squared_sum = float(values.sum()) ** 2
    error = float(np.sum(1.0 - values ** 2))
    denominator = squared_sum + error
    if denominator <= 0:
        return float("nan")
    return float(squared_sum / denominator)


def average_variance_extracted(loadings: Sequence[float]) -> float:
    """AVE = mean(lambda^2), the convergent-validity statistic of Eq. (8).

    With standardised indicators and unit latent variance, Eq. (8) collapses to the
    mean squared loading.
    """
    values = np.asarray([float(v) for v in loadings], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values ** 2))


def srmr(observed: pd.DataFrame, implied: pd.DataFrame) -> float:
    """Standardised root mean square residual between two correlation matrices.

    Averaged over the strictly-lower triangle, following the PLS-SEM convention of
    scoring only the off-diagonal (the diagonal is 1 by construction).
    """
    common = [c for c in observed.columns if c in implied.columns]
    if len(common) < 2:
        return float("nan")
    obs = observed.loc[common, common].to_numpy(dtype=float)
    imp = implied.loc[common, common].to_numpy(dtype=float)
    lower = np.tril_indices(len(common), k=-1)
    residual = obs[lower] - imp[lower]
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(residual ** 2)))


def overall_score(
    scores: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Weighted task aggregation from the supplement: sum(w_i s_i) / sum(w_i).

    ``weights`` defaults to uniform, which reproduces a plain mean. Pass the fitted
    outer loadings to weight each subtask by its task contribution.
    """
    keys = [k for k in scores if weights is None or k in weights]
    if not keys:
        return float("nan")
    s = np.asarray([float(scores[k]) for k in keys], dtype=float)
    w = (
        np.ones_like(s)
        if weights is None
        else np.asarray([abs(float(weights[k])) for k in keys], dtype=float)
    )
    mask = np.isfinite(s) & np.isfinite(w)
    if not mask.any() or w[mask].sum() <= 0:
        return float("nan")
    return float(np.sum(w[mask] * s[mask]) / np.sum(w[mask]))


def summarise(
    loadings: Mapping[str, float],
    vifs: Mapping[str, float],
    htmt: pd.DataFrame,
) -> Dict[str, float]:
    """Bundle the three benchmark-level diagnostics plus their unweighted sum.

    ``overall`` is the additive composite reported in the ablation table
    (TC + D_div + D_valid), not the weighted task aggregation of ``overall_score``.
    """
    tc = task_contribution(loadings)
    d_div = dimensional_diversity(htmt)
    d_valid = indicator_validity(vifs)
    parts = [tc, d_div, d_valid]
    overall = float(np.nansum(parts)) if any(np.isfinite(p) for p in parts) else float("nan")
    return {
        "task_num": len(loadings),
        "TC": tc,
        "D_div": d_div,
        "D_valid": d_valid,
        "overall": overall,
    }
