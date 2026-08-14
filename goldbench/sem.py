"""PLS-SEM estimation of the reflective measurement model, backed by ``plspm``.

The paper's estimator is standard PLS-SEM: alternate between latent-score estimation
(weighted composites of the indicators) and weight updating via inner/outer
approximation until the weights stabilise. We delegate that loop to ``plspm`` rather
than reimplementing it, and add the pieces the pipeline needs on top:

* the structural layouts of :mod:`goldbench.config` (anchor / two-stage HOC / chain);
* block-scoped VIF, HTMT and the reliability table;
* a model-implied correlation matrix so SRMR can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import plspm.mode as plspm_mode
import plspm.scale as plspm_scale
import plspm.scheme as plspm_scheme
from plspm.config import Config, MV, Structure
from plspm.plspm import Plspm

from .config import BenchmarkSpec
from . import metrics as M

SCHEMES = {
    "path": plspm_scheme.Scheme.PATH,
    "centroid": plspm_scheme.Scheme.CENTROID,
    "factorial": plspm_scheme.Scheme.FACTORIAL,
}


@dataclass
class SEMResult:
    """Everything a single PLS-SEM fit produces, in tidy form.

    Attributes:
        spec: the specification that was estimated.
        loadings: indicator -> outer loading (correlation with its own construct).
        weights: indicator -> outer weight (the composite coefficients).
        vifs: indicator -> VIF, computed with ``vif_scope``.
        htmt: construct x construct HTMT matrix.
        reliability: per-construct alpha / rho_A / CR / AVE table.
        scores: observations x construct latent scores.
        path_coefficients: structural coefficient matrix (rows = targets).
        r_squared: construct -> R^2 (0 for exogenous constructs).
        srmr: standardised root mean square residual of the indicator correlations.
        diagnostics: TC / D_div / D_valid / overall for this fit.
        converged: whether the PLS loop reached the tolerance.
        iterations: iteration cap used for the fit.
    """

    spec: BenchmarkSpec
    loadings: Dict[str, float]
    weights: Dict[str, float]
    vifs: Dict[str, float]
    htmt: pd.DataFrame
    reliability: pd.DataFrame
    scores: pd.DataFrame
    path_coefficients: pd.DataFrame
    r_squared: Dict[str, float]
    srmr: float
    diagnostics: Dict[str, float]
    converged: bool = True
    iterations: int = 300

    @property
    def task_loadings(self) -> Dict[str, float]:
        """Loadings restricted to task indicators (excludes the human-eval anchor)."""
        return {k: v for k, v in self.loadings.items() if k in set(self.spec.indicators)}

    def loading_table(self) -> pd.DataFrame:
        """Per-indicator report: construct, loading, weight, VIF and threshold verdicts."""
        rows = []
        for indicator in self.spec.indicators:
            loading = self.loadings.get(indicator, float("nan"))
            vif = self.vifs.get(indicator, float("nan"))
            rows.append(
                {
                    "indicator": indicator,
                    "construct": self.spec.construct_of(indicator),
                    "outer_loading": loading,
                    "outer_weight": self.weights.get(indicator, float("nan")),
                    "VIF": vif,
                    "loading_ok": bool(abs(loading) >= self.spec.thresholds.loading_min)
                    if np.isfinite(loading)
                    else False,
                    "vif_ok": bool(vif <= self.spec.thresholds.vif_max)
                    if np.isfinite(vif)
                    else False,
                }
            )
        return pd.DataFrame(rows)


def _build_path(spec: BenchmarkSpec, constructs: Sequence[str]) -> pd.DataFrame:
    """Structural (path) matrix for the requested layout.

    plspm reads ``path.loc[target, source] == 1`` as "source -> target".
    """
    constructs = list(constructs)

    if spec.structure == "anchor":
        structure = Structure()
        structure.add_path(constructs, [spec.human_name])
        return structure.path()

    if spec.structure == "chain":
        # Piaget ordering: every earlier layer predicts every later one.
        structure = Structure()
        for position, target in enumerate(constructs[1:], start=1):
            structure.add_path(constructs[:position], [target])
        return structure.path()

    # hoc_two_stage: stage 1 estimates the first-order blocks only. When a human anchor
    # is present we route the constructs through it so stage 1 is not path-less.
    if spec.human_column:
        structure = Structure()
        structure.add_path(constructs, [spec.human_name])
        return structure.path()
    structure = Structure()
    for position, target in enumerate(constructs[1:], start=1):
        structure.add_path(constructs[:position], [target])
    return structure.path()


def _fit_plspm(
    data: pd.DataFrame,
    spec: BenchmarkSpec,
    constructs: Mapping[str, Sequence[str]],
    scheme: str,
    iterations: int,
    tolerance: float,
) -> Plspm:
    """Configure and run one plspm estimation over the given construct blocks."""
    path = _build_path(spec, list(constructs))
    config = Config(path, scaled=True, default_scale=plspm_scale.Scale.NUM)
    for lv, indicators in constructs.items():
        # Mode A = correlation weights, the reflective specification of Section III-A.
        config.add_lv(lv, plspm_mode.Mode.A, *[MV(str(i)) for i in indicators])
    if spec.human_name in path.index and spec.human_name not in constructs:
        config.add_lv(spec.human_name, plspm_mode.Mode.A, MV(str(spec.human_column)))
    return Plspm(
        data,
        config,
        SCHEMES[scheme],
        iterations=iterations,
        tolerance=tolerance,
    )


def _implied_correlations(
    loadings: Mapping[str, float],
    constructs: Mapping[str, Sequence[str]],
    construct_corr: pd.DataFrame,
) -> pd.DataFrame:
    """Model-implied indicator correlations under the reflective model.

    For indicators i in construct a and j in construct b, the reflective model implies
    corr(x_i, x_j) = lambda_i * corr(xi_a, xi_b) * lambda_j for i != j, and 1 on the
    diagonal. Comparing this against the observed correlations yields SRMR.
    """
    indicators = [i for inds in constructs.values() for i in inds]
    owner = {i: lv for lv, inds in constructs.items() for i in inds}
    implied = pd.DataFrame(np.nan, index=indicators, columns=indicators, dtype=float)
    for i in indicators:
        implied.loc[i, i] = 1.0
        for j in indicators:
            if i == j:
                continue
            a, b = owner[i], owner[j]
            between = 1.0 if a == b else float(construct_corr.loc[a, b])
            implied.loc[i, j] = float(loadings.get(i, np.nan)) * between * float(
                loadings.get(j, np.nan)
            )
    return implied


def _reliability_table(
    data: pd.DataFrame,
    constructs: Mapping[str, Sequence[str]],
    loadings: Mapping[str, float],
    model: Plspm,
) -> pd.DataFrame:
    """Per-construct alpha, rho_A (Dillon-Goldstein), composite reliability and AVE."""
    unidim = model.unidimensionality()
    rows = []
    for lv, indicators in constructs.items():
        block = [abs(float(loadings.get(i, np.nan))) for i in indicators]
        rho = float("nan")
        if lv in unidim.index and "dillon_goldstein_rho" in unidim.columns:
            rho = float(unidim.loc[lv, "dillon_goldstein_rho"])
        rows.append(
            {
                "construct": lv,
                "n_indicators": len(indicators),
                "cronbach_alpha": M.cronbach_alpha(data, list(indicators)),
                "rho_A": rho,
                "composite_reliability": M.composite_reliability(block),
                "AVE": M.average_variance_extracted(block),
            }
        )
    return pd.DataFrame(rows).set_index("construct")


def fit(
    data: pd.DataFrame,
    spec: BenchmarkSpec,
    scheme: str = "path",
    iterations: int = 300,
    tolerance: float = 1e-7,
    vif_scope: str = "construct",
) -> SEMResult:
    """Estimate the reflective PLS-SEM model described by ``spec``.

    Args:
        data: models (rows) x task scores (columns). Missing values are mean-imputed
            per column by ``plspm``; drop or impute upstream if you want other
            behaviour. Indicators are standardised internally, so raw score scales
            (0-100, 0-200, Elo) may be mixed freely.
        spec: measurement model and thresholds.
        scheme: inner weighting scheme -- ``"path"``, ``"centroid"`` or ``"factorial"``.
        iterations: cap on the PLS alternation.
        tolerance: convergence tolerance on the weight change, i.e. the epsilon of
            ``max_ij |w_ij^(t+1) - w_ij^(t)| < epsilon``.
        vif_scope: ``"construct"`` (paper convention) or ``"benchmark"``.

    Returns:
        A populated :class:`SEMResult`.

    Raises:
        ValueError: if required columns are missing or a construct block is degenerate.
        RuntimeError: if the PLS loop fails to converge within ``iterations``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {sorted(SCHEMES)}")

    missing = [c for c in spec.required_columns() if c not in data.columns]
    if missing:
        raise ValueError(f"score matrix is missing required columns: {missing}")

    frame = data.loc[:, spec.required_columns()].apply(pd.to_numeric, errors="coerce")
    constant = [c for c in frame.columns if frame[c].std(ddof=1) == 0 or frame[c].isna().all()]
    if constant:
        raise ValueError(
            f"columns have zero variance and cannot be modelled: {constant}. "
            "Drop these tasks -- a saturated task carries no discriminative signal."
        )

    constructs = {lv: list(inds) for lv, inds in spec.constructs.items()}
    try:
        model = _fit_plspm(frame, spec, constructs, scheme, iterations, tolerance)
    except Exception as exc:  # plspm raises bare Exception on non-convergence
        if "converge" in str(exc).lower():
            raise RuntimeError(
                f"PLS-SEM did not converge in {iterations} iterations. Try scheme='centroid', "
                "raise iterations, or check for near-duplicate indicators."
            ) from exc
        raise

    outer = model.outer_model()
    loadings = {str(i): float(outer.loc[i, "loading"]) for i in outer.index}
    weights = {str(i): float(outer.loc[i, "weight"]) for i in outer.index}
    scores = model.scores()

    if spec.structure == "hoc_two_stage":
        scores, loadings, weights = _second_stage(
            frame, spec, constructs, scores, loadings, weights, scheme, iterations, tolerance
        )

    task_frame = frame.loc[:, spec.indicators]
    vifs = M.vif_scores(task_frame, constructs, scope=vif_scope)
    htmt = M.htmt_matrix(task_frame, constructs)
    reliability = _reliability_table(frame, constructs, loadings, model)

    inner_summary = model.inner_summary()
    r_squared = {
        str(lv): float(inner_summary.loc[lv, "r_squared"])
        for lv in inner_summary.index
        if "r_squared" in inner_summary.columns
    }

    construct_corr = scores.loc[:, list(constructs)].corr()
    implied = _implied_correlations(loadings, constructs, construct_corr)
    srmr_value = M.srmr(task_frame.corr(), implied)

    task_loadings = {k: v for k, v in loadings.items() if k in set(spec.indicators)}
    diagnostics = M.summarise(task_loadings, vifs, htmt)

    return SEMResult(
        spec=spec,
        loadings=loadings,
        weights=weights,
        vifs=vifs,
        htmt=htmt,
        reliability=reliability,
        scores=scores,
        path_coefficients=model.path_coefficients(),
        r_squared=r_squared,
        srmr=srmr_value,
        diagnostics=diagnostics,
        converged=True,
        iterations=iterations,
    )


def _second_stage(
    frame: pd.DataFrame,
    spec: BenchmarkSpec,
    constructs: Mapping[str, Sequence[str]],
    stage1_scores: pd.DataFrame,
    loadings: Dict[str, float],
    weights: Dict[str, float],
    scheme: str,
    iterations: int,
    tolerance: float,
):
    """Disjoint two-stage higher-order construct.

    Stage 1 estimates the first-order blocks; stage 2 treats their latent scores as the
    indicators of ``spec.top_name``. We implement this by hand because ``plspm``'s own
    ``add_higher_order`` reuses the first-order LV names as second-stage MV names, which
    its validator then rejects.
    """
    proxies = pd.DataFrame(
        {f"{lv}__score": stage1_scores[lv].to_numpy(dtype=float) for lv in constructs},
        index=frame.index,
    )
    top_block = {spec.top_name: list(proxies.columns)}

    if spec.human_column:
        proxies[spec.human_column] = frame[spec.human_column].to_numpy(dtype=float)
        path = pd.DataFrame(
            0,
            index=[spec.top_name, spec.human_name],
            columns=[spec.top_name, spec.human_name],
        )
        path.loc[spec.human_name, spec.top_name] = 1
    else:
        # A lone construct has no structural path; a single-block PLS fit still returns
        # the loadings we need, so estimate the composite directly from its correlations.
        path = None

    if path is None:
        standardised = (proxies - proxies.mean()) / proxies.std(ddof=1)
        composite = standardised.mean(axis=1)
        composite = (composite - composite.mean()) / composite.std(ddof=1)
        second_loadings = {
            column: float(standardised[column].corr(composite)) for column in standardised.columns
        }
        second_weights = {column: 1.0 / len(standardised.columns) for column in standardised.columns}
        top_scores = composite
    else:
        config = Config(path, scaled=True, default_scale=plspm_scale.Scale.NUM)
        config.add_lv(
            spec.top_name,
            plspm_mode.Mode.A,
            *[MV(str(c)) for c in top_block[spec.top_name]],
        )
        config.add_lv(spec.human_name, plspm_mode.Mode.A, MV(str(spec.human_column)))
        stage2 = Plspm(proxies, config, SCHEMES[scheme], iterations=iterations, tolerance=tolerance)
        outer2 = stage2.outer_model()
        second_loadings = {
            str(i): float(outer2.loc[i, "loading"])
            for i in outer2.index
            if str(i) in set(top_block[spec.top_name])
        }
        second_weights = {
            str(i): float(outer2.loc[i, "weight"])
            for i in outer2.index
            if str(i) in set(top_block[spec.top_name])
        }
        top_scores = stage2.scores()[spec.top_name]

    scores = stage1_scores.copy()
    scores[spec.top_name] = np.asarray(top_scores, dtype=float)

    # Surface the second-order loadings under their construct names, so the top-level
    # block is reported as Perception/Memory/Reasoning -> Gold.
    for column, value in second_loadings.items():
        loadings[column.replace("__score", "")] = value
    for column, value in second_weights.items():
        weights[column.replace("__score", "")] = value
    return scores, loadings, weights
