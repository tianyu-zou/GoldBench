"""Specification objects for GoldBench: constructs, thresholds and model structure.

A *benchmark specification* declares which observed task indicators belong to which
latent cognitive construct (Perception / Memory / Reasoning in GoldBench), plus the
external human-preference anchor used to align the top-level construct with human
judgement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Structural layouts supported when wiring the latent constructs together.
#   anchor        : first-order constructs -> a single-indicator anchor LV (human eval).
#                   This is the layout described in the paper, where the Chatbot Arena
#                   score is an external observed indicator of the top-level construct.
#   hoc_two_stage : first-order constructs -> second-order construct (disjoint two-stage),
#                   optionally anchored to human eval afterwards.
#   chain         : Perception -> Memory -> Reasoning, the Piaget developmental ordering.
#                   Use when no human-preference column is available.
STRUCTURES = ("anchor", "hoc_two_stage", "chain")


@dataclass(frozen=True)
class Thresholds:
    """Cut-offs driving the iterative pruning loop (Algorithm 1, Stage 2)."""

    vif_max: float = 5.0          # delta_VIF   : drop indicators above this
    loading_min: float = 0.75     # lambda_min  : drop indicators below this
    min_indicators: int = 2       # keep every construct measurable (VIF/HTMT need >= 2)

    def __post_init__(self) -> None:
        if self.vif_max <= 1.0:
            raise ValueError("vif_max must be > 1 (VIF is bounded below by 1)")
        if not 0.0 < self.loading_min < 1.0:
            raise ValueError("loading_min must lie in (0, 1)")
        if self.min_indicators < 1:
            raise ValueError("min_indicators must be >= 1")

    def as_dict(self) -> Dict[str, float]:
        return {
            "vif_max": self.vif_max,
            "loading_min": self.loading_min,
            "min_indicators": self.min_indicators,
        }


@dataclass
class BenchmarkSpec:
    """Declarative description of the measurement model to estimate and prune.

    Attributes:
        constructs: latent construct name -> list of observed indicator columns.
            Indicator names must match column names in the score matrix.
        human_column: column holding the human-preference score (e.g. Chatbot Arena
            Elo). Required for ``anchor``; optional elsewhere.
        structure: how latent constructs are wired together, see ``STRUCTURES``.
        top_name: name given to the aggregate construct (second-order or anchor target).
        human_name: latent-variable name wrapping ``human_column``. Kept distinct from
            the column name because plspm forbids an MV and an LV sharing a name.
        thresholds: pruning cut-offs.
        name: label used in reports and output filenames.
    """

    constructs: Dict[str, List[str]]
    human_column: Optional[str] = None
    structure: str = "anchor"
    top_name: str = "Benchmark"
    human_name: str = "HumanPref"
    thresholds: Thresholds = field(default_factory=Thresholds)
    name: str = "benchmark"

    def __post_init__(self) -> None:
        if not self.constructs:
            raise ValueError("spec must declare at least one latent construct")
        if self.structure not in STRUCTURES:
            raise ValueError(
                f"unknown structure {self.structure!r}; expected one of {STRUCTURES}"
            )
        # Preserve declaration order but copy so callers cannot mutate us in place.
        self.constructs = {k: list(v) for k, v in self.constructs.items()}

        for lv, indicators in self.constructs.items():
            if not indicators:
                raise ValueError(f"construct {lv!r} has no indicators")
            if len(set(indicators)) != len(indicators):
                raise ValueError(f"construct {lv!r} lists a duplicate indicator")

        seen: Dict[str, str] = {}
        for lv, indicators in self.constructs.items():
            for ind in indicators:
                if ind in seen:
                    raise ValueError(
                        f"indicator {ind!r} is assigned to both {seen[ind]!r} and {lv!r}; "
                        "a reflective indicator must load on exactly one construct"
                    )
                seen[ind] = lv

        reserved = set(self.constructs) | {self.top_name, self.human_name}
        clashes = reserved & set(seen)
        if clashes:
            raise ValueError(
                f"indicator names collide with latent-variable names: {sorted(clashes)}"
            )
        if self.top_name in self.constructs:
            raise ValueError(f"top_name {self.top_name!r} collides with a construct name")
        if self.human_name in self.constructs:
            raise ValueError(f"human_name {self.human_name!r} collides with a construct name")

        if self.structure == "anchor" and not self.human_column:
            raise ValueError(
                "structure='anchor' needs human_column; use structure='chain' or "
                "'hoc_two_stage' when no human-preference score is available"
            )
        if self.structure == "chain" and len(self.constructs) < 2:
            raise ValueError("structure='chain' needs at least two constructs")
        if self.human_column and self.human_column in seen:
            raise ValueError(
                f"human_column {self.human_column!r} is also listed as a task indicator"
            )

    # -- derived views ----------------------------------------------------------

    @property
    def indicators(self) -> List[str]:
        """All task indicators, in construct declaration order."""
        return [ind for inds in self.constructs.values() for ind in inds]

    @property
    def construct_names(self) -> List[str]:
        return list(self.constructs)

    def construct_of(self, indicator: str) -> str:
        for lv, indicators in self.constructs.items():
            if indicator in indicators:
                return lv
        raise KeyError(f"indicator {indicator!r} is not in this spec")

    def required_columns(self) -> List[str]:
        cols = list(self.indicators)
        if self.human_column:
            cols.append(self.human_column)
        return cols

    def without(self, indicators: Iterable[str]) -> "BenchmarkSpec":
        """Return a copy with ``indicators`` removed, dropping emptied constructs."""
        drop = set(indicators)
        pruned = {
            lv: [i for i in inds if i not in drop]
            for lv, inds in self.constructs.items()
        }
        pruned = {lv: inds for lv, inds in pruned.items() if inds}
        if not pruned:
            raise ValueError("pruning removed every construct; loosen the thresholds")
        return replace(self, constructs=pruned)

    # -- serialisation ----------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "structure": self.structure,
            "top_name": self.top_name,
            "human_name": self.human_name,
            "human_column": self.human_column,
            "thresholds": self.thresholds.as_dict(),
            "constructs": self.constructs,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "BenchmarkSpec":
        payload = dict(payload)
        thresholds = payload.pop("thresholds", None) or {}
        return cls(thresholds=Thresholds(**thresholds), **payload)

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkSpec":
        """Load a spec from JSON or YAML (YAML needs PyYAML installed)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "reading YAML specs requires PyYAML: pip install pyyaml"
                ) from exc
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        payload.setdefault("name", path.stem)
        return cls.from_dict(payload)

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
