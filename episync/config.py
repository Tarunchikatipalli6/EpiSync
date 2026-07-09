from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    dna_promoter_upstream: int
    dna_promoter_downstream: int
    dna_delta_cutoff: float
    dna_min_cpg_sites: int
    m6a_prob_threshold: float
    m6a_min_reads: int
    m6a_delta_mod_ratio_cutoff: float
    isoform_dif_cutoff: float
    isoform_q_value_cutoff: float
    expr_log2fc_cutoff: float
    expr_padj_cutoff: float
    overlap_dna_window: int
    scoring_strong: float
    scoring_moderate: float
    scoring_weak: float
    scoring_direction_multiplier: float
    sensitivity_dna_delta: List[float]
    sensitivity_m6a_delta: List[float]


def _require(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def _as_positive_int(value, key: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config key '{key}' must be a positive integer, got: {value!r}")
    return value


def _as_nonnegative_float(value, key: str) -> float:
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise ValueError(f"Config key '{key}' must be a non-negative number, got: {value!r}")
    return float(value)


def _as_float_list(value, key: str) -> List[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config key '{key}' must be a non-empty list of numbers")
    out: List[float] = []
    for v in value:
        if not isinstance(v, (int, float)):
            raise ValueError(f"Config key '{key}' contains non-numeric value: {v!r}")
        out.append(float(v))
    return out


def load_config(path: str = "config.yaml") -> Config:
    """Load and validate config.yaml into a Config dataclass."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    dna = _require(raw, "dna")
    m6a = _require(raw, "m6a")
    isoform = _require(raw, "isoform")
    expression = _require(raw, "expression")
    overlap = _require(raw, "overlap")
    scoring = _require(raw, "scoring")
    sensitivity = _require(raw, "sensitivity")

    cfg = Config(
        dna_promoter_upstream=_as_positive_int(_require(dna, "promoter_upstream"), "dna.promoter_upstream"),
        dna_promoter_downstream=_as_positive_int(_require(dna, "promoter_downstream"), "dna.promoter_downstream"),
        dna_delta_cutoff=_as_nonnegative_float(_require(dna, "delta_cutoff"), "dna.delta_cutoff"),
        dna_min_cpg_sites=_as_positive_int(_require(dna, "min_cpg_sites"), "dna.min_cpg_sites"),
        m6a_prob_threshold=_as_nonnegative_float(_require(m6a, "prob_threshold"), "m6a.prob_threshold"),
        m6a_min_reads=_as_positive_int(_require(m6a, "min_reads_per_condition"), "m6a.min_reads_per_condition"),
        m6a_delta_mod_ratio_cutoff=_as_nonnegative_float(
            _require(m6a, "delta_mod_ratio_cutoff"), "m6a.delta_mod_ratio_cutoff"
        ),
        isoform_dif_cutoff=_as_nonnegative_float(_require(isoform, "dif_cutoff"), "isoform.dif_cutoff"),
        isoform_q_value_cutoff=_as_nonnegative_float(_require(isoform, "q_value_cutoff"), "isoform.q_value_cutoff"),
        expr_log2fc_cutoff=_as_nonnegative_float(_require(expression, "log2fc_cutoff"), "expression.log2fc_cutoff"),
        expr_padj_cutoff=_as_nonnegative_float(_require(expression, "padj_cutoff"), "expression.padj_cutoff"),
        overlap_dna_window=_as_positive_int(_require(overlap, "dna_window"), "overlap.dna_window"),
        scoring_strong=_as_nonnegative_float(_require(scoring, "strong"), "scoring.strong"),
        scoring_moderate=_as_nonnegative_float(_require(scoring, "moderate"), "scoring.moderate"),
        scoring_weak=_as_nonnegative_float(_require(scoring, "weak"), "scoring.weak"),
        scoring_direction_multiplier=_as_nonnegative_float(
            _require(scoring, "direction_multiplier"), "scoring.direction_multiplier"
        ),
        sensitivity_dna_delta=_as_float_list(_require(sensitivity, "dna_delta"), "sensitivity.dna_delta"),
        sensitivity_m6a_delta=_as_float_list(_require(sensitivity, "m6a_delta"), "sensitivity.m6a_delta"),
    )

    if not (0.0 <= cfg.m6a_prob_threshold <= 1.0):
        raise ValueError("m6a.prob_threshold must be between 0 and 1")
    if not (0.0 <= cfg.isoform_q_value_cutoff <= 1.0):
        raise ValueError("isoform.q_value_cutoff must be between 0 and 1")
    if not (0.0 <= cfg.expr_padj_cutoff <= 1.0):
        raise ValueError("expression.padj_cutoff must be between 0 and 1")

    return cfg
