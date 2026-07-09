"""Isoform switching layer for EpiSync.

This module computes:
- per-gene switch properties from IsoformSwitchAnalyzeR switch_pairs data
- genomic gained/lost regions between switch isoform pairs
- region-type classification for changed regions
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyranges as pr

from episync.annotation import Annotation
from episync.config import Config


def _get_tx_exons(annotation: Annotation, transcript_id: str) -> pd.DataFrame:
    """Return exon intervals for transcript_id as DataFrame[Chromosome, Start, End, Strand]."""
    tx = annotation.genes

    # Primary source from transcript-indexed map if present in gene models
    for g in tx.values():
        if transcript_id in g.transcripts:
            ex = g.transcripts[transcript_id]
            if not ex:
                return pd.DataFrame(columns=["Chromosome", "Start", "End", "Strand"])
            # Accept tuple/list-like exon coordinates
            rows = []
            for e in ex:
                if isinstance(e, dict):
                    chrom = e.get("chromosome", g.chromosome)
                    start = int(e["start"])
                    end = int(e["end"])
                    strand = e.get("strand", g.strand)
                else:
                    # expected tuple forms: (start,end) or (chrom,start,end,strand)
                    if len(e) == 2:
                        chrom = g.chromosome
                        start, end = int(e[0]), int(e[1])
                        strand = g.strand
                    else:
                        chrom, start, end = e[0], int(e[1]), int(e[2])
                        strand = e[3] if len(e) > 3 else g.strand
                if end > start:
                    rows.append((chrom, start, end, strand))
            return pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Strand"])

    # Fallback from gene_models_pr if transcript annotation exists
    exons = annotation.gene_models_pr.df.copy()
    if exons.empty:
        return pd.DataFrame(columns=["Chromosome", "Start", "End", "Strand"])

    tx_col = "transcript_id" if "transcript_id" in exons.columns else None
    if tx_col is None and "transcript_id" not in exons.columns and "transcript_id" in exons:
        tx_col = "transcript_id"

    if tx_col is None:
        attrs = exons.get("Attributes")
        if attrs is not None:
            mask = attrs.astype(str).str.contains(f'transcript_id "{transcript_id}"', regex=False)
            exons = exons.loc[mask]
    else:
        exons = exons.loc[exons[tx_col].astype(str) == str(transcript_id)]

    if "Feature" in exons.columns:
        exons = exons.loc[exons["Feature"].astype(str).str.lower() == "exon"]

    keep = [c for c in ["Chromosome", "Start", "End", "Strand"] if c in exons.columns]
    if len(keep) < 4:
        return pd.DataFrame(columns=["Chromosome", "Start", "End", "Strand"])
    out = exons[keep].copy()
    out = out.loc[out["End"] > out["Start"]]
    return out


def identify_changed_regions(iso_df: pd.DataFrame, annotation: Annotation) -> pd.DataFrame:
    """Identify LOST/GAINED exon regions for each switch pair via set differences.

    LOST  = exons in isoformID_A not in isoformID_B
    GAINED = exons in isoformID_B not in isoformID_A
    """
    if iso_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "chr",
                "start",
                "end",
                "strand",
                "region_type",
                "length_bp",
                "isoformID_A",
                "isoformID_B",
            ]
        )

    records: List[Tuple[str, str, int, int, str, str, int, str, str]] = []

    for row in iso_df.itertuples(index=False):
        gene_id = str(getattr(row, "gene_id"))
        iso_a = str(getattr(row, "isoformID_A"))
        iso_b = str(getattr(row, "isoformID_B"))

        a = _get_tx_exons(annotation, iso_a)
        b = _get_tx_exons(annotation, iso_b)

        if a.empty and b.empty:
            continue

        pra = pr.PyRanges(a) if not a.empty else pr.PyRanges()
        prb = pr.PyRanges(b) if not b.empty else pr.PyRanges()

        lost = pra.subtract(prb).df if not a.empty else pd.DataFrame()
        gained = prb.subtract(pra).df if not b.empty else pd.DataFrame()

        if not lost.empty:
            for r in lost.itertuples(index=False):
                length = int(r.End) - int(r.Start)
                if length <= 0:
                    continue
                records.append(
                    (
                        gene_id,
                        str(r.Chromosome),
                        int(r.Start),
                        int(r.End),
                        str(r.Strand),
                        "LOST",
                        length,
                        iso_a,
                        iso_b,
                    )
                )

        if not gained.empty:
            for r in gained.itertuples(index=False):
                length = int(r.End) - int(r.Start)
                if length <= 0:
                    continue
                records.append(
                    (
                        gene_id,
                        str(r.Chromosome),
                        int(r.Start),
                        int(r.End),
                        str(r.Strand),
                        "GAINED",
                        length,
                        iso_a,
                        iso_b,
                    )
                )

    return pd.DataFrame(
        records,
        columns=[
            "gene_id",
            "chr",
            "start",
            "end",
            "strand",
            "region_type",
            "length_bp",
            "isoformID_A",
            "isoformID_B",
        ],
    )


def classify_region_type(region_df: pd.DataFrame, annotation: Annotation) -> pd.DataFrame:
    """Classify each changed region as 5_UTR/CDS/3_UTR/intron by overlap."""
    if region_df.empty:
        out = region_df.copy()
        out["genomic_region_type"] = []
        return out

    out = region_df.copy()
    out["genomic_region_type"] = "intron"

    gtf = annotation.gene_models_pr.df.copy()
    if gtf.empty or "Feature" not in gtf.columns:
        return out

    regions = out.rename(
        columns={
            "chr": "Chromosome",
            "start": "Start",
            "end": "End",
            "strand": "Strand",
        }
    )
    pr_regions = pr.PyRanges(regions[["Chromosome", "Start", "End", "Strand"]].copy())

    feature_map = [
        ("five_prime_utr", "5_UTR"),
        ("three_prime_utr", "3_UTR"),
        ("CDS", "CDS"),
    ]

    idx_map = out.reset_index()[["index", "chr", "start", "end", "strand"]].rename(
        columns={
            "chr": "Chromosome",
            "start": "Start",
            "end": "End",
            "strand": "Strand",
        }
    )

    for feature, label in feature_map:
        feat = gtf.loc[gtf["Feature"] == feature]
        if feat.empty:
            continue
        cols = [c for c in ["Chromosome", "Start", "End", "Strand"] if c in feat.columns]
        if len(cols) < 4:
            continue
        pr_feat = pr.PyRanges(feat[cols].copy())
        overlaps = pr_regions.join(pr_feat).df
        if overlaps.empty:
            continue

        merged = overlaps.merge(idx_map, on=["Chromosome", "Start", "End", "Strand"], how="left")
        hit_idx = merged["index"].dropna().astype(int).unique().tolist()
        if hit_idx:
            out.loc[hit_idx, "genomic_region_type"] = label

    return out


def calculate_switch_properties(iso_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Compute per-gene switching properties and triage flags."""
    if iso_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "switched",
                "dIF",
                "q_value",
                "length_change_bp",
                "switch_direction",
            ]
        )

    df = iso_df.copy()

    # Normalize expected columns
    rename_map = {
        "geneID": "gene_id",
        "isoform_switch_q_value": "q_value",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    req = {"gene_id", "dIF", "q_value", "length_A", "length_B"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Isoform dataframe missing required columns: {sorted(missing)}")

    df["length_change_bp"] = df["length_B"] - df["length_A"]
    df["switch_direction"] = np.where(
        df["length_change_bp"] > 0,
        "LONGER",
        np.where(df["length_change_bp"] < 0, "SHORTER", "SAME"),
    )

    # Pick strongest switch per gene (min q_value then max |dIF|)
    df = df.sort_values(by=["gene_id", "q_value", "dIF"], key=lambda s: s.abs() if s.name == "dIF" else s)
    best = df.groupby("gene_id", as_index=False).first()

    best["switched"] = (best["q_value"] < float(config.isoform_q_value_cutoff)) & (
        best["dIF"].abs() >= float(config.isoform_dif_cutoff)
    )

    return best[
        [
            "gene_id",
            "switched",
            "dIF",
            "q_value",
            "length_change_bp",
            "switch_direction",
        ]
    ].copy()


def run_isoform_layer(
    iso_df: pd.DataFrame,
    annotation: Annotation,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full isoform layer pipeline.

    Returns:
      (switch_properties_df, changed_regions_df)
    """
    switches = calculate_switch_properties(iso_df, config)

    # Use standardized columns for region finding
    work = iso_df.copy()
    if "geneID" in work.columns and "gene_id" not in work.columns:
        work = work.rename(columns={"geneID": "gene_id"})

    changed = identify_changed_regions(work, annotation)
    changed = classify_region_type(changed, annotation)
    return switches, changed
