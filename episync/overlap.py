"""Overlap engine for EpiSync.

This module implements positional coincidence checks between changed
isoform regions and other molecular-layer signals using PyRanges joins.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import pyranges as pr

from episync.config import Config


def _to_pr_intervals(
    df: pd.DataFrame,
    chr_col: str,
    start_col: str,
    end_col: str,
    strand_col: str,
) -> pr.PyRanges:
    """Build PyRanges from interval DataFrame with canonical columns."""
    if df.empty:
        return pr.PyRanges()
    tmp = df.rename(
        columns={
            chr_col: "Chromosome",
            start_col: "Start",
            end_col: "End",
            strand_col: "Strand",
        }
    )[["Chromosome", "Start", "End", "Strand"]].copy()
    return pr.PyRanges(tmp)


def check_m6a_in_region(switch_regions: pd.DataFrame, m6a_sites: pd.DataFrame) -> pd.DataFrame:
    """Check whether differential m6A sites fall inside changed switch regions."""
    genes = (
        pd.DataFrame({"gene_id": switch_regions["gene_id"].dropna().astype(str).unique()})
        if not switch_regions.empty and "gene_id" in switch_regions.columns
        else pd.DataFrame(columns=["gene_id"])
    )

    if switch_regions.empty or m6a_sites.empty:
        if genes.empty:
            return pd.DataFrame(columns=["gene_id", "contains_m6a", "n_m6a_in_region", "m6a_region_types"])
        out = genes.copy()
        out["contains_m6a"] = False
        out["n_m6a_in_region"] = 0
        out["m6a_region_types"] = [[] for _ in range(len(out))]
        return out

    sw = switch_regions.copy()
    m6 = m6a_sites.copy()

    sw["_gid"] = sw["gene_id"].astype(str)
    m6["_gid"] = m6["gene_id"].astype(str)

    sw_int = sw.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End", "strand": "Strand"})
    m6_int = m6.rename(columns={"chr": "Chromosome", "genomic_position": "Start", "strand": "Strand"})
    m6_int["End"] = m6_int["Start"].astype(int) + 1

    pr_sw = pr.PyRanges(sw_int[["Chromosome", "Start", "End", "Strand", "_gid", "region_type"]])
    keep_cols = ["Chromosome", "Start", "End", "Strand", "_gid"]
    if "region_type" in m6_int.columns:
        keep_cols.append("region_type")
    pr_m6 = pr.PyRanges(m6_int[keep_cols])

    ov = pr_sw.join(pr_m6).df
    if ov.empty:
        out = genes.copy()
        out["contains_m6a"] = False
        out["n_m6a_in_region"] = 0
        out["m6a_region_types"] = [[] for _ in range(len(out))]
        return out

    # Ensure same gene only
    if "_gid_b" in ov.columns:
        ov = ov.loc[ov["_gid"] == ov["_gid_b"]].copy()

    if ov.empty:
        out = genes.copy()
        out["contains_m6a"] = False
        out["n_m6a_in_region"] = 0
        out["m6a_region_types"] = [[] for _ in range(len(out))]
        return out

    type_col = "region_type_b" if "region_type_b" in ov.columns else ("region_type" if "region_type" in ov.columns else None)

    agg = (
        ov.groupby("_gid", as_index=False)
        .agg(
            n_m6a_in_region=("Start_b", "size"),
            m6a_region_types=(type_col, lambda s: sorted([x for x in s.dropna().astype(str).unique().tolist()]))
            if type_col is not None
            else ("Start_b", lambda s: []),
        )
        .rename(columns={"_gid": "gene_id"})
    )
    agg["contains_m6a"] = agg["n_m6a_in_region"] > 0

    out = genes.merge(agg[["gene_id", "contains_m6a", "n_m6a_in_region", "m6a_region_types"]], on="gene_id", how="left")
    out["contains_m6a"] = out["contains_m6a"].fillna(False)
    out["n_m6a_in_region"] = out["n_m6a_in_region"].fillna(0).astype(int)
    out["m6a_region_types"] = out["m6a_region_types"].apply(lambda x: x if isinstance(x, list) else [])
    return out


def check_dna_near_region(
    switch_regions: pd.DataFrame,
    dna_sites: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Check if differential DNA methylation sites are near changed regions."""
    genes = (
        pd.DataFrame({"gene_id": switch_regions["gene_id"].dropna().astype(str).unique()})
        if not switch_regions.empty and "gene_id" in switch_regions.columns
        else pd.DataFrame(columns=["gene_id"])
    )

    if switch_regions.empty or dna_sites.empty:
        if genes.empty:
            return pd.DataFrame(columns=["gene_id", "dna_near_region", "n_dna_sites_near"])
        out = genes.copy()
        out["dna_near_region"] = False
        out["n_dna_sites_near"] = 0
        return out

    sw = switch_regions.copy()
    dna = dna_sites.copy()

    sw["_gid"] = sw["gene_id"].astype(str)
    dna["_gid"] = dna["gene_id"].astype(str)

    sw_int = sw.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End", "strand": "Strand"})
    dna_int = dna.rename(columns={"chr": "Chromosome", "position": "Start", "strand": "Strand"})
    dna_int["End"] = dna_int["Start"].astype(int) + 1

    pr_sw = pr.PyRanges(sw_int[["Chromosome", "Start", "End", "Strand", "_gid"]]).slack(int(config.overlap_dna_window))
    pr_dna = pr.PyRanges(dna_int[["Chromosome", "Start", "End", "Strand", "_gid"]])

    ov = pr_sw.join(pr_dna).df
    if ov.empty:
        out = genes.copy()
        out["dna_near_region"] = False
        out["n_dna_sites_near"] = 0
        return out

    if "_gid_b" in ov.columns:
        ov = ov.loc[ov["_gid"] == ov["_gid_b"]].copy()

    agg = (
        ov.groupby("_gid", as_index=False)
        .agg(n_dna_sites_near=("Start_b", "size"))
        .rename(columns={"_gid": "gene_id"})
    )
    agg["dna_near_region"] = agg["n_dna_sites_near"] > 0

    out = genes.merge(agg, on="gene_id", how="left")
    out["dna_near_region"] = out["dna_near_region"].fillna(False)
    out["n_dna_sites_near"] = out["n_dna_sites_near"].fillna(0).astype(int)
    return out[["gene_id", "dna_near_region", "n_dna_sites_near"]]


def check_domain_in_region(switch_regions: pd.DataFrame, pfam_results: pd.DataFrame) -> pd.DataFrame:
    """Check whether CDS changed regions overlap Pfam/InterPro domains."""
    genes = (
        pd.DataFrame({"gene_id": switch_regions["gene_id"].dropna().astype(str).unique()})
        if not switch_regions.empty and "gene_id" in switch_regions.columns
        else pd.DataFrame(columns=["gene_id"])
    )

    if genes.empty:
        return pd.DataFrame(columns=["gene_id", "domain_affected", "domain_name"])

    if pfam_results is None or pfam_results.empty or switch_regions.empty:
        out = genes.copy()
        out["domain_affected"] = False
        out["domain_name"] = None
        return out

    sw = switch_regions.copy()
    sw = sw.loc[sw.get("genomic_region_type", "").astype(str).eq("CDS")]
    if sw.empty:
        out = genes.copy()
        out["domain_affected"] = False
        out["domain_name"] = None
        return out

    pf = pfam_results.copy()

    sw["_gid"] = sw["gene_id"].astype(str)
    pf["_gid"] = pf["gene_id"].astype(str)

    sw_int = sw.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End", "strand": "Strand"})

    if "strand" not in pf.columns:
        pf["strand"] = "."
    pf_int = pf.rename(columns={"chr": "Chromosome", "start": "Start", "end": "End", "strand": "Strand"})

    pr_sw = pr.PyRanges(sw_int[["Chromosome", "Start", "End", "Strand", "_gid"]])
    pr_pf = pr.PyRanges(pf_int[["Chromosome", "Start", "End", "Strand", "_gid", "domain_name"]])

    ov = pr_sw.join(pr_pf).df
    if ov.empty:
        out = genes.copy()
        out["domain_affected"] = False
        out["domain_name"] = None
        return out

    if "_gid_b" in ov.columns:
        ov = ov.loc[ov["_gid"] == ov["_gid_b"]].copy()

    agg = (
        ov.groupby("_gid", as_index=False)
        .agg(domain_name=("domain_name", lambda s: ";".join(sorted(set(s.dropna().astype(str).tolist())))))
        .rename(columns={"_gid": "gene_id"})
    )
    agg["domain_affected"] = True

    out = genes.merge(agg, on="gene_id", how="left")
    out["domain_affected"] = out["domain_affected"].fillna(False)
    out["domain_name"] = out["domain_name"].where(out["domain_name"].notna(), None)
    return out[["gene_id", "domain_affected", "domain_name"]]


def run_all_overlaps(
    switch_regions: pd.DataFrame,
    m6a_sites: pd.DataFrame,
    dna_sites: pd.DataFrame,
    config: Config,
    pfam_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all overlap checks and merge into one per-gene table."""
    m6a_overlap = check_m6a_in_region(switch_regions, m6a_sites)
    dna_overlap = check_dna_near_region(switch_regions, dna_sites, config)
    domain_overlap = check_domain_in_region(switch_regions, pfam_results)

    out = m6a_overlap.merge(dna_overlap, on="gene_id", how="outer")
    out = out.merge(domain_overlap, on="gene_id", how="outer")

    # Fill defaults
    if "contains_m6a" in out.columns:
        out["contains_m6a"] = out["contains_m6a"].fillna(False)
    if "n_m6a_in_region" in out.columns:
        out["n_m6a_in_region"] = out["n_m6a_in_region"].fillna(0).astype(int)
    if "m6a_region_types" in out.columns:
        out["m6a_region_types"] = out["m6a_region_types"].apply(lambda x: x if isinstance(x, list) else [])
    if "dna_near_region" in out.columns:
        out["dna_near_region"] = out["dna_near_region"].fillna(False)
    if "n_dna_sites_near" in out.columns:
        out["n_dna_sites_near"] = out["n_dna_sites_near"].fillna(0).astype(int)
    if "domain_affected" in out.columns:
        out["domain_affected"] = out["domain_affected"].fillna(False)
    if "domain_name" in out.columns:
        out["domain_name"] = out["domain_name"].where(out["domain_name"].notna(), None)

    return out
