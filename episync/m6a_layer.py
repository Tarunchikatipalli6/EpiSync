"""m6A layer processing for EpiSync.

This module implements the corrected m6A workflow:
1) high-confidence site filtering
2) anti-confounder coverage gate requiring support in BOTH conditions
3) transcript->genome spliced coordinate mapping
4) region annotation
5) matched-site stoichiometry comparison (mod_ratio deltas)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from episync.annotation import Annotation, tx_to_genome
from episync.config import Config


def filter_high_confidence(m6a_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Keep sites where probability_modified >= config.m6a_prob_threshold."""
    if m6a_df.empty:
        return m6a_df.copy()
    return m6a_df.loc[
        m6a_df["probability_modified"] >= config.m6a_prob_threshold
    ].copy()


def apply_coverage_gate(m6a_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply anti-confounder gate: require n_reads in BOTH normal and tumor.

    Expected input columns:
    - gene_id
    - transcript_id
    - tx_position
    - n_reads
    - condition (normal/tumor labels)

    A site key is (gene_id, transcript_id, tx_position).
    """
    if m6a_df.empty:
        return m6a_df.copy()

    key_cols = ["gene_id", "transcript_id", "tx_position"]

    # Keep rows with local read support first
    gated = m6a_df.loc[m6a_df["n_reads"] >= config.m6a_min_reads].copy()
    if gated.empty:
        return gated

    # Retain keys observed in >=2 distinct conditions after gating
    cond_counts = (
        gated.groupby(key_cols, dropna=False)["condition"]
        .nunique()
        .rename("n_conditions")
        .reset_index()
    )
    valid_keys = cond_counts.loc[cond_counts["n_conditions"] >= 2, key_cols]
    if valid_keys.empty:
        return gated.iloc[0:0].copy()

    out = gated.merge(valid_keys, on=key_cols, how="inner")
    return out


def map_to_genomic_coordinates(m6a_df: pd.DataFrame, annotation: Annotation) -> pd.DataFrame:
    """Map transcript_id + tx_position to genomic coordinate using spliced transform."""
    if m6a_df.empty:
        out = m6a_df.copy()
        out["chr"] = []
        out["genomic_position"] = []
        out["strand"] = []
        return out

    out = m6a_df.copy()

    coords: List[Tuple[str, int, str] | None] = []
    for row in out.itertuples(index=False):
        mapped = tx_to_genome(annotation, str(row.transcript_id), int(row.tx_position))
        coords.append(mapped)

    out["_mapped"] = coords
    out = out[out["_mapped"].notna()].copy()

    if out.empty:
        out = out.drop(columns=["_mapped"], errors="ignore")
        out["chr"] = []
        out["genomic_position"] = []
        out["strand"] = []
        return out

    out["chr"] = out["_mapped"].map(lambda x: x[0])
    out["genomic_position"] = out["_mapped"].map(lambda x: int(x[1]))
    out["strand"] = out["_mapped"].map(lambda x: x[2])
    out = out.drop(columns=["_mapped"])
    return out


def classify_site_region(mapped_df: pd.DataFrame, annotation: Annotation) -> pd.DataFrame:
    """Classify each site as 5_UTR/CDS/3_UTR using available annotation features.

    If feature-level annotation is unavailable in the parsed GTF model,
    default to "unknown".
    """
    if mapped_df.empty:
        out = mapped_df.copy()
        out["region_type"] = []
        return out

    out = mapped_df.copy()
    out["region_type"] = "unknown"

    gtf_df = annotation.gene_models_pr.df.copy()
    if gtf_df.empty or "Feature" not in gtf_df.columns:
        return out

    # Build simple feature intervals for UTR/CDS overlap
    region_map = {
        "five_prime_utr": "5_UTR",
        "three_prime_utr": "3_UTR",
        "CDS": "CDS",
    }

    # Convert sites to 1bp intervals for overlap checks
    sites = out.copy()
    sites["Start"] = sites["genomic_position"].astype(int)
    sites["End"] = sites["Start"] + 1
    sites["Chromosome"] = sites["chr"]
    sites["Strand"] = sites["strand"]

    # Priority order: UTR > CDS to avoid ambiguous assignments
    for feature, label in [("five_prime_utr", "5_UTR"), ("three_prime_utr", "3_UTR"), ("CDS", "CDS")]:
        feat = gtf_df.loc[gtf_df["Feature"] == feature, ["Chromosome", "Start", "End", "Strand"]]
        if feat.empty:
            continue

        # Fast interval containment check without requiring pyranges dependency here.
        # Group feature intervals by (chr, strand).
        grouped = {
            k: v[["Start", "End"]].to_numpy()
            for k, v in feat.groupby(["Chromosome", "Strand"], dropna=False)
        }

        mask = []
        for r in sites.itertuples(index=False):
            key = (r.Chromosome, r.Strand)
            intervals = grouped.get(key)
            if intervals is None:
                mask.append(False)
                continue
            pos = int(r.Start)
            hit = np.any((intervals[:, 0] <= pos) & (pos < intervals[:, 1]))
            mask.append(bool(hit))

        # Only set label where still unknown
        mask = np.array(mask, dtype=bool)
        unknown = out["region_type"].eq("unknown").to_numpy()
        out.loc[mask & unknown, "region_type"] = label

    return out


def match_sites_across_conditions(m6a_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only sites present in BOTH conditions at same genomic coordinate.

    Returns wide table with normal/tumor columns for mod_ratio and n_reads.
    """
    required = {
        "gene_id",
        "chr",
        "genomic_position",
        "region_type",
        "condition",
        "mod_ratio",
        "n_reads",
    }
    missing = required.difference(m6a_df.columns)
    if missing:
        raise ValueError(f"m6a_df missing required columns: {sorted(missing)}")

    if m6a_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "chr",
                "genomic_position",
                "region_type",
                "mod_ratio_normal",
                "mod_ratio_tumor",
                "n_reads_normal",
                "n_reads_tumor",
                "probability_modified_normal",
                "probability_modified_tumor",
            ]
        )

    # Infer control/tumor labels robustly
    labels = sorted(m6a_df["condition"].dropna().astype(str).unique().tolist())
    if len(labels) != 2:
        raise ValueError(
            f"Expected exactly 2 conditions for matching sites, found {len(labels)}: {labels}"
        )

    # Heuristic naming for output columns
    lower = [x.lower() for x in labels]
    if "normal" in lower and "tumor" in lower:
        normal_label = labels[lower.index("normal")]
        tumor_label = labels[lower.index("tumor")]
    else:
        normal_label, tumor_label = labels[0], labels[1]

    key = ["gene_id", "chr", "genomic_position", "region_type"]

    normal = m6a_df.loc[m6a_df["condition"] == normal_label].copy()
    tumor = m6a_df.loc[m6a_df["condition"] == tumor_label].copy()

    # Deduplicate per condition/site by averaging repeated calls
    agg_cols = {
        "mod_ratio": "mean",
        "n_reads": "mean",
    }
    if "probability_modified" in m6a_df.columns:
        agg_cols["probability_modified"] = "mean"

    normal = normal.groupby(key, dropna=False, as_index=False).agg(agg_cols)
    tumor = tumor.groupby(key, dropna=False, as_index=False).agg(agg_cols)

    normal = normal.rename(
        columns={
            "mod_ratio": "mod_ratio_normal",
            "n_reads": "n_reads_normal",
            "probability_modified": "probability_modified_normal",
        }
    )
    tumor = tumor.rename(
        columns={
            "mod_ratio": "mod_ratio_tumor",
            "n_reads": "n_reads_tumor",
            "probability_modified": "probability_modified_tumor",
        }
    )

    matched = normal.merge(tumor, on=key, how="inner")
    return matched


def calculate_gene_m6a(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched-site deltas to gene-level m6A statistics."""
    if matched_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "mean_delta_mod_ratio",
                "n_eligible_sites",
                "n_5utr_sites",
                "n_cds_sites",
                "n_3utr_sites",
                "mean_probability_modified",
                "site_positions",
            ]
        )

    df = matched_df.copy()
    df["site_delta_mod_ratio"] = df["mod_ratio_tumor"] - df["mod_ratio_normal"]

    # Region counts
    region_counts = (
        df.assign(
            is_5utr=df["region_type"].eq("5_UTR").astype(int),
            is_cds=df["region_type"].eq("CDS").astype(int),
            is_3utr=df["region_type"].eq("3_UTR").astype(int),
        )
        .groupby("gene_id", as_index=False)
        .agg(
            n_5utr_sites=("is_5utr", "sum"),
            n_cds_sites=("is_cds", "sum"),
            n_3utr_sites=("is_3utr", "sum"),
        )
    )

    core = (
        df.groupby("gene_id", as_index=False)
        .agg(
            mean_delta_mod_ratio=("site_delta_mod_ratio", "mean"),
            n_eligible_sites=("site_delta_mod_ratio", "size"),
        )
    )

    prob_cols = [c for c in ["probability_modified_normal", "probability_modified_tumor"] if c in df.columns]
    if prob_cols:
        tmp = df.copy()
        tmp["mean_probability_modified"] = tmp[prob_cols].mean(axis=1)
        prob = tmp.groupby("gene_id", as_index=False).agg(
            mean_probability_modified=("mean_probability_modified", "mean")
        )
    else:
        prob = pd.DataFrame({"gene_id": core["gene_id"], "mean_probability_modified": np.nan})

    site_positions = (
        df.groupby("gene_id", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "site_positions": [
                        (str(r.chr), int(r.genomic_position))
                        for r in g[["chr", "genomic_position"]].itertuples(index=False)
                    ]
                }
            )
        )
        .reset_index(drop=True)
    )

    out = core.merge(region_counts, on="gene_id", how="left").merge(prob, on="gene_id", how="left")
    out = out.merge(site_positions, on="gene_id", how="left")
    return out


def classify_m6a_change(gene_m6a_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Classify m6A gain/loss/retained from mean delta mod ratio."""
    if gene_m6a_df.empty:
        out = gene_m6a_df.copy()
        out["m6a_status"] = []
        return out

    out = gene_m6a_df.copy()
    delta = out["mean_delta_mod_ratio"]
    cutoff = float(config.m6a_delta_mod_ratio_cutoff)

    out["m6a_status"] = np.where(
        delta > cutoff,
        "GAINED",
        np.where(delta < -cutoff, "LOST", "RETAINED"),
    )
    return out


def run_m6a_layer(
    m6a_normal: pd.DataFrame,
    m6a_tumor: pd.DataFrame,
    annotation: Annotation,
    config: Config,
) -> pd.DataFrame:
    """Run full corrected m6A pipeline and return gene-level results."""
    if m6a_normal is None:
        m6a_normal = pd.DataFrame()
    if m6a_tumor is None:
        m6a_tumor = pd.DataFrame()

    a = m6a_normal.copy()
    b = m6a_tumor.copy()
    a["condition"] = "normal"
    b["condition"] = "tumor"

    combined = pd.concat([a, b], axis=0, ignore_index=True)
    if combined.empty:
        return calculate_gene_m6a(pd.DataFrame())

    step1 = filter_high_confidence(combined, config)
    step2 = apply_coverage_gate(step1, config)
    step3 = map_to_genomic_coordinates(step2, annotation)
    step4 = classify_site_region(step3, annotation)
    matched = match_sites_across_conditions(step4)
    gene = calculate_gene_m6a(matched)
    return classify_m6a_change(gene, config)
