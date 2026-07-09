from __future__ import annotations

import pandas as pd
import pyranges as pr

from episync.annotation import Annotation
from episync.config import Config


def map_cpg_to_promoters(
    meth_df: pd.DataFrame,
    annotation: Annotation,
    config: Config,
) -> pd.DataFrame:
    """
    For each CpG position map to gene promoter region.
    Use annotation.get_promoter() which is STRAND-AWARE.
    A CpG at position X belongs to gene Y if X falls inside Y's promoter window.
    Use pyranges for the interval join — do not hand-roll.

    Args:
        meth_df: methylation dataframe with chr, position, strand, methylation_freq, coverage
        annotation: Annotation object
        config: Config object with promoter window sizes

    Returns:
        methylation dataframe with added gene_id column
    """
    result = meth_df.copy()
    result["gene_id"] = None

    # Build promoter intervals for all genes
    promoter_data = []
    for gene_id, gene in annotation.genes.items():
        promo = (
            gene.chromosome,
            gene.tss - config.dna_promoter_upstream
            if gene.strand == "+"
            else gene.tes - config.dna_promoter_downstream,
            gene.tss + config.dna_promoter_downstream
            if gene.strand == "+"
            else gene.tes + config.dna_promoter_upstream,
            gene.strand,
        )
        if promo[1] >= 0:
            promoter_data.append(
                {
                    "Chromosome": promo[0],
                    "Start": promo[1],
                    "End": promo[2],
                    "Strand": promo[3],
                    "gene_id": gene_id,
                }
            )

    if not promoter_data:
        return result

    promo_df = pd.DataFrame(promoter_data)
    promo_pr = pr.PyRanges(promo_df)

    # Create PyRanges for CpG positions (as 0-width intervals or 1-bp intervals)
    cpg_data = []
    for idx, row in meth_df.iterrows():
        cpg_data.append(
            {
                "Chromosome": row["chr"],
                "Start": row["position"],
                "End": row["position"] + 1,
                "Strand": row["strand"],
            }
        )
    cpg_df = pd.DataFrame(cpg_data)
    if cpg_df.empty:
        return result

    cpg_pr = pr.PyRanges(cpg_df)

    # Join CpGs to promoter regions
    joined = cpg_pr.join(promo_pr, how="left")
    joined_df = joined.as_df()

    # Map back to result
    for i, row in joined_df.iterrows():
        if "gene_id" in row and pd.notna(row["gene_id"]):
            result.loc[i, "gene_id"] = row["gene_id"]

    return result


def calculate_gene_methylation(
    mapped_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Per gene per condition:
    - mean_methylation_freq weighted by coverage
    - n_cpg_sites covered
    Drop genes where n_cpg_sites < config.dna_min_cpg_sites.

    Args:
        mapped_df: methylation dataframe with gene_id, methylation_freq, coverage
        config: Config object

    Returns:
        gene-level dataframe with mean_methylation, n_cpg_sites per gene/condition
    """
    # Drop rows without gene_id
    mapped_df = mapped_df.dropna(subset=["gene_id"])

    if mapped_df.empty:
        return pd.DataFrame(
            columns=["gene_id", "condition", "mean_methylation", "n_cpg_sites"]
        )

    # Group by gene and condition
    grouped = mapped_df.groupby(["gene_id", "condition"])

    results = []
    for (gene_id, condition), group in grouped:
        # Weighted mean methylation
        total_cov = group["coverage"].sum()
        if total_cov > 0:
            mean_meth = (group["methylation_freq"] * group["coverage"]).sum() / total_cov
        else:
            mean_meth = group["methylation_freq"].mean()

        n_sites = len(group)

        results.append(
            {
                "gene_id": gene_id,
                "condition": condition,
                "mean_methylation": mean_meth,
                "n_cpg_sites": n_sites,
            }
        )

    result_df = pd.DataFrame(results)

    # Filter by minimum CpG sites
    result_df = result_df[result_df["n_cpg_sites"] >= config.dna_min_cpg_sites]

    return result_df


def classify_dna_change(
    gene_meth_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Per gene calculate delta_methylation and classify:
    delta < -cutoff  -> HYPOMETHYLATED
    delta > +cutoff  -> HYPERMETHYLATED
    else             -> UNCHANGED

    CARRY THE CONTINUOUS delta_methylation FORWARD.
    The label is for display only.
    The float feeds the continuous score in scorer.py.

    Args:
        gene_meth_df: gene-level methylation dataframe
        config: Config object

    Returns:
        dataframe with delta_methylation and dna_status columns
    """
    # Pivot to get normal and tumor columns
    pivot_df = gene_meth_df.pivot(
        index="gene_id", columns="condition", values="mean_methylation"
    ).reset_index()

    # Assume 'normal' and 'tumor' conditions
    if "normal" not in pivot_df.columns or "tumor" not in pivot_df.columns:
        # If not exactly these names, take first and second columns
        cols = [c for c in pivot_df.columns if c != "gene_id"]
        if len(cols) >= 2:
            pivot_df.rename(columns={cols[0]: "normal", cols[1]: "tumor"}, inplace=True)
        else:
            return pd.DataFrame(
                columns=[
                    "gene_id",
                    "delta_methylation",
                    "dna_status",
                    "mean_meth_normal",
                    "mean_meth_tumor",
                    "n_cpg_sites",
                ]
            )

    pivot_df["delta_methylation"] = pivot_df["tumor"].fillna(0) - pivot_df["normal"].fillna(0)

    def classify(delta):
        if delta < -config.dna_delta_cutoff:
            return "HYPOMETHYLATED"
        elif delta > config.dna_delta_cutoff:
            return "HYPERMETHYLATED"
        else:
            return "UNCHANGED"

    pivot_df["dna_status"] = pivot_df["delta_methylation"].apply(classify)

    # Get n_cpg_sites
    n_sites_df = gene_meth_df.groupby("gene_id")["n_cpg_sites"].max().reset_index()
    pivot_df = pivot_df.merge(n_sites_df, on="gene_id")

    result = pivot_df[
        [
            "gene_id",
            "delta_methylation",
            "dna_status",
            "normal",
            "tumor",
            "n_cpg_sites",
        ]
    ].copy()
    result.rename(columns={"normal": "mean_meth_normal", "tumor": "mean_meth_tumor"}, inplace=True)

    return result


def run_dna_layer(
    meth_df: pd.DataFrame,
    annotation: Annotation,
    config: Config,
) -> pd.DataFrame:
    """
    Run full DNA layer pipeline.
    Returns one row per gene with columns:
    gene_id, delta_methylation, dna_status,
    mean_meth_normal, mean_meth_tumor, n_cpg_sites

    Args:
        meth_df: methylation dataframe
        annotation: Annotation object
        config: Config object

    Returns:
        gene-level DNA methylation results
    """
    mapped = map_cpg_to_promoters(meth_df, annotation, config)
    gene_meth = calculate_gene_methylation(mapped, config)
    return classify_dna_change(gene_meth, config)
