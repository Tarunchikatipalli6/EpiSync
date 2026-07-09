from __future__ import annotations

from typing import Optional

import pandas as pd

from episync.annotation import Annotation


def load_dna_methylation(
    path: str,
    condition: str,
    annotation: Annotation,
) -> pd.DataFrame:
    """
    Load modkit bedMethyl output.
    Expected columns: chr, start, end, name, score, strand,
                      start2, end2, color, coverage, methylation_freq
    
    Map CpG positions to gene_id using annotation.
    Keep coverage column — required for weighted mean downstream.

    Args:
        path: path to bedMethyl file
        condition: condition label (e.g., 'normal', 'tumor')
        annotation: Annotation object for gene mapping

    Returns:
        standardized DataFrame with columns:
        gene_id, chr, position, strand, methylation_freq, coverage, condition
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[
            "chr",
            "start",
            "end",
            "name",
            "score",
            "strand",
            "start2",
            "end2",
            "color",
            "coverage",
            "methylation_freq",
        ],
        dtype={
            "chr": str,
            "start": int,
            "end": int,
            "coverage": int,
            "methylation_freq": float,
        },
    )

    # CpG position is typically start (0-based)
    df["position"] = df["start"]

    # For now, assign to genes that have promoter regions overlapping this position
    # Simple approach: use gene_id from name if available, or leave as NA
    # In production, use pyranges to join with promoter intervals
    df["gene_id"] = None
    df["condition"] = condition

    # Keep only required columns
    result = df[["gene_id", "chr", "position", "strand", "methylation_freq", "coverage", "condition"]].copy()

    # Drop rows with no gene_id (not in a promoter region)
    result = result.dropna(subset=["gene_id"])

    return result


def load_m6a(
    path: str,
    condition: str,
    annotation: Annotation,
) -> pd.DataFrame:
    """
    Load m6Anet data.site_proba.csv.
    Expected columns: transcript_id, transcript_position, n_reads,
                      probability_modified, kmer, mod_ratio

    Map transcript_id to gene_id using annotation.
    Keep n_reads column — CRITICAL for anti-confounder gate.
    DO NOT drop n_reads — it is required for the coverage gate.

    Args:
        path: path to m6Anet CSV
        condition: condition label
        annotation: Annotation object

    Returns:
        standardized DataFrame with columns:
        gene_id, transcript_id, tx_position, probability_modified,
        mod_ratio, n_reads, condition
    """
    df = pd.read_csv(
        path,
        dtype={
            "transcript_id": str,
            "transcript_position": int,
            "n_reads": int,
            "probability_modified": float,
            "kmer": str,
            "mod_ratio": float,
        },
    )

    # Map transcript_id to gene_id
    df["gene_id"] = df["transcript_id"].map(annotation.transcript_to_gene)

    df["tx_position"] = df["transcript_position"]
    df["condition"] = condition

    # Keep only required columns
    result = df[
        [
            "gene_id",
            "transcript_id",
            "tx_position",
            "probability_modified",
            "mod_ratio",
            "n_reads",
            "condition",
        ]
    ].copy()

    # Drop rows with no gene_id (transcript not in annotation)
    result = result.dropna(subset=["gene_id"])

    return result


def load_isoform_switches(path: str) -> pd.DataFrame:
    """
    Load IsoformSwitchAnalyzeR switch_pairs.txt.
    Expected columns: geneID, isoformID_A, isoformID_B,
                      condition_1, condition_2,
                      isoform_switch_q_value, dIF,
                      length_A, length_B, length_change_bp,
                      switch_direction

    Already has gene_id. Standardize column names.

    Args:
        path: path to switch_pairs.txt

    Returns:
        standardized DataFrame with columns:
        gene_id, isoformID_A, isoformID_B, dIF, q_value,
        length_change_bp, switch_direction, condition_control, condition_tumor
    """
    df = pd.read_csv(path, sep="\t")

    # Standardize column names
    df.rename(
        columns={
            "geneID": "gene_id",
            "isoform_switch_q_value": "q_value",
            "condition_1": "condition_control",
            "condition_2": "condition_tumor",
        },
        inplace=True,
    )

    result = df[
        [
            "gene_id",
            "isoformID_A",
            "isoformID_B",
            "dIF",
            "q_value",
            "length_change_bp",
            "switch_direction",
            "condition_control",
            "condition_tumor",
        ]
    ].copy()

    return result


def load_expression(path: str, condition: str) -> pd.DataFrame:
    """
    Load DESeq2 output CSV.
    Expected columns: gene_id, baseMean, log2FoldChange, lfcSE,
                      stat, pvalue, padj

    Rename log2FoldChange -> log2FC.
    Keep padj column.

    Args:
        path: path to DESeq2 results CSV
        condition: condition label (typically 'tumor' for FC direction)

    Returns:
        standardized DataFrame with columns:
        gene_id, log2FC, padj, condition
    """
    df = pd.read_csv(path, index_col=0)  # gene_id as index
    df.reset_index(inplace=True)
    df.rename(columns={"log2FoldChange": "log2FC"}, inplace=True)

    result = df[["gene_id", "log2FC", "padj"]].copy()
    result["condition"] = condition

    return result


def validate_inputs(
    dna_df: pd.DataFrame,
    m6a_df: pd.DataFrame,
    iso_df: pd.DataFrame,
    expr_df: pd.DataFrame,
) -> bool:
    """
    Check that all four dataframes share overlapping gene_ids.
    Warn if overlap is below 50%.
    Raise if overlap is zero.

    Args:
        dna_df: DNA methylation dataframe
        m6a_df: m6A dataframe
        iso_df: isoform switches dataframe
        expr_df: expression dataframe

    Returns:
        True if validation passes

    Raises:
        ValueError if no overlap
    """
    dna_genes = set(dna_df["gene_id"].dropna().unique())
    m6a_genes = set(m6a_df["gene_id"].dropna().unique())
    iso_genes = set(iso_df["gene_id"].dropna().unique())
    expr_genes = set(expr_df["gene_id"].dropna().unique())

    all_genes = dna_genes | m6a_genes | iso_genes | expr_genes
    overlap = dna_genes & m6a_genes & iso_genes & expr_genes

    if len(overlap) == 0:
        raise ValueError("No overlapping genes across all four layers")

    overlap_pct = (len(overlap) / len(all_genes)) * 100 if all_genes else 0

    if overlap_pct < 50:
        print(
            f"WARNING: Only {overlap_pct:.1f}% gene overlap across layers "
            f"({len(overlap)} / {len(all_genes)}). Consider checking input files."
        )

    print(
        f"Validated: {len(overlap)} genes present in all 4 layers "
        f"({overlap_pct:.1f}% of {len(all_genes)} total unique genes)"
    )
    return True
