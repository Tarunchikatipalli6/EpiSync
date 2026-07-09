from __future__ import annotations

import pandas as pd

from episync.config import Config


def classify_expression(expr_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Classify DESeq2 expression outputs into UP/DOWN/UNCHANGED.

    Rules:
      padj < cutoff AND log2FC > +fc_cutoff -> UPREGULATED
      padj < cutoff AND log2FC < -fc_cutoff -> DOWNREGULATED
      else -> UNCHANGED

    Returns one row per gene:
      gene_id, log2FC, padj, expression_status
    """
    if expr_df.empty:
        return pd.DataFrame(columns=["gene_id", "log2FC", "padj", "expression_status"])

    df = expr_df.copy()

    # Standardize expected DESeq2 names if needed
    if "log2FC" not in df.columns and "log2FoldChange" in df.columns:
        df = df.rename(columns={"log2FoldChange": "log2FC"})

    required = {"gene_id", "log2FC", "padj"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Expression dataframe missing required columns: {sorted(missing)}")

    fc_cutoff = float(config.expr_log2fc_cutoff)
    padj_cutoff = float(config.expr_padj_cutoff)

    significant = df["padj"] < padj_cutoff
    up = significant & (df["log2FC"] > fc_cutoff)
    down = significant & (df["log2FC"] < -fc_cutoff)

    df["expression_status"] = "UNCHANGED"
    df.loc[up, "expression_status"] = "UPREGULATED"
    df.loc[down, "expression_status"] = "DOWNREGULATED"

    return df[["gene_id", "log2FC", "padj", "expression_status"]].copy()
