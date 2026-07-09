from __future__ import annotations

import pandas as pd

from episync.config import Config


def classify_expression(
    expr_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Filter and classify expression changes:
    padj < padj_cutoff AND log2FC > +fc_cutoff  -> UPREGULATED
    padj < padj_cutoff AND log2FC < -fc_cutoff  -> DOWNREGULATED
    else                                         -> UNCHANGED

    CARRY THE CONTINUOUS log2FC FORWARD.
    The label is for display only.

    Args:
        expr_df: expression dataframe with gene_id, log2FC, padj
        config: Config object

    Returns:
        dataframe with log2FC, padj, expression_status columns
    """
    result = expr_df[["gene_id", "log2FC", "padj"]].copy()

    def classify(row):
        if pd.isna(row["padj"]) or pd.isna(row["log2FC"]):
            return "UNCHANGED"
        if row["padj"] < config.expr_padj_cutoff:
            if row["log2FC"] > config.expr_log2fc_cutoff:
                return "UPREGULATED"
            elif row["log2FC"] < -config.expr_log2fc_cutoff:
                return "DOWNREGULATED"
        return "UNCHANGED"

    result["expression_status"] = result.apply(classify, axis=1)

    return result
