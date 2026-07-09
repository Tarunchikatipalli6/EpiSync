from __future__ import annotations

import pandas as pd

from episync.config import Config
from episync.overlap import check_dna_near_region, check_m6a_in_region


def _cfg() -> Config:
    return Config(
        dna_promoter_upstream=2000,
        dna_promoter_downstream=500,
        dna_delta_cutoff=0.2,
        dna_min_cpg_sites=3,
        m6a_prob_threshold=0.9,
        m6a_min_reads=20,
        m6a_delta_mod_ratio_cutoff=0.1,
        isoform_dif_cutoff=0.1,
        isoform_q_value_cutoff=0.05,
        expr_log2fc_cutoff=1.0,
        expr_padj_cutoff=0.05,
        overlap_dna_window=10,
        scoring_strong=6.0,
        scoring_moderate=4.0,
        scoring_weak=2.0,
        scoring_direction_multiplier=1.5,
        sensitivity_dna_delta=[0.1, 0.2],
        sensitivity_m6a_delta=[0.05, 0.1],
    )


def test_check_m6a_in_region_inside_outside_and_boundary() -> None:
    switch_regions = pd.DataFrame(
        [
            # gene1 region [100, 200)
            {"gene_id": "gene1", "chr": "chr1", "start": 100, "end": 200, "strand": "+", "region_type": "LOST"},
            # gene2 region [500, 600)
            {"gene_id": "gene2", "chr": "chr1", "start": 500, "end": 600, "strand": "+", "region_type": "GAINED"},
        ]
    )

    m6a_sites = pd.DataFrame(
        [
            # inside gene1
            {"gene_id": "gene1", "chr": "chr1", "genomic_position": 150, "strand": "+", "region_type": "3_UTR"},
            # outside gene1
            {"gene_id": "gene1", "chr": "chr1", "genomic_position": 250, "strand": "+", "region_type": "CDS"},
            # boundary at start for gene2 (included)
            {"gene_id": "gene2", "chr": "chr1", "genomic_position": 500, "strand": "+", "region_type": "CDS"},
            # boundary at end for gene2 (excluded due to half-open end)
            {"gene_id": "gene2", "chr": "chr1", "genomic_position": 600, "strand": "+", "region_type": "5_UTR"},
        ]
    )

    out = check_m6a_in_region(switch_regions, m6a_sites)
    out = out.set_index("gene_id")

    assert bool(out.loc["gene1", "contains_m6a"]) is True
    # only position 150 should count for gene1
    assert int(out.loc["gene1", "n_m6a_in_region"]) == 1

    assert bool(out.loc["gene2", "contains_m6a"]) is True
    # only boundary-start should count, boundary-end excluded
    assert int(out.loc["gene2", "n_m6a_in_region"]) == 1


def test_check_m6a_in_region_strand_awareness() -> None:
    switch_regions = pd.DataFrame(
        [
            {"gene_id": "gene1", "chr": "chr1", "start": 100, "end": 200, "strand": "+", "region_type": "LOST"},
            {"gene_id": "gene1", "chr": "chr1", "start": 100, "end": 200, "strand": "-", "region_type": "GAINED"},
        ]
    )

    m6a_sites = pd.DataFrame(
        [
            {"gene_id": "gene1", "chr": "chr1", "genomic_position": 150, "strand": "+", "region_type": "CDS"},
        ]
    )

    out = check_m6a_in_region(switch_regions, m6a_sites)
    row = out.loc[out["gene_id"] == "gene1"].iloc[0]

    # Only same-strand overlap should count
    assert bool(row["contains_m6a"]) is True
    assert int(row["n_m6a_in_region"]) == 1


def test_check_dna_near_region_with_window() -> None:
    switch_regions = pd.DataFrame(
        [
            {"gene_id": "geneA", "chr": "chr2", "start": 1000, "end": 1100, "strand": "+", "region_type": "LOST"},
            {"gene_id": "geneB", "chr": "chr2", "start": 2000, "end": 2100, "strand": "+", "region_type": "GAINED"},
        ]
    )

    dna_sites = pd.DataFrame(
        [
            # Near geneA within 10bp slack (left side)
            {"gene_id": "geneA", "chr": "chr2", "position": 995, "strand": "+"},
            # Not near geneB
            {"gene_id": "geneB", "chr": "chr2", "position": 2200, "strand": "+"},
        ]
    )

    out = check_dna_near_region(switch_regions, dna_sites, _cfg()).set_index("gene_id")

    assert bool(out.loc["geneA", "dna_near_region"]) is True
    assert int(out.loc["geneA", "n_dna_sites_near"]) == 1

    assert bool(out.loc["geneB", "dna_near_region"]) is False
    assert int(out.loc["geneB", "n_dna_sites_near"]) == 0
