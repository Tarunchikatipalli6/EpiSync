from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyranges as pr


@dataclass
class GeneModel:
    """Represents a single gene with transcripts and exon structure."""

    gene_id: str
    gene_name: str
    chromosome: str
    strand: str  # '+' or '-'
    tss: int  # transcription start site (genomic coordinate)
    tes: int  # transcription end site (genomic coordinate)
    transcripts: Dict[str, List[Tuple[int, int]]] = field(
        default_factory=dict
    )  # transcript_id -> sorted list of (start, end) exon intervals


@dataclass
class Annotation:
    """Shared annotation object parsed once from GTF."""

    genes: Dict[str, GeneModel]  # gene_id -> GeneModel
    transcript_to_gene: Dict[str, str]  # transcript_id -> gene_id
    gene_models_pr: pr.PyRanges  # full exon table as PyRanges for interval operations


def build_annotation(gtf_path: str) -> Annotation:
    """
    Parse GTF once. Build gene models, transcript->gene map,
    and a PyRanges exon table for interval operations.

    Args:
        gtf_path: path to GTF file (gzip or plain text)

    Returns:
        Annotation object with genes, transcript_to_gene, and gene_models_pr
    """
    gtf_path = Path(gtf_path)
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")

    df = pr.read_gtf(str(gtf_path))
    genes: Dict[str, GeneModel] = {}
    transcript_to_gene: Dict[str, str] = {}

    # Extract gene and transcript features
    gene_features = df[df.Feature == "gene"]
    transcript_features = df[df.Feature == "transcript"]
    exon_features = df[df.Feature == "exon"]

    # Build gene models from gene features
    for _, row in gene_features.iterrows():
        gene_id = row.gene_id
        genes[gene_id] = GeneModel(
            gene_id=gene_id,
            gene_name=row.gene_name if hasattr(row, "gene_name") and row.gene_name else gene_id,
            chromosome=row.Chromosome,
            strand=row.Strand,
            tss=row.Start if row.Strand == "+" else row.End,
            tes=row.End if row.Strand == "+" else row.Start,
        )

    # Build transcript-to-gene map and collect exons per transcript
    for _, row in transcript_features.iterrows():
        gene_id = row.gene_id
        transcript_id = row.transcript_id
        transcript_to_gene[transcript_id] = gene_id
        if gene_id in genes:
            genes[gene_id].transcripts[transcript_id] = []

    # Populate exons per transcript (sorted by genomic position)
    for _, row in exon_features.iterrows():
        transcript_id = row.transcript_id
        if transcript_id in transcript_to_gene:
            gene_id = transcript_to_gene[transcript_id]
            if gene_id in genes and transcript_id in genes[gene_id].transcripts:
                genes[gene_id].transcripts[transcript_id].append((row.Start, row.End))

    # Sort exons per transcript by start position
    for gene_id in genes:
        for tx_id in genes[gene_id].transcripts:
            genes[gene_id].transcripts[tx_id].sort(key=lambda x: x[0])

    # Create PyRanges object for interval operations (exons only)
    exon_data = []
    for _, row in exon_features.iterrows():
        exon_data.append(
            {
                "Chromosome": row.Chromosome,
                "Start": row.Start,
                "End": row.End,
                "Strand": row.Strand,
                "gene_id": row.gene_id,
                "transcript_id": row.transcript_id,
            }
        )
    exon_df = pd.DataFrame(exon_data)
    gene_models_pr = pr.PyRanges(exon_df)

    return Annotation(
        genes=genes,
        transcript_to_gene=transcript_to_gene,
        gene_models_pr=gene_models_pr,
    )


def get_promoter(
    annotation: Annotation,
    gene_id: str,
    upstream: int,
    downstream: int,
) -> Optional[Tuple[str, int, int, str]]:
    """
    Return (chromosome, start, end, strand) of promoter region.
    STRAND-AWARE:
    + strand: start = TSS - upstream, end = TSS + downstream
    - strand: start = TSS - downstream, end = TSS + upstream
             where TSS on - strand is the END (max coordinate) of the gene.

    Args:
        annotation: Annotation object
        gene_id: gene ID to get promoter for
        upstream: bp upstream of TSS
        downstream: bp downstream of TSS

    Returns:
        (chromosome, start, end, strand) or None if gene not found
    """
    if gene_id not in annotation.genes:
        return None

    gene = annotation.genes[gene_id]
    chrom = gene.chromosome
    strand = gene.strand

    if strand == "+":
        # + strand: TSS is at the START of the gene
        tss = gene.tss
        start = max(0, tss - upstream)
        end = tss + downstream
    else:
        # - strand: TSS is at the END of the gene
        tss = gene.tes
        start = max(0, tss - downstream)
        end = tss + upstream

    return (chrom, start, end, strand)


def tx_to_genome(
    annotation: Annotation,
    transcript_id: str,
    tx_position: int,
) -> Optional[Tuple[str, int, str]]:
    """
    Convert transcript-level position to genomic coordinate.
    Walks the sorted exon blocks (strand-aware).
    This is a SPLICED transform — not a simple offset.

    Args:
        annotation: Annotation object
        transcript_id: transcript ID
        tx_position: position within transcript (0-based)

    Returns:
        (chromosome, genomic_position, strand) or None if not found
    """
    if transcript_id not in annotation.transcript_to_gene:
        return None

    gene_id = annotation.transcript_to_gene[transcript_id]
    if gene_id not in annotation.genes:
        return None

    gene = annotation.genes[gene_id]
    if transcript_id not in gene.transcripts:
        return None

    exons = gene.transcripts[transcript_id]
    if not exons:
        return None

    # Walk exons and accumulate transcript coordinate
    cumulative_tx_pos = 0
    for exon_start, exon_end in exons:
        exon_length = exon_end - exon_start
        if cumulative_tx_pos + exon_length > tx_position:
            # tx_position falls within this exon
            offset_in_exon = tx_position - cumulative_tx_pos
            genomic_pos = exon_start + offset_in_exon
            return (gene.chromosome, genomic_pos, gene.strand)
        cumulative_tx_pos += exon_length

    # tx_position beyond transcript length
    return None


def get_exon_boundaries(
    annotation: Annotation, transcript_id: str
) -> Optional[List[Tuple[int, int]]]:
    """
    Get sorted list of (start, end) exon boundaries for a transcript.

    Args:
        annotation: Annotation object
        transcript_id: transcript ID

    Returns:
        sorted list of exon tuples or None
    """
    if transcript_id not in annotation.transcript_to_gene:
        return None

    gene_id = annotation.transcript_to_gene[transcript_id]
    if gene_id not in annotation.genes:
        return None

    gene = annotation.genes[gene_id]
    if transcript_id not in gene.transcripts:
        return None

    return gene.transcripts[transcript_id]
