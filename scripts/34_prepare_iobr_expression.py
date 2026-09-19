from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_COUNTS = RAW_DIR / "GSE202537_Read_counts_NAc_Caudate_Putamen_psychosis_mcontrols.csv.gz"
ENSEMBL_GTF = RAW_DIR / "Homo_sapiens.GRCh38.110.gtf.gz"
OUTPUT_CPM = PROCESSED_DIR / "gse202537_gene_cpm_for_iobr.tsv"


def parse_gtf_gene_symbols(path: Path, needed_ids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern_gene_id = re.compile(r'gene_id "([^"]+)"')
    pattern_gene_name = re.compile(r'gene_name "([^"]+)"')
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            gene_id_match = pattern_gene_id.search(fields[8])
            gene_name_match = pattern_gene_name.search(fields[8])
            if not gene_id_match or not gene_name_match:
                continue
            gene_id = gene_id_match.group(1).split(".")[0]
            if gene_id in needed_ids:
                mapping[gene_id] = gene_name_match.group(1).upper()
    return mapping


def cpm_by_symbol(counts: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    mapped = counts.loc[counts.index.intersection(mapping.keys())].copy()
    mapped["gene_symbol"] = [mapping[gene_id] for gene_id in mapped.index]
    symbol_counts = mapped.groupby("gene_symbol").sum(numeric_only=True)
    keep_threshold = max(3, int(np.ceil(0.10 * symbol_counts.shape[1])))
    keep = (symbol_counts >= 10).sum(axis=1) >= keep_threshold
    symbol_counts = symbol_counts.loc[keep]
    library_size = symbol_counts.sum(axis=0)
    cpm = symbol_counts.div(library_size, axis=1) * 1_000_000
    return cpm


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(RAW_COUNTS, index_col=0)
    mapping = parse_gtf_gene_symbols(ENSEMBL_GTF, set(counts.index))
    cpm = cpm_by_symbol(counts, mapping)
    cpm.index.name = "gene_symbol"
    cpm.to_csv(OUTPUT_CPM, sep="\t", float_format="%.8g")
    print(f"Wrote {OUTPUT_CPM} with {cpm.shape[0]} genes and {cpm.shape[1]} samples.")


if __name__ == "__main__":
    main()
