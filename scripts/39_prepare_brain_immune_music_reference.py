from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"

MATRIX_DIR = (
    RAW_DIR
    / "filtered_feature_bc_matrix_HumanNewlyDiagnGBM"
    / "filtered_feature_bc_matrix_HumanNewlyDiagnGBM"
    / "filtered_feature_bc_matrix"
)
ANNOTATION = RAW_DIR / "annot_Human_ND_GBM_Full.csv"


T_MARKER_SETS = {
    "CD4_memory_resting": ["CD4", "IL7R", "CCR7", "LTB", "TCF7", "LEF1", "SELL", "MAL"],
    "CD4_memory_activated": ["CD4", "CD69", "CD40LG", "IL2RA", "TNFRSF4", "TNFRSF18", "ICOS", "CD28", "DPP4", "S100A4"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT", "TNFRSF18"],
    "CD8_cytotoxic": ["CD8A", "CD8B", "NKG7", "GZMB", "GZMK", "PRF1", "CCL5", "CX3CR1"],
    "NK_like_T": ["KLRD1", "GNLY", "NKG7", "KLRB1", "FCGR3A", "TRDC"],
}


def read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def read_features(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            rows.append({"gene_id": parts[0], "gene_symbol": parts[1].upper(), "feature_type": parts[2] if len(parts) > 2 else ""})
    return pd.DataFrame(rows)


def normalize_log1p_cp10k(counts: sparse.csc_matrix) -> sparse.csc_matrix:
    library = np.asarray(counts.sum(axis=0)).ravel()
    library[library == 0] = 1
    normalized = counts @ sparse.diags(10_000 / library)
    normalized.data = np.log1p(normalized.data)
    return normalized


def marker_scores(log_counts: sparse.csc_matrix, features: pd.DataFrame, barcodes: list[str]) -> pd.DataFrame:
    gene_to_indices: dict[str, list[int]] = {}
    for idx, gene in enumerate(features["gene_symbol"]):
        gene_to_indices.setdefault(gene, []).append(idx)

    scores = pd.DataFrame(index=barcodes)
    for state, markers in T_MARKER_SETS.items():
        indices = [idx for marker in markers for idx in gene_to_indices.get(marker, [])]
        indices = sorted(set(indices))
        if not indices:
            scores[state] = 0.0
            continue
        values = np.asarray(log_counts[indices, :].mean(axis=0)).ravel()
        scores[state] = values
    return scores


def assign_t_subtypes(annotation: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    assigned = pd.Series(index=annotation.index, data=annotation["cluster"].astype(str).str.replace(" ", "_", regex=False))
    t_mask = annotation["cluster"].astype(str).eq("T cells")
    t_scores = scores.loc[annotation.loc[t_mask, "cell"]].copy()
    top_state = t_scores.idxmax(axis=1)
    top_score = t_scores.max(axis=1)
    low_signal = top_score <= 0.02
    t_assignment = top_state.copy()
    t_assignment.loc[low_signal] = "T_cell_unresolved"

    # Resolve biologically specific classes before broad CD4 activated/resting assignment.
    t_assignment.loc[t_scores["Treg"] >= np.maximum.reduce([t_scores["CD4_memory_activated"], t_scores["CD4_memory_resting"], t_scores["CD8_cytotoxic"]])] = "Treg"
    t_assignment.loc[t_scores["CD8_cytotoxic"] >= np.maximum(t_scores["CD4_memory_activated"], t_scores["CD4_memory_resting"])] = "CD8_cytotoxic"
    nk_like = (t_scores["NK_like_T"] > t_scores["CD8_cytotoxic"]) & (t_scores["NK_like_T"] > t_scores["CD4_memory_activated"])
    t_assignment.loc[nk_like] = "NK_like_T"

    assigned.loc[t_mask] = t_assignment.to_numpy()
    return assigned


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    features = read_features(MATRIX_DIR / "features.tsv.gz")
    barcodes = read_gzip_lines(MATRIX_DIR / "barcodes.tsv.gz")
    counts = io.mmread(MATRIX_DIR / "matrix.mtx.gz").tocsc()
    annotation = pd.read_csv(ANNOTATION)

    annotation = annotation.loc[annotation["cell"].isin(barcodes)].copy()
    barcode_order = pd.Index(barcodes)
    annotation = annotation.set_index("cell").loc[barcode_order.intersection(annotation["cell"])].reset_index()
    annotation = annotation.rename(columns={"index": "cell"})
    selected_indices = [barcodes.index(cell) for cell in annotation["cell"]]
    counts = counts[:, selected_indices].tocsc()

    log_counts = normalize_log1p_cp10k(counts)
    scores = marker_scores(log_counts, features, annotation["cell"].tolist())
    annotation["music_cell_type"] = assign_t_subtypes(annotation, scores)
    annotation["music_sample_id"] = annotation["sample"].astype(str)
    annotation["reference_source"] = "Brain Immune Atlas human newly diagnosed GBM full aggregate"

    # MuSiC is unstable for extremely small classes; keep unresolved/rare classes grouped.
    class_counts = annotation["music_cell_type"].value_counts()
    rare_classes = set(class_counts[class_counts < 25].index)
    annotation.loc[annotation["music_cell_type"].isin(rare_classes), "music_cell_type"] = "Rare_or_unresolved"

    scores = scores.reset_index().rename(columns={"index": "cell"})
    annotation.merge(scores, on="cell", how="left").to_csv(
        TABLE_DIR / "npj_figure4_brain_immune_reference_annotation.tsv",
        sep="\t",
        index=False,
    )
    class_summary = (
        annotation.groupby(["music_cell_type", "sample"], observed=False)
        .size()
        .reset_index(name="n_cells")
        .groupby("music_cell_type", as_index=False)
        .agg(n_cells=("n_cells", "sum"), n_samples=("sample", "nunique"))
        .sort_values("n_cells", ascending=False)
    )
    class_summary.to_csv(TABLE_DIR / "npj_figure4_brain_immune_reference_celltype_summary.tsv", sep="\t", index=False)

    # Write a compact Matrix Market object containing the reference cells and matching metadata.
    # Rows are collapsed to unique gene symbols for compatibility with MuSiC.
    unique_genes = pd.Index(sorted(features["gene_symbol"].astype(str).unique()))
    gene_codes = unique_genes.get_indexer(features["gene_symbol"].astype(str))
    row_mapper = sparse.coo_matrix(
        (
            np.ones(len(gene_codes), dtype=np.float32),
            (gene_codes, np.arange(len(gene_codes))),
        ),
        shape=(len(unique_genes), len(gene_codes)),
    ).tocsr()
    symbol_counts = row_mapper @ counts
    io.mmwrite(PROCESSED_DIR / "brain_immune_music_reference_counts.mtx", symbol_counts.tocoo())
    pd.DataFrame({"gene_symbol": unique_genes}).to_csv(
        PROCESSED_DIR / "brain_immune_music_reference_features.tsv",
        sep="\t",
        index=False,
    )
    pd.Series(annotation["cell"]).to_csv(
        PROCESSED_DIR / "brain_immune_music_reference_barcodes.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    annotation[["cell", "music_cell_type", "music_sample_id", "cluster", "sample", "reference_source"]].to_csv(
        PROCESSED_DIR / "brain_immune_music_reference_metadata.tsv",
        sep="\t",
        index=False,
    )
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
