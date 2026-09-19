from __future__ import annotations

from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


ROOT = Path(__file__).resolve().parents[1]
RAW_H5AD = ROOT / "data" / "raw" / "cellxgene_csf_lmd_reference.h5ad"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"

MAX_CELLS_PER_TYPE = 250
MIN_CELLS_PER_TYPE = 100
MIN_DONORS_PER_TYPE = 3
RANDOM_SEED = 20260707


def read_h5ad_column(handle: h5py.File, path: str) -> np.ndarray:
    obj = handle[path]
    if isinstance(obj, h5py.Group) and "codes" in obj and "categories" in obj:
        codes = obj["codes"][:]
        categories = np.array(
            [item.decode() if isinstance(item, bytes) else str(item) for item in obj["categories"][:]],
            dtype=object,
        )
        return np.array([categories[code] if code >= 0 else None for code in codes], dtype=object)
    values = obj[:]
    return np.array([item.decode() if isinstance(item, bytes) else str(item) for item in values], dtype=object)


def sanitize_label(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", label.strip())
    return re.sub(r"_+", "_", value).strip("_")


def select_reference_cells(obs: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (obs["lv1_annot"] != "Non-immune")
        & ~obs["lv2_annot"].str.contains("Malignant|Tumor", case=False, na=False)
    )
    filtered = obs.loc[keep].copy()
    summary = (
        filtered.groupby("lv2_annot")
        .agg(n_cells=("lv2_annot", "size"), n_donors=("donor_id", "nunique"))
        .reset_index()
    )
    valid_types = summary.loc[
        (summary["n_cells"] >= MIN_CELLS_PER_TYPE) & (summary["n_donors"] >= MIN_DONORS_PER_TYPE),
        "lv2_annot",
    ]
    filtered = filtered.loc[filtered["lv2_annot"].isin(valid_types)].copy()

    rng = np.random.default_rng(RANDOM_SEED)
    selected_index: list[int] = []
    for _, group_df in filtered.groupby("lv2_annot", sort=False):
        indices = group_df.index.to_numpy()
        if len(indices) > MAX_CELLS_PER_TYPE:
            indices = rng.choice(indices, size=MAX_CELLS_PER_TYPE, replace=False)
        selected_index.extend(indices.tolist())

    selected = obs.loc[sorted(selected_index)].copy()
    selected["music_cell_type"] = selected["lv2_annot"].map(sanitize_label)
    return selected


def build_gene_subset(handle: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bulk_genes = pd.read_csv(PROCESSED_DIR / "gse202537_gene_cpm_for_iobr.tsv", sep="\t", usecols=[0]).iloc[:, 0]
    bulk_gene_set = set(bulk_genes.astype(str))
    gene_symbols = read_h5ad_column(handle, "raw/var/feature_name").astype(str)
    keep = np.array([symbol in bulk_gene_set for symbol in gene_symbols], dtype=bool)
    old_to_new = np.full(len(gene_symbols), -1, dtype=np.int64)
    old_to_new[np.where(keep)[0]] = np.arange(int(keep.sum()), dtype=np.int64)
    return gene_symbols[keep], keep, old_to_new


def write_sparse_reference(handle: h5py.File, selected: pd.DataFrame, old_to_new: np.ndarray, n_genes: int) -> None:
    indptr = handle["raw/X/indptr"]
    indices = handle["raw/X/indices"]
    data = handle["raw/X/data"]

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []

    for out_col, cell_idx in enumerate(selected["cell_index"].to_numpy(dtype=np.int64)):
        start = int(indptr[cell_idx])
        end = int(indptr[cell_idx + 1])
        gene_idx = indices[start:end].astype(np.int64)
        mapped = old_to_new[gene_idx]
        keep = mapped >= 0
        if not np.any(keep):
            continue
        row_chunks.append(mapped[keep])
        col_chunks.append(np.full(int(keep.sum()), out_col, dtype=np.int64))
        data_chunks.append(data[start:end][keep].astype(np.float32))

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    values = np.concatenate(data_chunks)
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(n_genes, len(selected))).tocsr()
    mmwrite(PROCESSED_DIR / "cellxgene_csf_music_reference_counts.mtx", matrix)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(RAW_H5AD, "r") as handle:
        obs = pd.DataFrame(
            {
                "cell_index": np.arange(handle["obs/_index"].shape[0], dtype=np.int64),
                "cell": read_h5ad_column(handle, "obs/_index").astype(str),
                "donor_id": read_h5ad_column(handle, "obs/donor_id").astype(str),
                "patient_sample": read_h5ad_column(handle, "obs/patient_sample").astype(str),
                "disease_group": read_h5ad_column(handle, "obs/disease_group").astype(str),
                "disease": read_h5ad_column(handle, "obs/disease").astype(str),
                "lv1_annot": read_h5ad_column(handle, "obs/lv1_annot").astype(str),
                "lv2_annot": read_h5ad_column(handle, "obs/lv2_annot").astype(str),
                "cell_type": read_h5ad_column(handle, "obs/cell_type").astype(str),
                "sex": read_h5ad_column(handle, "obs/sex").astype(str),
            }
        )
        selected = select_reference_cells(obs)
        gene_symbols, _, old_to_new = build_gene_subset(handle)
        write_sparse_reference(handle, selected, old_to_new, len(gene_symbols))

    features = pd.DataFrame({"gene_symbol": gene_symbols})
    features.to_csv(PROCESSED_DIR / "cellxgene_csf_music_reference_features.tsv", sep="\t", index=False)
    selected[["cell"]].to_csv(
        PROCESSED_DIR / "cellxgene_csf_music_reference_barcodes.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    selected.drop(columns=["cell_index"]).to_csv(
        PROCESSED_DIR / "cellxgene_csf_music_reference_metadata.tsv",
        sep="\t",
        index=False,
    )

    summary = (
        selected.groupby(["music_cell_type", "lv2_annot", "lv1_annot"])
        .agg(n_cells=("cell", "size"), n_donors=("donor_id", "nunique"))
        .reset_index()
        .sort_values(["lv1_annot", "lv2_annot"])
    )
    summary.to_csv(TABLE_DIR / "npj_figure4_csf_reference_celltype_summary.tsv", sep="\t", index=False)
    selected.to_csv(TABLE_DIR / "npj_figure4_csf_reference_cell_metadata.tsv", sep="\t", index=False)

    print(
        f"Wrote CELLxGENE CSF MuSiC reference with {len(selected)} cells, "
        f"{len(gene_symbols)} matched genes, and {summary.shape[0]} cell types."
    )


if __name__ == "__main__":
    main()
