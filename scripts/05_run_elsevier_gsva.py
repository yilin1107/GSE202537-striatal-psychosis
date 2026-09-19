from __future__ import annotations

import csv
import gzip
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"

RAW_COUNTS = RAW_DIR / "GSE202537_Read_counts_NAc_Caudate_Putamen_psychosis_mcontrols.csv.gz"
ELSEVIER_GMT = RAW_DIR / "Elsevier_Pathway_Collection.gmt"
ENSEMBL_GTF = RAW_DIR / "Homo_sapiens.GRCh38.110.gtf.gz"

ELSEVIER_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Elsevier_Pathway_Collection"
ENSEMBL_GTF_URL = "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


def write_checksums(output_path: Path) -> None:
    paths = []
    for folder in [RAW_DIR, PROCESSED_DIR, TABLE_DIR]:
        paths.extend(path for path in folder.glob("*") if path.is_file())
    lines = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_elsevier_gene_sets(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = [gene.strip().upper() for gene in parts[2:] if gene.strip()]
            if name and genes:
                gene_sets[name] = sorted(set(genes))
    return gene_sets


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


def logcpm_by_symbol(counts: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    mapped = counts.loc[counts.index.intersection(mapping.keys())].copy()
    mapped["gene_symbol"] = [mapping[gene_id] for gene_id in mapped.index]
    symbol_counts = mapped.groupby("gene_symbol").sum(numeric_only=True)
    keep = (symbol_counts >= 10).sum(axis=1) >= max(3, int(np.ceil(0.10 * symbol_counts.shape[1])))
    symbol_counts = symbol_counts.loc[keep]
    library_size = symbol_counts.sum(axis=0)
    cpm = symbol_counts.div(library_size, axis=1) * 1_000_000
    return np.log2(cpm + 1)


def filter_gene_sets(
    gene_sets: dict[str, list[str]],
    available_genes: set[str],
    min_size: int = 5,
    max_size: int = 500,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    filtered: dict[str, list[str]] = {}
    rows: list[dict[str, int | str]] = []
    for pathway, genes in gene_sets.items():
        overlap = sorted(set(genes) & available_genes)
        rows.append(
            {
                "pathway": pathway,
                "source_genes": len(set(genes)),
                "overlap_genes": len(overlap),
            }
        )
        if min_size <= len(overlap) <= max_size:
            filtered[pathway] = overlap
    return filtered, pd.DataFrame(rows)


def rank_based_scores(logcpm: pd.DataFrame, gene_sets: dict[str, list[str]], alpha: float = 0.25) -> pd.DataFrame:
    genes = np.asarray(logcpm.index)
    gene_index = {gene: idx for idx, gene in enumerate(genes)}
    set_indices = {
        pathway: np.asarray([gene_index[gene] for gene in genes_in_set], dtype=int)
        for pathway, genes_in_set in gene_sets.items()
    }
    n_genes = len(genes)
    scores = np.zeros((len(gene_sets), logcpm.shape[1]), dtype=float)
    pathway_names = list(gene_sets.keys())

    for sample_idx, sample in enumerate(logcpm.columns):
        values = logcpm[sample].to_numpy(dtype=float)
        order = np.argsort(values)[::-1]
        ranks = stats.rankdata(values, method="average")
        weights = np.power(np.abs(ranks), alpha)
        ordered_weights = weights[order]
        ordered_position = np.empty(n_genes, dtype=int)
        ordered_position[order] = np.arange(n_genes)

        for pathway_idx, pathway in enumerate(pathway_names):
            indices = set_indices[pathway]
            k_size = len(indices)
            hit_positions = np.sort(ordered_position[indices])
            hit_weights = ordered_weights[hit_positions]
            hit_weights = hit_weights / hit_weights.sum()
            running = np.full(n_genes, -1.0 / (n_genes - k_size), dtype=float)
            running[hit_positions] = hit_weights
            running = np.cumsum(running)
            max_es = running.max()
            min_es = running.min()
            scores[pathway_idx, sample_idx] = max_es if abs(max_es) >= abs(min_es) else min_es

    score_df = pd.DataFrame(scores, index=pathway_names, columns=logcpm.columns)
    zscores = score_df.sub(score_df.mean(axis=1), axis=0).div(score_df.std(axis=1).replace(0, np.nan), axis=0)
    return zscores.dropna(axis=0, how="any")


def design_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index]:
    data = metadata.copy()
    data["psychosis"] = (data["group"] != "Control").astype(float)
    data["sex_male"] = (data["gender"] == "Male").astype(float)
    data["tod_sin"] = np.sin(2 * np.pi * data["corrected_tod_24h"] / 24)
    data["tod_cos"] = np.cos(2 * np.pi * data["corrected_tod_24h"] / 24)
    continuous = ["age", "pmi", "rin", "ph", "count_depth_million"]
    for column in continuous:
        mean = data[column].mean()
        std = data[column].std()
        data[f"{column}_z"] = (data[column] - mean) / std if std and not np.isnan(std) else 0

    columns = [
        "intercept",
        "psychosis",
        "age_z",
        "sex_male",
        "pmi_z",
        "rin_z",
        "ph_z",
        "count_depth_million_z",
        "tod_sin",
        "tod_cos",
    ]
    data["intercept"] = 1.0
    model_data = data.dropna(subset=columns)
    return model_data[columns].to_numpy(dtype=float), columns, model_data["sample_key"]


def ols_psychosis_effect(y_values: pd.Series, metadata: pd.DataFrame) -> tuple[float, float, float]:
    x_matrix, columns, sample_keys = design_matrix(metadata)
    y = y_values.loc[sample_keys].to_numpy(dtype=float)
    if len(y) <= x_matrix.shape[1] + 2:
        return np.nan, np.nan, np.nan
    beta, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)
    residuals = y - x_matrix @ beta
    df_resid = len(y) - x_matrix.shape[1]
    sigma2 = float((residuals @ residuals) / df_resid)
    xtx_inv = np.linalg.pinv(x_matrix.T @ x_matrix)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    idx = columns.index("psychosis")
    t_value = beta[idx] / se[idx] if se[idx] > 0 else np.nan
    p_value = 2 * stats.t.sf(abs(t_value), df_resid) if not np.isnan(t_value) else np.nan
    return float(beta[idx]), float(t_value), float(p_value)


def fdr_bh(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if len(finite_values) == 0:
        return pd.Series(result, index=p_values.index)
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    temp = np.empty_like(finite_values)
    temp[order] = adjusted
    result[np.where(finite)[0]] = temp
    return pd.Series(result, index=p_values.index)


def differential_pathways(scores: pd.DataFrame, metadata: pd.DataFrame, pathway_sizes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for region in ["NAc", "Caudate", "Putamen"]:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        region_scores = scores[region_meta["sample_key"]]
        for pathway in scores.index:
            beta, t_value, p_value = ols_psychosis_effect(region_scores.loc[pathway], region_meta)
            rows.append(
                {
                    "pathway": pathway,
                    "brain_region": region,
                    "beta_psychosis": beta,
                    "t_value": t_value,
                    "p_value": p_value,
                }
            )
    results = pd.DataFrame(rows)
    results["fdr"] = results.groupby("brain_region")["p_value"].transform(fdr_bh)
    size_lookup = pathway_sizes.set_index("pathway")["overlap_genes"]
    results["overlap_genes"] = results["pathway"].map(size_lookup)
    return results.sort_values(["brain_region", "fdr", "p_value"])


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    download(ELSEVIER_URL, ELSEVIER_GMT)
    download(ENSEMBL_GTF_URL, ENSEMBL_GTF)

    sample_metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    expression_qc = pd.read_csv(PROCESSED_DIR / "gse202537_expression_qc.csv")
    metadata = sample_metadata.merge(
        expression_qc[["sample_key", "count_depth_million"]],
        on="sample_key",
        how="left",
    )
    counts = pd.read_csv(RAW_COUNTS, index_col=0)
    mapping = parse_gtf_gene_symbols(ENSEMBL_GTF, set(counts.index))
    logcpm = logcpm_by_symbol(counts, mapping)
    gene_sets = parse_elsevier_gene_sets(ELSEVIER_GMT)
    filtered_gene_sets, gene_set_summary = filter_gene_sets(gene_sets, set(logcpm.index))
    scores = rank_based_scores(logcpm, filtered_gene_sets)
    results = differential_pathways(scores, metadata, gene_set_summary)

    scores.to_csv(PROCESSED_DIR / "gse202537_elsevier_pathway_scores.csv")
    gene_set_summary.to_csv(TABLE_DIR / "figure2_elsevier_gene_set_coverage.tsv", sep="\t", index=False)
    results.to_csv(TABLE_DIR / "figure2_elsevier_gsva_results.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"metric": "mapped_expression_genes", "value": logcpm.shape[0]},
            {"metric": "elsevier_gene_sets_total", "value": len(gene_sets)},
            {"metric": "elsevier_gene_sets_used", "value": len(filtered_gene_sets)},
            {"metric": "samples_scored", "value": scores.shape[1]},
        ]
    ).to_csv(TABLE_DIR / "figure2_elsevier_gsva_summary.tsv", sep="\t", index=False)
    write_checksums(ROOT / "checksums.txt")

    print(f"Scored {len(filtered_gene_sets)} Elsevier pathways across {scores.shape[1]} samples.")


if __name__ == "__main__":
    main()
