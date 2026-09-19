from __future__ import annotations

import gzip
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
ENSEMBL_GTF = RAW_DIR / "Homo_sapiens.GRCh38.110.gtf.gz"
HALLMARK_GMT = RAW_DIR / "MSigDB_Hallmark_2020.gmt"
HALLMARK_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"


HALLMARK_CATEGORIES = {
    "Immune-inflammatory": {
        "TNFA_SIGNALING_VIA_NFKB",
        "IL6_JAK_STAT3_SIGNALING",
        "INTERFERON_ALPHA_RESPONSE",
        "INTERFERON_GAMMA_RESPONSE",
        "INFLAMMATORY_RESPONSE",
        "COMPLEMENT",
        "IL2_STAT5_SIGNALING",
        "ALLOGRAFT_REJECTION",
    },
    "Stress and cell death": {
        "APOPTOSIS",
        "P53_PATHWAY",
        "HYPOXIA",
        "UNFOLDED_PROTEIN_RESPONSE",
        "REACTIVE_OXYGEN_SPECIES_PATHWAY",
        "UV_RESPONSE_UP",
        "UV_RESPONSE_DN",
    },
    "Metabolism and mitochondria": {
        "OXIDATIVE_PHOSPHORYLATION",
        "GLYCOLYSIS",
        "FATTY_ACID_METABOLISM",
        "CHOLESTEROL_HOMEOSTASIS",
        "BILE_ACID_METABOLISM",
        "XENOBIOTIC_METABOLISM",
        "PEROXISOME",
        "ADIPOGENESIS",
        "HEME_METABOLISM",
    },
    "Cell cycle and genome": {
        "E2F_TARGETS",
        "G2M_CHECKPOINT",
        "MITOTIC_SPINDLE",
        "MYC_TARGETS_V1",
        "MYC_TARGETS_V2",
        "DNA_REPAIR",
    },
    "Developmental signaling": {
        "WNT_BETA_CATENIN_SIGNALING",
        "HEDGEHOG_SIGNALING",
        "NOTCH_SIGNALING",
        "TGF_BETA_SIGNALING",
        "APICAL_SURFACE",
        "APICAL_JUNCTION",
    },
    "Growth and kinase signaling": {
        "PI3K_AKT_MTOR_SIGNALING",
        "MTORC1_SIGNALING",
        "KRAS_SIGNALING_UP",
        "KRAS_SIGNALING_DN",
        "PROTEIN_SECRETION",
    },
    "Hormone and tissue remodeling": {
        "ANDROGEN_RESPONSE",
        "ESTROGEN_RESPONSE_EARLY",
        "ESTROGEN_RESPONSE_LATE",
        "EPITHELIAL_MESENCHYMAL_TRANSITION",
        "ANGIOGENESIS",
        "COAGULATION",
        "MYOGENESIS",
        "PANCREAS_BETA_CELLS",
        "SPERMATOGENESIS",
    },
}

CATEGORY_ORDER = list(HALLMARK_CATEGORIES)
REGION_ORDER = ["NAc", "Caudate", "Putamen"]


def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=180) as response:
        path.write_bytes(response.read())


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


def parse_hallmark_gene_sets(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw_name = parts[0].strip()
            hallmark_id = normalize_hallmark_id(raw_name)
            genes = sorted({gene.strip().upper() for gene in parts[2:] if gene.strip()})
            rows.append(
                {
                    "hallmark": raw_name,
                    "hallmark_id": hallmark_id,
                    "label": hallmark_id.replace("_", " ").title().replace("Nfkb", "NF-kB").replace("Mtor", "mTOR"),
                    "genes": ";".join(genes),
                    "source_gene_count": len(genes),
                }
            )
    data = pd.DataFrame(rows)
    data["category"] = data["hallmark_id"].map(category_for_hallmark)
    missing = data.loc[data["category"].isna(), "hallmark_id"].tolist()
    if missing:
        raise ValueError(f"Hallmark categories missing for: {missing}")
    return data.sort_values(["category", "hallmark_id"]).reset_index(drop=True)


def normalize_hallmark_id(raw_name: str) -> str:
    hallmark_id = raw_name.upper().replace("HALLMARK_", "")
    hallmark_id = re.sub(r"[^A-Z0-9]+", "_", hallmark_id)
    hallmark_id = re.sub(r"_+", "_", hallmark_id).strip("_")
    fixes = {
        "G2_M_CHECKPOINT": "G2M_CHECKPOINT",
        "IL_2_STAT5_SIGNALING": "IL2_STAT5_SIGNALING",
        "IL_6_JAK_STAT3_SIGNALING": "IL6_JAK_STAT3_SIGNALING",
        "PI3K_AKT_MTOR_SIGNALING": "PI3K_AKT_MTOR_SIGNALING",
        "PPEROXISOME": "PEROXISOME",
        "TGF_BETA_SIGNALING": "TGF_BETA_SIGNALING",
        "TNF_ALPHA_SIGNALING_VIA_NF_KB": "TNFA_SIGNALING_VIA_NFKB",
        "WNT_BETA_CATENIN_SIGNALING": "WNT_BETA_CATENIN_SIGNALING",
    }
    return fixes.get(hallmark_id, hallmark_id)


def category_for_hallmark(hallmark_id: str) -> str:
    for category, members in HALLMARK_CATEGORIES.items():
        if hallmark_id in members:
            return category
    return ""


def model_metadata(sample_metadata: pd.DataFrame, expression_qc: pd.DataFrame) -> pd.DataFrame:
    metadata = sample_metadata.merge(
        expression_qc[["sample_key", "count_depth_million"]],
        on="sample_key",
        how="left",
    )
    metadata["sex_male"] = (metadata["gender"] == "Male").astype(float)
    metadata["tod_sin"] = np.sin(2 * np.pi * metadata["corrected_tod_24h"] / 24)
    metadata["tod_cos"] = np.cos(2 * np.pi * metadata["corrected_tod_24h"] / 24)
    metadata["is_scz"] = (metadata["group"] == "SCZ").astype(float)
    metadata["is_bd_psychosis"] = (metadata["group"] == "BD with psychosis").astype(float)
    return metadata


def design_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index]:
    data = metadata.copy()
    continuous = ["age", "pmi", "rin", "ph", "count_depth_million"]
    for column in continuous:
        std = data[column].std()
        data[f"{column}_z"] = (data[column] - data[column].mean()) / std if std and not np.isnan(std) else 0
    columns = [
        "intercept",
        "is_scz",
        "is_bd_psychosis",
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


def differential_genes(logcpm: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for region in REGION_ORDER:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        x_matrix, columns, sample_keys = design_matrix(region_meta)
        expr = logcpm[sample_keys].to_numpy(dtype=float)
        xtx_inv = np.linalg.pinv(x_matrix.T @ x_matrix)
        betas = xtx_inv @ x_matrix.T @ expr.T
        fitted = x_matrix @ betas
        residuals = expr.T - fitted
        df_resid = x_matrix.shape[0] - x_matrix.shape[1]
        sigma2 = np.sum(residuals**2, axis=0) / df_resid
        se = np.sqrt(np.outer(np.diag(xtx_inv), sigma2))
        mean_expr = expr.mean(axis=1)

        for contrast, column in [("SCZ_vs_Control", "is_scz"), ("BD_psychosis_vs_Control", "is_bd_psychosis")]:
            idx = columns.index(column)
            beta = betas[idx, :]
            t_values = beta / se[idx, :]
            p_values = 2 * stats.t.sf(np.abs(t_values), df_resid)
            data = pd.DataFrame(
                {
                    "gene": logcpm.index,
                    "brain_region": region,
                    "contrast": contrast,
                    "n_samples": len(sample_keys),
                    "mean_logcpm": mean_expr,
                    "log2fc": beta,
                    "t_value": t_values,
                    "p_value": p_values,
                }
            )
            data["fdr"] = fdr_bh(data["p_value"])
            data["rank_by_p"] = data["p_value"].rank(method="first")
            rows.append(data)
    return pd.concat(rows, ignore_index=True)


def hallmark_enrichment(
    gene_results: pd.DataFrame,
    hallmark: pd.DataFrame,
    expressed_genes: set[str],
    n_perm: int = 2000,
    seed: int = 17,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (region, contrast), contrast_results in gene_results.groupby(["brain_region", "contrast"]):
        values = contrast_results.set_index("gene")["t_value"].dropna()
        universe = values.index.to_numpy()
        t_array = values.to_numpy(dtype=float)
        for _, row in hallmark.iterrows():
            genes = [gene for gene in str(row["genes"]).split(";") if gene in expressed_genes and gene in values.index]
            if len(genes) < 5:
                continue
            observed = float(values.loc[genes].mean())
            k_size = len(genes)
            null_means = np.empty(n_perm, dtype=float)
            for i in range(n_perm):
                idx = rng.choice(len(universe), size=k_size, replace=False)
                null_means[i] = t_array[idx].mean()
            null_mean = float(null_means.mean())
            null_sd = float(null_means.std(ddof=1))
            z_score = (observed - null_mean) / null_sd if null_sd > 0 else np.nan
            p_value = (np.sum(np.abs(null_means - null_mean) >= abs(observed - null_mean)) + 1) / (n_perm + 1)
            rows.append(
                {
                    "brain_region": region,
                    "contrast": contrast,
                    "hallmark": row["hallmark"],
                    "hallmark_id": row["hallmark_id"],
                    "label": row["label"],
                    "category": row["category"],
                    "overlap_genes": k_size,
                    "mean_t": observed,
                    "enrichment_z": z_score,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = result.groupby(["brain_region", "contrast"])["p_value"].transform(fdr_bh)
    return result.sort_values(["brain_region", "contrast", "category", "hallmark_id"]).reset_index(drop=True)


def write_category_table(hallmark: pd.DataFrame) -> pd.DataFrame:
    return (
        hallmark.groupby("category", as_index=False)
        .agg(n_gene_sets=("hallmark_id", "count"), gene_sets=("hallmark_id", lambda values: ";".join(values)))
        .assign(category=lambda data: pd.Categorical(data["category"], categories=CATEGORY_ORDER, ordered=True))
        .sort_values("category")
    )


def main() -> None:
    download_if_missing(HALLMARK_URL, HALLMARK_GMT)
    sample_metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    expression_qc = pd.read_csv(PROCESSED_DIR / "gse202537_expression_qc.csv")
    counts = pd.read_csv(RAW_COUNTS, index_col=0)
    mapping = parse_gtf_gene_symbols(ENSEMBL_GTF, set(counts.index))
    logcpm = logcpm_by_symbol(counts, mapping)
    metadata = model_metadata(sample_metadata, expression_qc)
    hallmark = parse_hallmark_gene_sets(HALLMARK_GMT)
    gene_results = differential_genes(logcpm, metadata)
    enrichment = hallmark_enrichment(gene_results, hallmark, set(logcpm.index))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(PROCESSED_DIR / "gse202537_region_model_metadata.csv", index=False)
    hallmark.drop(columns=["genes"]).to_csv(TABLE_DIR / "npj_figure1_hallmark_definitions.tsv", sep="\t", index=False)
    write_category_table(hallmark).to_csv(TABLE_DIR / "npj_figure1_hallmark_categories.tsv", sep="\t", index=False)
    gene_results.to_csv(TABLE_DIR / "npj_figure1_gene_ma_results.tsv", sep="\t", index=False)
    enrichment.to_csv(TABLE_DIR / "npj_figure1_hallmark_enrichment.tsv", sep="\t", index=False)
    print(f"Wrote {gene_results.shape[0]} region-specific gene-level contrast rows and {enrichment.shape[0]} Hallmark enrichment rows.")


if __name__ == "__main__":
    main()
