from __future__ import annotations

import importlib.util
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

HPA_CLUSTER_TYPE_URL = "https://www.proteinatlas.org/download/tsv/rna_single_nuclei_cluster_type.tsv.zip"
HPA_CLUSTER_TYPES_URL = "https://www.proteinatlas.org/download/tsv/rna_single_nuclei_cluster_type_cluster_types.tsv.zip"
HPA_CLUSTER_TYPE_ZIP = RAW_DIR / "rna_single_nuclei_cluster_type.tsv.zip"
HPA_CLUSTER_TYPES_ZIP = RAW_DIR / "rna_single_nuclei_cluster_type_cluster_types.tsv.zip"

EXTERNAL_GROUPS: dict[str, list[str]] = {
    "MSN": ["medium spiny neuron", "eccentric medium spiny neuron"],
    "Interneuron": ["CGE interneuron", "MGE interneuron", "LAMP5-LHX6 and Chandelier"],
    "Astrocyte": ["astrocyte"],
    "Oligodendrocyte": ["oligodendrocyte"],
    "OPC": ["oligodendrocyte precursor cell", "committed oligodendrocyte precursor"],
    "Microglia-immune": ["central nervous system macrophage", "leukocyte"],
    "Vascular": ["endothelial cell", "pericyte", "vascular associated smooth muscle cell"],
}
EXTERNAL_ORDER = list(EXTERNAL_GROUPS)

INTERNAL_TO_EXTERNAL = {
    "D1 MSN": "MSN",
    "D2 MSN": "MSN",
    "Interneuron": "Interneuron",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodendrocyte",
    "OPC": "OPC",
    "Microglia-immune": "Microglia-immune",
    "Vascular": "Vascular",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "GSE202537-HPA-snbrain-validation/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        path.write_bytes(response.read())


def load_gsva_module():
    module_path = ROOT / "scripts" / "05_run_elsevier_gsva.py"
    spec = importlib.util.spec_from_file_location("elsevier_gsva", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Elsevier GSVA helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_metadata() -> pd.DataFrame:
    sample_metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    expression_qc = pd.read_csv(PROCESSED_DIR / "gse202537_expression_qc.csv")
    return sample_metadata.merge(
        expression_qc[["sample_key", "count_depth_million"]],
        on="sample_key",
        how="left",
    )


def load_logcpm(gsva_module) -> pd.DataFrame:
    counts = pd.read_csv(RAW_COUNTS, index_col=0)
    mapping = gsva_module.parse_gtf_gene_symbols(ENSEMBL_GTF, set(counts.index))
    return gsva_module.logcpm_by_symbol(counts, mapping)


def build_external_marker_sets(available_genes: set[str], top_n: int = 40) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    download(HPA_CLUSTER_TYPE_URL, HPA_CLUSTER_TYPE_ZIP)
    download(HPA_CLUSTER_TYPES_URL, HPA_CLUSTER_TYPES_ZIP)
    hpa = pd.read_csv(HPA_CLUSTER_TYPE_ZIP, sep="\t", compression="zip")
    hpa["Gene name"] = hpa["Gene name"].str.upper()
    expr = hpa.pivot_table(index="Gene name", columns="Cluster type", values="nCPM", aggfunc="mean").fillna(0.0)
    all_clusters = list(expr.columns)

    rows: list[dict[str, object]] = []
    marker_sets: dict[str, list[str]] = {}
    for group, clusters in EXTERNAL_GROUPS.items():
        present_clusters = [cluster for cluster in clusters if cluster in expr.columns]
        group_expr = expr[present_clusters].mean(axis=1)
        other_clusters = [cluster for cluster in all_clusters if cluster not in present_clusters]
        other_expr = expr[other_clusters].max(axis=1)
        specificity = np.log2((group_expr + 1.0) / (other_expr + 1.0))
        marker_table = pd.DataFrame(
            {
                "gene": expr.index,
                "external_cell_type": group,
                "group_nCPM": group_expr,
                "max_other_nCPM": other_expr,
                "specificity_log2_ratio": specificity,
            }
        )
        marker_table = marker_table.loc[marker_table["group_nCPM"] >= 1.0]
        marker_table = marker_table.sort_values(
            ["specificity_log2_ratio", "group_nCPM"], ascending=[False, False]
        ).head(top_n)
        source_markers = marker_table["gene"].tolist()
        available_markers = [gene for gene in source_markers if gene in available_genes]
        marker_sets[group] = available_markers
        for _, row in marker_table.iterrows():
            rows.append(
                {
                    "external_cell_type": group,
                    "source_clusters": ";".join(present_clusters),
                    "gene": row["gene"],
                    "group_nCPM": row["group_nCPM"],
                    "max_other_nCPM": row["max_other_nCPM"],
                    "specificity_log2_ratio": row["specificity_log2_ratio"],
                    "available_in_gse202537": row["gene"] in available_genes,
                }
            )
    return pd.DataFrame(rows), marker_sets


def internal_marker_sets() -> dict[str, set[str]]:
    definitions = pd.read_csv(TABLE_DIR / "figure6_cell_type_marker_definitions.tsv", sep="\t")
    sets: dict[str, set[str]] = {}
    for external_cell_type in EXTERNAL_ORDER:
        markers: set[str] = set()
        internal_rows = definitions.loc[
            definitions["cell_type"].map(INTERNAL_TO_EXTERNAL).fillna("") == external_cell_type
        ]
        for marker_string in internal_rows["available_markers"]:
            markers.update(gene for gene in str(marker_string).split(";") if gene)
        sets[external_cell_type] = markers
    return sets


def marker_concordance(external_markers: dict[str, list[str]], universe: set[str], gsva_module) -> pd.DataFrame:
    internal = internal_marker_sets()
    rows: list[dict[str, object]] = []
    for cell_type in EXTERNAL_ORDER:
        internal_set = internal[cell_type] & universe
        external_set = set(external_markers[cell_type]) & universe
        overlap = sorted(internal_set & external_set)
        union_size = len(internal_set | external_set)
        a = len(overlap)
        b = len(internal_set - external_set)
        c = len(external_set - internal_set)
        d = max(len(universe) - a - b - c, 0)
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "external_cell_type": cell_type,
                "internal_marker_count": len(internal_set),
                "external_marker_count": len(external_set),
                "overlap_count": a,
                "jaccard": a / union_size if union_size else 0.0,
                "overlap_genes": ";".join(overlap),
                "odds_ratio": odds_ratio if np.isfinite(odds_ratio) else np.nan,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["fdr"] = gsva_module.fdr_bh(result["p_value"])
    return result


def score_external_markers(logcpm: pd.DataFrame, marker_sets: dict[str, list[str]], gsva_module) -> pd.DataFrame:
    filtered = {cell_type: genes for cell_type, genes in marker_sets.items() if len(genes) >= 4}
    scores = gsva_module.rank_based_scores(logcpm, filtered)
    return scores.reindex(EXTERNAL_ORDER)


def differential_scores(scores: pd.DataFrame, metadata: pd.DataFrame, gsva_module) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in ["NAc", "Caudate", "Putamen"]:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        region_scores = scores[region_meta["sample_key"]]
        for cell_type in scores.index:
            beta, t_value, p_value = gsva_module.ols_psychosis_effect(region_scores.loc[cell_type], region_meta)
            rows.append(
                {
                    "external_cell_type": cell_type,
                    "brain_region": region,
                    "external_beta_psychosis": beta,
                    "external_t_value": t_value,
                    "external_p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["external_fdr"] = result.groupby("brain_region")["external_p_value"].transform(gsva_module.fdr_bh)
    return result.sort_values(["brain_region", "external_fdr", "external_p_value"])


def internal_external_beta_concordance(external_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = pd.read_csv(TABLE_DIR / "figure6_cell_type_score_results.tsv", sep="\t")
    internal = internal.loc[internal["cell_type"].isin(INTERNAL_TO_EXTERNAL)].copy()
    internal["external_cell_type"] = internal["cell_type"].map(INTERNAL_TO_EXTERNAL)
    internal_summary = (
        internal.groupby(["external_cell_type", "brain_region"], as_index=False)
        .agg(internal_beta_psychosis=("beta_psychosis", "mean"))
    )
    concordance = internal_summary.merge(
        external_results,
        on=["external_cell_type", "brain_region"],
        how="inner",
    )
    valid = concordance.dropna(subset=["internal_beta_psychosis", "external_beta_psychosis"])
    if valid.shape[0] >= 3:
        spearman_r, spearman_p = stats.spearmanr(valid["internal_beta_psychosis"], valid["external_beta_psychosis"])
        pearson_r, pearson_p = stats.pearsonr(valid["internal_beta_psychosis"], valid["external_beta_psychosis"])
    else:
        spearman_r = spearman_p = pearson_r = pearson_p = np.nan
    metrics = pd.DataFrame(
        [
            {"metric": "external_reference_marker_sets", "value": len(EXTERNAL_ORDER)},
            {"metric": "external_markers_available", "value": int(concordance["external_cell_type"].nunique())},
            {"metric": "concordance_points", "value": int(valid.shape[0])},
            {"metric": "spearman_r", "value": spearman_r},
            {"metric": "spearman_p", "value": spearman_p},
            {"metric": "pearson_r", "value": pearson_r},
            {"metric": "pearson_p", "value": pearson_p},
        ]
    )
    return concordance, metrics


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gsva_module = load_gsva_module()
    metadata = load_metadata()
    logcpm = load_logcpm(gsva_module)

    marker_table, marker_sets = build_external_marker_sets(set(logcpm.index))
    marker_table.to_csv(TABLE_DIR / "supplementary_figure2_hpa_snbrain_markers.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "external_cell_type": cell_type,
                "available_marker_count": len(genes),
                "available_markers": ";".join(genes),
            }
            for cell_type, genes in marker_sets.items()
        ]
    ).to_csv(TABLE_DIR / "supplementary_figure2_external_marker_definitions.tsv", sep="\t", index=False)

    concordance = marker_concordance(marker_sets, set(logcpm.index), gsva_module)
    concordance.to_csv(TABLE_DIR / "supplementary_figure2_marker_concordance.tsv", sep="\t", index=False)

    scores = score_external_markers(logcpm, marker_sets, gsva_module)
    scores.to_csv(PROCESSED_DIR / "gse202537_hpa_snbrain_cell_type_scores.csv")
    external_results = differential_scores(scores, metadata, gsva_module)
    external_results.to_csv(TABLE_DIR / "supplementary_figure2_external_cell_type_score_results.tsv", sep="\t", index=False)

    beta_concordance, metrics = internal_external_beta_concordance(external_results)
    beta_concordance.to_csv(TABLE_DIR / "supplementary_figure2_beta_concordance.tsv", sep="\t", index=False)
    metrics.to_csv(TABLE_DIR / "supplementary_figure2_external_validation_metrics.tsv", sep="\t", index=False)
    print(
        "Validated Figure 6 with HPA/Siletti single-nuclei brain markers: "
        f"{len(EXTERNAL_ORDER)} external cell-type marker sets."
    )


if __name__ == "__main__":
    main()
