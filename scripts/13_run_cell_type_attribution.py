from __future__ import annotations

import importlib.util
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

CELL_TYPE_MARKERS: dict[str, dict[str, object]] = {
    "D1 MSN": {
        "class": "striatal neuron",
        "markers": ["DRD1", "TAC1", "PDYN", "ISL1", "EBF1", "GPR88", "RGS9", "PPP1R1B", "PDE10A"],
    },
    "D2 MSN": {
        "class": "striatal neuron",
        "markers": ["DRD2", "ADORA2A", "PENK", "GPR6", "OPRM1", "GPR88", "RGS9", "PPP1R1B", "PDE10A"],
    },
    "Interneuron": {
        "class": "striatal neuron",
        "markers": ["GAD1", "GAD2", "PVALB", "SST", "VIP", "NOS1", "NPY", "CHAT", "LHX6"],
    },
    "Dopaminergic axon": {
        "class": "neurotransmitter input",
        "markers": ["TH", "DDC", "SLC6A3", "SLC18A2", "ALDH1A1", "MAOA", "MAOB", "DBH"],
    },
    "Astrocyte": {
        "class": "glia",
        "markers": ["AQP4", "ALDH1L1", "GFAP", "SLC1A2", "SLC1A3", "GJA1", "SLC4A4", "CLU", "SOX9"],
    },
    "Oligodendrocyte": {
        "class": "glia",
        "markers": ["MBP", "MOBP", "PLP1", "MOG", "MAG", "CNP", "OPALIN", "OLIG1", "ERMN"],
    },
    "OPC": {
        "class": "glia",
        "markers": ["PDGFRA", "CSPG4", "VCAN", "OLIG2", "SOX10", "BCAN", "PTPRZ1", "GPR17"],
    },
    "Microglia-immune": {
        "class": "immune",
        "markers": ["CX3CR1", "P2RY12", "TMEM119", "AIF1", "C1QA", "C1QB", "CSF1R", "TYROBP", "TREM2", "CXCL8", "CCL2", "IL1B"],
    },
    "Vascular": {
        "class": "vascular",
        "markers": ["CLDN5", "PECAM1", "VWF", "FLT1", "KDR", "PDGFRB", "RGS5", "NOTCH3", "COL4A1", "COL4A2"],
    },
}

CELL_TYPE_ORDER = list(CELL_TYPE_MARKERS)


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


def marker_definition_table(available_genes: set[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    filtered: dict[str, list[str]] = {}
    for cell_type, definition in CELL_TYPE_MARKERS.items():
        source_markers = [str(gene).upper() for gene in definition["markers"]]
        available_markers = sorted(set(source_markers) & available_genes)
        filtered[cell_type] = available_markers
        rows.append(
            {
                "cell_type": cell_type,
                "cell_class": definition["class"],
                "source_marker_count": len(set(source_markers)),
                "available_marker_count": len(available_markers),
                "source_markers": ";".join(sorted(set(source_markers))),
                "available_markers": ";".join(available_markers),
            }
        )
    return pd.DataFrame(rows), filtered


def signature_celltype_enrichment(
    signature_defs: pd.DataFrame,
    marker_sets: dict[str, list[str]],
    universe: set[str],
    gsva_module,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signature, group in signature_defs.groupby("signature", sort=False):
        signature_genes = set(group["gene"].str.upper()) & universe
        for cell_type, marker_genes in marker_sets.items():
            markers = set(marker_genes) & universe
            overlap = sorted(signature_genes & markers)
            a = len(overlap)
            b = len(signature_genes - markers)
            c = len(markers - signature_genes)
            d = max(len(universe) - a - b - c, 0)
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(
                {
                    "signature": signature,
                    "cell_type": cell_type,
                    "signature_gene_count": len(signature_genes),
                    "marker_gene_count": len(markers),
                    "overlap_count": a,
                    "overlap_genes": ";".join(overlap),
                    "odds_ratio": odds_ratio if np.isfinite(odds_ratio) else np.nan,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = gsva_module.fdr_bh(result["p_value"])
    result["neg_log10_fdr"] = -np.log10(result["fdr"].clip(lower=1e-300))
    return result


def score_cell_types(logcpm: pd.DataFrame, marker_sets: dict[str, list[str]], gsva_module) -> pd.DataFrame:
    scored_sets = {cell_type: genes for cell_type, genes in marker_sets.items() if len(genes) >= 4}
    scores = gsva_module.rank_based_scores(logcpm, scored_sets)
    return scores.reindex(CELL_TYPE_ORDER)


def differential_cell_type_scores(scores: pd.DataFrame, metadata: pd.DataFrame, gsva_module) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in ["NAc", "Caudate", "Putamen"]:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        region_scores = scores[region_meta["sample_key"]]
        for cell_type in scores.index:
            beta, t_value, p_value = gsva_module.ols_psychosis_effect(region_scores.loc[cell_type], region_meta)
            rows.append(
                {
                    "cell_type": cell_type,
                    "brain_region": region,
                    "beta_psychosis": beta,
                    "t_value": t_value,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr"] = result.groupby("brain_region")["p_value"].transform(gsva_module.fdr_bh)
    return result.sort_values(["brain_region", "fdr", "p_value"])


def cellular_attribution_summary(enrichment: pd.DataFrame, score_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signature, group in enrichment.groupby("signature", sort=False):
        top = group.sort_values(["p_value", "fdr", "cell_type"]).head(3)
        rows.append(
            {
                "signature": signature,
                "top_cell_types": ";".join(top["cell_type"]),
                "top_overlap_genes": ";".join([item for item in top["overlap_genes"] if item]),
                "max_neg_log10_fdr": float(top["neg_log10_fdr"].max()) if not top.empty else 0.0,
            }
        )
    score_summary = (
        score_results.assign(abs_beta=lambda data: data["beta_psychosis"].abs())
        .sort_values(["fdr", "p_value", "abs_beta"], ascending=[True, True, False])
        .groupby("cell_type", as_index=False)
        .head(1)
        .rename(
            columns={
                "brain_region": "best_region",
                "beta_psychosis": "best_beta",
                "p_value": "best_p_value",
                "fdr": "best_fdr",
            }
        )
    )
    summary_rows = []
    for _, row in score_summary.iterrows():
        summary_rows.append(
            {
                "cell_type": row["cell_type"],
                "best_region": row["best_region"],
                "best_beta": row["best_beta"],
                "best_p_value": row["best_p_value"],
                "best_fdr": row["best_fdr"],
            }
        )
    cell_summary = pd.DataFrame(summary_rows)
    return pd.DataFrame(rows), cell_summary


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gsva_module = load_gsva_module()
    metadata = load_metadata()
    logcpm = load_logcpm(gsva_module)
    signature_defs = pd.read_csv(TABLE_DIR / "figure4_signature_definitions.tsv", sep="\t")

    marker_table, marker_sets = marker_definition_table(set(logcpm.index))
    marker_table.to_csv(TABLE_DIR / "figure6_cell_type_marker_definitions.tsv", sep="\t", index=False)

    enrichment = signature_celltype_enrichment(signature_defs, marker_sets, set(logcpm.index), gsva_module)
    enrichment.to_csv(TABLE_DIR / "figure6_signature_celltype_enrichment.tsv", sep="\t", index=False)

    scores = score_cell_types(logcpm, marker_sets, gsva_module)
    scores.to_csv(PROCESSED_DIR / "gse202537_cell_type_marker_scores.csv")

    score_results = differential_cell_type_scores(scores, metadata, gsva_module)
    score_results.to_csv(TABLE_DIR / "figure6_cell_type_score_results.tsv", sep="\t", index=False)

    signature_summary, cell_summary = cellular_attribution_summary(enrichment, score_results)
    signature_summary.to_csv(TABLE_DIR / "figure6_signature_cellular_attribution.tsv", sep="\t", index=False)
    cell_summary.to_csv(TABLE_DIR / "figure6_cell_type_score_summary.tsv", sep="\t", index=False)

    metrics = pd.DataFrame(
        [
            {"metric": "cell_type_marker_sets", "value": len(marker_sets)},
            {"metric": "cell_type_markers_available", "value": int(marker_table["available_marker_count"].sum())},
            {"metric": "signature_celltype_enrichment_tests", "value": enrichment.shape[0]},
            {"metric": "cell_type_region_score_tests", "value": score_results.shape[0]},
            {"metric": "nominal_cell_type_score_effects", "value": int((score_results["p_value"] < 0.05).sum())},
        ]
    )
    metrics.to_csv(TABLE_DIR / "figure6_cell_type_metrics.tsv", sep="\t", index=False)
    print(
        f"Scored {len(marker_sets)} cell-type marker sets and tested "
        f"{enrichment.shape[0]} signature-cell type enrichments."
    )


if __name__ == "__main__":
    main()
