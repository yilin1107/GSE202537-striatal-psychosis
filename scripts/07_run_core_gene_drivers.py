from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"

RAW_COUNTS = RAW_DIR / "GSE202537_Read_counts_NAc_Caudate_Putamen_psychosis_mcontrols.csv.gz"
ELSEVIER_GMT = RAW_DIR / "Elsevier_Pathway_Collection.gmt"
ENSEMBL_GTF = RAW_DIR / "Homo_sapiens.GRCh38.110.gtf.gz"

AXIS_ORDER = [
    "Dopamine/adenosine",
    "Glutamate/calcium",
    "Immune/cytokine",
    "TGF-beta/SMAD",
    "Neurodegeneration",
]

SELECTED_PATHWAYS = [
    {
        "pathway": "Dopamine Metabolism in Parkinson's Disease",
        "axis": "Dopamine/adenosine",
        "short_label": "Dopamine metabolism",
    },
    {
        "pathway": "ADK Expression Downregulation after Acute Seizures",
        "axis": "Dopamine/adenosine",
        "short_label": "ADK / adenosine",
    },
    {
        "pathway": "GRM2-4/6-8 (Presynaptic) -> Glutamate Release Attenuation",
        "axis": "Glutamate/calcium",
        "short_label": "Presynaptic GRM",
    },
    {
        "pathway": "Timothy Syndrome",
        "axis": "Glutamate/calcium",
        "short_label": "Timothy / Ca2+",
    },
    {
        "pathway": "Wolfram Syndrome Progression (Hypothesis)",
        "axis": "Glutamate/calcium",
        "short_label": "Wolfram stress",
    },
    {
        "pathway": "N2 Neutrophils in Tumor-Promoting Inflammation and Tumor Progression",
        "axis": "Immune/cytokine",
        "short_label": "N2 neutrophils",
    },
    {
        "pathway": "MacrophageR -> CEBPB -> NF-kB Signaling",
        "axis": "Immune/cytokine",
        "short_label": "Macrophage NF-kB",
    },
    {
        "pathway": "IL6ST -> STAT5B Signaling",
        "axis": "Immune/cytokine",
        "short_label": "IL6ST-STAT5B",
    },
    {
        "pathway": "Lipoxin A4/FPR2-Related Neutrophil Depression",
        "axis": "Immune/cytokine",
        "short_label": "Lipoxin-FPR2",
    },
    {
        "pathway": "ActivinR -> SMAD2/3 Signaling",
        "axis": "TGF-beta/SMAD",
        "short_label": "Activin-SMAD2/3",
    },
    {
        "pathway": "ActivinR/BMPR -> SMAD1/5/9 Signaling",
        "axis": "TGF-beta/SMAD",
        "short_label": "BMPR-SMAD1/5/9",
    },
    {
        "pathway": "APP Processing in Alzheimer Disease",
        "axis": "Neurodegeneration",
        "short_label": "APP processing",
    },
]


def load_gsva_module():
    module_path = ROOT / "scripts" / "05_run_elsevier_gsva.py"
    spec = importlib.util.spec_from_file_location("elsevier_gsva", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Elsevier GSVA helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_logcpm(gsva_module) -> pd.DataFrame:
    counts = pd.read_csv(RAW_COUNTS, index_col=0)
    mapping = gsva_module.parse_gtf_gene_symbols(ENSEMBL_GTF, set(counts.index))
    return gsva_module.logcpm_by_symbol(counts, mapping)


def load_metadata() -> pd.DataFrame:
    sample_metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    expression_qc = pd.read_csv(PROCESSED_DIR / "gse202537_expression_qc.csv")
    return sample_metadata.merge(
        expression_qc[["sample_key", "count_depth_million"]],
        on="sample_key",
        how="left",
    )


def selected_pathway_table(
    gene_sets: dict[str, list[str]],
    figure2_results: pd.DataFrame,
    available_genes: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in SELECTED_PATHWAYS:
        pathway = item["pathway"]
        if pathway not in gene_sets:
            continue
        genes = sorted(set(gene_sets[pathway]) & available_genes)
        pathway_rows = figure2_results.loc[figure2_results["pathway"] == pathway]
        best_row = pathway_rows.sort_values(["fdr", "p_value"]).head(1)
        if best_row.empty:
            best_region = ""
            best_beta = np.nan
            best_fdr = np.nan
        else:
            best_region = str(best_row.iloc[0]["brain_region"])
            best_beta = float(best_row.iloc[0]["beta_psychosis"])
            best_fdr = float(best_row.iloc[0]["fdr"])
        rows.append(
            {
                "pathway": pathway,
                "short_label": item["short_label"],
                "axis": item["axis"],
                "best_region": best_region,
                "pathway_beta": best_beta,
                "pathway_fdr": best_fdr,
                "n_genes": len(genes),
                "genes": ";".join(genes),
            }
        )
    return pd.DataFrame(rows)


def gene_axis(gene: str, selected_pathways: pd.DataFrame) -> str:
    memberships = selected_pathways.loc[selected_pathways["genes"].str.split(";").map(lambda genes: gene in genes)]
    if memberships.empty:
        return "Neurodegeneration"
    axes = memberships["axis"].tolist()
    if len(set(axes)) > 1:
        for axis in AXIS_ORDER:
            if axis in axes:
                return axis
    return axes[0]


def gene_model_results(
    logcpm: pd.DataFrame,
    metadata: pd.DataFrame,
    selected_pathways: pd.DataFrame,
    gsva_module,
) -> pd.DataFrame:
    candidate_genes = sorted(set().union(*[set(row.split(";")) for row in selected_pathways["genes"] if row]))
    rows: list[dict[str, object]] = []
    for region in ["NAc", "Caudate", "Putamen"]:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        for gene in candidate_genes:
            beta, t_value, p_value = gsva_module.ols_psychosis_effect(logcpm.loc[gene], region_meta)
            rows.append(
                {
                    "gene": gene,
                    "brain_region": region,
                    "beta_psychosis": beta,
                    "t_value": t_value,
                    "p_value": p_value,
                }
            )
    results = pd.DataFrame(rows)
    results["fdr"] = results.groupby("brain_region")["p_value"].transform(gsva_module.fdr_bh)
    return results


def gene_summary(gene_results: pd.DataFrame, selected_pathways: pd.DataFrame) -> pd.DataFrame:
    pathway_lookup = {
        row["pathway"]: set(str(row["genes"]).split(";")) if row["genes"] else set()
        for _, row in selected_pathways.iterrows()
    }
    rows: list[dict[str, object]] = []
    for gene, group in gene_results.groupby("gene"):
        memberships = [pathway for pathway, genes in pathway_lookup.items() if gene in genes]
        best = group.sort_values(["p_value", "fdr"]).iloc[0]
        min_p = max(float(best["p_value"]), 1e-300)
        evidence_score = -np.log10(min_p) + np.log2(len(memberships) + 1)
        rows.append(
            {
                "gene": gene,
                "axis": gene_axis(gene, selected_pathways),
                "pathway_count": len(memberships),
                "pathways": ";".join(memberships),
                "best_region": best["brain_region"],
                "best_beta": best["beta_psychosis"],
                "best_p_value": best["p_value"],
                "best_fdr": best["fdr"],
                "evidence_score": evidence_score,
            }
        )
    summary = pd.DataFrame(rows)
    axis_rank = {axis: idx for idx, axis in enumerate(AXIS_ORDER)}
    summary["axis_rank"] = summary["axis"].map(axis_rank).fillna(len(AXIS_ORDER))
    return summary.sort_values(["evidence_score", "pathway_count"], ascending=[False, False]).drop(columns="axis_rank")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gsva_module = load_gsva_module()
    metadata = load_metadata()
    logcpm = load_logcpm(gsva_module)
    gene_sets = gsva_module.parse_elsevier_gene_sets(ELSEVIER_GMT)
    figure2_results = pd.read_csv(TABLE_DIR / "figure2_elsevier_gsva_results.tsv", sep="\t")
    selected_pathways = selected_pathway_table(gene_sets, figure2_results, set(logcpm.index))
    selected_pathways.to_csv(TABLE_DIR / "figure3_selected_molecular_pathways.tsv", sep="\t", index=False)

    candidate_genes = sorted(set().union(*[set(row.split(";")) for row in selected_pathways["genes"] if row]))
    logcpm.loc[candidate_genes].to_csv(PROCESSED_DIR / "gse202537_core_gene_logcpm.csv")

    gene_results = gene_model_results(logcpm, metadata, selected_pathways, gsva_module)
    gene_results.to_csv(TABLE_DIR / "figure3_core_gene_model_results.tsv", sep="\t", index=False)
    summary = gene_summary(gene_results, selected_pathways)
    summary.to_csv(TABLE_DIR / "figure3_core_gene_summary.tsv", sep="\t", index=False)

    pd.DataFrame(
        [
            {"metric": "selected_molecular_pathways", "value": selected_pathways.shape[0]},
            {"metric": "candidate_core_genes", "value": len(candidate_genes)},
            {"metric": "genes_with_nominal_p_lt_0.05", "value": int((gene_results["p_value"] < 0.05).sum())},
        ]
    ).to_csv(TABLE_DIR / "figure3_core_gene_metrics.tsv", sep="\t", index=False)
    print(f"Modeled {len(candidate_genes)} candidate genes from {selected_pathways.shape[0]} molecular pathways.")


if __name__ == "__main__":
    main()
