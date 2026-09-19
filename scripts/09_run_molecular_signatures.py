from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"


SIGNATURES: dict[str, list[str]] = {
    "Activin-SMAD remodeling": ["INHBA", "SMAD1", "SMAD3", "ACVR1C", "SMURF1", "GDF5", "GDF11"],
    "Dopamine-adenosine dysregulation": ["TH", "MAOB", "ADK", "ADORA1", "ADORA2A", "IL6"],
    "Glutamate-calcium stress": ["GRM8", "GNAO1", "XBP1", "SLC2A1", "CACNA2D1", "HSPA5", "CACNA1C"],
    "Immune-NF-kB cytokine tone": ["NFKBIA", "CEBPB", "CXCL8", "PRKCZ", "TIMP2", "CCL2", "IL1B"],
    "APP processing shift": ["PSEN1", "KAT5", "ADAM17", "APP", "BACE1", "ADAM10", "PSEN2"],
}

SIGNATURE_ORDER = list(SIGNATURES)


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


def signature_definitions(logcpm: pd.DataFrame, gene_summary: pd.DataFrame) -> pd.DataFrame:
    beta_lookup = gene_summary.set_index("gene")["best_beta"].to_dict()
    rows: list[dict[str, object]] = []
    for signature, genes in SIGNATURES.items():
        for gene in genes:
            if gene not in logcpm.index or gene not in beta_lookup:
                continue
            direction = 1 if beta_lookup[gene] >= 0 else -1
            rows.append(
                {
                    "signature": signature,
                    "gene": gene,
                    "direction": direction,
                    "direction_label": "higher_in_psychosis_axis" if direction > 0 else "lower_in_psychosis_axis",
                    "best_beta": beta_lookup[gene],
                }
            )
    return pd.DataFrame(rows)


def score_signatures(logcpm: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    z = logcpm.sub(logcpm.mean(axis=1), axis=0).div(logcpm.std(axis=1).replace(0, np.nan), axis=0)
    rows: dict[str, pd.Series] = {}
    for signature, group in definitions.groupby("signature", sort=False):
        signed_genes = []
        for _, row in group.iterrows():
            signed_genes.append(z.loc[row["gene"]] * int(row["direction"]))
        if signed_genes:
            rows[signature] = pd.concat(signed_genes, axis=1).mean(axis=1)
    score_df = pd.DataFrame(rows).T
    score_df = score_df.reindex(SIGNATURE_ORDER)
    return score_df


def differential_signatures(scores: pd.DataFrame, metadata: pd.DataFrame, gsva_module) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in ["NAc", "Caudate", "Putamen"]:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        region_scores = scores[region_meta["sample_key"]]
        for signature in scores.index:
            beta, t_value, p_value = gsva_module.ols_psychosis_effect(region_scores.loc[signature], region_meta)
            rows.append(
                {
                    "signature": signature,
                    "brain_region": region,
                    "beta_psychosis": beta,
                    "t_value": t_value,
                    "p_value": p_value,
                }
            )
    results = pd.DataFrame(rows)
    results["fdr"] = results.groupby("brain_region")["p_value"].transform(gsva_module.fdr_bh)
    return results.sort_values(["brain_region", "fdr", "p_value"])


def vulnerability_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region, group in results.groupby("brain_region"):
        rows.append(
            {
                "brain_region": region,
                "mean_positive_beta": group["beta_psychosis"].clip(lower=0).mean(),
                "n_nominal": int((group["p_value"] < 0.05).sum()),
                "n_fdr_0_10": int((group["fdr"] < 0.10).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gsva_module = load_gsva_module()
    metadata = load_metadata()
    logcpm = pd.read_csv(PROCESSED_DIR / "gse202537_core_gene_logcpm.csv", index_col=0)
    gene_summary = pd.read_csv(TABLE_DIR / "figure3_core_gene_summary.tsv", sep="\t")
    definitions = signature_definitions(logcpm, gene_summary)
    definitions.to_csv(TABLE_DIR / "figure4_signature_definitions.tsv", sep="\t", index=False)
    scores = score_signatures(logcpm, definitions)
    scores.to_csv(PROCESSED_DIR / "gse202537_molecular_signature_scores.csv")
    results = differential_signatures(scores, metadata, gsva_module)
    results.to_csv(TABLE_DIR / "figure4_molecular_signature_results.tsv", sep="\t", index=False)
    summary = vulnerability_summary(results)
    summary.to_csv(TABLE_DIR / "figure4_region_vulnerability_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"metric": "molecular_signatures", "value": len(SIGNATURES)},
            {"metric": "signature_genes_used", "value": definitions["gene"].nunique()},
            {"metric": "region_signature_tests", "value": results.shape[0]},
        ]
    ).to_csv(TABLE_DIR / "figure4_molecular_signature_metrics.tsv", sep="\t", index=False)
    print(f"Scored {len(SIGNATURES)} directional molecular signatures.")


if __name__ == "__main__":
    main()
