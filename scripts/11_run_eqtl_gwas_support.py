from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"

GTEX_BASE = "https://gtexportal.org/api/v2"
GWAS_BASE = "https://www.ebi.ac.uk/gwas/rest/api/v2"
GTEX_DATASET = "gtex_v8"

SIGNATURE_ORDER = [
    "Activin-SMAD remodeling",
    "Dopamine-adenosine dysregulation",
    "Glutamate-calcium stress",
    "Immune-NF-kB cytokine tone",
    "APP processing shift",
]

BRAIN_TISSUES = [
    ("Brain_Caudate_basal_ganglia", "Caudate"),
    ("Brain_Cortex", "Cortex"),
    ("Brain_Frontal_Cortex_BA9", "Frontal cortex"),
    ("Brain_Nucleus_accumbens_basal_ganglia", "NAc"),
    ("Brain_Putamen_basal_ganglia", "Putamen"),
]

STRIATAL_TISSUES = {
    "Brain_Caudate_basal_ganglia",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
}

PSYCHIATRIC_TRAITS = [
    ("Schizophrenia", "MONDO_0005090"),
    ("Bipolar disorder", "MONDO_0004985"),
    ("Major depressive disorder", "MONDO_0002009"),
    ("Autism spectrum disorder", "MONDO_0005258"),
    ("ADHD", "MONDO_0007743"),
    ("Psychotic disorder", "MONDO_0005485"),
]


def output_paths() -> list[Path]:
    return [
        TABLE_DIR / "figure5_gene_reference.tsv",
        TABLE_DIR / "figure5_gtex_brain_eqtl.tsv",
        TABLE_DIR / "figure5_gwas_catalog_support.tsv",
        TABLE_DIR / "figure5_integrated_eqtl_gwas_summary.tsv",
        TABLE_DIR / "figure5_eqtl_gwas_metrics.tsv",
    ]


def fetch_json(url: str, retries: int = 3, delay: float = 0.25) -> dict[str, Any]:
    headers = {"User-Agent": "GSE202537-eQTL-GWAS-analysis/1.0", "Accept": "application/json"}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


def url_join(base: str, path: str, params: dict[str, object]) -> str:
    encoded = urllib.parse.urlencode(params, doseq=True)
    return f"{base}{path}?{encoded}"


def load_gene_reference() -> pd.DataFrame:
    definitions = pd.read_csv(TABLE_DIR / "figure4_signature_definitions.tsv", sep="\t")
    gene_summary = pd.read_csv(TABLE_DIR / "figure3_core_gene_summary.tsv", sep="\t")
    reference = definitions.merge(
        gene_summary[
            [
                "gene",
                "axis",
                "pathway_count",
                "best_region",
                "best_beta",
                "best_p_value",
                "best_fdr",
                "evidence_score",
            ]
        ],
        on="gene",
        how="left",
    )
    signature_rank = {signature: idx for idx, signature in enumerate(SIGNATURE_ORDER)}
    reference["signature_rank"] = reference["signature"].map(signature_rank)
    reference = reference.sort_values(["signature_rank", "evidence_score"], ascending=[True, False])
    return reference.drop(columns=["signature_rank"]).reset_index(drop=True)


def resolve_gtex_gene(gene: str) -> dict[str, object]:
    params = {
        "geneId": gene,
        "genomeBuild": "GRCh38/hg38",
        "page": 0,
        "itemsPerPage": 10,
    }
    url = url_join(GTEX_BASE, "/reference/gene", params)
    try:
        payload = fetch_json(url)
        records = payload.get("data", [])
        query_status = "ok"
    except Exception as exc:
        records = []
        query_status = f"error: {type(exc).__name__}"
    exact = [record for record in records if record.get("geneSymbol") == gene]
    record = exact[0] if exact else (records[0] if records else {})
    return {
        "gene": gene,
        "gtex_gencode_id": record.get("gencodeId", ""),
        "gtex_gene_symbol": record.get("geneSymbol", ""),
        "gtex_description": record.get("description", ""),
        "gtex_gene_query_status": query_status,
    }


def fetch_gtex_eqtl(gene_reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_records = [resolve_gtex_gene(gene) for gene in gene_reference["gene"]]
    gene_map = pd.DataFrame(gene_records)
    tasks: list[tuple[str, str, str, str]] = []
    for _, record in gene_map.iterrows():
        gene = str(record["gene"])
        gencode_id = str(record["gtex_gencode_id"])
        for tissue_id, tissue_label in BRAIN_TISSUES:
            tasks.append((gene, gencode_id, tissue_id, tissue_label))

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_task = {
            executor.submit(fetch_gtex_eqtl_record, gene, gencode_id, tissue_id, tissue_label): (gene, tissue_id)
            for gene, gencode_id, tissue_id, tissue_label in tasks
        }
        for future in as_completed(future_to_task):
            rows.append(future.result())
    return gene_map, pd.DataFrame(rows)


def fetch_gtex_eqtl_record(gene: str, gencode_id: str, tissue_id: str, tissue_label: str) -> dict[str, object]:
    if not gencode_id:
        return {
            "gene": gene,
            "gtex_gencode_id": "",
            "tissue_id": tissue_id,
            "tissue_label": tissue_label,
            "is_striatal_tissue": tissue_id in STRIATAL_TISSUES,
            "n_independent_eqtl": 0,
            "min_p_value": np.nan,
            "top_variant_id": "",
            "query_status": "missing_gencode_id",
        }
    params = {
        "gencodeId": gencode_id,
        "tissueSiteDetailId": tissue_id,
        "datasetId": GTEX_DATASET,
        "page": 0,
        "itemsPerPage": 20,
    }
    url = url_join(GTEX_BASE, "/association/independentEqtl", params)
    try:
        payload = fetch_json(url)
        associations = payload.get("data", [])
        total = int(payload.get("paging_info", {}).get("totalNumberOfItems", len(associations)) or 0)
        query_status = "ok"
    except Exception as exc:
        associations = []
        total = 0
        query_status = f"error: {type(exc).__name__}"
    p_values = [float(item["pValue"]) for item in associations if item.get("pValue") is not None]
    if p_values:
        best = min(associations, key=lambda item: float(item.get("pValue", math.inf)))
        min_p = min(p_values)
        top_variant = best.get("variantId", "")
    else:
        min_p = np.nan
        top_variant = ""
    return {
        "gene": gene,
        "gtex_gencode_id": gencode_id,
        "tissue_id": tissue_id,
        "tissue_label": tissue_label,
        "is_striatal_tissue": tissue_id in STRIATAL_TISSUES,
        "n_independent_eqtl": total,
        "min_p_value": min_p,
        "top_variant_id": top_variant,
        "query_status": query_status,
    }


def association_rs_id(association: dict[str, Any]) -> str:
    alleles = association.get("snp_allele") or []
    rs_ids = [item.get("rs_id", "") for item in alleles if item.get("rs_id")]
    if rs_ids:
        return ";".join(sorted(set(rs_ids)))
    effect_alleles = association.get("snp_effect_allele") or []
    return ";".join(sorted(set(str(item).split("-")[0] for item in effect_alleles if item)))


def association_traits(association: dict[str, Any]) -> str:
    efo_traits = [item.get("efo_trait", "") for item in association.get("efo_traits", []) if item.get("efo_trait")]
    reported = association.get("reported_trait") or []
    all_traits = [*efo_traits, *reported]
    return "; ".join(sorted(set(str(item) for item in all_traits if item)))


def fetch_gwas_gene_trait(gene: str, trait_name: str, efo_id: str) -> dict[str, object]:
    all_associations: list[dict[str, Any]] = []
    total_associations = 0
    query_status = "ok"
    params = {
        "mapped_gene": gene,
        "efo_id": efo_id,
        "show_child_trait": "true",
        "extended_geneset": "false",
        "page": 0,
        "size": 100,
    }
    url = url_join(GWAS_BASE, "/associations", params)
    try:
        payload = fetch_json(url)
        page_info = payload.get("page", {})
        total_associations = int(page_info.get("totalElements", 0) or 0)
        associations = payload.get("_embedded", {}).get("associations", [])
        all_associations = [
            association
            for association in associations
            if gene in {str(item) for item in association.get("mapped_genes", [])}
        ]
        if total_associations > len(associations):
            query_status = "ok_first_page_min_p"
    except Exception as exc:
        query_status = f"error: {type(exc).__name__}"
    if not all_associations:
        return {
            "gene": gene,
            "trait": trait_name,
            "efo_id": efo_id,
            "n_associations": 0,
            "min_p_value": np.nan,
            "neg_log10_min_p": 0.0,
            "top_rs_id": "",
            "top_reported_trait": "",
            "top_accession_id": "",
            "top_pubmed_id": "",
            "top_first_author": "",
            "query_status": query_status,
        }
    best = min(all_associations, key=lambda item: float(item.get("p_value", math.inf)))
    min_p = float(best.get("p_value", np.nan))
    return {
        "gene": gene,
        "trait": trait_name,
        "efo_id": efo_id,
        "n_associations": max(total_associations, len(all_associations)),
        "min_p_value": min_p,
        "neg_log10_min_p": -math.log10(max(min_p, 1e-300)) if np.isfinite(min_p) else 0.0,
        "top_rs_id": association_rs_id(best),
        "top_reported_trait": association_traits(best),
        "top_accession_id": best.get("accession_id", ""),
        "top_pubmed_id": best.get("pubmed_id", ""),
        "top_first_author": best.get("first_author", ""),
        "query_status": query_status,
    }


def fetch_gwas_support(gene_reference: pd.DataFrame) -> pd.DataFrame:
    tasks = [(str(gene), trait_name, efo_id) for gene in gene_reference["gene"] for trait_name, efo_id in PSYCHIATRIC_TRAITS]
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_task = {
            executor.submit(fetch_gwas_gene_trait, gene, trait_name, efo_id): (gene, trait_name)
            for gene, trait_name, efo_id in tasks
        }
        for future in as_completed(future_to_task):
            rows.append(future.result())
    return pd.DataFrame(rows)


def summarize_support(
    gene_reference: pd.DataFrame,
    gtex_gene_map: pd.DataFrame,
    eqtl: pd.DataFrame,
    gwas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eqtl_summary = (
        eqtl.assign(has_eqtl=lambda data: data["n_independent_eqtl"] > 0)
        .groupby("gene", as_index=False)
        .agg(
            brain_eqtl_tissue_count=("has_eqtl", "sum"),
            striatal_eqtl_tissue_count=("has_eqtl", lambda values: int(values[eqtl.loc[values.index, "is_striatal_tissue"]].sum())),
            independent_eqtl_count=("n_independent_eqtl", "sum"),
            min_brain_eqtl_p=("min_p_value", "min"),
        )
    )
    eqtl_summary["brain_eqtl_score"] = eqtl_summary.apply(
        lambda row: (-math.log10(max(float(row["min_brain_eqtl_p"]), 1e-300)) + math.log2(row["brain_eqtl_tissue_count"] + 1))
        if np.isfinite(row["min_brain_eqtl_p"])
        else 0.0,
        axis=1,
    )

    gwas_summary = (
        gwas.assign(has_gwas=lambda data: data["n_associations"] > 0)
        .groupby("gene", as_index=False)
        .agg(
            psychiatric_gwas_trait_count=("has_gwas", "sum"),
            psychiatric_gwas_association_count=("n_associations", "sum"),
            min_gwas_p=("min_p_value", "min"),
            max_gwas_neg_log10_p=("neg_log10_min_p", "max"),
        )
    )
    gwas_summary["gwas_score"] = gwas_summary.apply(
        lambda row: (float(row["max_gwas_neg_log10_p"]) + math.log2(row["psychiatric_gwas_trait_count"] + 1))
        if row["psychiatric_gwas_trait_count"] > 0
        else 0.0,
        axis=1,
    )

    summary = gene_reference.merge(gtex_gene_map, on="gene", how="left")
    summary = summary.merge(eqtl_summary, on="gene", how="left").merge(gwas_summary, on="gene", how="left")
    fill_zero = [
        "brain_eqtl_tissue_count",
        "striatal_eqtl_tissue_count",
        "independent_eqtl_count",
        "brain_eqtl_score",
        "psychiatric_gwas_trait_count",
        "psychiatric_gwas_association_count",
        "max_gwas_neg_log10_p",
        "gwas_score",
    ]
    summary[fill_zero] = summary[fill_zero].fillna(0)
    summary["has_brain_eqtl"] = summary["brain_eqtl_tissue_count"] > 0
    summary["has_psychiatric_gwas"] = summary["psychiatric_gwas_trait_count"] > 0
    summary["has_integrated_support"] = summary["has_brain_eqtl"] & summary["has_psychiatric_gwas"]
    summary["integrated_support_score"] = (
        summary["evidence_score"].fillna(0)
        + summary["brain_eqtl_score"] * 0.5
        + summary["gwas_score"] * 0.5
    )

    metrics = pd.DataFrame(
        [
            {"metric": "figure5_core_signature_genes", "value": summary.shape[0]},
            {"metric": "genes_with_gtex_brain_eqtl", "value": int(summary["has_brain_eqtl"].sum())},
            {"metric": "genes_with_striatal_eqtl", "value": int((summary["striatal_eqtl_tissue_count"] > 0).sum())},
            {"metric": "genes_with_psychiatric_gwas_catalog_support", "value": int(summary["has_psychiatric_gwas"].sum())},
            {"metric": "genes_with_integrated_eqtl_gwas_support", "value": int(summary["has_integrated_support"].sum())},
            {"metric": "psychiatric_traits_tested", "value": len(PSYCHIATRIC_TRAITS)},
            {"metric": "gtex_brain_tissues_tested", "value": len(BRAIN_TISSUES)},
        ]
    )
    sort_cols = ["has_integrated_support", "psychiatric_gwas_trait_count", "brain_eqtl_tissue_count", "integrated_support_score"]
    summary = summary.sort_values(sort_cols, ascending=[False, False, False, False])
    return summary, metrics


def write_tables(refresh: bool) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = output_paths()
    if not refresh and all(path.exists() and path.stat().st_size > 0 for path in paths):
        print("Figure 5 eQTL/GWAS support tables already exist; use --refresh to query public APIs again.")
        return

    gene_reference = load_gene_reference()
    gtex_gene_map, eqtl = fetch_gtex_eqtl(gene_reference)
    gwas = fetch_gwas_support(gene_reference)
    summary, metrics = summarize_support(gene_reference, gtex_gene_map, eqtl, gwas)

    gene_reference.merge(gtex_gene_map, on="gene", how="left").to_csv(
        TABLE_DIR / "figure5_gene_reference.tsv", sep="\t", index=False
    )
    eqtl.to_csv(TABLE_DIR / "figure5_gtex_brain_eqtl.tsv", sep="\t", index=False)
    gwas.to_csv(TABLE_DIR / "figure5_gwas_catalog_support.tsv", sep="\t", index=False)
    summary.to_csv(TABLE_DIR / "figure5_integrated_eqtl_gwas_summary.tsv", sep="\t", index=False)
    metrics.to_csv(TABLE_DIR / "figure5_eqtl_gwas_metrics.tsv", sep="\t", index=False)
    print(
        "Wrote Figure 5 eQTL/GWAS support tables for "
        f"{gene_reference.shape[0]} genes, {len(BRAIN_TISSUES)} brain tissues, "
        f"and {len(PSYCHIATRIC_TRAITS)} psychiatric GWAS traits."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 5 GTEx brain eQTL and GWAS Catalog support tables.")
    parser.add_argument("--refresh", action="store_true", help="Query GTEx and GWAS Catalog even if cached tables exist.")
    args = parser.parse_args()
    write_tables(refresh=args.refresh)


if __name__ == "__main__":
    main()
