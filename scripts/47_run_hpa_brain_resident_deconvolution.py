from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"

HPA_CLUSTER_TYPE_ZIP = RAW_DIR / "rna_single_nuclei_cluster_type.tsv.zip"
CPM_FILE = PROCESSED_DIR / "gse202537_gene_cpm_for_iobr.tsv"
METADATA_FILE = PROCESSED_DIR / "gse202537_sample_metadata.csv"

REGION_ORDER = ["NAc", "Caudate", "Putamen"]
GROUP_ORDER = ["Control", "SCZ"]

BRAIN_CLASSES: dict[str, list[str]] = {
    "MSN": ["medium spiny neuron", "eccentric medium spiny neuron"],
    "Interneuron": ["CGE interneuron", "MGE interneuron", "LAMP5-LHX6 and Chandelier"],
    "Astrocyte": ["astrocyte"],
    "Oligodendrocyte": ["oligodendrocyte"],
    "OPC": ["oligodendrocyte precursor cell", "committed oligodendrocyte precursor"],
    "Microglia": ["central nervous system macrophage"],
    "Vascular": ["endothelial cell", "pericyte", "vascular associated smooth muscle cell"],
}
BRAIN_CLASS_ORDER = list(BRAIN_CLASSES)


def load_gsva_module():
    module_path = ROOT / "scripts" / "05_run_elsevier_gsva.py"
    spec = importlib.util.spec_from_file_location("elsevier_gsva", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Elsevier GSVA helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def load_hpa_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    hpa = pd.read_csv(HPA_CLUSTER_TYPE_ZIP, sep="\t", compression="zip")
    hpa["Gene name"] = hpa["Gene name"].astype(str).str.upper()
    expr = hpa.pivot_table(index="Gene name", columns="Cluster type", values="nCPM", aggfunc="mean").fillna(0.0)

    ref = pd.DataFrame(index=expr.index)
    rows: list[dict[str, object]] = []
    all_clusters = list(expr.columns)
    for brain_class, clusters in BRAIN_CLASSES.items():
        present = [cluster for cluster in clusters if cluster in expr.columns]
        if not present:
            raise ValueError(f"No HPA clusters found for {brain_class}: {clusters}")
        ref[brain_class] = expr[present].mean(axis=1)
        other = [cluster for cluster in all_clusters if cluster not in present]
        group_expr = ref[brain_class]
        other_expr = expr[other].max(axis=1)
        specificity = np.log2((group_expr + 1.0) / (other_expr + 1.0))
        marker_table = (
            pd.DataFrame(
                {
                    "gene": expr.index,
                    "brain_cell_class": brain_class,
                    "source_clusters": ";".join(present),
                    "group_nCPM": group_expr,
                    "max_other_nCPM": other_expr,
                    "specificity_log2_ratio": specificity,
                }
            )
            .loc[lambda frame: frame["group_nCPM"] >= 1.0]
            .sort_values(["specificity_log2_ratio", "group_nCPM"], ascending=[False, False])
            .head(80)
        )
        rows.extend(marker_table.to_dict("records"))
    return ref[BRAIN_CLASS_ORDER], pd.DataFrame(rows)


def load_bulk_logcpm() -> pd.DataFrame:
    cpm = pd.read_csv(CPM_FILE, sep="\t", index_col=0)
    cpm.index = cpm.index.astype(str).str.upper()
    cpm = cpm.groupby(cpm.index).mean()
    return np.log2(cpm + 1.0)


def nnls_weights(reference: pd.DataFrame, bulk_logcpm: pd.DataFrame, markers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_markers = markers.loc[markers["gene"].isin(bulk_logcpm.index)].copy()
    marker_genes = sorted(available_markers["gene"].unique())
    if len(marker_genes) < 20:
        raise ValueError("Too few HPA/Siletti marker genes overlapped with GSE202537.")

    ref_log = np.log2(reference.loc[marker_genes] + 1.0)
    ref_log = ref_log.loc[:, BRAIN_CLASS_ORDER]
    bulk = bulk_logcpm.loc[marker_genes]

    weights: dict[str, np.ndarray] = {}
    ref_matrix = ref_log.to_numpy(dtype=float)
    for sample_key in bulk.columns:
        target = bulk[sample_key].to_numpy(dtype=float)
        coef, _ = optimize.nnls(ref_matrix, target)
        if coef.sum() > 0:
            coef = coef / coef.sum()
        else:
            coef = np.repeat(1.0 / len(BRAIN_CLASS_ORDER), len(BRAIN_CLASS_ORDER))
        weights[sample_key] = coef

    weight_df = pd.DataFrame(weights, index=BRAIN_CLASS_ORDER)
    marker_counts = (
        available_markers.groupby("brain_cell_class")
        .agg(
            source_marker_count=("gene", "size"),
            available_marker_count=("gene", "nunique"),
            available_markers=("gene", lambda value: ";".join(sorted(set(value)))),
            source_clusters=("source_clusters", "first"),
        )
        .reindex(BRAIN_CLASS_ORDER)
        .reset_index()
    )
    return weight_df, marker_counts


def adjusted_ols_effect(subset: pd.DataFrame) -> tuple[float, float]:
    model_df = subset.loc[subset["group"].isin(GROUP_ORDER)].copy()
    if model_df["group"].nunique() < 2:
        return np.nan, np.nan
    model_df["psychosis"] = (model_df["group"].astype(str) == "SCZ").astype(float)
    model_df["tod_sin"] = np.sin(2 * np.pi * model_df["corrected_tod_24h"].astype(float) / 24)
    model_df["tod_cos"] = np.cos(2 * np.pi * model_df["corrected_tod_24h"].astype(float) / 24)
    y = np.arcsin(np.sqrt(np.clip(model_df["weight"].astype(float).to_numpy(), 0, 1)))
    covariates = ["psychosis", "age", "pmi", "rin", "ph", "library_size", "tod_sin", "tod_cos"]
    design = model_df[covariates].astype(float)
    design["gender_male"] = (model_df["gender"].astype(str).str.lower() == "male").astype(float)
    for column in [item for item in design.columns if item != "psychosis"]:
        sd = design[column].std(ddof=0)
        design[column] = (design[column] - design[column].mean()) / sd if np.isfinite(sd) and sd > 0 else 0.0
    x = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    rank = np.linalg.matrix_rank(x)
    if len(y) <= rank:
        return np.nan, np.nan
    coef = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ coef
    df_resid = len(y) - rank
    sigma2 = float((residual @ residual) / df_resid)
    cov_beta = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(cov_beta))
    if se[1] == 0 or not np.isfinite(se[1]):
        return float(coef[1]), np.nan
    p_value = 2 * stats.t.sf(abs(coef[1] / se[1]), df_resid)
    return float(coef[1]), float(p_value)


def summarize_and_test(weight_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(METADATA_FILE)
    long = (
        weight_df.reset_index(names="brain_cell_class")
        .melt(id_vars="brain_cell_class", var_name="sample_key", value_name="weight")
        .merge(
            metadata[
                [
                    "sample_key",
                    "patients_id",
                    "group",
                    "brain_region",
                    "age",
                    "gender",
                    "pmi",
                    "rin",
                    "ph",
                    "library_size",
                    "corrected_tod_24h",
                ]
            ],
            on="sample_key",
            how="left",
        )
    )
    long = long.loc[long["brain_region"].isin(REGION_ORDER) & long["group"].isin(GROUP_ORDER)].copy()
    long["brain_cell_class"] = pd.Categorical(long["brain_cell_class"], categories=BRAIN_CLASS_ORDER, ordered=True)
    long["brain_region"] = pd.Categorical(long["brain_region"], categories=REGION_ORDER, ordered=True)
    long["group"] = pd.Categorical(long["group"], categories=GROUP_ORDER, ordered=True)

    summary = (
        long.groupby(["brain_region", "group", "brain_cell_class"], observed=True)["weight"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        for brain_class in BRAIN_CLASS_ORDER:
            subset = long.loc[(long["brain_region"] == region) & (long["brain_cell_class"] == brain_class)]
            control = subset.loc[subset["group"] == "Control", "weight"].to_numpy(dtype=float)
            scz = subset.loc[subset["group"] == "SCZ", "weight"].to_numpy(dtype=float)
            beta, p_value = adjusted_ols_effect(subset)
            rows.append(
                {
                    "brain_region": region,
                    "brain_cell_class": brain_class,
                    "control_mean": np.nanmean(control),
                    "scz_mean": np.nanmean(scz),
                    "mean_difference_scz_minus_control": np.nanmean(scz) - np.nanmean(control),
                    "adjusted_beta_arcsine": beta,
                    "p_value": p_value,
                }
            )
    tests = pd.DataFrame(rows)
    tests["fdr_region"] = tests.groupby("brain_region")["p_value"].transform(fdr_bh)
    tests["fdr_global"] = fdr_bh(tests["p_value"])
    return long, summary, tests


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    load_gsva_module()

    reference, source_markers = load_hpa_reference()
    bulk_logcpm = load_bulk_logcpm()
    weight_df, marker_counts = nnls_weights(reference, bulk_logcpm, source_markers)
    long, summary, tests = summarize_and_test(weight_df)

    source_markers["available_in_gse202537"] = source_markers["gene"].isin(bulk_logcpm.index)
    source_markers.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_source_markers.tsv", sep="\t", index=False)
    marker_counts.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_marker_counts.tsv", sep="\t", index=False)
    reference.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_reference_profiles.tsv", sep="\t")
    weight_df.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_nnls_weights_wide.tsv", sep="\t")
    long.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_nnls_weights_long.tsv", sep="\t", index=False)
    summary.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_group_summary.tsv", sep="\t", index=False)
    tests.to_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_group_tests.tsv", sep="\t", index=False)
    print(
        "Wrote HPA/Siletti brain-resident NNLS weights for "
        f"{len(BRAIN_CLASS_ORDER)} classes and {weight_df.shape[1]} bulk samples."
    )


if __name__ == "__main__":
    main()
