from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy import stats

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import mm_to_inches, set_publication_style  # noqa: E402


TABLE_DIR = ROOT / "results" / "tables"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}
NPJ_FIG_DIR = ROOT / "NPJ" / "Figures"

REGION_ORDER = ["NAc", "Caudate", "Putamen"]
GROUP_ORDER = ["Control", "SCZ"]
GROUP_COLORS = {"Control": "#4F86C6", "SCZ": "#D94873"}

RESTING_COL = "T_cells_CD4_memory_resting_CIBERSORT"
ACTIVATED_COL = "T_cells_CD4_memory_activated_CIBERSORT"
LM22_RESTING = "T cells CD4 memory resting"
LM22_ACTIVATED = "T cells CD4 memory activated"

IMMUNE_HALLMARKS = [
    "TNFA_SIGNALING_VIA_NFKB",
    "IL6_JAK_STAT3_SIGNALING",
    "IL2_STAT5_SIGNALING",
    "INTERFERON_GAMMA_RESPONSE",
    "INTERFERON_ALPHA_RESPONSE",
    "INFLAMMATORY_RESPONSE",
    "COMPLEMENT",
]

HALLMARK_LABELS = {
    "TNFA_SIGNALING_VIA_NFKB": "TNFA/NF-kB",
    "IL6_JAK_STAT3_SIGNALING": "IL6/JAK/STAT3",
    "IL2_STAT5_SIGNALING": "IL2/STAT5",
    "INTERFERON_GAMMA_RESPONSE": "IFN-gamma",
    "INTERFERON_ALPHA_RESPONSE": "IFN-alpha",
    "INFLAMMATORY_RESPONSE": "Inflammatory",
    "COMPLEMENT": "Complement",
}


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


def normalize_hallmark_id(raw_name: str) -> str:
    hallmark_id = raw_name.upper().replace("HALLMARK_", "")
    hallmark_id = re.sub(r"[^A-Z0-9]+", "_", hallmark_id)
    hallmark_id = re.sub(r"_+", "_", hallmark_id).strip("_")
    fixes = {
        "TNF_ALPHA_SIGNALING_VIA_NF_KB": "TNFA_SIGNALING_VIA_NFKB",
        "IL_6_JAK_STAT3_SIGNALING": "IL6_JAK_STAT3_SIGNALING",
        "IL_2_STAT5_SIGNALING": "IL2_STAT5_SIGNALING",
    }
    return fixes.get(hallmark_id, hallmark_id)


def parse_hallmark_gene_sets(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            hallmark_id = normalize_hallmark_id(parts[0].strip())
            if hallmark_id in IMMUNE_HALLMARKS:
                gene_sets[hallmark_id] = sorted({gene.strip().upper() for gene in parts[2:] if gene.strip()})
    return gene_sets


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, np.nan), axis=0)


def score_gene_sets(logcpm: pd.DataFrame, gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    expr_z = zscore_rows(logcpm)
    rows: list[pd.Series] = []
    names: list[str] = []
    for name, genes in gene_sets.items():
        available = sorted(set(genes) & set(expr_z.index))
        if len(available) < 3:
            continue
        rows.append(expr_z.loc[available].mean(axis=0))
        names.append(name)
    return pd.DataFrame(rows, index=names)


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    qc = pd.read_csv(PROCESSED_DIR / "gse202537_expression_qc.csv")
    metadata = metadata.merge(qc[["sample_key", "count_depth_million"]], on="sample_key", how="left")
    metadata = metadata.loc[metadata["group"].isin(GROUP_ORDER) & metadata["brain_region"].isin(REGION_ORDER)].copy()
    metadata["group"] = pd.Categorical(metadata["group"], categories=GROUP_ORDER, ordered=True)
    metadata["brain_region"] = pd.Categorical(metadata["brain_region"], categories=REGION_ORDER, ordered=True)
    return metadata


def load_cd4_cibersort(metadata: pd.DataFrame) -> pd.DataFrame:
    wide = pd.read_csv(TABLE_DIR / "npj_figure2_iobr_cibersort_wide.tsv", sep="\t").rename(columns={"ID": "sample_key"})
    data = metadata.merge(wide[["sample_key", RESTING_COL, ACTIVATED_COL]], on="sample_key", how="inner")
    data = data.rename(columns={RESTING_COL: "cibersort_resting", ACTIVATED_COL: "cibersort_activated"})
    data["cibersort_balance"] = np.log2((data["cibersort_activated"] + 0.001) / (data["cibersort_resting"] + 0.001))
    data["cibersort_activation_index"] = data["cibersort_activated"] / (
        data["cibersort_activated"] + data["cibersort_resting"] + 1e-9
    )
    return data


def derive_lm22_marker_sets(lm22: pd.DataFrame, available_genes: set[str], top_n: int = 30) -> tuple[dict[str, list[str]], pd.DataFrame]:
    numeric = lm22.drop(columns=["gene_symbol"])
    log_sig = np.log2(numeric + 1)
    rows: list[pd.DataFrame] = []
    marker_sets: dict[str, list[str]] = {}
    for state, column in [("Resting", LM22_RESTING), ("Activated", LM22_ACTIVATED)]:
        other_max = log_sig.drop(columns=[column]).max(axis=1)
        specificity = log_sig[column] - other_max
        table = pd.DataFrame(
            {
                "state": state,
                "gene_symbol": lm22["gene_symbol"].str.upper(),
                "target_log2_signature": log_sig[column],
                "specificity_vs_other_max": specificity,
            }
        )
        table = table.loc[table["gene_symbol"].isin(available_genes)].sort_values(
            ["specificity_vs_other_max", "target_log2_signature"],
            ascending=[False, False],
        )
        marker_sets[state] = table.head(top_n)["gene_symbol"].tolist()
        rows.append(table.head(top_n).assign(rank=np.arange(1, min(top_n, len(table)) + 1)))
    return marker_sets, pd.concat(rows, ignore_index=True)


def compute_marker_and_projection_scores(
    logcpm: pd.DataFrame,
    lm22: pd.DataFrame,
    marker_sets: dict[str, list[str]],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    expr_z = zscore_rows(logcpm)
    score_data = pd.DataFrame({"sample_key": metadata["sample_key"].to_numpy()})
    for state, genes in marker_sets.items():
        available = sorted(set(genes) & set(expr_z.index))
        score_data[f"marker_{state.lower()}"] = expr_z.loc[available, metadata["sample_key"]].mean(axis=0).to_numpy()

    lm22_expr = lm22.set_index(lm22["gene_symbol"].str.upper()).drop(columns=["gene_symbol"])
    common = sorted(set(lm22_expr.index) & set(logcpm.index))
    sample_expr = logcpm.loc[common, metadata["sample_key"]]
    for state, column in [("resting", LM22_RESTING), ("activated", LM22_ACTIVATED)]:
        target = np.log2(lm22_expr.loc[common, column].astype(float) + 1)
        projections: list[float] = []
        for sample in sample_expr.columns:
            r_value, _ = stats.spearmanr(sample_expr[sample].to_numpy(dtype=float), target.to_numpy(dtype=float))
            projections.append(r_value)
        score_data[f"projection_{state}"] = projections

    for prefix in ["marker", "projection"]:
        score_data[f"{prefix}_balance"] = score_data[f"{prefix}_activated"] - score_data[f"{prefix}_resting"]
    return metadata.merge(score_data, on="sample_key", how="left")


def design_matrix(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index]:
    data = metadata.copy()
    data["intercept"] = 1.0
    data["is_scz"] = (data["group"] == "SCZ").astype(float)
    data["sex_male"] = (data["gender"] == "Male").astype(float)
    data["tod_sin"] = np.sin(2 * np.pi * data["corrected_tod_24h"] / 24)
    data["tod_cos"] = np.cos(2 * np.pi * data["corrected_tod_24h"] / 24)
    for column in ["age", "pmi", "rin", "ph", "count_depth_million"]:
        std = data[column].std()
        data[f"{column}_z"] = (data[column] - data[column].mean()) / std if std and not np.isnan(std) else 0
    columns = [
        "intercept",
        "is_scz",
        "age_z",
        "sex_male",
        "pmi_z",
        "rin_z",
        "ph_z",
        "count_depth_million_z",
        "tod_sin",
        "tod_cos",
    ]
    model_data = data.dropna(subset=columns)
    return model_data[columns].to_numpy(dtype=float), columns, model_data.index


def adjusted_effects(data: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        region_data = data.loc[data["brain_region"] == region].copy()
        x_matrix, columns, idx = design_matrix(region_data)
        for metric in metric_columns:
            y = region_data.loc[idx, metric].to_numpy(dtype=float)
            valid = np.isfinite(y)
            x = x_matrix[valid]
            y = y[valid]
            if len(y) <= x.shape[1] + 2:
                beta = t_value = p_value = np.nan
            else:
                coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
                residuals = y - x @ coef
                df_resid = len(y) - x.shape[1]
                sigma2 = float((residuals @ residuals) / df_resid)
                xtx_inv = np.linalg.pinv(x.T @ x)
                se = np.sqrt(np.diag(xtx_inv) * sigma2)
                coef_idx = columns.index("is_scz")
                beta = float(coef[coef_idx])
                t_value = float(coef[coef_idx] / se[coef_idx]) if se[coef_idx] > 0 else np.nan
                p_value = float(2 * stats.t.sf(abs(t_value), df_resid)) if np.isfinite(t_value) else np.nan
            rows.append({"brain_region": region, "metric": metric, "beta_scz": beta, "t_value": t_value, "p_value": p_value})
    result = pd.DataFrame(rows)
    result["fdr_region"] = result.groupby("brain_region")["p_value"].transform(fdr_bh)
    return result


def correlation_with_hallmarks(cd4_data: pd.DataFrame, hallmark_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cd4 = cd4_data.set_index("sample_key")
    for region in REGION_ORDER:
        samples = cd4_data.loc[cd4_data["brain_region"] == region, "sample_key"].tolist()
        activation = cd4.loc[samples, "cibersort_balance"]
        for hallmark_id in IMMUNE_HALLMARKS:
            if hallmark_id not in hallmark_scores.index:
                continue
            score = hallmark_scores.loc[hallmark_id, samples]
            r_value, p_value = stats.spearmanr(activation, score)
            rows.append(
                {
                    "brain_region": region,
                    "hallmark_id": hallmark_id,
                    "hallmark_label": HALLMARK_LABELS[hallmark_id],
                    "spearman_r": r_value,
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_region"] = result.groupby("brain_region")["p_value"].transform(fdr_bh)
    return result


def panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def annotate_group_tests(ax, data: pd.DataFrame, metric: str, y_base: float) -> None:
    for xi, region in enumerate(REGION_ORDER):
        subset = data.loc[data["brain_region"] == region]
        control = subset.loc[subset["group"] == "Control", metric].to_numpy(dtype=float)
        scz = subset.loc[subset["group"] == "SCZ", metric].to_numpy(dtype=float)
        _, p_value = stats.mannwhitneyu(scz, control, alternative="two-sided")
        if p_value < 0.05:
            ax.text(xi, y_base, f"p={p_value:.2f}", ha="center", va="bottom", fontsize=4.8, color="#333333")


def draw_single_fraction_panel(ax, cd4_data: pd.DataFrame, metric: str, title: str, show_ylabel: bool, show_legend: bool) -> None:
    sns.boxplot(
        data=cd4_data,
        x="brain_region",
        y=metric,
        hue="group",
        order=REGION_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        width=0.58,
        fliersize=0,
        linewidth=0.55,
        ax=ax,
    )
    sns.stripplot(
        data=cd4_data,
        x="brain_region",
        y=metric,
        hue="group",
        order=REGION_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        dodge=True,
        size=1.8,
        alpha=0.42,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(title, fontsize=6.5, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction" if show_ylabel else "")
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if show_legend:
        ax.legend(handles[:2], labels[:2], frameon=False, fontsize=5.2, loc="upper right")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(axis="both", labelsize=5.6, width=0.5, length=2)


def draw_balance_panel(ax, cd4_data: pd.DataFrame) -> None:
    sns.boxplot(
        data=cd4_data,
        x="brain_region",
        y="cibersort_balance",
        hue="group",
        order=REGION_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        width=0.58,
        fliersize=0,
        linewidth=0.55,
        ax=ax,
    )
    sns.stripplot(
        data=cd4_data,
        x="brain_region",
        y="cibersort_balance",
        hue="group",
        order=REGION_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        dodge=True,
        size=1.8,
        alpha=0.42,
        linewidth=0,
        ax=ax,
    )
    y_base = float(cd4_data["cibersort_balance"].max()) + 0.25
    annotate_group_tests(ax, cd4_data, "cibersort_balance", y_base)
    ax.set_title("Activated/resting CIBERSORT balance", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("log2((activated+0.001)/(resting+0.001))")
    ax.spines[["top", "right"]].set_visible(False)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(axis="both", labelsize=5.6, width=0.5, length=2)


def draw_effect_heatmap(ax, effects: pd.DataFrame) -> None:
    metric_labels = {
        "cibersort_resting": "CIBERSORT resting",
        "cibersort_activated": "CIBERSORT activated",
        "cibersort_balance": "CIBERSORT balance",
        "marker_resting": "LM22 marker resting",
        "marker_activated": "LM22 marker activated",
        "marker_balance": "LM22 marker balance",
        "projection_resting": "LM22 projection resting",
        "projection_activated": "LM22 projection activated",
        "projection_balance": "LM22 projection balance",
    }
    matrix = (
        effects.assign(metric_label=lambda data: data["metric"].map(metric_labels))
        .pivot(index="metric_label", columns="brain_region", values="beta_scz")
        .reindex(index=list(metric_labels.values()), columns=REGION_ORDER)
    )
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Adjusted SCZ beta", "shrink": 0.72},
        ax=ax,
    )
    p_lookup = effects.assign(metric_label=lambda data: data["metric"].map(metric_labels)).set_index(["metric_label", "brain_region"])
    for y, metric in enumerate(matrix.index):
        for x, region in enumerate(matrix.columns):
            row = p_lookup.loc[(metric, region)]
            if row["fdr_region"] < 0.10:
                ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", fontsize=7.0, color="black")
            elif row["p_value"] < 0.05:
                ax.text(x + 0.5, y + 0.5, "+", ha="center", va="center", fontsize=6.0, color="black")
    ax.set_title("Covariate-adjusted SCZ effects across CD4 state metrics", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.2, width=0.5, length=2)


def draw_marker_gene_heatmap(ax, logcpm: pd.DataFrame, metadata: pd.DataFrame, marker_table: pd.DataFrame) -> None:
    selected = (
        marker_table.sort_values(["state", "specificity_vs_other_max"], ascending=[True, False])
        .groupby("state", as_index=False, sort=False)
        .head(8)
        .copy()
    )
    genes = selected["gene_symbol"].tolist()
    group_cols: list[str] = []
    values: list[pd.Series] = []
    expr_z = zscore_rows(logcpm)
    for region in REGION_ORDER:
        for group in GROUP_ORDER:
            samples = metadata.loc[(metadata["brain_region"] == region) & (metadata["group"] == group), "sample_key"]
            values.append(expr_z.loc[genes, samples].mean(axis=1))
            group_cols.append(f"{region}\n{group}")
    matrix = pd.concat(values, axis=1)
    matrix.columns = group_cols
    matrix = matrix.loc[genes]
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.20,
        linecolor="white",
        cbar_kws={"label": "Mean z expression", "shrink": 0.72},
        ax=ax,
    )
    ax.set_title("LM22 CD4 memory marker-gene expression", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.0, width=0.5, length=2)


def draw_hallmark_correlation(ax, correlations: pd.DataFrame) -> None:
    matrix = (
        correlations.pivot(index="hallmark_label", columns="brain_region", values="spearman_r")
        .reindex(index=[HALLMARK_LABELS[item] for item in IMMUNE_HALLMARKS], columns=REGION_ORDER)
    )
    sns.heatmap(
        matrix,
        cmap="vlag",
        vmin=-0.6,
        vmax=0.6,
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Spearman r", "shrink": 0.72},
        ax=ax,
    )
    lookup = correlations.set_index(["hallmark_label", "brain_region"])
    for y, hallmark in enumerate(matrix.index):
        for x, region in enumerate(matrix.columns):
            row = lookup.loc[(hallmark, region)]
            if row["fdr_region"] < 0.10:
                ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", fontsize=7.0, color="black")
            elif row["p_value"] < 0.05:
                ax.text(x + 0.5, y + 0.5, "+", ha="center", va="center", fontsize=6.0, color="black")
    ax.set_title("CD4 activation balance versus immune Hallmark scores", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.2, width=0.5, length=2)


def main() -> None:
    set_publication_style()
    metadata = load_metadata()
    cd4_data = load_cd4_cibersort(metadata)
    logcpm = np.log2(pd.read_csv(PROCESSED_DIR / "gse202537_gene_cpm_for_iobr.tsv", sep="\t", index_col=0) + 1)
    logcpm.index = logcpm.index.str.upper()
    lm22 = pd.read_csv(PROCESSED_DIR / "iobr_lm22_signature_matrix.tsv", sep="\t")
    marker_sets, marker_table = derive_lm22_marker_sets(lm22, set(logcpm.index))
    score_data = compute_marker_and_projection_scores(logcpm, lm22, marker_sets, metadata)
    combined = cd4_data.merge(
        score_data[
            [
                "sample_key",
                "marker_resting",
                "marker_activated",
                "marker_balance",
                "projection_resting",
                "projection_activated",
                "projection_balance",
            ]
        ],
        on="sample_key",
        how="left",
    )
    metrics = [
        "cibersort_resting",
        "cibersort_activated",
        "cibersort_balance",
        "marker_resting",
        "marker_activated",
        "marker_balance",
        "projection_resting",
        "projection_activated",
        "projection_balance",
    ]
    effects = adjusted_effects(combined, metrics)
    hallmark_scores = score_gene_sets(logcpm, parse_hallmark_gene_sets(RAW_DIR / "MSigDB_Hallmark_2020.gmt"))
    correlations = correlation_with_hallmarks(combined, hallmark_scores)

    marker_table.to_csv(TABLE_DIR / "npj_figure3_lm22_cd4_marker_genes.tsv", sep="\t", index=False)
    combined.to_csv(TABLE_DIR / "npj_figure3_cd4_memory_sample_metrics.tsv", sep="\t", index=False)
    effects.to_csv(TABLE_DIR / "npj_figure3_cd4_memory_adjusted_effects.tsv", sep="\t", index=False)
    hallmark_scores.to_csv(PROCESSED_DIR / "gse202537_hallmark_immune_scores.csv")
    correlations.to_csv(TABLE_DIR / "npj_figure3_cd4_hallmark_correlations.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=mm_to_inches(190, 300), constrained_layout=False)
    fig.text(
        0.5,
        0.985,
        "Multi-layer evaluation of CD4 memory T-cell state in striatal psychosis",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax_a1 = fig.add_axes([0.075, 0.725, 0.180, 0.185])
    draw_single_fraction_panel(ax_a1, combined, "cibersort_resting", "Resting", True, True)
    ax_a2 = fig.add_axes([0.300, 0.725, 0.180, 0.185])
    draw_single_fraction_panel(ax_a2, combined, "cibersort_activated", "Activated", False, False)
    fig.text(0.280, 0.927, "CIBERSORT CD4 memory fractions", ha="center", va="center", fontsize=7.0)
    ax_b = fig.add_axes([0.570, 0.725, 0.36, 0.185])
    draw_balance_panel(ax_b, combined)
    ax_c = fig.add_axes([0.075, 0.405, 0.38, 0.225])
    draw_effect_heatmap(ax_c, effects)
    ax_d = fig.add_axes([0.555, 0.405, 0.38, 0.225])
    draw_marker_gene_heatmap(ax_d, logcpm, metadata, marker_table)
    ax_e = fig.add_axes([0.250, 0.105, 0.50, 0.205])
    draw_hallmark_correlation(ax_e, correlations)

    fig.text(0.5, 0.065, "+ nominal P<0.05; * region-level FDR<0.10", ha="center", va="center", fontsize=5.5)

    panel_label(fig, "A", 0.035, 0.935)
    panel_label(fig, "B", 0.525, 0.935)
    panel_label(fig, "C", 0.035, 0.655)
    panel_label(fig, "D", 0.515, 0.655)
    panel_label(fig, "E", 0.035, 0.330)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"npj_figure3_cd4_memory_state.{ext}", dpi=600, bbox_inches="tight")

    NPJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_FIG_DIR / "Figure-03.png", dpi=600, bbox_inches="tight")
    fig.savefig(NPJ_FIG_DIR / "Figure-03.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote NPJ Figure 3 CD4 memory state figure.")


if __name__ == "__main__":
    main()
