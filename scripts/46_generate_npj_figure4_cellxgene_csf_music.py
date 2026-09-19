from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import mm_to_inches, set_publication_style  # noqa: E402


TABLE_DIR = ROOT / "results" / "tables"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}
NPJ_FIG_DIR = ROOT / "NPJ" / "Figures"

REGION_ORDER = ["NAc", "Caudate", "Putamen"]
GROUP_ORDER = ["Control", "SCZ"]
CD4_FOCUS = [
    "CD4_Naive",
    "CD4_CM",
    "CD4_T_reg",
    "CD4_Th17",
    "CD4_T_helper",
    "CD4_IFN_Response",
    "T_cells_Proliferative",
]

BROAD_ORDER = ["CD4 T", "CD8 T", "NK/gd T", "B/plasma", "DC", "Myeloid/microglia"]
BROAD_COLORS = {
    "CD4 T": "#7A63C7",
    "CD8 T": "#3E78B2",
    "NK/gd T": "#1B998B",
    "B/plasma": "#F28E2B",
    "DC": "#59A14F",
    "Myeloid/microglia": "#D64F6F",
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


def broad_class(cell_type: str) -> str:
    if cell_type.startswith("CD4_") or cell_type == "T_cells_Proliferative":
        return "CD4 T"
    if cell_type.startswith("CD8_"):
        return "CD8 T"
    if cell_type == "NK_gd":
        return "NK/gd T"
    if cell_type.startswith("B_cell") or cell_type == "Plasma":
        return "B/plasma"
    if cell_type.startswith("DC") or cell_type == "pDC":
        return "DC"
    return "Myeloid/microglia"


def load_reference() -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    ref = pd.read_csv(TABLE_DIR / "npj_figure4_csf_reference_celltype_summary.tsv", sep="\t")
    ref["broad_class"] = ref["music_cell_type"].map(broad_class)
    order = []
    for broad in BROAD_ORDER:
        items = ref.loc[ref["broad_class"] == broad].sort_values("n_cells", ascending=False)["music_cell_type"].tolist()
        order.extend(items)
    labels = dict(zip(ref["music_cell_type"], ref["lv2_annot"]))
    return ref, order, labels


def load_music_long(table_name: str, cell_order: list[str]) -> pd.DataFrame:
    wide = pd.read_csv(TABLE_DIR / table_name, sep="\t")
    metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    available = [cell for cell in cell_order if cell in wide.columns]
    merged = wide[["sample_key", *available]].merge(
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
    merged = merged.loc[merged["group"].isin(GROUP_ORDER) & merged["brain_region"].isin(REGION_ORDER)].copy()
    long = merged.melt(
        id_vars=[
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
        ],
        value_vars=available,
        var_name="cell_type",
        value_name="proportion",
    )
    long["cell_type"] = pd.Categorical(long["cell_type"], categories=available, ordered=True)
    long["group"] = pd.Categorical(long["group"], categories=GROUP_ORDER, ordered=True)
    long["brain_region"] = pd.Categorical(long["brain_region"], categories=REGION_ORDER, ordered=True)
    return long


def adjusted_ols_effect(subset: pd.DataFrame) -> tuple[float, float]:
    model_df = subset.loc[subset["group"].isin(GROUP_ORDER)].copy()
    if model_df["group"].nunique() < 2:
        return np.nan, np.nan
    model_df["psychosis"] = (model_df["group"].astype(str) == "SCZ").astype(float)
    model_df["tod_sin"] = np.sin(2 * np.pi * model_df["corrected_tod_24h"].astype(float) / 24)
    model_df["tod_cos"] = np.cos(2 * np.pi * model_df["corrected_tod_24h"].astype(float) / 24)
    y = np.arcsin(np.sqrt(np.clip(model_df["proportion"].astype(float).to_numpy(), 0, 1)))
    covariates = ["psychosis", "age", "pmi", "rin", "ph", "library_size", "tod_sin", "tod_cos"]
    design = model_df[covariates].astype(float)
    design["gender_male"] = (model_df["gender"].astype(str).str.lower() == "male").astype(float)
    for col in [item for item in design.columns if item != "psychosis"]:
        sd = design[col].std(ddof=0)
        design[col] = (design[col] - design[col].mean()) / sd if np.isfinite(sd) and sd > 0 else 0.0
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


def summarize_and_test(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        long.groupby(["brain_region", "group", "cell_type"], observed=False)["proportion"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    rows = []
    for region in REGION_ORDER:
        for cell_type in long["cell_type"].cat.categories:
            subset = long.loc[(long["brain_region"] == region) & (long["cell_type"] == cell_type)]
            control = subset.loc[subset["group"] == "Control", "proportion"].to_numpy(dtype=float)
            scz = subset.loc[subset["group"] == "SCZ", "proportion"].to_numpy(dtype=float)
            beta, p_value = adjusted_ols_effect(subset)
            rows.append(
                {
                    "brain_region": region,
                    "cell_type": cell_type,
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
    return summary, tests


def panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def draw_reference_panel(ax, ref: pd.DataFrame, cell_order: list[str], labels: dict[str, str]) -> None:
    ref = ref.set_index("music_cell_type").reindex(cell_order).reset_index()
    colors = [BROAD_COLORS[broad_class(cell)] for cell in ref["music_cell_type"]]
    ax.barh(np.arange(len(ref)), ref["n_cells"], color=colors, edgecolor="none")
    ax.set_yticks(np.arange(len(ref)))
    ax.set_yticklabels([labels[cell] for cell in ref["music_cell_type"]])
    ax.invert_yaxis()
    ax.set_xlabel("Reference cells")
    ax.set_title("CELLxGENE CSF reference lv2 states", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=4.6, width=0.5, length=2)


def draw_broad_stacked(ax, long: pd.DataFrame) -> None:
    data = long.copy()
    data["broad_class"] = data["cell_type"].astype(str).map(broad_class)
    broad = (
        data.groupby(["sample_key", "brain_region", "group", "broad_class"], observed=True)["proportion"]
        .sum()
        .reset_index()
    )
    summary = (
        broad.groupby(["brain_region", "group", "broad_class"], observed=True)["proportion"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot_table(
        index=["brain_region", "group"],
        columns="broad_class",
        values="proportion",
        fill_value=0,
        observed=False,
    ).reindex(pd.MultiIndex.from_product([REGION_ORDER, GROUP_ORDER], names=["brain_region", "group"]))
    positions = np.array([0, 0.82, 2.15, 2.97, 4.30, 5.12])
    bottoms = np.zeros(len(pivot), dtype=float)
    for broad_class_name in BROAD_ORDER:
        values = pivot.get(broad_class_name, pd.Series(0, index=pivot.index)).to_numpy(dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=BROAD_COLORS[broad_class_name],
            edgecolor="white",
            linewidth=0.18,
            label=broad_class_name,
        )
        bottoms += values
    ax.set_xticks(positions)
    ax.set_xticklabels(["Control", "SCZ"] * 3)
    for center, region in zip([0.41, 2.56, 4.71], REGION_ORDER):
        ax.text(center, 1.035, region, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean MuSiC proportion")
    ax.set_title("Estimated CNS-immune composition", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.3, width=0.5, length=2)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.03), loc="upper left", fontsize=5.1)


def draw_full_effect_heatmap(ax, tests: pd.DataFrame, labels: dict[str, str]) -> None:
    top = (
        tests.groupby("cell_type")["mean_difference_scz_minus_control"]
        .apply(lambda value: np.nanmax(np.abs(value.to_numpy(dtype=float))))
        .sort_values(ascending=False)
        .head(14)
        .index.tolist()
    )
    matrix = tests.pivot(index="cell_type", columns="brain_region", values="mean_difference_scz_minus_control")
    matrix = matrix.reindex(index=top, columns=REGION_ORDER)
    matrix.index = [labels.get(item, item) for item in matrix.index]
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "SCZ-control mean difference", "shrink": 0.78},
        ax=ax,
    )
    lookup = tests.set_index(["cell_type", "brain_region"])
    for y, cell in enumerate(top):
        for x, region in enumerate(REGION_ORDER):
            row = lookup.loc[(cell, region)]
            if row["fdr_region"] < 0.10:
                ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", fontsize=7.0, color="black")
            elif row["p_value"] < 0.05:
                ax.text(x + 0.5, y + 0.5, "+", ha="center", va="center", fontsize=6.0, color="black")
    ax.set_title("Top SCZ-control lv2-state differences", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.0, width=0.5, length=2)


def draw_cd4_dotplot(ax, long: pd.DataFrame, tests: pd.DataFrame, labels: dict[str, str]) -> None:
    data = long.loc[long["cell_type"].isin(CD4_FOCUS)].copy()
    order = [f"{region}\n{group}" for region in REGION_ORDER for group in GROUP_ORDER]
    rows = []
    for x_idx, region_group in enumerate(order):
        region, group = region_group.split("\n")
        for y_idx, cell_type in enumerate(CD4_FOCUS):
            subset = data.loc[
                (data["brain_region"].astype(str) == region)
                & (data["group"].astype(str) == group)
                & (data["cell_type"].astype(str) == cell_type),
                "proportion",
            ]
            rows.append(
                {
                    "x": x_idx,
                    "y": y_idx,
                    "cell_type": cell_type,
                    "region": region,
                    "group": group,
                    "mean": float(subset.mean()),
                    "detected_fraction": float((subset > 1e-4).mean()),
                }
            )
    plot_df = pd.DataFrame(rows)
    scatter = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        c=np.log10(plot_df["mean"] + 1e-5),
        s=12 + 180 * plot_df["detected_fraction"],
        cmap="mako_r",
        vmin=-5,
        vmax=0,
        edgecolor="#333333",
        linewidth=0.25,
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.030, pad=0.012)
    cbar.set_label("log10(mean proportion + 1e-5)", fontsize=5.0)
    cbar.ax.tick_params(labelsize=4.8, length=2)
    for size_value, label, xpos in [(0.0, "0%", 5.55), (0.5, "50%", 5.95), (1.0, "100%", 6.40)]:
        ax.scatter(xpos, -0.55, s=12 + 180 * size_value, color="#BDBDBD", edgecolor="#333333", linewidth=0.25)
        ax.text(xpos, -0.98, label, ha="center", va="top", fontsize=4.4)
    ax.text(6.0, -1.42, "nonzero samples", ha="center", va="top", fontsize=4.5, color="#333333")
    lookup = tests.set_index(["brain_region", "cell_type"])
    for x_idx, region_group in enumerate(order):
        region, group = region_group.split("\n")
        if group != "SCZ":
            continue
        for y_idx, cell_type in enumerate(CD4_FOCUS):
            row = lookup.loc[(region, cell_type)]
            if row["fdr_region"] < 0.10:
                ax.text(x_idx + 0.33, y_idx, "*", ha="center", va="center", fontsize=6.0)
            elif row["p_value"] < 0.05:
                ax.text(x_idx + 0.33, y_idx, "+", ha="center", va="center", fontsize=5.6)
    ax.set_title("CD4-conditioned MuSiC estimates", fontsize=7.0, pad=3)
    ax.set_xlim(-0.6, 6.75)
    ax.set_ylim(len(CD4_FOCUS) - 0.4, -1.65)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(range(len(CD4_FOCUS)))
    ax.set_yticklabels([labels.get(cell, cell) for cell in CD4_FOCUS])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=5.0, width=0.5, length=2)
    ax.tick_params(axis="y", labelsize=5.2, width=0.5, length=2)


def draw_cd4_effect_heatmap(ax, tests: pd.DataFrame, labels: dict[str, str]) -> None:
    matrix = tests.pivot(index="cell_type", columns="brain_region", values="mean_difference_scz_minus_control")
    matrix = matrix.reindex(index=CD4_FOCUS, columns=REGION_ORDER)
    matrix.index = [labels.get(item, item) for item in matrix.index]
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "SCZ-control conditioned difference", "shrink": 0.78},
        ax=ax,
    )
    lookup = tests.set_index(["cell_type", "brain_region"])
    for y, cell in enumerate(CD4_FOCUS):
        for x, region in enumerate(REGION_ORDER):
            row = lookup.loc[(cell, region)]
            if row["fdr_region"] < 0.10:
                ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", fontsize=7.0, color="black")
            elif row["p_value"] < 0.05:
                ax.text(x + 0.5, y + 0.5, "+", ha="center", va="center", fontsize=6.0, color="black")
    ax.set_title("SCZ-control CD4-conditioned differences", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.3, width=0.5, length=2)


def main() -> None:
    set_publication_style()
    ref, cell_order, labels = load_reference()
    full_long = load_music_long("npj_figure4_csf_music_proportions_wide.tsv", cell_order)
    full_summary, full_tests = summarize_and_test(full_long)
    cd4_long = load_music_long("npj_figure4_csf_music_cd4_conditioned_proportions_wide.tsv", CD4_FOCUS)
    cd4_summary, cd4_tests = summarize_and_test(cd4_long)

    full_long.to_csv(TABLE_DIR / "npj_figure4_csf_music_proportions_long.tsv", sep="\t", index=False)
    full_summary.to_csv(TABLE_DIR / "npj_figure4_csf_music_group_summary.tsv", sep="\t", index=False)
    full_tests.to_csv(TABLE_DIR / "npj_figure4_csf_music_group_tests.tsv", sep="\t", index=False)
    cd4_long.to_csv(TABLE_DIR / "npj_figure4_csf_music_cd4_conditioned_proportions_long.tsv", sep="\t", index=False)
    cd4_summary.to_csv(TABLE_DIR / "npj_figure4_csf_music_cd4_conditioned_group_summary.tsv", sep="\t", index=False)
    cd4_tests.to_csv(TABLE_DIR / "npj_figure4_csf_music_cd4_conditioned_group_tests.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=mm_to_inches(190, 330), constrained_layout=False)
    fig.text(
        0.5,
        0.985,
        "CELLxGENE CSF single-cell reference deconvolution of striatal bulk RNA-seq",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax_a = fig.add_axes([0.08, 0.730, 0.30, 0.220])
    draw_reference_panel(ax_a, ref, cell_order, labels)
    ax_b = fig.add_axes([0.48, 0.730, 0.34, 0.220])
    draw_broad_stacked(ax_b, full_long)
    ax_c = fig.add_axes([0.24, 0.505, 0.52, 0.155])
    draw_full_effect_heatmap(ax_c, full_tests, labels)
    ax_d = fig.add_axes([0.08, 0.285, 0.84, 0.145])
    draw_cd4_dotplot(ax_d, cd4_long, cd4_tests, labels)
    ax_e = fig.add_axes([0.28, 0.095, 0.44, 0.125])
    draw_cd4_effect_heatmap(ax_e, cd4_tests, labels)

    fig.text(
        0.5,
        0.055,
        "+ nominal P<0.05; * region-level FDR<0.10 by covariate-adjusted OLS on arcsine-transformed proportions",
        ha="center",
        fontsize=5.4,
    )
    fig.text(
        0.5,
        0.035,
        "Reference: CELLxGENE Discover CSF leptomeningeal disease single-cell dataset; malignant/tumor/non-immune cells were excluded before MuSiC.",
        ha="center",
        fontsize=5.1,
        color="#333333",
    )

    panel_label(fig, "A", 0.035, 0.955)
    panel_label(fig, "B", 0.420, 0.955)
    panel_label(fig, "C", 0.035, 0.665)
    panel_label(fig, "D", 0.035, 0.440)
    panel_label(fig, "E", 0.035, 0.230)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"npj_figure4_cellxgene_csf_music.{ext}", dpi=600, bbox_inches="tight")

    NPJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_FIG_DIR / "Figure-04.png", dpi=600, bbox_inches="tight")
    fig.savefig(NPJ_FIG_DIR / "Figure-04.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote NPJ Figure 4 CELLxGENE CSF MuSiC figure.")


if __name__ == "__main__":
    main()
