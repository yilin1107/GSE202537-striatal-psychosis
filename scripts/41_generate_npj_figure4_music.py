from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import sys

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
GROUP_COLORS = {"Control": "#4F86C6", "SCZ": "#D94873"}

CELL_ORDER = [
    "TAM_2",
    "TAM_1",
    "prol._TAM",
    "Monocytes",
    "DC",
    "B_cells",
    "NK_cells",
    "NK_like_T",
    "CD8_cytotoxic",
    "CD4_memory_resting",
    "CD4_memory_activated",
    "Treg",
]

DISPLAY_LABELS = {
    "TAM_2": "TAM 2",
    "TAM_1": "TAM 1",
    "prol._TAM": "Prolif. TAM",
    "Monocytes": "Monocytes",
    "DC": "DC",
    "B_cells": "B cells",
    "NK_cells": "NK cells",
    "NK_like_T": "NK-like T",
    "CD8_cytotoxic": "CD8 cytotoxic",
    "CD4_memory_resting": "CD4 mem resting",
    "CD4_memory_activated": "CD4 mem activated",
    "Treg": "Treg",
}

CD4_FOCUS = ["CD4_memory_resting", "CD4_memory_activated", "Treg", "CD8_cytotoxic", "NK_like_T"]


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


def load_music_long(table_name: str, cell_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return long, merged


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
        if np.isfinite(sd) and sd > 0:
            design[col] = (design[col] - design[col].mean()) / sd
        else:
            design[col] = 0.0
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
    psychosis_idx = 1
    if se[psychosis_idx] == 0 or not np.isfinite(se[psychosis_idx]):
        return float(coef[psychosis_idx]), np.nan
    t_value = coef[psychosis_idx] / se[psychosis_idx]
    p_value = 2 * stats.t.sf(abs(t_value), df_resid)
    return float(coef[psychosis_idx]), float(p_value)


def summarize_and_test(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        long.groupby(["brain_region", "group", "cell_type"], observed=False)["proportion"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        for cell_type in long["cell_type"].cat.categories:
            subset = long.loc[(long["brain_region"] == region) & (long["cell_type"] == cell_type)]
            control = subset.loc[subset["group"] == "Control", "proportion"].to_numpy(dtype=float)
            scz = subset.loc[subset["group"] == "SCZ", "proportion"].to_numpy(dtype=float)
            if len(control) == 0 or len(scz) == 0:
                statistic = adjusted_beta = p_value = np.nan
            else:
                statistic, p_value = stats.mannwhitneyu(scz, control, alternative="two-sided")
                adjusted_beta, adjusted_p_value = adjusted_ols_effect(subset)
            rows.append(
                {
                    "brain_region": region,
                    "cell_type": cell_type,
                    "control_mean": np.nanmean(control) if len(control) else np.nan,
                    "scz_mean": np.nanmean(scz) if len(scz) else np.nan,
                    "mean_difference_scz_minus_control": (
                        np.nanmean(scz) - np.nanmean(control) if len(control) and len(scz) else np.nan
                    ),
                    "mannwhitney_u": statistic,
                    "mannwhitney_p_value": p_value,
                    "adjusted_beta_arcsine": adjusted_beta,
                    "p_value": adjusted_p_value if len(control) and len(scz) else np.nan,
                }
            )
    tests = pd.DataFrame(rows)
    tests["fdr_region"] = tests.groupby("brain_region")["p_value"].transform(fdr_bh)
    tests["fdr_global"] = fdr_bh(tests["p_value"])
    return summary, tests


def cell_colors() -> dict[str, str]:
    colors = list(sns.color_palette("tab20", 20))
    return {cell_type: mpl.colors.to_hex(colors[idx]) for idx, cell_type in enumerate(CELL_ORDER)}


def panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def draw_reference_panel(ax, colors: dict[str, str]) -> None:
    ref = pd.read_csv(TABLE_DIR / "npj_figure4_brain_immune_reference_celltype_summary.tsv", sep="\t")
    ref = ref.loc[ref["music_cell_type"].isin(CELL_ORDER)].set_index("music_cell_type").reindex(CELL_ORDER).reset_index()
    ax.barh(
        np.arange(len(ref)),
        ref["n_cells"],
        color=[colors[cell] for cell in ref["music_cell_type"]],
        edgecolor="none",
    )
    ax.set_yticks(np.arange(len(ref)))
    ax.set_yticklabels([DISPLAY_LABELS[item] for item in ref["music_cell_type"]])
    ax.invert_yaxis()
    ax.set_xlabel("Reference cells")
    ax.set_title("Brain Immune Atlas reference used for MuSiC", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.3, width=0.5, length=2)


def draw_stacked_panel(ax, summary: pd.DataFrame, colors: dict[str, str]) -> None:
    pivot = summary.pivot_table(
        index=["brain_region", "group"],
        columns="cell_type",
        values="mean",
        fill_value=0,
        observed=False,
    ).reindex(pd.MultiIndex.from_product([REGION_ORDER, GROUP_ORDER], names=["brain_region", "group"]))
    positions = np.array([0, 0.82, 2.15, 2.97, 4.30, 5.12])
    bottoms = np.zeros(len(pivot), dtype=float)
    for cell_type in CELL_ORDER:
        if cell_type not in pivot.columns:
            continue
        values = pivot[cell_type].to_numpy(dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=colors[cell_type],
            edgecolor="white",
            linewidth=0.18,
            label=DISPLAY_LABELS[cell_type],
        )
        bottoms += values
    ax.set_xticks(positions)
    ax.set_xticklabels(["Control", "SCZ"] * 3)
    for center, region in zip([0.41, 2.56, 4.71], REGION_ORDER):
        ax.text(center, 1.035, region, ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean MuSiC proportion")
    ax.set_title("Estimated brain-immune composition in bulk striatum", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.4, width=0.5, length=2)
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.03),
        loc="upper left",
        fontsize=4.8,
        ncol=1,
        handlelength=0.9,
        labelspacing=0.35,
    )


def draw_tcell_dotplot(ax, long: pd.DataFrame, tests: pd.DataFrame) -> None:
    data = long.loc[long["cell_type"].isin(CD4_FOCUS)].copy()
    order = [f"{region}\n{group}" for region in REGION_ORDER for group in GROUP_ORDER]
    records = []
    for x_idx, region_group in enumerate(order):
        region, group = region_group.split("\n")
        for y_idx, cell_type in enumerate(CD4_FOCUS):
            subset = data.loc[
                (data["brain_region"].astype(str) == region)
                & (data["group"].astype(str) == group)
                & (data["cell_type"].astype(str) == cell_type),
                "proportion",
            ]
            records.append(
                {
                    "x": x_idx,
                    "y": y_idx,
                    "region": region,
                    "group": group,
                    "cell_type": cell_type,
                    "mean": float(subset.mean()),
                    "detected_fraction": float((subset > 1e-4).mean()),
                }
            )
    plot_df = pd.DataFrame(records)
    color_value = np.log10(plot_df["mean"] + 1e-5)
    sizes = 12 + 180 * plot_df["detected_fraction"]
    scatter = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        c=color_value,
        s=sizes,
        cmap="mako_r",
        vmin=-5,
        vmax=0,
        edgecolor="#333333",
        linewidth=0.25,
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.030, pad=0.012)
    cbar.set_label("log10(mean proportion + 1e-5)", fontsize=5.1)
    cbar.ax.tick_params(labelsize=4.8, length=2)

    for size_value, label, xpos in [(0.0, "0%", 5.55), (0.5, "50%", 5.95), (1.0, "100%", 6.40)]:
        ax.scatter(xpos, -0.55, s=12 + 180 * size_value, color="#BDBDBD", edgecolor="#333333", linewidth=0.25)
        ax.text(xpos, -0.98, label, ha="center", va="top", fontsize=4.4)
    ax.text(6.0, -1.42, "nonzero samples", ha="center", va="top", fontsize=4.5, color="#333333")

    test_lookup = tests.set_index(["brain_region", "cell_type"])
    for x_idx, region_group in enumerate(order):
        region, group = region_group.split("\n")
        if group != "SCZ":
            continue
        for y_idx, cell_type in enumerate(CD4_FOCUS):
            row = test_lookup.loc[(region, cell_type)]
            if row["fdr_region"] < 0.10:
                ax.text(x_idx + 0.33, y_idx, "*", ha="center", va="center", fontsize=6.0)
            elif row["p_value"] < 0.05:
                ax.text(x_idx + 0.33, y_idx, "+", ha="center", va="center", fontsize=5.6)

    ax.set_title("T/NK-conditioned MuSiC estimates", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-0.6, 6.75)
    ax.set_ylim(len(CD4_FOCUS) - 0.4, -1.65)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(range(len(CD4_FOCUS)))
    ax.set_yticklabels([DISPLAY_LABELS[cell] for cell in CD4_FOCUS])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=5.0, width=0.5, length=2, rotation=0)
    ax.tick_params(axis="y", labelsize=5.2, width=0.5, length=2)


def draw_effect_heatmap(ax, tests: pd.DataFrame) -> None:
    matrix = tests.pivot(index="cell_type", columns="brain_region", values="mean_difference_scz_minus_control")
    matrix = matrix.reindex(index=CELL_ORDER, columns=REGION_ORDER)
    matrix.index = [DISPLAY_LABELS[item] for item in matrix.index]
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "SCZ-control mean difference", "shrink": 0.74},
        ax=ax,
    )
    lookup = tests.set_index(["cell_type", "brain_region"])
    for y, cell in enumerate(CELL_ORDER):
        for x, region in enumerate(REGION_ORDER):
            row = lookup.loc[(cell, region)]
            if row["fdr_region"] < 0.10:
                ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", fontsize=7.0, color="black")
            elif row["p_value"] < 0.05:
                ax.text(x + 0.5, y + 0.5, "+", ha="center", va="center", fontsize=6.0, color="black")
    ax.set_title("SCZ-control MuSiC proportion differences", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.2, width=0.5, length=2)


def draw_tcell_effect_heatmap(ax, tests: pd.DataFrame) -> None:
    matrix = tests.pivot(index="cell_type", columns="brain_region", values="mean_difference_scz_minus_control")
    matrix = matrix.reindex(index=CD4_FOCUS, columns=REGION_ORDER)
    matrix.index = [DISPLAY_LABELS[item] for item in matrix.index]
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
    ax.set_title("SCZ-control T/NK-conditioned differences", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.5, width=0.5, length=2)


def main() -> None:
    set_publication_style()
    long, _ = load_music_long("npj_figure4_brain_immune_music_proportions_wide.tsv", CELL_ORDER)
    summary, tests = summarize_and_test(long)
    tcell_long, _ = load_music_long(
        "npj_figure4_brain_immune_music_tcell_conditioned_proportions_wide.tsv",
        CD4_FOCUS,
    )
    tcell_summary, tcell_tests = summarize_and_test(tcell_long)
    long.to_csv(TABLE_DIR / "npj_figure4_brain_immune_music_proportions_long.tsv", sep="\t", index=False)
    summary.to_csv(TABLE_DIR / "npj_figure4_brain_immune_music_group_summary.tsv", sep="\t", index=False)
    tests.to_csv(TABLE_DIR / "npj_figure4_brain_immune_music_group_tests.tsv", sep="\t", index=False)
    tcell_long.to_csv(
        TABLE_DIR / "npj_figure4_brain_immune_music_tcell_conditioned_proportions_long.tsv",
        sep="\t",
        index=False,
    )
    tcell_summary.to_csv(
        TABLE_DIR / "npj_figure4_brain_immune_music_tcell_conditioned_group_summary.tsv",
        sep="\t",
        index=False,
    )
    tcell_tests.to_csv(
        TABLE_DIR / "npj_figure4_brain_immune_music_tcell_conditioned_group_tests.tsv",
        sep="\t",
        index=False,
    )

    colors = cell_colors()
    fig = plt.figure(figsize=mm_to_inches(190, 335), constrained_layout=False)
    fig.text(
        0.5,
        0.985,
        "Brain immune single-cell reference deconvolution of striatal bulk RNA-seq",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax_a = fig.add_axes([0.08, 0.760, 0.28, 0.180])
    draw_reference_panel(ax_a, colors)
    ax_b = fig.add_axes([0.44, 0.760, 0.40, 0.180])
    draw_stacked_panel(ax_b, summary, colors)
    ax_c = fig.add_axes([0.25, 0.540, 0.50, 0.155])
    draw_effect_heatmap(ax_c, tests)
    ax_d = fig.add_axes([0.08, 0.295, 0.84, 0.160])
    draw_tcell_dotplot(ax_d, tcell_long, tcell_tests)
    ax_e = fig.add_axes([0.27, 0.095, 0.46, 0.140])
    draw_tcell_effect_heatmap(ax_e, tcell_tests)

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
        "Reference: Brain Immune Atlas human newly diagnosed GBM full aggregate; T-cell labels were marker-refined before MuSiC.",
        ha="center",
        fontsize=5.1,
        color="#333333",
    )

    panel_label(fig, "A", 0.035, 0.950)
    panel_label(fig, "B", 0.390, 0.950)
    panel_label(fig, "C", 0.035, 0.700)
    panel_label(fig, "D", 0.035, 0.465)
    panel_label(fig, "E", 0.035, 0.245)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"npj_figure4_brain_immune_music.{ext}", dpi=600, bbox_inches="tight")

    NPJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_FIG_DIR / "Figure-04.png", dpi=600, bbox_inches="tight")
    fig.savefig(NPJ_FIG_DIR / "Figure-04.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote NPJ Figure 4 Brain Immune Atlas MuSiC figure.")


if __name__ == "__main__":
    main()
