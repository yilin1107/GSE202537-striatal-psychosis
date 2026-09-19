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
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}
NPJ_FIG_DIR = ROOT / "NPJ" / "Figures"

WIDE_PATH = TABLE_DIR / "npj_figure2_iobr_cibersort_wide.tsv"
META_PATH = ROOT / "data" / "processed" / "gse202537_sample_metadata.csv"

REGION_ORDER = ["NAc", "Caudate", "Putamen"]
GROUP_ORDER = ["Control", "SCZ"]
GROUP_COLORS = {"Control": "#4F86C6", "SCZ": "#D94873"}

METRIC_COLUMNS = {"P-value_CIBERSORT", "Correlation_CIBERSORT", "RMSE_CIBERSORT"}


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = p[order] * n / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted_sorted, 0, 1)
    return pd.Series(out, index=p_values.index)


def clean_cell_type(column: str) -> str:
    name = column.replace("_CIBERSORT", "").replace("_", " ")
    replacements = {
        "B cells naive": "B naive",
        "B cells memory": "B memory",
        "Plasma cells": "Plasma",
        "T cells CD8": "CD8 T",
        "T cells CD4 naive": "CD4 naive T",
        "T cells CD4 memory resting": "CD4 memory resting T",
        "T cells CD4 memory activated": "CD4 memory activated T",
        "T cells follicular helper": "T follicular helper",
        "T cells regulatory (Tregs)": "Tregs",
        "T cells gamma delta": "Gamma-delta T",
        "NK cells resting": "NK resting",
        "NK cells activated": "NK activated",
        "Dendritic cells resting": "DC resting",
        "Dendritic cells activated": "DC activated",
        "Mast cells resting": "Mast resting",
        "Mast cells activated": "Mast activated",
    }
    return replacements.get(name, name)


def load_iobr_long() -> tuple[pd.DataFrame, list[str]]:
    wide = pd.read_csv(WIDE_PATH, sep="\t")
    metadata = pd.read_csv(META_PATH)
    cell_columns = [column for column in wide.columns if column.endswith("_CIBERSORT") and column not in METRIC_COLUMNS]
    cell_order = [clean_cell_type(column) for column in cell_columns]
    rename = {column: clean_cell_type(column) for column in cell_columns}

    merged = wide.rename(columns={"ID": "sample_key", **rename}).merge(
        metadata[["sample_key", "group", "brain_region", "patients_id"]],
        on="sample_key",
        how="left",
    )
    merged = merged.loc[merged["group"].isin(GROUP_ORDER) & merged["brain_region"].isin(REGION_ORDER)].copy()

    long = merged.melt(
        id_vars=["sample_key", "patients_id", "group", "brain_region"],
        value_vars=cell_order,
        var_name="cell_type",
        value_name="fraction",
    )
    long["cell_type"] = pd.Categorical(long["cell_type"], categories=cell_order, ordered=True)
    long["group"] = pd.Categorical(long["group"], categories=GROUP_ORDER, ordered=True)
    long["brain_region"] = pd.Categorical(long["brain_region"], categories=REGION_ORDER, ordered=True)
    return long, cell_order


def summarize_and_test(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        long.groupby(["brain_region", "group", "cell_type"], observed=False)["fraction"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        for cell_type in long["cell_type"].cat.categories:
            subset = long.loc[(long["brain_region"] == region) & (long["cell_type"] == cell_type)]
            control = subset.loc[subset["group"] == "Control", "fraction"].to_numpy(dtype=float)
            scz = subset.loc[subset["group"] == "SCZ", "fraction"].to_numpy(dtype=float)
            if len(control) == 0 or len(scz) == 0:
                p_value = np.nan
                statistic = np.nan
            else:
                statistic, p_value = stats.mannwhitneyu(scz, control, alternative="two-sided")
            rows.append(
                {
                    "brain_region": region,
                    "cell_type": str(cell_type),
                    "control_mean": float(np.nanmean(control)) if len(control) else np.nan,
                    "scz_mean": float(np.nanmean(scz)) if len(scz) else np.nan,
                    "mean_difference_scz_minus_control": (
                        float(np.nanmean(scz) - np.nanmean(control)) if len(control) and len(scz) else np.nan
                    ),
                    "mannwhitney_u": statistic,
                    "p_value": p_value,
                }
            )
    tests = pd.DataFrame(rows)
    tests["fdr_region"] = tests.groupby("brain_region")["p_value"].transform(fdr_bh)
    tests["fdr_global"] = fdr_bh(tests["p_value"])
    return summary, tests


def panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def cell_type_colors(cell_order: list[str]) -> dict[str, str]:
    colors = list(sns.color_palette("tab20", 20)) + list(sns.color_palette("Set2", 8))
    return {cell_type: mpl.colors.to_hex(colors[idx]) for idx, cell_type in enumerate(cell_order)}


def draw_stacked_composition(ax, summary: pd.DataFrame, cell_order: list[str], colors: dict[str, str]) -> None:
    plot_data = summary.pivot_table(
        index=["brain_region", "group"],
        columns="cell_type",
        values="mean",
        fill_value=0,
        observed=False,
    ).reindex(pd.MultiIndex.from_product([REGION_ORDER, GROUP_ORDER], names=["brain_region", "group"]))

    positions = np.array([0, 0.82, 2.15, 2.97, 4.30, 5.12])
    bottoms = np.zeros(len(plot_data), dtype=float)
    for cell_type in cell_order:
        values = plot_data[cell_type].to_numpy(dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=colors[cell_type],
            edgecolor="white",
            linewidth=0.18,
            label=cell_type,
        )
        bottoms += values

    ax.set_xticks(positions)
    ax.set_xticklabels(["Control", "SCZ"] * 3, rotation=0)
    for center, region in zip([0.41, 2.56, 4.71], REGION_ORDER):
        ax.text(center, 1.035, region, ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean CIBERSORT fraction")
    ax.set_title("IOBR-CIBERSORT LM22 composition by diagnosis and region", fontsize=7.4, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.8, width=0.6, length=2.4)
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.03),
        loc="upper left",
        ncol=2,
        fontsize=4.8,
        handlelength=0.8,
        columnspacing=0.8,
        labelspacing=0.35,
    )


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def draw_boxplot_panel(
    ax,
    long: pd.DataFrame,
    tests: pd.DataFrame,
    region: str,
    cell_order: list[str],
    show_ylabels: bool,
    x_max: float,
) -> None:
    data = long.loc[long["brain_region"] == region].copy()
    sns.boxplot(
        data=data,
        x="fraction",
        y="cell_type",
        hue="group",
        order=cell_order,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        dodge=True,
        width=0.62,
        showfliers=False,
        linewidth=0.45,
        boxprops={"edgecolor": "#333333", "linewidth": 0.45},
        whiskerprops={"color": "#333333", "linewidth": 0.45},
        capprops={"color": "#333333", "linewidth": 0.45},
        medianprops={"color": "#111111", "linewidth": 0.55},
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="fraction",
        y="cell_type",
        hue="group",
        order=cell_order,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        dodge=True,
        jitter=0.18,
        size=1.15,
        alpha=0.33,
        linewidth=0,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    region_tests = tests.loc[tests["brain_region"] == region].set_index("cell_type")
    for yi, cell_type in enumerate(cell_order):
        if cell_type not in region_tests.index:
            continue
        row = region_tests.loc[cell_type]
        has_stat_label = row["p_value"] < 0.05 or row["fdr_region"] < 0.10
        if row["fdr_region"] < 0.10:
            marker = "**" if row["fdr_region"] < 0.05 else "*"
            stat_text = f"{marker} P={format_p_value(row['p_value'])}, q={format_p_value(row['fdr_region'])}"
            ax.text(x_max * 1.035, yi, stat_text, ha="left", va="center", fontsize=4.7, color="#111111")
        elif row["p_value"] < 0.05:
            stat_text = f"\u2020 P={format_p_value(row['p_value'])}, q={format_p_value(row['fdr_region'])}"
            ax.text(x_max * 1.035, yi, stat_text, ha="left", va="center", fontsize=4.7, color="#333333")
        elif has_stat_label:
            ax.text(
                x_max * 1.035,
                yi,
                f"P={format_p_value(row['p_value'])}, q={format_p_value(row['fdr_region'])}",
                ha="left",
                va="center",
                fontsize=4.7,
                color="#333333",
            )

    ax.set_title(region, fontsize=7.0, pad=2.5)
    ax.set_xlim(0, x_max * 1.42)
    ax.set_xlabel("CIBERSORT fraction")
    ax.set_ylabel("")
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=5.4, width=0.5, length=2)
    ax.tick_params(axis="y", labelsize=4.9, width=0.5, length=2, pad=1)


def main() -> None:
    set_publication_style()
    long, cell_order = load_iobr_long()
    summary, tests = summarize_and_test(long)

    long.to_csv(TABLE_DIR / "npj_figure2_iobr_cibersort_long.tsv", sep="\t", index=False)
    summary.to_csv(TABLE_DIR / "npj_figure2_iobr_cibersort_group_summary.tsv", sep="\t", index=False)
    tests.to_csv(TABLE_DIR / "npj_figure2_iobr_cibersort_group_tests.tsv", sep="\t", index=False)

    colors = cell_type_colors(cell_order)
    x_max = min(max(float(long["fraction"].quantile(0.995)) * 1.18, 0.20), 0.55)

    fig = plt.figure(figsize=mm_to_inches(190, 280), constrained_layout=False)
    fig.text(
        0.5,
        0.985,
        "IOBR-CIBERSORT inference of immune-cell composition in striatal psychosis",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax_a = fig.add_axes([0.07, 0.725, 0.58, 0.195])
    draw_stacked_composition(ax_a, summary, cell_order, colors)

    fig.text(
        0.50,
        0.655,
        "SCZ-control differences across 22 LM22 cell types",
        ha="center",
        va="center",
        fontsize=7.4,
    )
    fig.text(
        0.50,
        0.636,
        "\u2020 nominal P<0.05; * region-level FDR<0.10; ** region-level FDR<0.05 by Mann-Whitney U test; labels show P and region-level q",
        ha="center",
        va="center",
        fontsize=5.5,
        color="#333333",
    )
    b_axes = [
        [0.155, 0.105, 0.245, 0.500],
        [0.435, 0.105, 0.245, 0.500],
        [0.715, 0.105, 0.245, 0.500],
    ]
    for idx, (region, rect) in enumerate(zip(REGION_ORDER, b_axes)):
        ax_b = fig.add_axes(rect)
        draw_boxplot_panel(ax_b, long, tests, region, cell_order, show_ylabels=idx == 0, x_max=x_max)

    handles = [
        mpl.patches.Patch(facecolor=GROUP_COLORS[group], edgecolor="none", label=group)
        for group in GROUP_ORDER
    ]
    fig.legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(0.95, 0.660), fontsize=6.0)

    panel_label(fig, "A", 0.035, 0.940)
    panel_label(fig, "B", 0.035, 0.665)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"npj_figure2_iobr_cibersort.{ext}", dpi=600, bbox_inches="tight")

    NPJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_FIG_DIR / "Figure-02.png", dpi=600, bbox_inches="tight")
    fig.savefig(NPJ_FIG_DIR / "Figure-02.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote NPJ Figure 2 IOBR-CIBERSORT figure.")


if __name__ == "__main__":
    main()
