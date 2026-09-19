from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import (  # noqa: E402
    GROUP_COLORS,
    GROUP_ORDER,
    REGION_COLORS,
    REGION_ORDER,
    mm_to_inches,
    panel_label,
    set_publication_style,
)


PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}


DISPLAY_GROUPS = {
    "Control": "Control",
    "SCZ": "SCZ",
    "BD with psychosis": "BD psychosis",
}


def style_axis(ax) -> None:
    ax.tick_params(axis="both", width=0.6, length=2.5)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def draw_qc_box(ax, data: pd.DataFrame, y_column: str, ylabel: str, title: str) -> None:
    sns.boxplot(
        data=data,
        x="brain_region",
        y=y_column,
        hue="brain_region",
        order=REGION_ORDER,
        hue_order=REGION_ORDER,
        palette=REGION_COLORS,
        width=0.55,
        fliersize=0,
        linewidth=0.6,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="brain_region",
        y=y_column,
        order=REGION_ORDER,
        color="black",
        size=2.1,
        alpha=0.48,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)


def pc_label(variance: pd.DataFrame, pc: str) -> str:
    value = variance.loc[variance["pc"] == pc, "variance_explained"].iloc[0] * 100
    return f"{pc} ({value:.1f}%)"


def draw_variance(ax, variance: pd.DataFrame) -> None:
    display = variance.head(8).copy()
    display["percent"] = display["variance_explained"] * 100
    x = np.arange(len(display))
    ax.bar(x, display["percent"], color="#7F7F7F", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(display["pc"], rotation=45, ha="right")
    ax.set_ylim(0, display["percent"].max() * 1.12)
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA variance")
    style_axis(ax)


def draw_pca(ax, pca: pd.DataFrame, variance: pd.DataFrame, color_column: str, title: str) -> None:
    if color_column == "brain_region":
        order = REGION_ORDER
        palette = REGION_COLORS
        labels = {region: region for region in REGION_ORDER}
    else:
        order = GROUP_ORDER
        palette = GROUP_COLORS
        labels = DISPLAY_GROUPS
    for level in order:
        subset = pca.loc[pca[color_column] == level]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=17,
            color=palette[level],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.82,
            label=labels[level],
        )
    ax.axhline(0, color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.set_xlabel(pc_label(variance, "PC1"))
    ax.set_ylabel(pc_label(variance, "PC2"))
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", handlelength=1.0, labelspacing=0.25)
    style_axis(ax)


def draw_association_heatmap(ax, associations: pd.DataFrame) -> None:
    covariate_order = [
        "Brain region",
        "Diagnostic group",
        "Sex",
        "Sequence run",
        "Age",
        "PMI",
        "RIN",
        "Brain pH",
        "Count depth",
        "Corrected TOD",
    ]
    pc_order = [f"PC{i}" for i in range(1, 6)]
    heatmap_data = (
        associations.pivot(index="covariate", columns="pc", values="minus_log10_p")
        .reindex(index=covariate_order, columns=pc_order)
        .fillna(0)
        .clip(upper=12)
    )
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=12,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "-log10 P", "shrink": 0.78},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("PC-covariate association")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def main() -> None:
    set_publication_style()
    qc = pd.read_csv(PROCESSED_DIR / "gse202537_expression_qc.csv")
    pca = pd.read_csv(PROCESSED_DIR / "gse202537_pca_coordinates.csv")
    variance = pd.read_csv(TABLE_DIR / "supplementary_figure1_pca_variance.tsv", sep="\t")
    associations = pd.read_csv(TABLE_DIR / "supplementary_figure1_pc_covariate_associations.tsv", sep="\t")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=mm_to_inches(180, 120),
        constrained_layout=True,
    )
    axes = axes.ravel()

    draw_qc_box(axes[0], qc, "count_depth_million", "Count depth (million)", "Sequencing depth")
    draw_qc_box(axes[1], qc, "detected_genes", "Detected genes", "Detected genes")
    draw_variance(axes[2], variance)
    draw_pca(axes[3], pca, variance, "brain_region", "PCA by brain region")
    draw_pca(axes[4], pca, variance, "group", "PCA by group")
    draw_association_heatmap(axes[5], associations)

    for label, ax in zip(list("ABCDEF"), axes):
        panel_label(ax, label)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"supplementary_figure1_qc_pca.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Wrote Supplementary Figure 1 QC/PCA.")


if __name__ == "__main__":
    main()
