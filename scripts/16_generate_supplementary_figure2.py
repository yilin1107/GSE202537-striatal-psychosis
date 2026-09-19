from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import REGION_ORDER, mm_to_inches, set_publication_style  # noqa: E402


TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}

CELL_ORDER = ["MSN", "Interneuron", "Astrocyte", "Oligodendrocyte", "OPC", "Microglia-immune", "Vascular"]
CELL_LABELS = {
    "MSN": "MSN",
    "Interneuron": "Interneuron",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodend.",
    "OPC": "OPC",
    "Microglia-immune": "Microglia",
    "Vascular": "Vascular",
}
CELL_COLORS = {
    "MSN": "#4E79A7",
    "Interneuron": "#76B7B2",
    "Astrocyte": "#B07AA1",
    "Oligodendrocyte": "#59A14F",
    "OPC": "#8CD17D",
    "Microglia-immune": "#E15759",
    "Vascular": "#F28E2B",
}
EFFECT_CMAP = sns.diverging_palette(240, 10, as_cmap=True)


def figure_panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def metric_value(metrics: pd.DataFrame, metric: str) -> float:
    values = metrics.loc[metrics["metric"] == metric, "value"]
    return float(values.iloc[0]) if not values.empty else np.nan


def draw_workflow_box(ax, x: float, color: str, title: str, text: str) -> None:
    rect = patches.FancyBboxPatch(
        (x, 0.24),
        0.21,
        0.50,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(rect)
    ax.text(x + 0.105, 0.57, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=color)
    ax.text(x + 0.105, 0.38, text, ha="center", va="center", fontsize=6.0, color="#222222")


def draw_workflow(ax, metrics: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_points = int(metric_value(metrics, "concordance_points"))
    r_value = metric_value(metrics, "spearman_r")
    boxes = [
        (0.02, "#333333", "HPA/Siletti snRNA", "34 human brain\ncluster types"),
        (0.275, "#2C7FB8", "External markers", "Top specificity genes\nfor 7 cell classes"),
        (0.53, "#59A14F", "GSE202537 scores", "Recomputed\nmarker scores"),
        (0.765, "#54278F", "Concordance", f"{n_points} matched effects\nSpearman r={r_value:.2f}"),
    ]
    for x, color, title, text in boxes:
        draw_workflow_box(ax, x, color, title, text)
    for x in [0.235, 0.49, 0.745]:
        ax.annotate("", xy=(x + 0.035, 0.49), xytext=(x, 0.49), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"})
    ax.text(
        0.5,
        0.06,
        "External markers are derived from the Human Protein Atlas single-nuclei brain resource based on Siletti et al.",
        ha="center",
        va="center",
        fontsize=5.7,
        color="#444444",
    )


def draw_marker_overlap(ax, concordance: pd.DataFrame) -> None:
    plot_df = concordance.set_index("external_cell_type").reindex(CELL_ORDER).reset_index()
    y = np.arange(plot_df.shape[0])
    colors = [CELL_COLORS[cell] for cell in plot_df["external_cell_type"]]
    ax.barh(y, plot_df["jaccard"], color=colors, height=0.58, alpha=0.85)
    for idx, row in plot_df.iterrows():
        ax.text(
            row["jaccard"] + 0.006,
            idx,
            f"{int(row['overlap_count'])}/{int(row['internal_marker_count'])}",
            va="center",
            ha="left",
            fontsize=5.8,
            color="#222222",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([CELL_LABELS[cell] for cell in plot_df["external_cell_type"]])
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.16, plot_df["jaccard"].max() + 0.04))
    ax.set_xlabel("Jaccard overlap")
    ax.set_title("Internal versus external marker overlap")
    ax.text(
        0.99,
        0.02,
        "label = overlap / internal markers",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=5.6,
        color="#555555",
    )
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def draw_external_beta_heatmap(ax, cbar_ax, external_results: pd.DataFrame) -> None:
    beta = (
        external_results.pivot(index="external_cell_type", columns="brain_region", values="external_beta_psychosis")
        .reindex(index=CELL_ORDER, columns=REGION_ORDER)
    )
    fdr = (
        external_results.pivot(index="external_cell_type", columns="brain_region", values="external_fdr")
        .reindex(index=CELL_ORDER, columns=REGION_ORDER)
    )
    sns.heatmap(
        beta,
        ax=ax,
        cmap=EFFECT_CMAP,
        center=0,
        vmin=-0.70,
        vmax=0.70,
        linewidths=0,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
    )
    ax.set_title("External-marker psychosis effects")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(REGION_ORDER, rotation=0)
    ax.set_yticklabels([CELL_LABELS[cell] for cell in CELL_ORDER], rotation=0)
    ax.tick_params(axis="both", length=0, labelsize=5.8)
    for y_idx, cell_type in enumerate(CELL_ORDER):
        for x_idx, region in enumerate(REGION_ORDER):
            value = fdr.loc[cell_type, region]
            if np.isfinite(value) and value < 0.10:
                ax.text(x_idx + 0.5, y_idx + 0.5, "*", ha="center", va="center", fontsize=8, color="black")
    cbar_ax.set_title("Psychosis beta", fontsize=6.0, pad=2)
    cbar_ax.tick_params(axis="x", labelsize=5.6, length=2, pad=1)


def draw_beta_concordance(ax, concordance: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plot_df = concordance.dropna(subset=["internal_beta_psychosis", "external_beta_psychosis"]).copy()
    for cell_type in CELL_ORDER:
        data = plot_df.loc[plot_df["external_cell_type"] == cell_type]
        ax.scatter(
            data["internal_beta_psychosis"],
            data["external_beta_psychosis"],
            s=34,
            color=CELL_COLORS[cell_type],
            edgecolor="white",
            linewidth=0.45,
            label=CELL_LABELS[cell_type],
            alpha=0.9,
        )
    lim = max(
        0.75,
        float(np.nanmax(np.abs(plot_df[["internal_beta_psychosis", "external_beta_psychosis"]].to_numpy()))) + 0.08,
    )
    ax.plot([-lim, lim], [-lim, lim], color="#999999", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#D0D0D0", linewidth=0.6)
    ax.axvline(0, color="#D0D0D0", linewidth=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Internal marker beta")
    ax.set_ylabel("External marker beta")
    ax.set_title("Internal-external effect-size concordance")
    r_value = metric_value(metrics, "spearman_r")
    p_value = metric_value(metrics, "spearman_p")
    ax.text(
        0.04,
        0.96,
        f"Spearman r={r_value:.2f}\np={p_value:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=5.8, handletextpad=0.2, borderaxespad=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def main() -> None:
    set_publication_style()
    marker_concordance = pd.read_csv(TABLE_DIR / "supplementary_figure2_marker_concordance.tsv", sep="\t")
    external_results = pd.read_csv(TABLE_DIR / "supplementary_figure2_external_cell_type_score_results.tsv", sep="\t")
    beta_concordance = pd.read_csv(TABLE_DIR / "supplementary_figure2_beta_concordance.tsv", sep="\t")
    metrics = pd.read_csv(TABLE_DIR / "supplementary_figure2_external_validation_metrics.tsv", sep="\t")

    fig = plt.figure(figsize=mm_to_inches(180, 185), constrained_layout=False)
    ax_workflow = fig.add_axes([0.07, 0.78, 0.86, 0.15])
    draw_workflow(ax_workflow, metrics)

    ax_overlap = fig.add_axes([0.14, 0.45, 0.31, 0.23])
    draw_marker_overlap(ax_overlap, marker_concordance)

    ax_heatmap = fig.add_axes([0.60, 0.45, 0.28, 0.23])
    cbar_ax = fig.add_axes([0.64, 0.405, 0.20, 0.012])
    draw_external_beta_heatmap(ax_heatmap, cbar_ax, external_results)

    ax_concordance = fig.add_axes([0.23, 0.11, 0.54, 0.23])
    draw_beta_concordance(ax_concordance, beta_concordance, metrics)

    figure_panel_label(fig, "A", 0.035, 0.955)
    figure_panel_label(fig, "B", 0.035, 0.705)
    figure_panel_label(fig, "C", 0.525, 0.705)
    figure_panel_label(fig, "D", 0.035, 0.365)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"supplementary_figure2_external_snbrain_validation.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Supplementary Figure 2 external snRNA-seq validation.")


if __name__ == "__main__":
    main()
