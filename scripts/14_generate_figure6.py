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

SIGNATURE_ORDER = [
    "Activin-SMAD remodeling",
    "Dopamine-adenosine dysregulation",
    "Glutamate-calcium stress",
    "Immune-NF-kB cytokine tone",
    "APP processing shift",
]
SIGNATURE_LABELS = {
    "Activin-SMAD remodeling": "Activin-SMAD",
    "Dopamine-adenosine dysregulation": "Dopamine-adenosine",
    "Glutamate-calcium stress": "Glutamate-calcium",
    "Immune-NF-kB cytokine tone": "Immune-NF-kB",
    "APP processing shift": "APP processing",
}
SIGNATURE_COLORS = {
    "Activin-SMAD remodeling": "#B07AA1",
    "Dopamine-adenosine dysregulation": "#4E79A7",
    "Glutamate-calcium stress": "#59A14F",
    "Immune-NF-kB cytokine tone": "#E15759",
    "APP processing shift": "#F28E2B",
}
CELL_TYPE_ORDER = [
    "D1 MSN",
    "D2 MSN",
    "Interneuron",
    "Dopaminergic axon",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia-immune",
    "Vascular",
]
CELL_TYPE_LABELS = {
    "D1 MSN": "D1 MSN",
    "D2 MSN": "D2 MSN",
    "Interneuron": "Interneuron",
    "Dopaminergic axon": "DA axon",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodend.",
    "OPC": "OPC",
    "Microglia-immune": "Microglia",
    "Vascular": "Vascular",
}
EFFECT_CMAP = sns.diverging_palette(240, 10, as_cmap=True)


def figure_panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def metric_value(metrics: pd.DataFrame, metric: str) -> int:
    values = metrics.loc[metrics["metric"] == metric, "value"]
    return int(values.iloc[0]) if not values.empty else 0


def draw_workflow_box(ax, x: float, color: str, title: str, text: str) -> None:
    rect = patches.FancyBboxPatch(
        (x, 0.23),
        0.20,
        0.52,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(rect)
    ax.text(x + 0.10, 0.57, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=color)
    ax.text(x + 0.10, 0.37, text, ha="center", va="center", fontsize=6.0, color="#222222")


def draw_workflow(ax, metrics: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_markers = metric_value(metrics, "cell_type_markers_available")
    boxes = [
        (0.02, "#333333", "Signature genes", "34 Figure 4\nmolecular genes"),
        (0.27, "#2C7FB8", "Marker panels", f"9 cell types\n{n_markers} expressed markers"),
        (0.52, "#59A14F", "Sample scores", "Rank-based\ncell-type marker scores"),
        (0.77, "#54278F", "Cellular basis", "Molecular axes\nassigned to compartments"),
    ]
    for x, color, title, text in boxes:
        draw_workflow_box(ax, x, color, title, text)
    for x in [0.225, 0.475, 0.725]:
        ax.annotate("", xy=(x + 0.035, 0.49), xytext=(x, 0.49), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"})
    ax.text(
        0.5,
        0.06,
        "Marker scores are interpreted as cell-type or cell-state attribution, not absolute cell fractions.",
        ha="center",
        va="center",
        fontsize=5.7,
        color="#444444",
    )


def draw_enrichment_heatmap(ax, cbar_ax, enrichment: pd.DataFrame) -> None:
    matrix = (
        enrichment.pivot(index="signature", columns="cell_type", values="neg_log10_fdr")
        .reindex(index=SIGNATURE_ORDER, columns=CELL_TYPE_ORDER)
        .fillna(0)
    )
    overlaps = (
        enrichment.pivot(index="signature", columns="cell_type", values="overlap_genes")
        .reindex(index=SIGNATURE_ORDER, columns=CELL_TYPE_ORDER)
        .fillna("")
    )
    cmap = sns.light_palette("#54278F", as_cmap=True)
    sns.heatmap(
        matrix.clip(upper=12),
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=12,
        linewidths=0,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "vertical"},
    )
    ax.set_title("Signature gene cell-type enrichment")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([CELL_TYPE_LABELS[item] for item in CELL_TYPE_ORDER], rotation=40, ha="right")
    ax.set_yticklabels([SIGNATURE_LABELS[item] for item in SIGNATURE_ORDER], rotation=0)
    ax.tick_params(axis="both", length=0, labelsize=5.8)
    for y_idx, signature in enumerate(SIGNATURE_ORDER):
        for x_idx, cell_type in enumerate(CELL_TYPE_ORDER):
            genes = overlaps.loc[signature, cell_type]
            if genes:
                label = genes.replace(";", "\n")
                ax.text(x_idx + 0.5, y_idx + 0.5, label, ha="center", va="center", fontsize=4.8, color="black")
    cbar_ax.set_title("-log10\nFDR", fontsize=5.4, pad=2)
    cbar_ax.tick_params(axis="y", labelsize=5.6, length=2, pad=1)


def draw_score_heatmap(ax, cbar_ax, score_results: pd.DataFrame) -> None:
    beta = (
        score_results.pivot(index="cell_type", columns="brain_region", values="beta_psychosis")
        .reindex(index=CELL_TYPE_ORDER, columns=REGION_ORDER)
    )
    fdr = (
        score_results.pivot(index="cell_type", columns="brain_region", values="fdr")
        .reindex(index=CELL_TYPE_ORDER, columns=REGION_ORDER)
    )
    sns.heatmap(
        beta,
        ax=ax,
        cmap=EFFECT_CMAP,
        center=0,
        vmin=-0.85,
        vmax=0.85,
        linewidths=0,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
    )
    ax.set_title("Psychosis effects on cell-type marker scores")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(REGION_ORDER, rotation=0)
    ax.set_yticklabels([CELL_TYPE_LABELS[item] for item in CELL_TYPE_ORDER], rotation=0)
    ax.tick_params(axis="both", length=0, labelsize=5.8)
    for y_idx, cell_type in enumerate(CELL_TYPE_ORDER):
        for x_idx, region in enumerate(REGION_ORDER):
            value = fdr.loc[cell_type, region]
            if np.isfinite(value) and value < 0.10:
                ax.text(x_idx + 0.5, y_idx + 0.5, "*", ha="center", va="center", fontsize=8, color="black")
    cbar_ax.set_title("Psychosis beta", fontsize=6.0, pad=2)
    cbar_ax.tick_params(axis="x", labelsize=5.6, length=2, pad=1)


def model_box(ax, xy: tuple[float, float], w: float, h: float, color: str, title: str, text: str) -> None:
    rect = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.14,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.66, title, ha="center", va="center", fontsize=6.7, fontweight="bold", color=color)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.34, text, ha="center", va="center", fontsize=5.7, color="#222222")


def draw_cellular_model(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    model_box(ax, (0.37, 0.38), 0.26, 0.24, "#333333", "Cellular psychosis state", "Striatal molecular signatures\nmapped to cellular compartments")
    boxes = [
        ((0.04, 0.66), "Dopamine-adenosine", "TH/MAOB -> DA axon\nADORA2A -> D2 MSN", "Dopamine-adenosine dysregulation", (0.37, 0.55)),
        ((0.04, 0.13), "Immune-NF-kB", "CCL2/CXCL8/IL1B\nmicroglia-immune tone", "Immune-NF-kB cytokine tone", (0.38, 0.44)),
        ((0.70, 0.66), "MSN compartment", "Caudate D1 MSN score lower\nD2 signal linked to ADORA2A", "#4E79A7", (0.63, 0.56)),
        ((0.70, 0.13), "Glial compartment", "Putamen oligodendrocyte lower\nmicroglia score higher", "#59A14F", (0.63, 0.44)),
    ]
    for xy, title, text, key, target in boxes:
        color = SIGNATURE_COLORS[key] if key in SIGNATURE_COLORS else key
        model_box(ax, xy, 0.24, 0.22, color, title, text)
        start = (xy[0] + (0.24 if xy[0] < 0.5 else 0), xy[1] + 0.11)
        ax.annotate("", xy=target, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": color, "alpha": 0.75})

    ax.text(
        0.5,
        0.05,
        "Cellular attribution supports a combined neuronal, glial, and immune interpretation of the molecular basis model.",
        ha="center",
        va="center",
        fontsize=5.9,
        color="#444444",
    )


def main() -> None:
    set_publication_style()
    enrichment = pd.read_csv(TABLE_DIR / "figure6_signature_celltype_enrichment.tsv", sep="\t")
    score_results = pd.read_csv(TABLE_DIR / "figure6_cell_type_score_results.tsv", sep="\t")
    metrics = pd.read_csv(TABLE_DIR / "figure6_cell_type_metrics.tsv", sep="\t")

    fig = plt.figure(figsize=mm_to_inches(180, 205), constrained_layout=False)
    ax_workflow = fig.add_axes([0.07, 0.78, 0.86, 0.15])
    draw_workflow(ax_workflow, metrics)

    ax_enrich = fig.add_axes([0.09, 0.45, 0.43, 0.22])
    cbar_enrich = fig.add_axes([0.535, 0.485, 0.012, 0.15])
    draw_enrichment_heatmap(ax_enrich, cbar_enrich, enrichment)

    ax_score = fig.add_axes([0.65, 0.45, 0.26, 0.22])
    cbar_score = fig.add_axes([0.68, 0.405, 0.19, 0.012])
    draw_score_heatmap(ax_score, cbar_score, score_results)

    ax_model = fig.add_axes([0.07, 0.10, 0.86, 0.23])
    draw_cellular_model(ax_model)

    figure_panel_label(fig, "A", 0.035, 0.955)
    figure_panel_label(fig, "B", 0.035, 0.695)
    figure_panel_label(fig, "C", 0.585, 0.695)
    figure_panel_label(fig, "D", 0.035, 0.355)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"figure6_cellular_basis_model.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 6 cellular basis model.")


if __name__ == "__main__":
    main()
