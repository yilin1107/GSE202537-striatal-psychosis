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
TRAIT_ORDER = [
    "Schizophrenia",
    "Psychotic disorder",
    "Bipolar disorder",
    "Major depressive disorder",
    "Autism spectrum disorder",
    "ADHD",
]
TRAIT_LABELS = {
    "Schizophrenia": "SCZ",
    "Psychotic disorder": "Psychosis",
    "Bipolar disorder": "BD",
    "Major depressive disorder": "MDD",
    "Autism spectrum disorder": "ASD",
    "ADHD": "ADHD",
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


def draw_box(ax, x: float, y: float, w: float, h: float, color: str, title: str, text: str) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.0, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.34, text, ha="center", va="center", fontsize=6.1, color="#222222")


def draw_workflow(ax, metrics: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    total_genes = metric_value(metrics, "figure5_core_signature_genes")
    n_eqtl = metric_value(metrics, "genes_with_gtex_brain_eqtl")
    n_striatal = metric_value(metrics, "genes_with_striatal_eqtl")
    n_gwas = metric_value(metrics, "genes_with_psychiatric_gwas_catalog_support")
    n_both = metric_value(metrics, "genes_with_integrated_eqtl_gwas_support")
    n_traits = metric_value(metrics, "psychiatric_traits_tested")
    n_tissues = metric_value(metrics, "gtex_brain_tissues_tested")

    boxes = [
        (0.02, "#333333", "Core genes", f"{total_genes} Figure 4\nsignature genes"),
        (0.27, "#2C7FB8", "Brain eQTL", f"GTEx v8, {n_tissues} tissues\n{n_eqtl} genes; {n_striatal} striatal"),
        (0.52, "#F28E2B", "GWAS Catalog", f"{n_traits} psychiatric traits\n{n_gwas} mapped genes"),
        (0.77, "#54278F", "Integrated support", f"{n_both} genes with\nboth evidence layers"),
    ]
    for x, color, title, text in boxes:
        draw_box(ax, x, 0.20, 0.20, 0.56, color, title, text)
    for x in [0.225, 0.475, 0.725]:
        ax.annotate("", xy=(x + 0.035, 0.48), xytext=(x, 0.48), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"})
    ax.text(
        0.5,
        0.04,
        "Mapped-gene support is used as an eQTL-informed prioritization layer; formal colocalization requires trait summary statistics.",
        ha="center",
        va="center",
        fontsize=5.7,
        color="#444444",
    )


def selected_heatmap_genes(summary: pd.DataFrame) -> list[str]:
    focused = summary.loc[
        (summary["psychiatric_gwas_trait_count"] > 0)
        | (summary["brain_eqtl_tissue_count"] >= 2)
        | (summary["gene"].isin(["INHBA", "TH", "SMAD1", "PSEN1", "KAT5", "XBP1"]))
    ].copy()
    focused = focused.sort_values(
        ["has_integrated_support", "psychiatric_gwas_trait_count", "brain_eqtl_tissue_count", "evidence_score"],
        ascending=[False, False, False, False],
    )
    return focused["gene"].head(22).tolist()


def draw_gwas_heatmap(ax, cbar_ax, summary: pd.DataFrame, gwas: pd.DataFrame) -> None:
    genes = selected_heatmap_genes(summary)
    matrix = (
        gwas.pivot_table(index="gene", columns="trait", values="neg_log10_min_p", aggfunc="max", fill_value=0)
        .reindex(index=genes, columns=TRAIT_ORDER)
        .fillna(0)
    )
    cmap = sns.light_palette("#B35806", as_cmap=True)
    sns.heatmap(
        matrix.clip(upper=24),
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=24,
        linewidths=0,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
    )
    ax.set_title("Psychiatric GWAS Catalog support")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([TRAIT_LABELS[trait] for trait in TRAIT_ORDER], rotation=35, ha="right")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis="y", labelsize=5.8, length=0)
    ax.tick_params(axis="x", labelsize=5.8, length=0)
    cbar_ax.set_title("-log10 min P", fontsize=6.0, pad=2)
    cbar_ax.tick_params(axis="x", labelsize=5.6, length=2, pad=1)


def draw_evidence_scatter(ax, summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["size"] = 22 + np.sqrt(plot_df["psychiatric_gwas_association_count"].clip(lower=0)) * 9
    for signature in SIGNATURE_ORDER:
        data = plot_df.loc[plot_df["signature"] == signature]
        ax.scatter(
            data["evidence_score"],
            data["brain_eqtl_tissue_count"],
            s=data["size"],
            color=SIGNATURE_COLORS[signature],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            label=SIGNATURE_LABELS[signature],
        )
    labels = ["CACNA1C", "GRM8", "INHBA", "TH", "SMAD1", "XBP1", "ADAM17", "PSEN1", "CACNA2D1", "ADAM10"]
    offsets = {
        "CACNA1C": (-0.92, 0.18),
        "GRM8": (-0.70, 0.20),
        "INHBA": (-0.85, 0.14),
        "TH": (0.24, 0.42),
        "SMAD1": (0.10, 0.14),
        "XBP1": (0.06, -0.22),
        "ADAM17": (-0.72, 0.02),
        "PSEN1": (0.08, 0.28),
        "CACNA2D1": (0.25, -0.44),
        "ADAM10": (-0.85, -0.45),
    }
    for gene in labels:
        row = plot_df.loc[plot_df["gene"] == gene]
        if row.empty:
            continue
        x = float(row.iloc[0]["evidence_score"])
        y = float(row.iloc[0]["brain_eqtl_tissue_count"])
        dx, dy = offsets[gene]
        ax.text(x + dx, y + dy, gene, fontsize=5.7, ha="left", va="center")
    ax.set_title("Transcriptomic priority versus brain eQTL breadth")
    ax.set_xlabel("Transcriptomic evidence score")
    ax.set_ylabel("GTEx brain tissues with eQTL")
    ax.set_ylim(-0.35, 5.35)
    ax.set_yticks(range(0, 6))
    ax.set_xlim(0, max(9.6, plot_df["evidence_score"].max() + 0.6))
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.03, 1.03), ncol=1, handletextpad=0.3, borderaxespad=0)


def draw_axis_support(ax, summary: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for signature in SIGNATURE_ORDER:
        data = summary.loc[summary["signature"] == signature]
        rows.append(
            {
                "signature": signature,
                "Neither": int((~data["has_brain_eqtl"] & ~data["has_psychiatric_gwas"]).sum()),
                "eQTL only": int((data["has_brain_eqtl"] & ~data["has_psychiatric_gwas"]).sum()),
                "GWAS only": int((~data["has_brain_eqtl"] & data["has_psychiatric_gwas"]).sum()),
                "Both": int((data["has_brain_eqtl"] & data["has_psychiatric_gwas"]).sum()),
            }
        )
    plot_df = pd.DataFrame(rows).set_index("signature").reindex(SIGNATURE_ORDER)
    categories = ["Neither", "eQTL only", "GWAS only", "Both"]
    colors = {"Neither": "#D9D9D9", "eQTL only": "#2C7FB8", "GWAS only": "#F28E2B", "Both": "#54278F"}
    left = np.zeros(plot_df.shape[0])
    y = np.arange(plot_df.shape[0])
    for category in categories:
        values = plot_df[category].values
        ax.barh(y, values, left=left, color=colors[category], height=0.62, label=category)
        for idx, value in enumerate(values):
            if value > 0:
                ax.text(left[idx] + value / 2, idx, str(int(value)), ha="center", va="center", fontsize=6, color="white" if category in {"Both", "eQTL only"} else "#222222")
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([SIGNATURE_LABELS[item] for item in SIGNATURE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Genes per molecular signature")
    ax.set_title("Integrated genetic-regulatory support by axis")
    ax.set_xlim(0, max(7, int(left.max()) + 1))
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, handlelength=1.2, columnspacing=0.8)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def draw_cell_enrichment_heatmap(ax, cbar_ax, enrichment: pd.DataFrame) -> None:
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
                ax.text(x_idx + 0.5, y_idx + 0.5, genes.replace(";", "\n"), ha="center", va="center", fontsize=4.8, color="black")
    cbar_ax.set_title("-log10\nFDR", fontsize=5.4, pad=2)
    cbar_ax.tick_params(axis="y", labelsize=5.6, length=2, pad=1)


def draw_cell_score_heatmap(ax, cbar_ax, score_results: pd.DataFrame) -> None:
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


def main() -> None:
    set_publication_style()
    summary = pd.read_csv(TABLE_DIR / "figure5_integrated_eqtl_gwas_summary.tsv", sep="\t")
    gwas = pd.read_csv(TABLE_DIR / "figure5_gwas_catalog_support.tsv", sep="\t")
    enrichment = pd.read_csv(TABLE_DIR / "figure6_signature_celltype_enrichment.tsv", sep="\t")
    score_results = pd.read_csv(TABLE_DIR / "figure6_cell_type_score_results.tsv", sep="\t")

    fig = plt.figure(figsize=mm_to_inches(180, 222), constrained_layout=False)
    ax_heatmap = fig.add_axes([0.10, 0.66, 0.37, 0.25])
    cbar_ax = fig.add_axes([0.18, 0.625, 0.22, 0.012])
    draw_gwas_heatmap(ax_heatmap, cbar_ax, summary, gwas)

    ax_scatter = fig.add_axes([0.57, 0.66, 0.34, 0.25])
    draw_evidence_scatter(ax_scatter, summary)

    ax_axis = fig.add_axes([0.14, 0.42, 0.64, 0.16])
    draw_axis_support(ax_axis, summary)

    ax_enrich = fig.add_axes([0.10, 0.095, 0.43, 0.20])
    cbar_enrich = fig.add_axes([0.545, 0.130, 0.012, 0.14])
    draw_cell_enrichment_heatmap(ax_enrich, cbar_enrich, enrichment)

    ax_score = fig.add_axes([0.66, 0.095, 0.25, 0.20])
    cbar_score = fig.add_axes([0.69, 0.055, 0.19, 0.012])
    draw_cell_score_heatmap(ax_score, cbar_score, score_results)

    fig.text(0.50, 0.605, "Genetic-regulatory prioritization", ha="center", va="center", fontsize=7.0, color="#555555")
    fig.text(0.50, 0.335, "Cell-type attribution of molecular signatures", ha="center", va="center", fontsize=7.0, color="#555555")

    figure_panel_label(fig, "A", 0.035, 0.94)
    figure_panel_label(fig, "B", 0.515, 0.94)
    figure_panel_label(fig, "C", 0.035, 0.605)
    figure_panel_label(fig, "D", 0.035, 0.335)
    figure_panel_label(fig, "E", 0.595, 0.335)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"figure5_eqtl_gwas_support.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 5 eQTL-informed GWAS support.")


if __name__ == "__main__":
    main()
