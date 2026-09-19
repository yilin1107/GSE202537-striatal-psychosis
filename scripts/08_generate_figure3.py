from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import GROUP_COLORS, REGION_ORDER, mm_to_inches, set_publication_style, story_step  # noqa: E402


TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}

AXIS_ORDER = [
    "Dopamine/adenosine",
    "Glutamate/calcium",
    "Immune/cytokine",
    "TGF-beta/SMAD",
    "Neurodegeneration",
]
AXIS_COLORS = {
    "Dopamine/adenosine": "#4E79A7",
    "Glutamate/calcium": "#59A14F",
    "Immune/cytokine": "#E15759",
    "TGF-beta/SMAD": "#B07AA1",
    "Neurodegeneration": "#F28E2B",
}
EFFECT_CMAP = sns.diverging_palette(240, 10, as_cmap=True)
TOP_N_GENES = 18
DRIVER_LABEL_OFFSETS = {
    "INHBA": (8, 6),
    "SMAD1": (6, 8),
    "TH": (8, 4),
    "PSEN1": (8, 8),
    "GRM8": (8, -14),
    "IL6": (-8, 8),
}
REGION_LABELS = {
    "NAc": ["INHBA", "NFKBIA", "CEBPB", "CXCL8"],
    "Caudate": ["TH", "INHBA", "PSEN1", "KAT5"],
    "Putamen": ["INHBA", "SMAD1", "TH", "KAT5"],
}
VOLCANO_LABEL_OFFSETS = {
    ("NAc", "INHBA"): (-10, 10),
    ("NAc", "NFKBIA"): (8, 14),
    ("NAc", "CEBPB"): (8, 2),
    ("NAc", "CXCL8"): (8, -10),
    ("Caudate", "TH"): (-8, 8),
    ("Caudate", "INHBA"): (8, 10),
    ("Caudate", "PSEN1"): (-8, 8),
    ("Caudate", "KAT5"): (-8, -8),
    ("Putamen", "INHBA"): (8, 8),
    ("Putamen", "SMAD1"): (8, 8),
    ("Putamen", "TH"): (-8, 10),
    ("Putamen", "KAT5"): (-8, 8),
}


def figure_panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def select_top_genes(summary: pd.DataFrame, n_genes: int = TOP_N_GENES) -> list[str]:
    axis_rank = {axis: idx for idx, axis in enumerate(AXIS_ORDER)}
    ranked = summary.copy()
    ranked["axis_rank"] = ranked["axis"].map(axis_rank).fillna(len(AXIS_ORDER))
    ranked = ranked.sort_values(["evidence_score", "pathway_count"], ascending=[False, False])
    seeds: list[str] = []
    for axis in AXIS_ORDER:
        axis_genes = ranked.loc[ranked["axis"] == axis, "gene"].head(3).tolist()
        for gene in axis_genes:
            if gene not in seeds:
                seeds.append(gene)
    for gene in ranked["gene"]:
        if gene not in seeds:
            seeds.append(gene)
        if len(seeds) >= n_genes:
            break
    ordered = (
        summary.loc[summary["gene"].isin(seeds)]
        .assign(axis_rank=lambda frame: frame["axis"].map(axis_rank).fillna(len(AXIS_ORDER)))
        .sort_values(["axis_rank", "evidence_score"], ascending=[True, False])
    )
    return ordered["gene"].tolist()


def draw_membership_map(ax, selected_pathways: pd.DataFrame, summary: pd.DataFrame, genes: list[str]) -> None:
    pathway_labels = selected_pathways["short_label"].tolist()
    pathway_axes = selected_pathways["axis"].tolist()
    gene_lookup = summary.set_index("gene")
    x_positions: list[int] = []
    y_positions: list[int] = []
    colors: list[str] = []
    for x_idx, row in selected_pathways.reset_index(drop=True).iterrows():
        members = set(str(row["genes"]).split(";"))
        for y_idx, gene in enumerate(genes):
            if gene in members:
                x_positions.append(x_idx)
                y_positions.append(y_idx)
                colors.append(AXIS_COLORS[row["axis"]])
    ax.scatter(x_positions, y_positions, s=34, c=colors, edgecolor="white", linewidth=0.35)
    ax.set_xticks(range(len(pathway_labels)))
    ax.set_xticklabels(pathway_labels, rotation=35, ha="right", fontsize=5.6)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=6.2)
    ax.set_xlim(-0.5, len(pathway_labels) - 0.5)
    ax.set_ylim(len(genes) - 0.5, -0.5)
    ax.set_title("Genes drawn from altered Figure 2 pathways")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)
    for y_idx, gene in enumerate(genes):
        axis = gene_lookup.loc[gene, "axis"]
        ax.get_yticklabels()[y_idx].set_color(AXIS_COLORS.get(axis, "black"))


def draw_gene_beta_heatmap(ax, gene_results: pd.DataFrame, genes: list[str]) -> None:
    beta = (
        gene_results.loc[gene_results["gene"].isin(genes)]
        .pivot(index="gene", columns="brain_region", values="beta_psychosis")
        .reindex(index=genes, columns=REGION_ORDER)
    )
    fdr = (
        gene_results.loc[gene_results["gene"].isin(genes)]
        .pivot(index="gene", columns="brain_region", values="fdr")
        .reindex(index=genes, columns=REGION_ORDER)
    )
    sns.heatmap(
        beta,
        ax=ax,
        cmap=EFFECT_CMAP,
        center=0,
        vmin=-0.75,
        vmax=0.75,
        linewidths=0,
        cbar=False,
    )
    ax.set_title("Component-gene psychosis effects")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(REGION_ORDER)) + 0.5)
    ax.set_yticks(np.arange(len(genes)) + 0.5)
    ax.set_xticklabels(REGION_ORDER, rotation=0)
    ax.set_yticklabels(genes, rotation=0, fontsize=5.6)
    for y_idx, gene in enumerate(beta.index):
        for x_idx, region in enumerate(REGION_ORDER):
            value = fdr.loc[gene, region]
            if np.isfinite(value) and value < 0.10:
                ax.text(x_idx + 0.5, y_idx + 0.5, "*", ha="center", va="center", fontsize=7, color="black")


def draw_driver_evidence(ax, summary: pd.DataFrame, genes: list[str]) -> None:
    plot_df = summary.loc[summary["gene"].isin(genes)].copy()
    plot_df["neg_log10_p"] = -np.log10(plot_df["best_p_value"].clip(lower=1e-300))
    rng = np.random.default_rng(3)
    jitter = rng.normal(0, 0.035, size=len(plot_df))
    sizes = 30 + 120 * plot_df["best_beta"].abs().clip(upper=0.8)
    for axis in AXIS_ORDER:
        axis_df = plot_df.loc[plot_df["axis"] == axis]
        if axis_df.empty:
            continue
        axis_jitter = jitter[plot_df["axis"].to_numpy() == axis]
        ax.scatter(
            axis_df["pathway_count"] + axis_jitter,
            axis_df["neg_log10_p"],
            s=sizes.loc[axis_df.index],
            c=AXIS_COLORS[axis],
            alpha=0.86,
            edgecolor="black",
            linewidth=0.25,
            label=axis,
        )
    labels = plot_df.loc[plot_df["gene"].isin(DRIVER_LABEL_OFFSETS)]
    for _, row in labels.iterrows():
        offset = DRIVER_LABEL_OFFSETS[row["gene"]]
        ax.annotate(
            row["gene"],
            xy=(row["pathway_count"], row["neg_log10_p"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.6,
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            arrowprops={"arrowstyle": "-", "color": "#9E9E9E", "linewidth": 0.35},
        )
    ax.set_xlim(0.6, max(3.4, plot_df["pathway_count"].max() + 0.4))
    ax.set_xlabel("Pathway memberships")
    ax.set_ylabel("-log10 best P")
    ax.set_title("Driver evidence for signature building")
    ax.set_xticks([1, 2, 3])
    ax.legend(frameon=False, loc="upper right", fontsize=5.1, handletextpad=0.25, borderaxespad=0.1)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def gene_label(gene: str) -> str:
    replacements = {
        "INHBA": "INHBA",
        "TH": "TH",
        "SMAD1": "SMAD1",
        "PSEN1": "PSEN1",
        "KAT5": "KAT5",
        "XBP1": "XBP1",
        "GRM8": "GRM8",
        "SLC2A1": "SLC2A1",
        "NFKBIA": "NFKBIA",
        "CEBPB": "CEBPB",
    }
    return replacements.get(gene, gene)


def draw_region_gene_volcano(ax, gene_results: pd.DataFrame, region: str) -> None:
    subset = gene_results.loc[gene_results["brain_region"] == region].copy()
    subset["neg_log10_p"] = -np.log10(subset["p_value"].clip(lower=1e-300))
    subset["significant"] = subset["fdr"] < 0.10
    colors = np.where(
        subset["significant"] & (subset["beta_psychosis"] >= 0),
        GROUP_COLORS["SCZ"],
        np.where(subset["significant"], "#4E79A7", "#CFCFCF"),
    )
    ax.scatter(
        subset["beta_psychosis"],
        subset["neg_log10_p"],
        s=np.where(subset["significant"], 21, 12),
        c=colors,
        alpha=np.where(subset["significant"], 0.9, 0.48),
        edgecolor="none",
    )
    ax.axvline(0, color="#BDBDBD", linewidth=0.7)
    ax.set_xlim(-0.85, 0.85)
    ax.set_xlabel("Psychosis beta")
    ax.set_ylabel("-log10 P")
    ax.set_title(region)
    label_genes = REGION_LABELS.get(region, [])
    labels = subset.loc[subset["gene"].isin(label_genes)].copy()
    y_max = subset["neg_log10_p"].max()
    ax.set_ylim(0, y_max * 1.18)
    for _, row in labels.iterrows():
        x_value = row["beta_psychosis"]
        y_value = row["neg_log10_p"]
        offset = VOLCANO_LABEL_OFFSETS.get((region, row["gene"]), (8 if x_value >= 0 else -8, 8))
        ax.annotate(
            gene_label(row["gene"]),
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.6,
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            arrowprops={"arrowstyle": "-", "color": "#9E9E9E", "linewidth": 0.35},
        )
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def main() -> None:
    set_publication_style()
    selected_pathways = pd.read_csv(TABLE_DIR / "figure3_selected_molecular_pathways.tsv", sep="\t")
    gene_results = pd.read_csv(TABLE_DIR / "figure3_core_gene_model_results.tsv", sep="\t")
    summary = pd.read_csv(TABLE_DIR / "figure3_core_gene_summary.tsv", sep="\t")
    genes = select_top_genes(summary, TOP_N_GENES)

    fig = plt.figure(figsize=mm_to_inches(180, 225), constrained_layout=False)
    story_step(
        fig,
        "Step 2",
        "Pathway-to-gene decomposition",
        "Significant pathway signals are resolved into candidate driver genes",
    )
    ax_map = fig.add_axes([0.19, 0.66, 0.75, 0.27])
    draw_membership_map(ax_map, selected_pathways, summary, genes)

    ax_heatmap = fig.add_axes([0.11, 0.38, 0.36, 0.19])
    draw_gene_beta_heatmap(ax_heatmap, gene_results, genes)
    cbar_ax = fig.add_axes([0.13, 0.335, 0.23, 0.012])
    norm = plt.Normalize(-0.75, 0.75)
    sm = plt.cm.ScalarMappable(cmap=EFFECT_CMAP, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_title("Gene beta", fontsize=6.2, pad=2)
    cbar_ax.tick_params(axis="x", labelsize=5.8, pad=1, length=2)

    ax_evidence = fig.add_axes([0.58, 0.38, 0.36, 0.19])
    draw_driver_evidence(ax_evidence, summary, genes)

    volcano_axes = [
        fig.add_axes([0.07, 0.08, 0.26, 0.20]),
        fig.add_axes([0.39, 0.08, 0.26, 0.20]),
        fig.add_axes([0.71, 0.08, 0.26, 0.20]),
    ]
    for ax, region in zip(volcano_axes, REGION_ORDER):
        draw_region_gene_volcano(ax, gene_results, region)

    figure_panel_label(fig, "A", 0.035, 0.965)
    figure_panel_label(fig, "B", 0.035, 0.60)
    figure_panel_label(fig, "C", 0.505, 0.60)
    figure_panel_label(fig, "D", 0.035, 0.305)
    figure_panel_label(fig, "E", 0.355, 0.305)
    figure_panel_label(fig, "F", 0.675, 0.305)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"figure3_core_molecular_drivers.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 3 core molecular drivers.")


if __name__ == "__main__":
    main()
