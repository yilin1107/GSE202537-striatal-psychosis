from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import GROUP_COLORS, REGION_ORDER, mm_to_inches, panel_label, set_publication_style, story_step  # noqa: E402


TABLE_DIR = ROOT / "results" / "tables"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}


REGION_COLORS = {"NAc": "#76B7B2", "Caudate": "#F28E2B", "Putamen": "#EDC948"}
EFFECT_CMAP = sns.diverging_palette(240, 10, as_cmap=True)
CATEGORY_ORDER = [
    "Neurotransmission",
    "Immune/inflammation",
    "Mitochondrial/metabolic",
    "Neurodegeneration",
    "Circadian/sleep",
    "Other",
]
CATEGORY_LABELS = {
    "Neurotransmission": "Neurotrans.",
    "Immune/inflammation": "Immune/inflamm.",
    "Mitochondrial/metabolic": "Mito/metabolic",
    "Neurodegeneration": "Neurodegen.",
    "Circadian/sleep": "Circadian/sleep",
    "Other": "Other",
}
HEATMAP_LABELS = {
    "Dopamine Metabolism in Parkinson's Disease": "Dopamine metabolism in PD",
    "Timothy Syndrome": "Timothy syndrome",
    "Wolfram Syndrome Progression (Hypothesis)": "Wolfram progression",
    "N2 Neutrophils in Tumor-Promoting Inflammation and Tumor Progression": "N2 neutrophils / inflammation",
    "IL6ST -> STAT5B Signaling": "IL6ST-STAT5B signaling",
    "Lipoxin A4/FPR2-Related Neutrophil Depression": "Lipoxin A4-FPR2 neutrophils",
    "ADK Expression Downregulation after Acute Seizures": "ADK after acute seizures",
    "MacrophageR -> CEBPB -> NF-kB Signaling": "Macrophage CEBPB-NF-kB",
    "APP Processing in Alzheimer Disease": "APP processing",
    "GRM2-4/6-8 (Presynaptic) -> Glutamate Release Attenuation": "Presynaptic GRM glutamate",
    "Proteins Involved in B-Cell Chronic Lymphocytic Leukemia": "B-cell signaling proteins",
    "ICAM2 -> CTNNB/FOXO/STAT3 Signaling": "ICAM2-STAT3 signaling",
}
PRIORITY_PATTERN = (
    r"dopamine|glutamate|gaba|synap|mitochond|immune|interleukin|"
    r"inflamm|nf-kb|stat[0-9]|sleep|circadian|melatonin|adenosine|"
    r"adk|seizure|alzheimer|parkinson|wolfram|timothy|toll-like|"
    r"macrophage|neutrophil|t-cell|t cell|b-cell|b cell|kynurenine"
)


def short_label(label: str, width: int = 34) -> str:
    cleaned = label.replace(" -> ", " -> ")
    if len(cleaned) > 54:
        cleaned = cleaned[:51] + "..."
    return "\n".join(textwrap.wrap(cleaned, width=width, break_long_words=False))


def heatmap_label(pathway: str) -> str:
    return HEATMAP_LABELS.get(pathway, short_label(pathway, width=32))


def pathway_category(pathway: str) -> str:
    value = pathway.lower()
    if any(term in value for term in ["dopamine", "glutamate", "gaba", "synap", "adenosine"]):
        return "Neurotransmission"
    if any(term in value for term in ["immune", "inflamm", "nf-kb", "interleukin", "macrophage", "neutrophil", "t-cell", "t cell", "b-cell", "b cell", "toll-like", "stat"]):
        return "Immune/inflammation"
    if any(term in value for term in ["mitochond", "tricarboxylic", "kynurenine", "metabolism"]):
        return "Mitochondrial/metabolic"
    if any(term in value for term in ["alzheimer", "parkinson", "dementia", "wolfram", "timothy", "seizure"]):
        return "Neurodegeneration"
    if any(term in value for term in ["sleep", "circadian", "melatonin"]):
        return "Circadian/sleep"
    return "Other"


def style_axis(ax) -> None:
    ax.tick_params(axis="both", width=0.6, length=2.5)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def panel_label_high(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.18,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def figure_panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def select_top_pathways(results: pd.DataFrame, n_pathways: int = 18) -> list[str]:
    priority = results.loc[results["pathway"].str.contains(PRIORITY_PATTERN, case=False, regex=True, na=False)]
    ranked = priority.assign(abs_beta=priority["beta_psychosis"].abs()).sort_values(
        ["fdr", "p_value", "abs_beta"], ascending=[True, True, False]
    )
    pathways: list[str] = []
    for pathway in ranked["pathway"]:
        if pathway not in pathways:
            pathways.append(pathway)
        if len(pathways) == n_pathways:
            break
    if len(pathways) < n_pathways:
        fallback = (
            results.assign(abs_beta=results["beta_psychosis"].abs())
            .sort_values(["fdr", "p_value", "abs_beta"], ascending=[True, True, False])
        )
        for pathway in fallback["pathway"]:
            if pathway not in pathways:
                pathways.append(pathway)
            if len(pathways) == n_pathways:
                break
    return pathways


def draw_category_bubble(ax, results: pd.DataFrame) -> None:
    data = results.copy()
    data["category"] = data["pathway"].map(pathway_category)
    significant = data.loc[data["fdr"] < 0.10]
    summary = (
        significant.groupby(["brain_region", "category"], as_index=False)
        .agg(n_pathways=("pathway", "count"), mean_beta=("beta_psychosis", "mean"))
    )
    x_lookup = {region: index for index, region in enumerate(REGION_ORDER)}
    y_lookup = {category: index for index, category in enumerate(CATEGORY_ORDER[::-1])}
    if not summary.empty:
        bubble_sizes = np.clip(np.sqrt(summary["n_pathways"]) * 34 + 20, 35, 760)
        ax.scatter(
            summary["brain_region"].map(x_lookup),
            summary["category"].map(y_lookup),
            s=bubble_sizes,
            c=summary["mean_beta"],
            cmap=EFFECT_CMAP,
            vmin=-0.8,
            vmax=0.8,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.92,
        )
        for _, row in summary.iterrows():
            ax.text(
                x_lookup[row["brain_region"]],
                y_lookup[row["category"]],
                str(int(row["n_pathways"])),
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if abs(row["mean_beta"]) > 0.42 else "black",
            )
    ax.set_xticks(range(len(REGION_ORDER)))
    ax.set_xticklabels(REGION_ORDER)
    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels([CATEGORY_LABELS[category] for category in CATEGORY_ORDER[::-1]])
    ax.set_xlim(-0.5, len(REGION_ORDER) - 0.5)
    ax.set_ylim(-0.5, len(CATEGORY_ORDER) - 0.5)
    ax.set_title("FDR<0.10 pathway classes")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    style_axis(ax)


def draw_heatmap(ax, results: pd.DataFrame, pathways: list[str], cbar_ax=None, cbar_orientation: str = "vertical") -> None:
    plot_data = (
        results.loc[results["pathway"].isin(pathways)]
        .pivot(index="pathway", columns="brain_region", values="beta_psychosis")
        .reindex(index=pathways, columns=REGION_ORDER)
    )
    labels = [heatmap_label(pathway) for pathway in plot_data.index]
    sns.heatmap(
        plot_data,
        ax=ax,
        cmap=EFFECT_CMAP,
        center=0,
        linewidths=0.35,
        linecolor="white",
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": cbar_orientation},
    )
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xticklabels(REGION_ORDER, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Pathway-level discovery screen")

    fdr_data = (
        results.loc[results["pathway"].isin(pathways)]
        .pivot(index="pathway", columns="brain_region", values="fdr")
        .reindex(index=pathways, columns=REGION_ORDER)
    )
    for y_idx, pathway in enumerate(plot_data.index):
        for x_idx, region in enumerate(REGION_ORDER):
            fdr = fdr_data.loc[pathway, region]
            if np.isfinite(fdr) and fdr < 0.10:
                ax.text(x_idx + 0.5, y_idx + 0.5, "*", ha="center", va="center", fontsize=8, color="black")


def draw_representative_scores(
    ax, results: pd.DataFrame, scores: pd.DataFrame, metadata: pd.DataFrame, compact: bool = False
) -> None:
    selected = [
        ("Putamen", "Dopamine Metabolism in Parkinson's Disease", "Dopamine\nPutamen"),
        ("Caudate", "GRM2-4/6-8 (Presynaptic) -> Glutamate Release Attenuation", "Glutamate\nCaudate"),
        ("Putamen", "IL6ST -> STAT5B Signaling", "IL6ST-STAT5B\nPutamen"),
    ]
    rows: list[dict[str, str | float]] = []
    for region, pathway, label in selected:
        region_meta = metadata.loc[metadata["brain_region"] == region].copy()
        if pathway not in scores.index:
            continue
        for _, sample in region_meta.iterrows():
            sample_key = sample["sample_key"]
            if sample_key not in scores.columns:
                continue
            rows.append(
                {
                    "pathway": label,
                    "group": "Psychosis" if sample["group"] != "Control" else "Control",
                    "score": float(scores.loc[pathway, sample_key]),
                    "brain_region": region,
                    "full_pathway": pathway,
                }
            )
    plot_df = pd.DataFrame(rows)
    palette = {"Control": GROUP_COLORS["Control"], "Psychosis": GROUP_COLORS["SCZ"]}
    sns.boxplot(
        data=plot_df,
        x="pathway",
        y="score",
        hue="group",
        hue_order=["Control", "Psychosis"],
        palette=palette,
        width=0.65,
        fliersize=0,
        linewidth=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="pathway",
        y="score",
        hue="group",
        hue_order=["Control", "Psychosis"],
        palette=palette,
        dodge=True,
        size=2.0,
        alpha=0.55,
        linewidth=0,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    if compact:
        ax.legend(
            handles[:2],
            labels[:2],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=2,
            borderaxespad=0.0,
            columnspacing=0.9,
            handlelength=1.4,
            fontsize=5.8,
        )
        ax.tick_params(axis="x", labelsize=5.8)
    else:
        ax.legend(handles[:2], labels[:2], frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.axhline(0, color="#BDBDBD", linewidth=0.6, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Pathway score (z)")
    ax.set_title("Representative significant pathways")
    style_axis(ax)

    y_top = plot_df["score"].max()
    y_bottom = plot_df["score"].min()
    y_range = y_top - y_bottom
    for x_idx, (_, pathway, _) in enumerate(selected):
        row = results.loc[(results["pathway"] == pathway) & (results["brain_region"] == selected[x_idx][0])]
        if row.empty:
            continue
        fdr = float(row.iloc[0]["fdr"])
        label = f"q={fdr:.2f}" if fdr >= 0.01 else "q<0.01"
        ax.text(x_idx, y_top + y_range * 0.06, label, ha="center", va="bottom", fontsize=5.8)
    ax.set_ylim(y_bottom - y_range * 0.08, y_top + y_range * 0.18)


def volcano_label(label: str) -> str:
    replacements = {
        "Dopamine Metabolism in Parkinson's Disease": "Dopamine metabolism",
        "Timothy Syndrome": "Timothy syndrome",
        "Wolfram Syndrome Progression (Hypothesis)": "Wolfram syndrome",
        "N2 Neutrophils in Tumor-Promoting Inflammation and Tumor Progression": "N2 neutrophils",
        "MacrophageR -> CEBPB -> NF-kB Signaling": "Macrophage NF-kB",
        "APP Processing in Alzheimer Disease": "APP processing",
        "GRM2-4/6-8 (Presynaptic) -> Glutamate Release Attenuation": "GRM glutamate release",
        "ADK Expression Downregulation after Acute Seizures": "ADK after seizures",
        "IL6ST -> STAT5B Signaling": "IL6ST-STAT5B",
        "Lipoxin A4/FPR2-Related Neutrophil Depression": "Lipoxin A4/FPR2",
        "ICAM2 -> CTNNB/FOXO/STAT3 Signaling": "ICAM2-STAT3",
        "Proteins Involved in Alzheimer's Disease": "Alzheimer proteins",
    }
    if label in replacements:
        return replacements[label]
    compact = label.replace(" -> ", "-")
    return compact if len(compact) <= 28 else compact[:25] + "..."


def draw_region_volcano(ax, results: pd.DataFrame, region: str) -> None:
    subset = results.loc[results["brain_region"] == region].copy()
    subset["neg_log10_p"] = -np.log10(subset["p_value"].clip(lower=1e-300))
    subset["priority"] = subset["pathway"].str.contains(PRIORITY_PATTERN, case=False, regex=True, na=False)
    subset["significant"] = subset["fdr"] < 0.10
    colors = np.where(
        subset["significant"] & (subset["beta_psychosis"] >= 0),
        GROUP_COLORS["SCZ"],
        np.where(subset["significant"], "#4E79A7", "#BDBDBD"),
    )
    sizes = np.where(subset["priority"], 13, 8)
    ax.scatter(
        subset["beta_psychosis"],
        subset["neg_log10_p"],
        s=sizes,
        c=colors,
        alpha=np.where(subset["significant"], 0.86, 0.35),
        edgecolor="none",
    )
    ax.axvline(0, color="#BDBDBD", linewidth=0.7, zorder=0)
    ax.set_xlim(-1.35, 1.35)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel("Psychosis beta")
    ax.set_ylabel("-log10 P")
    ax.set_title(region, pad=8)
    style_axis(ax)

    significant_priority = (
        subset.loc[subset["priority"] & subset["significant"]]
        .assign(abs_beta=lambda frame: frame["beta_psychosis"].abs())
        .sort_values(["fdr", "p_value", "abs_beta"], ascending=[True, True, False])
        .head(3)
    )
    if significant_priority.empty:
        significant_priority = (
            subset.loc[subset["priority"]]
            .assign(abs_beta=lambda frame: frame["beta_psychosis"].abs())
            .sort_values(["p_value", "abs_beta"], ascending=[True, False])
            .head(2)
        )
    y_max = subset["neg_log10_p"].max()
    ax.set_ylim(0, y_max * 1.15)
    for _, row in significant_priority.iterrows():
        x_value = row["beta_psychosis"]
        y_value = row["neg_log10_p"]
        ha = "left" if x_value >= 0 else "right"
        x_offset = 0.04 if x_value >= 0 else -0.04
        ax.text(
            x_value + x_offset,
            y_value + y_max * 0.025,
            volcano_label(row["pathway"]),
            ha=ha,
            va="bottom",
            fontsize=5.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
        )


def main() -> None:
    set_publication_style()
    results = pd.read_csv(TABLE_DIR / "figure2_elsevier_gsva_results.tsv", sep="\t")
    scores = pd.read_csv(PROCESSED_DIR / "gse202537_elsevier_pathway_scores.csv", index_col=0)
    metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    top_pathways = select_top_pathways(results, n_pathways=12)

    fig = plt.figure(figsize=mm_to_inches(180, 220), constrained_layout=False)
    story_step(
        fig,
        "Step 1",
        "Elsevier pathway discovery",
        "FDR-supported pathways nominate axes for gene-level follow-up",
    )

    ax_heatmap = fig.add_axes([0.25, 0.45, 0.42, 0.47])
    cbar_ax = fig.add_axes([0.41, 0.385, 0.20, 0.012])
    draw_heatmap(ax_heatmap, results, top_pathways, cbar_ax=cbar_ax, cbar_orientation="horizontal")
    cbar_ax.set_title("Psychosis beta", fontsize=6.2, pad=2)
    cbar_ax.tick_params(axis="x", labelsize=5.8, pad=1, length=2)

    ax_category = fig.add_axes([0.75, 0.73, 0.23, 0.19])
    draw_category_bubble(ax_category, results)
    ax_scores = fig.add_axes([0.75, 0.45, 0.23, 0.21])
    draw_representative_scores(ax_scores, results, scores, metadata, compact=True)

    volcano_axes = [
        fig.add_axes([0.07, 0.08, 0.26, 0.24]),
        fig.add_axes([0.39, 0.08, 0.26, 0.24]),
        fig.add_axes([0.71, 0.08, 0.26, 0.24]),
    ]
    for ax, region in zip(volcano_axes, REGION_ORDER):
        draw_region_volcano(ax, results, region)

    figure_panel_label(fig, "A", 0.17, 0.97)
    figure_panel_label(fig, "B", 0.705, 0.97)
    figure_panel_label(fig, "C", 0.705, 0.69)
    figure_panel_label(fig, "D", 0.035, 0.345)
    figure_panel_label(fig, "E", 0.355, 0.345)
    figure_panel_label(fig, "F", 0.675, 0.345)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"figure2_elsevier_gsva.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Wrote Figure 2 Elsevier pathway GSVA.")


if __name__ == "__main__":
    main()
