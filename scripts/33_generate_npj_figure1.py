from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

CONTRAST_LABELS = {
    "SCZ_vs_Control": "SCZ vs control",
    "BD_psychosis_vs_Control": "BD psychosis vs control",
}
REGION_ORDER = ["NAc", "Caudate", "Putamen"]
REGION_COLORS = {"NAc": "#76B7B2", "Caudate": "#F28E2B", "Putamen": "#EDC948"}

CATEGORY_ORDER = [
    "Immune-inflammatory",
    "Stress and cell death",
    "Metabolism and mitochondria",
    "Cell cycle and genome",
    "Developmental signaling",
    "Growth and kinase signaling",
    "Hormone and tissue remodeling",
]

CATEGORY_COLORS = {
    "Immune-inflammatory": "#D84A6A",
    "Stress and cell death": "#E88C30",
    "Metabolism and mitochondria": "#4B9CD3",
    "Cell cycle and genome": "#8E63CE",
    "Developmental signaling": "#51A65B",
    "Growth and kinase signaling": "#B68E2F",
    "Hormone and tissue remodeling": "#7A7A7A",
}

ENRICHMENT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "hallmark_enrichment",
    ["#2C7BB6", "#F7F7F7", "#D7191C"],
)


def panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def style_axis(ax) -> None:
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", width=0.6, length=2.5)


def draw_ma_plot(ax, results: pd.DataFrame, contrast: str, region: str) -> None:
    data = results.loc[(results["contrast"] == contrast) & (results["brain_region"] == region)].copy()
    data["neg_log10_p"] = -np.log10(data["p_value"].clip(lower=1e-300))
    significant = data["fdr"] < 0.10
    positive = data["log2fc"] > 0
    colors = np.where(
        significant & positive,
        "#D94B63",
        np.where(significant & ~positive, "#2C7BB6", "#BDBDBD"),
    )
    ax.scatter(
        data["mean_logcpm"],
        data["log2fc"],
        c=colors,
        s=np.where(significant, 10, 5),
        alpha=np.where(significant, 0.82, 0.34),
        edgecolor="none",
        rasterized=True,
    )
    ax.axhline(0, color="#9E9E9E", linewidth=0.7)
    ax.set_xlabel("Mean expression, log2(CPM + 1)")
    ax.set_ylabel("Adjusted log2 fold-change")
    ax.set_title(region)
    style_axis(ax)

    x_min = max(-0.5, float(data["mean_logcpm"].min()) - 0.35)
    x_max = float(data["mean_logcpm"].max()) + 0.45
    y_abs = max(abs(float(data["log2fc"].min())), abs(float(data["log2fc"].max())), 0.55)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_abs * 1.08, y_abs * 1.08)

    top = data.loc[significant].sort_values("p_value").head(20).copy()
    up = top.loc[top["log2fc"] >= 0].sort_values("log2fc", ascending=False)
    down = top.loc[top["log2fc"] < 0].sort_values("log2fc", ascending=True)

    def add_label_column(label_data: pd.DataFrame, side: str) -> None:
        if label_data.empty:
            return
        n_labels = label_data.shape[0]
        y_positions = np.linspace(y_abs * 0.83, -y_abs * 0.83, n_labels)
        x_text = x_max - 0.04 * (x_max - x_min) if side == "right" else x_min + 0.04 * (x_max - x_min)
        ha = "right" if side == "right" else "left"
        for y_text, (_, row) in zip(y_positions, label_data.iterrows()):
            ax.annotate(
                row["gene"],
                xy=(row["mean_logcpm"], row["log2fc"]),
                xytext=(x_text, y_text),
                fontsize=4.25,
                ha=ha,
                va="center",
                arrowprops={"arrowstyle": "-", "linewidth": 0.24, "color": "#777777", "alpha": 0.68},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.18},
                clip_on=True,
            )

    add_label_column(up, "right")
    add_label_column(down, "left")

    n_fdr = int(significant.sum())
    top_fdr = float(data["fdr"].min())
    note = f"{n_fdr} FDR<0.10 genes" if n_fdr else f"min FDR={top_fdr:.2f}"
    ax.text(
        0.02,
        0.96,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color="#444444",
    )


def short_hallmark_label(hallmark_id: str) -> str:
    replacements = {
        "TNFA_SIGNALING_VIA_NFKB": "TNFA/NF-kB",
        "IL6_JAK_STAT3_SIGNALING": "IL6/JAK/STAT3",
        "IL2_STAT5_SIGNALING": "IL2/STAT5",
        "INTERFERON_ALPHA_RESPONSE": "IFN-alpha",
        "INTERFERON_GAMMA_RESPONSE": "IFN-gamma",
        "INFLAMMATORY_RESPONSE": "Inflammatory",
        "UNFOLDED_PROTEIN_RESPONSE": "UPR",
        "REACTIVE_OXYGEN_SPECIES_PATHWAY": "ROS",
        "OXIDATIVE_PHOSPHORYLATION": "OxPhos",
        "FATTY_ACID_METABOLISM": "Fatty acid",
        "BILE_ACID_METABOLISM": "Bile acid",
        "CHOLESTEROL_HOMEOSTASIS": "Cholesterol",
        "XENOBIOTIC_METABOLISM": "Xenobiotic",
        "G2M_CHECKPOINT": "G2M",
        "E2F_TARGETS": "E2F",
        "MYC_TARGETS_V1": "MYC V1",
        "MYC_TARGETS_V2": "MYC V2",
        "MITOTIC_SPINDLE": "Mitotic",
        "WNT_BETA_CATENIN_SIGNALING": "WNT/beta-cat.",
        "TGF_BETA_SIGNALING": "TGF-beta",
        "PI3K_AKT_MTOR_SIGNALING": "PI3K/AKT/mTOR",
        "MTORC1_SIGNALING": "mTORC1",
        "KRAS_SIGNALING_UP": "KRAS up",
        "KRAS_SIGNALING_DN": "KRAS down",
        "EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
        "ESTROGEN_RESPONSE_EARLY": "Estrogen early",
        "ESTROGEN_RESPONSE_LATE": "Estrogen late",
        "PANCREAS_BETA_CELLS": "Beta cells",
    }
    return replacements.get(hallmark_id, hallmark_id.replace("_", " ").title())


def ordered_hallmarks(enrichment: pd.DataFrame) -> pd.DataFrame:
    base = (
        enrichment[["hallmark_id", "hallmark", "category"]]
        .drop_duplicates()
        .assign(category=lambda frame: pd.Categorical(frame["category"], categories=CATEGORY_ORDER, ordered=True))
        .sort_values(["category", "hallmark_id"])
        .reset_index(drop=True)
    )
    return base


def draw_region_hallmark_circle(
    ax,
    enrichment: pd.DataFrame,
    region: str,
    show_geneset_labels: bool = False,
    label_fontsize: float = 2.45,
) -> None:
    ordered = ordered_hallmarks(enrichment)
    n_items = ordered.shape[0]
    width = 2 * np.pi / n_items
    theta = np.arange(n_items) * width
    norm = mpl.colors.Normalize(vmin=-6, vmax=6)

    ring_height = 0.28
    ring_start = 0.70
    lookup = enrichment.set_index(["brain_region", "contrast", "hallmark_id"])

    colors = []
    sig_marks: list[tuple[float, float]] = []
    for idx, row in ordered.iterrows():
        value = float(lookup.loc[(region, "SCZ_vs_Control", row["hallmark_id"]), "enrichment_z"])
        colors.append(ENRICHMENT_CMAP(norm(np.clip(value, -6, 6))))
        fdr = float(lookup.loc[(region, "SCZ_vs_Control", row["hallmark_id"]), "fdr"])
        if fdr < 0.10:
            sig_marks.append((theta[idx] + width / 2, ring_start + ring_height / 2))
    ax.bar(
        theta + width / 2,
        ring_height,
        width=width * 0.96,
        bottom=ring_start,
        color=colors,
        edgecolor="white",
        linewidth=0.24,
    )
    if sig_marks:
        ax.scatter(
            [item[0] for item in sig_marks],
            [item[1] for item in sig_marks],
            s=5.2,
            color="black",
            marker="o",
            zorder=5,
        )

    for idx, row in ordered.iterrows():
        color = CATEGORY_COLORS[str(row["category"])]
        ax.bar(
            theta[idx] + width / 2,
            0.055,
            width=width * 0.98,
            bottom=1.025,
            color=color,
            edgecolor="white",
            linewidth=0.20,
        )

    if show_geneset_labels:
        for idx, row in ordered.iterrows():
            angle = theta[idx] + width / 2
            label = short_hallmark_label(str(row["hallmark_id"]))
            rotation = np.degrees(angle)
            ha = "left"
            if 90 < rotation < 270:
                rotation += 180
                ha = "right"
            ax.text(
                angle,
                1.155,
                label,
                rotation=rotation,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=label_fontsize,
                color="#222222",
                clip_on=False,
            )

    n_sig = int((enrichment.loc[(enrichment["brain_region"] == region) & (enrichment["contrast"] == "SCZ_vs_Control"), "fdr"] < 0.10).sum())
    ax.text(
        0.5,
        0.54,
        region,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.435,
        f"{n_sig} FDR<0.10\nHallmark sets",
        ha="center",
        va="center",
        fontsize=4.8,
        color="#333333",
        transform=ax.transAxes,
    )
    ax.set_ylim(0, 1.25 if show_geneset_labels else 1.14)
    ax.set_axis_off()


def significant_positive_enrichment(enrichment: pd.DataFrame) -> pd.DataFrame:
    score_col = "NES" if "NES" in enrichment.columns else "enrichment_z"
    data = enrichment.loc[
        (enrichment["contrast"] == "SCZ_vs_Control")
        & (enrichment["fdr"] < 0.10)
        & (enrichment[score_col] > 0)
    ].copy()
    data["plot_score"] = data[score_col].astype(float)
    data["short_label"] = data["hallmark_id"].map(short_hallmark_label)
    return data


def max_category_stack_total(enrichment: pd.DataFrame) -> float:
    data = significant_positive_enrichment(enrichment)
    if data.empty:
        return 1.0
    totals = data.groupby(["brain_region", "category"], observed=False)["plot_score"].sum()
    return float(totals.max())


def draw_region_category_stack(
    ax,
    enrichment: pd.DataFrame,
    region: str,
    x_max: float,
    show_ylabels: bool = True,
) -> None:
    data = significant_positive_enrichment(enrichment)
    region_data = data.loc[data["brain_region"] == region].copy()
    y = np.arange(len(CATEGORY_ORDER))

    for yi, category in zip(y, CATEGORY_ORDER):
        subset = (
            region_data.loc[region_data["category"] == category]
            .sort_values("plot_score", ascending=False)
            .reset_index(drop=True)
        )
        left = 0.0
        for _, row in subset.iterrows():
            width = float(row["plot_score"])
            ax.barh(
                yi,
                width,
                left=left,
                height=0.62,
                color=CATEGORY_COLORS[category],
                edgecolor="white",
                linewidth=0.30,
                alpha=0.92,
            )
            left += width
        if left > 0:
            ax.text(left + x_max * 0.015, yi, f"{left:.1f}", ha="left", va="center", fontsize=4.8, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(CATEGORY_ORDER if show_ylabels else [""] * len(CATEGORY_ORDER), fontsize=4.8)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_title(region, fontsize=6.8, pad=2)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.35, alpha=0.70)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=4.8, width=0.5, length=2)
    ax.tick_params(axis="y", width=0.5, length=2, pad=1.5)
    style_axis(ax)


def add_circular_legends(fig) -> None:
    cax = fig.add_axes([0.39, 0.002, 0.22, 0.010])
    sm = mpl.cm.ScalarMappable(cmap=ENRICHMENT_CMAP, norm=mpl.colors.Normalize(vmin=-6, vmax=6))
    fig.colorbar(sm, cax=cax, orientation="horizontal")
    cax.set_title("Hallmark enrichment z score", fontsize=5.6, pad=1)
    cax.tick_params(axis="x", labelsize=5.2, length=2, pad=1)
    fig.text(0.62, 0.008, "black dot: FDR<0.10", ha="left", va="center", fontsize=5.4, color="#333333")

    handles = [
        mpl.patches.Patch(facecolor=CATEGORY_COLORS[category], edgecolor="none", label=category)
        for category in CATEGORY_ORDER
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=4,
        fontsize=5.3,
        handlelength=1.1,
        columnspacing=1.0,
    )


def main() -> None:
    set_publication_style()
    gene_results = pd.read_csv(TABLE_DIR / "npj_figure1_gene_ma_results.tsv", sep="\t")
    enrichment = pd.read_csv(TABLE_DIR / "npj_figure1_hallmark_enrichment.tsv", sep="\t")

    fig = plt.figure(figsize=mm_to_inches(190, 350), constrained_layout=False)
    fig.text(
        0.5,
        0.985,
        "Diagnosis-specific transcriptome and Hallmark systems discovery",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(0.27, 0.935, "SCZ vs control", ha="center", va="center", fontsize=7.5, fontweight="bold")
    fig.text(0.73, 0.935, "BD psychosis vs control", ha="center", va="center", fontsize=7.5, fontweight="bold")
    y_positions = [0.765, 0.595, 0.425]
    for idx, (region, y_pos) in enumerate(zip(REGION_ORDER, y_positions)):
        ax_scz = fig.add_axes([0.07, y_pos, 0.40, 0.135])
        ax_bd = fig.add_axes([0.55, y_pos, 0.40, 0.135])
        draw_ma_plot(ax_scz, gene_results, "SCZ_vs_Control", region)
        draw_ma_plot(ax_bd, gene_results, "BD_psychosis_vs_Control", region)
        if region != REGION_ORDER[-1]:
            ax_scz.set_xlabel("")
            ax_bd.set_xlabel("")
        if idx != 1:
            ax_scz.set_ylabel("")
            ax_bd.set_ylabel("")

    fig.text(
        0.50,
        0.365,
        "SCZ Hallmark systems enrichment by striatal region",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    fig.text(
        0.50,
        0.346,
        "Each circle displays the same 50 Hallmark gene sets ordered by seven system classes.",
        ha="center",
        va="center",
        fontsize=5.7,
        color="#333333",
    )
    circle_axes = [
        [0.020, 0.145, 0.285, 0.205],
        [0.357, 0.145, 0.285, 0.205],
        [0.694, 0.145, 0.285, 0.205],
    ]
    for region, rect in zip(REGION_ORDER, circle_axes):
        ax_circ = fig.add_axes(rect, projection="polar")
        draw_region_hallmark_circle(
            ax_circ,
            enrichment,
            region,
            show_geneset_labels=True,
            label_fontsize=2.35,
        )

    fig.text(
        0.50,
        0.132,
        "Cumulative positive enrichment by Hallmark system class",
        ha="center",
        va="center",
        fontsize=7.0,
    )
    x_max = max_category_stack_total(enrichment) * 1.18
    d_axes = [
        [0.125, 0.042, 0.245, 0.070],
        [0.420, 0.042, 0.245, 0.070],
        [0.715, 0.042, 0.245, 0.070],
    ]
    for idx, (region, rect) in enumerate(zip(REGION_ORDER, d_axes)):
        ax_d = fig.add_axes(rect)
        draw_region_category_stack(ax_d, enrichment, region, x_max, show_ylabels=idx == 0)
    fig.text(
        0.54,
        0.029,
        "Cumulative positive enrichment score",
        ha="center",
        va="center",
        fontsize=5.5,
        color="#222222",
    )
    add_circular_legends(fig)

    panel_label(fig, "A", 0.035, 0.945)
    panel_label(fig, "B", 0.515, 0.945)
    panel_label(fig, "C", 0.035, 0.365)
    panel_label(fig, "D", 0.035, 0.128)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"npj_figure1_hallmark_systems_discovery.{ext}", dpi=600, bbox_inches="tight")

    NPJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_FIG_DIR / "Figure-01.png", dpi=600, bbox_inches="tight")
    fig.savefig(NPJ_FIG_DIR / "Figure-01.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote NPJ Figure 1 systems discovery.")


if __name__ == "__main__":
    main()
