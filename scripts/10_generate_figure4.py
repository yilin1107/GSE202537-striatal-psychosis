from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting_theme import GROUP_COLORS, REGION_ORDER, mm_to_inches, set_publication_style  # noqa: E402


PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
ASSET_DIR = ROOT / "assets"
FIGURE4D_IMAGE = ASSET_DIR / "figure4d_model.png"
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
EFFECT_CMAP = sns.diverging_palette(240, 10, as_cmap=True)


def figure_panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def draw_signature_heatmap(ax, results: pd.DataFrame) -> None:
    beta = (
        results.pivot(index="signature", columns="brain_region", values="beta_psychosis")
        .reindex(index=SIGNATURE_ORDER, columns=REGION_ORDER)
    )
    fdr = (
        results.pivot(index="signature", columns="brain_region", values="fdr")
        .reindex(index=SIGNATURE_ORDER, columns=REGION_ORDER)
    )
    sns.heatmap(
        beta,
        ax=ax,
        cmap=EFFECT_CMAP,
        center=0,
        vmin=-0.50,
        vmax=0.50,
        linewidths=0,
        cbar=False,
    )
    ax.set_title("Directional molecular signature effects")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(REGION_ORDER)) + 0.5)
    ax.set_xticklabels(REGION_ORDER, rotation=0)
    ax.set_yticks(np.arange(len(SIGNATURE_ORDER)) + 0.5)
    ax.set_yticklabels([SIGNATURE_LABELS[item] for item in SIGNATURE_ORDER], rotation=0)
    for y_idx, signature in enumerate(SIGNATURE_ORDER):
        for x_idx, region in enumerate(REGION_ORDER):
            value = fdr.loc[signature, region]
            if np.isfinite(value) and value < 0.10:
                ax.text(x_idx + 0.5, y_idx + 0.5, "*", ha="center", va="center", fontsize=8, color="black")


def selected_score_records(scores: pd.DataFrame, metadata: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    selected = [
        ("Putamen", "Dopamine-adenosine dysregulation", "Dopamine\nPutamen"),
        ("Putamen", "Activin-SMAD remodeling", "SMAD\nPutamen"),
        ("Caudate", "Glutamate-calcium stress", "Glutamate\nCaudate"),
        ("NAc", "Immune-NF-kB cytokine tone", "Immune\nNAc"),
    ]
    rows: list[dict[str, object]] = []
    for region, signature, label in selected:
        region_meta = metadata.loc[metadata["brain_region"] == region]
        for _, sample in region_meta.iterrows():
            sample_key = sample["sample_key"]
            rows.append(
                {
                    "label": label,
                    "signature": signature,
                    "brain_region": region,
                    "group": "Psychosis" if sample["group"] != "Control" else "Control",
                    "score": float(scores.loc[signature, sample_key]),
                }
            )
    plot_df = pd.DataFrame(rows)
    q_lookup = results.set_index(["brain_region", "signature"])["fdr"].to_dict()
    plot_df["q_value"] = plot_df.apply(lambda row: q_lookup[(row["brain_region"], row["signature"])], axis=1)
    return plot_df


def draw_score_distributions(ax, scores: pd.DataFrame, metadata: pd.DataFrame, results: pd.DataFrame) -> None:
    plot_df = selected_score_records(scores, metadata, results)
    palette = {"Control": GROUP_COLORS["Control"], "Psychosis": GROUP_COLORS["SCZ"]}
    sns.boxplot(
        data=plot_df,
        x="label",
        y="score",
        hue="group",
        hue_order=["Control", "Psychosis"],
        palette=palette,
        width=0.62,
        fliersize=0,
        linewidth=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="label",
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
    ax.legend(handles[:2], labels[:2], frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.axhline(0, color="#BDBDBD", linewidth=0.6)
    ax.set_title("Representative significant signatures")
    ax.set_xlabel("")
    ax.set_ylabel("Directional score")
    ax.tick_params(axis="x", labelsize=5.8)
    y_top = plot_df["score"].max()
    y_bottom = plot_df["score"].min()
    y_range = y_top - y_bottom
    for x_idx, (label, group) in enumerate(plot_df.groupby("label", sort=False)):
        q_value = group["q_value"].iloc[0]
        text = f"q={q_value:.2f}" if q_value >= 0.01 else "q<0.01"
        ax.text(x_idx, y_top + y_range * 0.05, text, ha="center", va="bottom", fontsize=5.6)
    ax.set_ylim(y_bottom - y_range * 0.06, y_top + y_range * 0.17)


def draw_region_vulnerability(ax, summary: pd.DataFrame) -> None:
    plot_df = summary.set_index("brain_region").reindex(REGION_ORDER).reset_index()
    colors = ["#C9C9C9" if n == 0 else "#E15759" for n in plot_df["n_fdr_0_10"]]
    ax.barh(plot_df["brain_region"], plot_df["mean_positive_beta"], color=colors, height=0.55)
    for _, row in plot_df.iterrows():
        ax.text(
            row["mean_positive_beta"] + 0.015,
            row["brain_region"],
            f"{int(row['n_fdr_0_10'])} q<0.10",
            va="center",
            ha="left",
            fontsize=6,
        )
    ax.set_xlim(0, max(0.45, plot_df["mean_positive_beta"].max() + 0.12))
    ax.set_xlabel("Mean positive beta")
    ax.set_ylabel("")
    ax.set_title("Regional convergence index")
    ax.invert_yaxis()
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def draw_model_box(ax, xy: tuple[float, float], width: float, height: float, color: str, title: str, text: str) -> None:
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=color,
        facecolor=color,
        alpha=0.13,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.68, title, ha="center", va="center", fontsize=6.7, fontweight="bold", color=color)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.33, text, ha="center", va="center", fontsize=5.8, color="black")


def draw_mechanism_model(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ((0.03, 0.70), "Activin-SMAD", "INHBA, SMAD1 up\nSMURF1 down", "Activin-SMAD remodeling"),
        ((0.67, 0.70), "Dopamine-adenosine", "TH down\nADK/ADORA shift", "Dopamine-adenosine dysregulation"),
        ((0.03, 0.10), "Glutamate-calcium", "GRM8, XBP1 up\nGNAO1 down", "Glutamate-calcium stress"),
        ((0.67, 0.10), "APP processing", "PSEN1, KAT5 down\nprocessing shift", "APP processing shift"),
        ((0.35, 0.08), "Immune-NF-kB", "NFKBIA, CEBPB,\nCXCL8 tone", "Immune-NF-kB cytokine tone"),
    ]
    arrow_targets = {
        "Activin-SMAD remodeling": (0.42, 0.54),
        "Dopamine-adenosine dysregulation": (0.58, 0.54),
        "Glutamate-calcium stress": (0.42, 0.42),
        "Immune-NF-kB cytokine tone": (0.50, 0.39),
        "APP processing shift": (0.58, 0.42),
    }
    for xy, title, text, signature in boxes:
        draw_model_box(ax, xy, 0.27, 0.17, SIGNATURE_COLORS[signature], title, text)
        start = (xy[0] + 0.135, xy[1] + 0.085)
        end = arrow_targets[signature]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": SIGNATURE_COLORS[signature], "linewidth": 1.0, "alpha": 0.75},
        )
    draw_model_box(ax, (0.35, 0.38), 0.30, 0.18, "#333333", "Striatal psychosis state", "Putamen > Caudate\nsignature convergence")
    ax.text(0.5, 0.95, "Proposed molecular basis model", ha="center", va="center", fontsize=8)


def draw_external_model_image(ax) -> None:
    image = mpimg.imread(FIGURE4D_IMAGE)
    ax.imshow(image)
    ax.set_axis_off()


def main() -> None:
    set_publication_style()
    results = pd.read_csv(TABLE_DIR / "figure4_molecular_signature_results.tsv", sep="\t")
    summary = pd.read_csv(TABLE_DIR / "figure4_region_vulnerability_summary.tsv", sep="\t")
    scores = pd.read_csv(PROCESSED_DIR / "gse202537_molecular_signature_scores.csv", index_col=0)
    sample_metadata = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")

    fig = plt.figure(figsize=mm_to_inches(180, 185), constrained_layout=False)
    ax_heatmap = fig.add_axes([0.17, 0.64, 0.32, 0.25])
    draw_signature_heatmap(ax_heatmap, results)
    cbar_ax = fig.add_axes([0.20, 0.59, 0.22, 0.012])
    sm = plt.cm.ScalarMappable(cmap=EFFECT_CMAP, norm=plt.Normalize(-0.50, 0.50))
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_title("Signature beta", fontsize=6.2, pad=2)
    cbar_ax.tick_params(axis="x", labelsize=5.8, pad=1, length=2)

    ax_scores = fig.add_axes([0.58, 0.64, 0.31, 0.25])
    draw_score_distributions(ax_scores, scores, sample_metadata, results)

    ax_vulnerability = fig.add_axes([0.11, 0.25, 0.28, 0.23])
    draw_region_vulnerability(ax_vulnerability, summary)

    ax_model = fig.add_axes([0.45, 0.17, 0.50, 0.40])
    draw_external_model_image(ax_model)

    figure_panel_label(fig, "A", 0.045, 0.935)
    figure_panel_label(fig, "B", 0.515, 0.935)
    figure_panel_label(fig, "C", 0.045, 0.52)
    figure_panel_label(fig, "D", 0.405, 0.53)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"figure4_molecular_signature_model.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote Figure 4 molecular signature model.")


if __name__ == "__main__":
    main()
