from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

REGION_ORDER = ["NAc", "Caudate", "Putamen"]
GROUP_ORDER = ["Control", "SCZ"]
BRAIN_CLASS_ORDER = ["MSN", "Interneuron", "Astrocyte", "Oligodendrocyte", "OPC", "Microglia", "Vascular"]
BRAIN_COLORS = {
    "MSN": "#4F86C6",
    "Interneuron": "#66C2A5",
    "Astrocyte": "#9B6BD3",
    "Oligodendrocyte": "#5FBF49",
    "OPC": "#8AF06A",
    "Microglia": "#D94873",
    "Vascular": "#F28E2B",
}


def panel_label(fig, label: str, x: float, y: float) -> None:
    fig.text(x, y, label, ha="left", va="top", fontsize=10, fontweight="bold")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    markers = pd.read_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_marker_counts.tsv", sep="\t")
    summary = pd.read_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_group_summary.tsv", sep="\t")
    tests = pd.read_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_group_tests.tsv", sep="\t")
    long = pd.read_csv(TABLE_DIR / "npj_figure4_hpa_brain_resident_nnls_weights_long.tsv", sep="\t")
    return markers, summary, tests, long


def draw_reference_panel(ax, markers: pd.DataFrame) -> None:
    data = markers.set_index("brain_cell_class").reindex(BRAIN_CLASS_ORDER).reset_index()
    ax.barh(
        np.arange(len(data)),
        data["available_marker_count"],
        color=[BRAIN_COLORS[item] for item in data["brain_cell_class"]],
        edgecolor="none",
    )
    for y, row in data.iterrows():
        ax.text(
            row["available_marker_count"] + 1.5,
            y,
            str(int(row["available_marker_count"])),
            ha="left",
            va="center",
            fontsize=5.2,
            color="#333333",
        )
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["brain_cell_class"])
    ax.invert_yaxis()
    ax.set_xlabel("Available HPA/Siletti markers")
    ax.set_title("Pure brain snRNA reference classes", fontsize=7.0, pad=3)
    ax.set_xlim(0, max(data["available_marker_count"]) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.2, width=0.5, length=2)


def draw_stacked_panel(ax, summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(
        index=["brain_region", "group"],
        columns="brain_cell_class",
        values="mean",
        fill_value=0,
        observed=True,
    ).reindex(pd.MultiIndex.from_product([REGION_ORDER, GROUP_ORDER], names=["brain_region", "group"]))
    positions = np.array([0, 0.82, 2.15, 2.97, 4.30, 5.12])
    bottoms = np.zeros(len(pivot), dtype=float)
    for brain_class in BRAIN_CLASS_ORDER:
        values = pivot.get(brain_class, pd.Series(0, index=pivot.index)).to_numpy(dtype=float)
        ax.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=BRAIN_COLORS[brain_class],
            edgecolor="white",
            linewidth=0.18,
            label=brain_class,
        )
        bottoms += values
    ax.set_xticks(positions)
    ax.set_xticklabels(["Control", "SCZ"] * 3)
    for center, region in zip([0.41, 2.56, 4.71], REGION_ORDER):
        ax.text(center, 1.035, region, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean NNLS reference weight")
    ax.set_title("Estimated brain-resident composition", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.3, width=0.5, length=2)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.03), loc="upper left", fontsize=5.2, labelspacing=0.35)


def draw_effect_heatmap(ax, tests: pd.DataFrame) -> None:
    matrix = tests.pivot(index="brain_cell_class", columns="brain_region", values="mean_difference_scz_minus_control")
    matrix = matrix.reindex(index=BRAIN_CLASS_ORDER, columns=REGION_ORDER)
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "SCZ-control mean weight difference", "shrink": 0.78},
        ax=ax,
    )
    lookup = tests.set_index(["brain_cell_class", "brain_region"])
    for y, brain_class in enumerate(BRAIN_CLASS_ORDER):
        for x, region in enumerate(REGION_ORDER):
            row = lookup.loc[(brain_class, region)]
            if row["fdr_region"] < 0.10:
                ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", fontsize=7.0, color="black")
            elif row["p_value"] < 0.05:
                ax.text(x + 0.5, y + 0.5, "+", ha="center", va="center", fontsize=6.0, color="black")
    ax.set_title("SCZ-control composition-weight differences", fontsize=7.0, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=5.5, width=0.5, length=2)


def draw_global_violin_panel(fig, long: pd.DataFrame, tests: pd.DataFrame, y0: float) -> None:
    test_lookup = tests.set_index(["brain_region", "brain_cell_class"])
    x_positions = [0.070, 0.365, 0.660]
    width = 0.260
    height = 0.135
    for region_idx, (region, x0) in enumerate(zip(REGION_ORDER, x_positions)):
        ax = fig.add_axes([x0, y0, width, height])
        plot_df = long.loc[
            (long["brain_region"] == region)
            & (long["brain_cell_class"].isin(BRAIN_CLASS_ORDER))
            & (long["group"].isin(GROUP_ORDER))
        ].copy()
        sns.violinplot(
            data=plot_df,
            x="brain_cell_class",
            y="weight",
            hue="group",
            order=BRAIN_CLASS_ORDER,
            hue_order=GROUP_ORDER,
            palette={"Control": "#4F86C6", "SCZ": "#D94873"},
            cut=0,
            inner=None,
            linewidth=0.35,
            width=0.90,
            ax=ax,
        )
        sns.boxplot(
            data=plot_df,
            x="brain_cell_class",
            y="weight",
            hue="group",
            order=BRAIN_CLASS_ORDER,
            hue_order=GROUP_ORDER,
            palette={"Control": "#4F86C6", "SCZ": "#D94873"},
            width=0.34,
            dodge=True,
            showfliers=False,
            showcaps=True,
            linewidth=0.55,
            boxprops={"facecolor": "white", "alpha": 0.70, "zorder": 3},
            medianprops={"color": "#111111", "linewidth": 0.65, "zorder": 4},
            whiskerprops={"color": "#333333", "linewidth": 0.45, "zorder": 4},
            capprops={"color": "#333333", "linewidth": 0.45, "zorder": 4},
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="brain_cell_class",
            y="weight",
            hue="group",
            order=BRAIN_CLASS_ORDER,
            hue_order=GROUP_ORDER,
            dodge=True,
            jitter=0.18,
            palette={"Control": "#4F86C6", "SCZ": "#D94873"},
            size=1.15,
            alpha=0.34,
            linewidth=0.12,
            edgecolor="white",
            zorder=5,
            ax=ax,
        )
        for violin in ax.collections:
            if isinstance(violin, mpl.collections.PolyCollection):
                violin.set_alpha(0.60)
        handles, labels = ax.get_legend_handles_labels()
        if region_idx == 2:
            ax.legend(handles[:2], labels[:2], frameon=False, fontsize=4.8, loc="upper right")
        else:
            ax.get_legend().remove()
        ymax = max(float(plot_df["weight"].max()), 0.01)
        ax.set_ylim(-0.015, ymax * 1.20)
        for class_idx, brain_class in enumerate(BRAIN_CLASS_ORDER):
            row = test_lookup.loc[(region, brain_class)]
            label = "*" if row["fdr_region"] < 0.10 else "+" if row["p_value"] < 0.05 else ""
            if label:
                class_max = plot_df.loc[plot_df["brain_cell_class"] == brain_class, "weight"].max()
                ax.text(class_idx, class_max + ymax * 0.045, label, ha="center", va="bottom", fontsize=6.4)
        ax.set_title(region, fontsize=6.4, pad=2.0, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("NNLS reference weight" if region_idx == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=4.8, width=0.5, length=2)
        ax.tick_params(axis="x", labelsize=4.4, width=0.5, length=2, rotation=42)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.text(0.5, y0 + height + 0.020, "Whole-panel distribution of brain-resident reference weights", ha="center", fontsize=7.0)


def draw_representative_boxplots(ax, long: pd.DataFrame, tests: pd.DataFrame) -> None:
    selected = [
        ("NAc", "Astrocyte"),
        ("NAc", "Oligodendrocyte"),
        ("Caudate", "OPC"),
        ("Putamen", "OPC"),
        ("Putamen", "Oligodendrocyte"),
        ("Putamen", "MSN"),
    ]
    rows = []
    test_lookup = tests.set_index(["brain_region", "brain_cell_class"])
    for idx, (region, brain_class) in enumerate(selected):
        subset = long.loc[
            (long["brain_region"] == region)
            & (long["brain_cell_class"] == brain_class)
            & (long["group"].isin(GROUP_ORDER))
        ].copy()
        subset["panel"] = f"{brain_class}\n{region}"
        subset["panel_order"] = idx
        rows.append(subset)
    plot_df = pd.concat(rows, ignore_index=True)
    sns.boxplot(
        data=plot_df,
        x="panel",
        y="weight",
        hue="group",
        order=[f"{brain_class}\n{region}" for region, brain_class in selected],
        hue_order=GROUP_ORDER,
        palette={"Control": "#4F86C6", "SCZ": "#D94873"},
        width=0.72,
        fliersize=0,
        linewidth=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="panel",
        y="weight",
        hue="group",
        order=[f"{brain_class}\n{region}" for region, brain_class in selected],
        hue_order=GROUP_ORDER,
        dodge=True,
        palette={"Control": "#4F86C6", "SCZ": "#D94873"},
        size=1.6,
        alpha=0.35,
        linewidth=0,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, fontsize=5.4, loc="upper right")
    ymax = plot_df["weight"].max()
    for idx, (region, brain_class) in enumerate(selected):
        row = test_lookup.loc[(region, brain_class)]
        label = "*" if row["fdr_region"] < 0.10 else "+" if row["p_value"] < 0.05 else ""
        if label:
            y = plot_df.loc[plot_df["panel_order"] == idx, "weight"].max() + ymax * 0.035
            ax.text(idx, y, label, ha="center", va="bottom", fontsize=7.0, color="black")
    ax.set_xlabel("")
    ax.set_ylabel("NNLS reference weight")
    ax.set_title("Representative region-specific shifts", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=5.0, width=0.5, length=2)
    ax.tick_params(axis="y", labelsize=5.2, width=0.5, length=2)


def draw_putamen_rank_panel(ax, tests: pd.DataFrame) -> None:
    data = tests.loc[tests["brain_region"] == "Putamen"].copy()
    data = data.set_index("brain_cell_class").reindex(BRAIN_CLASS_ORDER).reset_index()
    ax.axvline(0, color="#777777", lw=0.45)
    y = np.arange(len(data))
    ax.barh(
        y,
        data["mean_difference_scz_minus_control"],
        color=[
            BRAIN_COLORS[item] if np.isfinite(value) else "#CCCCCC"
            for item, value in zip(data["brain_cell_class"], data["mean_difference_scz_minus_control"])
        ],
        edgecolor="none",
    )
    for idx, row in data.iterrows():
        label = "*" if row["fdr_region"] < 0.10 else "+" if row["p_value"] < 0.05 else ""
        if label:
            x = row["mean_difference_scz_minus_control"]
            ax.text(x * 0.55, idx, label, ha="center", va="center", fontsize=6.5, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(data["brain_cell_class"])
    ax.invert_yaxis()
    ax.set_xlabel("SCZ-control mean weight difference")
    ax.set_title("Putamen composition signature", fontsize=7.0, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=5.2, width=0.5, length=2)


def main() -> None:
    set_publication_style()
    markers, summary, tests, long = load_tables()
    fig = plt.figure(figsize=mm_to_inches(190, 285), constrained_layout=False)
    fig.text(
        0.5,
        0.985,
        "Pure brain single-nucleus reference resolves brain-resident composition in striatal bulk RNA-seq",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax_a = fig.add_axes([0.08, 0.735, 0.27, 0.210])
    draw_reference_panel(ax_a, markers)
    ax_b = fig.add_axes([0.45, 0.735, 0.37, 0.210])
    draw_stacked_panel(ax_b, summary)
    draw_global_violin_panel(fig, long, tests, y0=0.425)
    ax_d = fig.add_axes([0.13, 0.150, 0.34, 0.165])
    draw_effect_heatmap(ax_d, tests)
    ax_e = fig.add_axes([0.58, 0.150, 0.30, 0.165])
    draw_putamen_rank_panel(ax_e, tests)

    fig.text(
        0.5,
        0.070,
        "+ nominal P<0.05; * region-level FDR<0.10 by covariate-adjusted OLS on arcsine-transformed NNLS weights",
        ha="center",
        fontsize=5.4,
    )
    fig.text(
        0.5,
        0.050,
        "Reference: Human Protein Atlas/Siletti human brain single-nucleus cluster-type expression; values are normalized NNLS reference weights.",
        ha="center",
        fontsize=5.1,
        color="#333333",
    )

    panel_label(fig, "A", 0.035, 0.955)
    panel_label(fig, "B", 0.395, 0.955)
    panel_label(fig, "C", 0.035, 0.645)
    panel_label(fig, "D", 0.035, 0.330)
    panel_label(fig, "E", 0.510, 0.330)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"npj_figure4_hpa_brain_resident.{ext}", dpi=600, bbox_inches="tight")

    NPJ_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(NPJ_FIG_DIR / "Figure-04.png", dpi=600, bbox_inches="tight")
    fig.savefig(NPJ_FIG_DIR / "Figure-04.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Wrote NPJ Figure 4 HPA/Siletti brain-resident composition figure.")


if __name__ == "__main__":
    main()
