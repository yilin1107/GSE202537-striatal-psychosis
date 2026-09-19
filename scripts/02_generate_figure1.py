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
FIGURE_DIRS = {
    "pdf": ROOT / "results" / "figures" / "pdf",
    "svg": ROOT / "results" / "figures" / "svg",
    "png": ROOT / "results" / "figures" / "png",
}

DISPLAY_LABELS = {
    "Control": "Control",
    "SCZ": "SCZ",
    "BD with psychosis": "BD psychosis",
}


def p_value_for(variable: str) -> str:
    tests = pd.read_csv(ROOT / "results" / "tables" / "figure1_clinical_tests.tsv", sep="\t", dtype=str)
    row = tests.loc[tests["variable"] == variable]
    if row.empty:
        return ""
    return f"P = {row.iloc[0]['p_value']}"


def add_p_value(ax, variable: str) -> None:
    ax.text(
        0.03,
        0.96,
        p_value_for(variable),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )


def style_axis(ax) -> None:
    ax.tick_params(axis="both", width=0.6, length=2.5)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.6)


def draw_sample_counts(ax, samples: pd.DataFrame) -> None:
    counts = (
        samples.groupby(["brain_region", "group"])
        .size()
        .reset_index(name="n")
        .pivot(index="brain_region", columns="group", values="n")
        .reindex(REGION_ORDER)
        .fillna(0)
    )
    x = np.arange(len(REGION_ORDER))
    width = 0.22
    offsets = np.linspace(-width, width, len(GROUP_ORDER))
    for offset, group in zip(offsets, GROUP_ORDER):
        values = counts[group].values if group in counts.columns else np.zeros(len(REGION_ORDER))
        ax.bar(
            x + offset,
            values,
            width=width,
            color=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=0.4,
            label=DISPLAY_LABELS[group],
        )
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.8, str(int(value)), ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER)
    ax.set_ylabel("Samples")
    ax.set_title("Brain-region samples")
    ax.set_ylim(0, max(counts.max()) + 10)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.30, 1.00),
        borderaxespad=0.0,
        handlelength=1.1,
        labelspacing=0.25,
    )
    style_axis(ax)


def draw_donor_counts(ax, subjects: pd.DataFrame) -> None:
    counts = subjects["group"].value_counts().reindex(GROUP_ORDER).fillna(0)
    bars = ax.bar(
        GROUP_ORDER,
        counts.values,
        color=[GROUP_COLORS[group] for group in GROUP_ORDER],
        edgecolor="black",
        linewidth=0.4,
    )
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, str(int(value)), ha="center", fontsize=6)
    ax.set_ylabel("Donors")
    ax.set_title("Donors by group")
    ax.set_ylim(0, max(counts.values) + 6)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(["Control", "SCZ", "BD\npsychosis"])
    style_axis(ax)


def box_swarm(ax, subjects: pd.DataFrame, column: str, ylabel: str, title: str, test_label: str) -> None:
    sns.boxplot(
        data=subjects,
        x="group",
        y=column,
        hue="group",
        order=GROUP_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        width=0.52,
        fliersize=0,
        linewidth=0.6,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=subjects,
        x="group",
        y=column,
        order=GROUP_ORDER,
        color="black",
        size=2.5,
        alpha=0.65,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(["Control", "SCZ", "BD\npsychosis"])
    add_p_value(ax, test_label)
    style_axis(ax)


def draw_sex_fraction(ax, subjects: pd.DataFrame) -> None:
    table = pd.crosstab(subjects["group"], subjects["sex"]).reindex(GROUP_ORDER).fillna(0)
    fractions = table.div(table.sum(axis=1), axis=0) * 100
    bottoms = np.zeros(len(GROUP_ORDER))
    x = np.arange(len(GROUP_ORDER))
    for sex in ["Female", "Male"]:
        values = fractions[sex].values if sex in fractions.columns else np.zeros(len(GROUP_ORDER))
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color={"Female": "#B07AA1", "Male": "#4E79A7"}[sex],
            edgecolor="black",
            linewidth=0.4,
            label=sex,
        )
        raw_counts = table[sex].values if sex in table.columns else np.zeros(len(GROUP_ORDER))
        for xpos, bottom, value, count in zip(x, bottoms, values, raw_counts):
            if value >= 12:
                ax.text(xpos, bottom + value / 2, str(int(count)), ha="center", va="center", fontsize=6, color="white")
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels(["Control", "SCZ", "BD\npsychosis"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Donors (%)")
    ax.set_title("Sex")
    add_p_value(ax, "Sex")
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02),
        borderaxespad=0.0,
        handlelength=1.0,
        labelspacing=0.25,
    )
    style_axis(ax)


def draw_time_of_death(ax, subjects: pd.DataFrame) -> None:
    rng = np.random.default_rng(202537)
    y_positions = {"Control": 2, "SCZ": 1, "BD with psychosis": 0}
    for group in GROUP_ORDER:
        values = subjects.loc[subjects["group"] == group, "corrected_tod_24h"].dropna()
        jitter = rng.normal(0, 0.055, len(values))
        ax.scatter(
            values,
            np.full(len(values), y_positions[group]) + jitter,
            s=18,
            color=GROUP_COLORS[group],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.25,
        )
    for x_value in [6, 12, 18]:
        ax.axvline(x_value, color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["Control", "SCZ", "BD\npsychosis"])
    ax.set_xlabel("Corrected time of death (h)")
    ax.set_title("Diurnal coverage")
    add_p_value(ax, "Corrected time of death")
    style_axis(ax)


def main() -> None:
    set_publication_style()
    samples = pd.read_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv")
    subjects = pd.read_csv(PROCESSED_DIR / "gse202537_subject_metadata.csv")

    fig, axes = plt.subplots(
        2,
        4,
        figsize=mm_to_inches(180, 130),
        constrained_layout=True,
    )
    axes = axes.ravel()

    draw_sample_counts(axes[0], samples)
    draw_donor_counts(axes[1], subjects)
    box_swarm(axes[2], subjects, "age", "Age (years)", "Age", "Age")
    draw_sex_fraction(axes[3], subjects)
    box_swarm(axes[4], subjects, "pmi", "PMI (h)", "Postmortem interval", "Postmortem interval")
    box_swarm(axes[5], subjects, "rin", "RIN", "RNA integrity", "RNA integrity number")
    box_swarm(axes[6], subjects, "ph", "pH", "Brain pH", "Brain pH")
    draw_time_of_death(axes[7], subjects)

    for label, ax in zip(list("ABCDEFGH"), axes):
        panel_label(ax, label)

    for output_dir in FIGURE_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    for ext, output_dir in FIGURE_DIRS.items():
        fig.savefig(output_dir / f"figure1_clinical_overview.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Wrote Figure 1 clinical overview.")


if __name__ == "__main__":
    main()
