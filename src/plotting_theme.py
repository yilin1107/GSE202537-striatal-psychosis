from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.font_manager as fm


MM_PER_INCH = 25.4


GROUP_ORDER = ["Control", "SCZ", "BD with psychosis"]

GROUP_COLORS = {
    "Control": "#4E79A7",
    "SCZ": "#E15759",
    "BD with psychosis": "#59A14F",
}

REGION_ORDER = ["NAc", "Caudate", "Putamen"]

REGION_COLORS = {
    "NAc": "#76B7B2",
    "Caudate": "#F28E2B",
    "Putamen": "#EDC948",
}

SEX_COLORS = {
    "Female": "#B07AA1",
    "Male": "#4E79A7",
}


def mm_to_inches(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def choose_font(preferred: Iterable[str] = ("Arial", "Helvetica", "DejaVu Sans")) -> str:
    available = {Path(font.fname).stem.lower(): font.name for font in fm.fontManager.ttflist}
    for family in preferred:
        if family.lower() in {name.lower() for name in available.values()}:
            return family
    return "DejaVu Sans"


def set_publication_style() -> str:
    family = choose_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.title_fontsize": 7,
            "figure.titlesize": 10,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )
    return family


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def story_step(fig, step: str, title: str, detail: str = "", x: float = 0.5, y: float = 0.985) -> None:
    text = f"{step}: {title}" if step else title
    if detail:
        text = f"{text} | {detail}"
    fig.text(
        x,
        y,
        text,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.08",
            "facecolor": "#F4F4F4",
            "edgecolor": "#BDBDBD",
            "linewidth": 0.45,
        },
    )
