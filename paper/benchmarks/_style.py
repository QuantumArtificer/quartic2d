"""Shared publication plotting style for Quartic2D paper figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

SINGLE_COLUMN = (3.35, 2.55)
DOUBLE_COLUMN = (7.0, 3.05)
DOUBLE_COLUMN_TALL = (7.0, 5.0)

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "savefig.dpi": 600,
            "figure.dpi": 120,
            "mathtext.fontset": "dejavusans",
        }
    )


def finish_axis(ax, *, legend: bool = True) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.55)
    if legend:
        ax.legend(frameon=False)


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
