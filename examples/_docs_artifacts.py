"""Private helpers used only when generating documentation artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from matplotlib import pyplot as plt

_ARTIFACT_ENV = "QUARTIC2D_DOC_FIGURE_DIR"


def save_plot(plotter, filename: str) -> None:
    """Save a convergence plot when documentation artifact capture is enabled."""
    directory = os.environ.get(_ARTIFACT_ENV)
    if not directory:
        return

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = plotter(title="")
    figure = result[0] if isinstance(result, tuple) else result
    figure.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(figure)
