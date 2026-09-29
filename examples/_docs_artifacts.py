"""Private helpers used only when generating documentation artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from matplotlib import pyplot as plt

_ARTIFACT_ENV = "QUARTIC2D_DOC_FIGURE_DIR"
_SVG_HASH_SALT = "quartic2d-docs"


def save_plot(plotter, filename: str) -> None:
    """Save a convergence plot when documentation artifact capture is enabled."""
    directory = os.environ.get(_ARTIFACT_ENV)
    if not directory:
        return

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    with plt.rc_context({"svg.hashsalt": _SVG_HASH_SALT}):
        result = plotter(title="")
        figure = result[0] if isinstance(result, tuple) else result
        figure.savefig(output_path, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)

    if output_path.suffix.lower() == ".svg":
        text = "\n".join(
            line.rstrip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ) + "\n"
        output_path.write_text(text, encoding="utf-8")
