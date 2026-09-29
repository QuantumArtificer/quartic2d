"""Execute guided examples and capture terminal output and convergence plots."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples"
OUTPUT_DIR = ROOT / "docs" / "source" / "_generated" / "examples"
FIGURE_DIR = ROOT / "docs" / "source" / "_static" / "examples"

EXAMPLES = {
    "isotropic_interaction": "isotropic_interaction.py",
    "direct_exchange": "four_center_interaction.py",
    "anisotropic_interaction": "anisotropic_interaction.py",
    "sampled_data": "sampled_data.py",
    "complex_transition_field": "complex_transition_field.py",
    "screening_family_sweep": "screening_family_sweep.py",
    "large_separation": "large_separation.py",
}


def _run_example(name: str, filename: str) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["QUARTIC2D_DOC_FIGURE_DIR"] = str(FIGURE_DIR)

    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / filename)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        raise RuntimeError(f"Guided example failed: {filename}")

    (OUTPUT_DIR / f"{name}.txt").write_text(completed.stdout)
    print(f"docs/source/_generated/examples/{name}.txt")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, filename in EXAMPLES.items():
        _run_example(name, filename)


if __name__ == "__main__":
    main()
