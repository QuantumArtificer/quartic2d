"""Regenerate all tracked documentation evidence from one publication bundle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results",
        help="canonical consolidated publication result directory",
    )
    args = parser.parse_args()
    results = str(args.results_dir.resolve())

    _run(sys.executable, "docs/scripts/generate_figures.py", "validation", "--results-dir", results)
    _run(sys.executable, "docs/scripts/generate_evidence.py", "--results-dir", results)
    _run(sys.executable, "docs/scripts/check_artifacts.py", "--results-dir", results)


if __name__ == "__main__":
    main()
