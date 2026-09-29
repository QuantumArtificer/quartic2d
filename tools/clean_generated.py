#!/usr/bin/env python3
"""Remove generated QUARTIC2D artifacts without deleting Git-tracked files."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / ".pytest_cache",
    ROOT / "build",
    ROOT / "dist",
    ROOT / "docs" / "_build",
    ROOT / "benchmarks" / "results",
    ROOT / "validation",
]


def is_tracked(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def remove_untracked_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        if not is_tracked(path):
            path.unlink()
        return

    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            if not is_tracked(child):
                child.unlink()
        elif child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def main():
    for target in TARGETS:
        remove_untracked_tree(target)

    for pattern in ("*.egg-info", "__pycache__", ".ruff_cache", ".mypy_cache"):
        for path in ROOT.rglob(pattern):
            remove_untracked_tree(path)

    handoff_patterns = (
        "quartic2d_patch*.patch",
        "q2d_patch*.zip",
        "PATCH*_README.txt",
        "README.txt",
    )
    for pattern in handoff_patterns:
        for path in ROOT.glob(pattern):
            remove_untracked_tree(path)


if __name__ == "__main__":
    main()
