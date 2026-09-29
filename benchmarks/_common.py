"""Shared helpers for publication benchmark scripts."""

from __future__ import annotations

import hashlib
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata() -> dict:
    """Describe repository provenance without restricting benchmark execution.

    A clean tree is exactly reproducible from ``commit``.  A dirty tree is still a
    valid benchmark source state, but reproducing it requires both the commit and
    the recorded working-tree changes.  Benchmark runners therefore *record* this
    distinction rather than refusing to run.
    """
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    try:
        # Do not use ``_git_output`` here: its whitespace stripping would destroy
        # the two-column porcelain status prefix (for example ``" M"``).
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).rstrip("\n")
    except (OSError, subprocess.CalledProcessError):
        status = None

    if status is None:
        return {
            "commit": commit,
            "branch": branch,
            "dirty": None,
            "source_state": "git_unavailable",
            "reproducibility": "unknown",
            "changed_paths": [],
            "diff_sha256": None,
        }

    status_lines = [line for line in status.splitlines() if line.strip()]
    dirty = bool(status_lines)
    changed_paths = []
    for line in status_lines:
        code = line[:2]
        path = line[3:] if len(line) > 3 else ""
        changed_paths.append({"status": code, "path": path})

    diff_sha256 = None
    if dirty:
        try:
            # ``git diff HEAD`` captures both staged and unstaged tracked changes.
            # Include untracked file contents as well so the fingerprint describes
            # the complete working-tree source state rather than only tracked edits.
            hasher = hashlib.sha256()
            diff = subprocess.check_output(
                ["git", "diff", "HEAD", "--binary"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            hasher.update(diff)
            untracked = _git_output("ls-files", "--others", "--exclude-standard")
            for rel in sorted(untracked.splitlines() if untracked else []):
                path = ROOT / rel
                hasher.update(rel.encode("utf-8", errors="surrogateescape"))
                hasher.update(b"\0")
                try:
                    hasher.update(path.read_bytes())
                except OSError:
                    hasher.update(b"<unreadable>")
                hasher.update(b"\0")
            diff_sha256 = hasher.hexdigest()
        except (OSError, subprocess.CalledProcessError):
            diff_sha256 = None

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "source_state": "working_tree_modified" if dirty else "exact_commit",
        "reproducibility": "working_tree_dependent" if dirty else "exact_commit",
        "changed_paths": changed_paths,
        "diff_sha256": diff_sha256,
    }


def environment_metadata() -> dict:
    try:
        import scipy
    except ImportError:  # pragma: no cover
        scipy = None
    try:
        import quartic2d
    except ImportError:  # pragma: no cover
        quartic2d = None
    try:
        import petal2d
    except ImportError:  # pragma: no cover
        petal2d = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": None if scipy is None else scipy.__version__,
        "quartic2d": getattr(quartic2d, "__version__", None),
        "petal2d": getattr(petal2d, "__version__", None),
        "git": git_metadata(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def timed_call(func, *, warmups: int, repeats: int):
    for _ in range(int(warmups)):
        func()
    samples = []
    value = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        value = func()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=float)
    return value, {
        "median_seconds": float(statistics.median(samples)),
        "q25_seconds": float(np.quantile(arr, 0.25)),
        "q75_seconds": float(np.quantile(arr, 0.75)),
        "samples_seconds": [float(x) for x in samples],
    }


def write_json(path: str | Path, data: dict) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
