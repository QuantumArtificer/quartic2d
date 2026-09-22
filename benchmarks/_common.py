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
    """Return the exact repository state associated with a benchmark result."""
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    dirty = None if status is None else bool(status)
    diff_sha256 = None
    if dirty:
        try:
            diff = subprocess.check_output(
                ["git", "diff", "HEAD", "--binary"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            diff_sha256 = hashlib.sha256(diff).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            diff_sha256 = None
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "diff_sha256": diff_sha256,
    }


def environment_metadata() -> dict:
    try:
        import scipy
    except Exception:  # pragma: no cover
        scipy = None
    try:
        import quartic2d
    except Exception:  # pragma: no cover
        quartic2d = None
    try:
        import petal2d
    except Exception:  # pragma: no cover
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
