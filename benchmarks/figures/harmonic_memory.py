#!/usr/bin/env python3
"""Plot peak-memory scaling benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.figures._style import DOUBLE_COLUMN, apply_style, finish_axis, save_figure
MIB = 1024.0**2


def values(rows, axis):
    rr = sorted((row for row in rows if row["axis"] == axis), key=lambda row: row["value"])
    x = np.asarray([row["value"] if axis == "N_q" else row["N_s"] for row in rr], dtype=float)
    y = np.asarray([row["incremental_peak_rss_median_bytes"] / MIB for row in rr])
    lo = np.asarray([row["incremental_peak_rss_q25_bytes"] / MIB for row in rr])
    hi = np.asarray([row["incremental_peak_rss_q75_bytes"] / MIB for row in rr])
    return rr, x, y, np.vstack((y - lo, hi - y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmarks/results/publication/harmonic_memory.json")
    parser.add_argument("--output-dir", default="benchmarks/results/publication/figures")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    rows = data["rows"]
    B = int(data["settings"]["B"])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)

    _, xq, yq, eq = values(rows, "N_q")
    ax = axes[0]
    ax.errorbar(xq, yq, yerr=eq, fmt="o-", capsize=2)
    ax.axvline(B, linestyle="--", linewidth=1.0, label=rf"$B={B}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$N_q$")
    ax.set_ylabel(r"$\Delta M_{\rm RSS}$ (MiB)")
    finish_axis(ax, legend=True)

    _, xr, yr, er = values(rows, "N_r")
    ax = axes[1]
    ax.errorbar(xr, yr, yerr=er, fmt="o-", capsize=2, label="data")
    fit = data["fits"]["N_r"]
    xx = np.geomspace(xr.min(), xr.max(), 200)
    yy = (fit["intercept_bytes"] + fit["slope_bytes_per_x"] * xx) / MIB
    ax.plot(xx, yy, linestyle="--", label=rf"$R^2={fit['r2']:.4f}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"$N_s$")
    ax.set_ylabel(r"$\Delta M_{\rm RSS}$ (MiB)")
    finish_axis(ax, legend=True)

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, Path(args.output_dir) / "harmonic_transform_memory_scaling")


if __name__ == "__main__":
    main()
