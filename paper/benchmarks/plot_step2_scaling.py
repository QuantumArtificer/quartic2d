#!/usr/bin/env python3
"""Publication figures for the Step-2 HarmonicTransform scaling benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from _style import DOUBLE_COLUMN, DOUBLE_COLUMN_TALL, MARKERS, apply_style, finish_axis, save_figure
except ImportError:  # pragma: no cover
    from paper.benchmarks._style import DOUBLE_COLUMN, DOUBLE_COLUMN_TALL, MARKERS, apply_style, finish_axis, save_figure


AXIS_LABELS = {
    "N_q": r"$N_q$",
    "N_r": r"$N_r$",
    "N_m": r"$N_m$",
    "s_r": r"$s_r$",
}


def load(path):
    return json.loads(Path(path).read_text())


def rows_for(data, axis):
    return sorted([row for row in data["rows"] if row["axis"] == axis], key=lambda row: row["value"])


def panel(ax, data, axis):
    rr = rows_for(data, axis)
    x = np.asarray([row["value"] for row in rr], dtype=float)
    y = 1e3 * np.asarray([row["median_seconds"] for row in rr], dtype=float)
    q25 = 1e3 * np.asarray([row["q25_seconds"] for row in rr], dtype=float)
    q75 = 1e3 * np.asarray([row["q75_seconds"] for row in rr], dtype=float)
    ax.errorbar(x, y, yerr=np.vstack((y - q25, q75 - y)), marker="o", capsize=2, label="data")

    fit = data["fits"][axis]["asymptotic_tail"]
    if axis in ("N_r", "s_r"):
        fit_x = np.asarray([row["fit_x"] for row in rr], dtype=float)
        guide_x = fit_x
        # Plot against the displayed variable while evaluating the fitted
        # effective-node model at each point.
        guide = 1e3 * fit["prefactor"] * guide_x ** fit["alpha"]
    else:
        guide = 1e3 * fit["prefactor"] * x ** fit["alpha"]
    ax.plot(x, guide, "--", label=rf"$\alpha={fit['alpha']:.2f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(AXIS_LABELS[axis])
    ax.set_ylabel(r"$t$ (ms)")
    finish_axis(ax)


def figure_axis_scaling(data, outdir):
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COLUMN_TALL)
    for label, axis, ax in zip("abcd", ("N_q", "N_r", "N_m", "s_r"), axes.ravel()):
        panel(ax, data, axis)
        ax.text(0.03, 0.94, f"({label})", transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    save_figure(fig, Path(outdir) / "step2_harmonic_transform_scaling")


def figure_work_collapse(data, outdir):
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    for i, axis in enumerate(data.get("work_collapse_axes", ("N_q", "N_r", "s_r"))):
        rr = rows_for(data, axis)
        x = np.asarray([row["work_units"] for row in rr], dtype=float)
        y = 1e3 * np.asarray([row["median_seconds"] for row in rr], dtype=float)
        ax.plot(x, y, ls="none", marker=MARKERS[i], label=AXIS_LABELS[axis])

    fit = data["work_collapse_fit"]["asymptotic_tail"]
    axes = set(data.get("work_collapse_axes", ("N_q", "N_r", "s_r")))
    all_w = np.asarray([row["work_units"] for row in data["rows"] if row["axis"] in axes], dtype=float)
    xx = np.geomspace(all_w.min(), all_w.max(), 256)
    yy = 1e3 * fit["prefactor"] * xx ** fit["alpha"]
    ax.plot(xx, yy, "--", label=rf"$\alpha={fit['alpha']:.2f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{W}=N_mN_qN_s$")
    ax.set_ylabel(r"$t$ (ms)")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, Path(outdir) / "step2_harmonic_transform_work_collapse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="paper/benchmarks/results/step2_scaling.json")
    ap.add_argument("--output-dir", default="paper/benchmarks/results/figures")
    args = ap.parse_args()
    apply_style()
    data = load(args.input)
    figure_axis_scaling(data, args.output_dir)
    figure_work_collapse(data, args.output_dir)
    print(f"Figures: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
