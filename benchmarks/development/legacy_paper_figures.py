#!/usr/bin/env python3
"""Build publication-ready figures from Quartic2D paper benchmark JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.figures._style import DOUBLE_COLUMN, DOUBLE_COLUMN_TALL, MARKERS, apply_style, finish_axis, save_figure
METHOD_LABELS = {
    "trapezoid": "Trapezoid",
    "simpson": "Simpson",
    "gl4": "GL4",
    "gl8": "GL8",
    "ogata": "Ogata",
    "fftlog": "FFTLog",
}


def load(path):
    return json.loads(Path(path).read_text())


def calibrated_rows(release):
    out = {}
    rows = release["interaction_validation_rows"]
    for method, cal in release["interaction_calibration"].items():
        selected = cal["selected_parameters"]
        chosen = []
        for row in rows:
            if row["method"] != method:
                continue
            params = row["parameters"]
            if all(params.get(k) == v for k, v in selected.items()):
                chosen.append(row)
        if chosen:
            out[method] = chosen
    return out


def figure_interaction_pareto(release, outdir):
    groups = calibrated_rows(release)
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
    for i, method in enumerate(("trapezoid", "simpson", "gl4", "gl8", "ogata", "fftlog")):
        rr = groups.get(method, [])
        if not rr:
            continue
        runtimes = np.asarray([r["runtime_median_seconds"] for r in rr]) * 1e3
        errors = np.asarray([r["discretization_relative_l2_error"] for r in rr])
        x = float(np.median(runtimes))
        y = float(np.max(errors))
        xerr = np.array([[x - np.quantile(runtimes, 0.25)], [np.quantile(runtimes, 0.75) - x]])
        ax.errorbar(x, y, xerr=xerr, marker=MARKERS[i], ls="none", capsize=2, label=METHOD_LABELS[method])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$\max\,\epsilon_2$")
    finish_axis(ax)
    fig.tight_layout()
    save_figure(fig, Path(outdir) / "interaction_backend_pareto")


def _best_rows_by_workload_target(data, method="simpson"):
    groups = {}
    for row in data["rows"]:
        if row.get("status") != "complete" or row.get("stress"):
            continue
        mr = row.get("methods", {}).get(method)
        if not mr or mr.get("status") != "complete":
            continue
        key = (row["workload"], float(row["target"]))
        old = groups.get(key)
        # Prefer a higher-resolution input when multiple N_r values are present.
        if old is None or row["n_r"] > old["n_r"]:
            groups[key] = row
    return groups


def figure_harmonic_tolerance(data, outdir):
    groups = _best_rows_by_workload_target(data)
    workloads = sorted({k[0] for k in groups})
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    for i, name in enumerate(workloads):
        rr = [groups[(name, tol)] for _, tol in groups if _ == name]
        rr = sorted(rr, key=lambda r: r["target"])
        if not rr:
            continue
        x = np.asarray([r["target"] for r in rr])
        err = np.asarray([
            r["methods"]["simpson"].get(
                "quartic2d_total_relative_l2",
                r["methods"]["simpson"].get("worst_end_to_end_relative_l2"),
            )
            for r in rr
        ])
        runtime = 1e3 * np.asarray([r["methods"]["simpson"]["timing"]["median_seconds"] for r in rr])
        axes[0].plot(x, err, marker=MARKERS[i % len(MARKERS)], label=name.replace("_", " "))
        axes[1].plot(x, runtime, marker=MARKERS[i % len(MARKERS)], label=name.replace("_", " "))
    if groups:
        vals = np.asarray([k[1] for k in groups])
        axes[0].plot([vals.min(), vals.max()], [vals.min(), vals.max()], "--", lw=1.0, label=r"$\epsilon_2=\epsilon_{\rm req}$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel(r"$\epsilon_{\rm req}$")
    axes[0].set_ylabel(r"$\epsilon_2$")
    axes[1].set_xlabel(r"$\epsilon_{\rm req}$")
    axes[1].set_ylabel(r"$t$ (ms)")
    finish_axis(axes[0], legend=False)
    finish_axis(axes[1], legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, Path(outdir) / "harmonic_tolerance_accuracy_runtime")


def _plot_scaling_panel(ax, rows, axis, xlabel, guide=None):
    rr = sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["value"])
    x = np.asarray([r["value"] for r in rr], dtype=float)
    y = 1e3 * np.asarray([r["median_seconds"] for r in rr], dtype=float)
    q25 = 1e3 * np.asarray([r["q25_seconds"] for r in rr], dtype=float)
    q75 = 1e3 * np.asarray([r["q75_seconds"] for r in rr], dtype=float)
    ax.errorbar(x, y, yerr=np.vstack((y-q25, q75-y)), marker="o", capsize=2, label="Measured")
    if guide is not None and x.size:
        gy = np.asarray(guide(x), dtype=float)
        scale = y[-1] / gy[-1]
        ax.plot(x, scale * gy, "--", label="Asymptotic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$t$ (ms)")
    finish_axis(ax)


def figure_scaling(data, outdir):
    fig, axes = plt.subplots(2, 3, figsize=DOUBLE_COLUMN_TALL)
    ht = data["harmonic_transform_rows"]
    inter = data["interaction_fftlog_rows"]
    _plot_scaling_panel(axes[0, 0], ht, "N_q", r"$N_q$", guide=lambda x: x)
    _plot_scaling_panel(axes[0, 1], ht, "N_r", r"$N_r$", guide=lambda x: x)
    _plot_scaling_panel(axes[0, 2], ht, "N_m", r"$N_m$", guide=lambda x: x)
    _plot_scaling_panel(axes[1, 0], inter, "N_F", r"$N_F$", guide=lambda x: x*np.log2(x))
    _plot_scaling_panel(axes[1, 1], inter, "N_D", r"$N_D$", guide=None)
    _plot_scaling_panel(axes[1, 2], inter, "N_p", r"$N_p$", guide=lambda x: x)
    for label, ax in zip("abcdef", axes.ravel()):
        ax.text(0.03, 0.94, f"({label})", transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    save_figure(fig, Path(outdir) / "asymptotic_scaling")


def figure_stress(data, outdir):
    rows = [r for r in data["rows"] if r.get("stress")]
    complete = [
        r for r in rows
        if r.get("status") == "complete"
        and r.get("methods", {}).get("simpson", {}).get("status") == "complete"
    ]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    names = sorted({r["workload"] for r in rows})
    xpos = np.arange(len(names))
    err = []
    runtime = []
    for name in names:
        rr = [r for r in complete if r["workload"] == name]
        if rr:
            # Use the loosest requested tolerance as the representative stress point.
            row = max(rr, key=lambda r: r["target"])
            mr = row["methods"]["simpson"]
            err.append(mr.get("quartic2d_total_relative_l2", mr["worst_end_to_end_relative_l2"]))
            runtime.append(1e3 * mr["timing"]["median_seconds"])
        else:
            err.append(np.nan)
            runtime.append(np.nan)
    axes[0].bar(xpos, err)
    axes[1].bar(xpos, runtime)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$\epsilon_2$")
    axes[1].set_ylabel(r"$t$ (ms)")
    for ax in axes:
        ax.set_xticks(xpos, [n.replace("_stress", "").replace("_", " ") for n in names], rotation=20, ha="right")
        finish_axis(ax, legend=False)
    fig.tight_layout()
    save_figure(fig, Path(outdir) / "stress_cases")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--harmonic")
    ap.add_argument("--scaling")
    ap.add_argument("--output-dir", default="benchmarks/results/publication/figures")
    args = ap.parse_args()
    apply_style()
    release = load(args.release)
    figure_interaction_pareto(release, args.output_dir)
    if args.harmonic and Path(args.harmonic).exists():
        harmonic = load(args.harmonic)
        figure_harmonic_tolerance(harmonic, args.output_dir)
        figure_stress(harmonic, args.output_dir)
    if args.scaling and Path(args.scaling).exists():
        figure_scaling(load(args.scaling), args.output_dir)
    print(f"Figures: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
