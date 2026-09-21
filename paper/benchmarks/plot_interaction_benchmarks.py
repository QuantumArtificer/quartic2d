#!/usr/bin/env python3
"""Publication figures for interaction accuracy/runtime and scaling results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from _style import DOUBLE_COLUMN, save_figure
except ImportError:  # pragma: no cover
    from paper.benchmarks._style import DOUBLE_COLUMN, save_figure


METHODS = ("trapezoid", "simpson", "gl4", "gl8", "ogata", "fftlog")


def accuracy_figure(data, outdir):
    summary = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    for method in METHODS:
        rows = [r for r in summary if r["method"] == method and r["median_production_seconds"] is not None]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["requested_tolerance"], reverse=True)
        t = np.array([1e3 * r["median_production_seconds"] for r in rows])
        e2 = np.array([r["worst_relative_l2"] for r in rows])
        ep = np.array([r["worst_relative_peak"] for r in rows])
        axes[0].loglog(t, e2, marker="o", label=method)
        axes[1].loglog(t, ep, marker="o", label=method)
    axes[0].set_xlabel(r"$t$ (ms)")
    axes[0].set_ylabel(r"$\max\,\epsilon_2$")
    axes[1].set_xlabel(r"$t$ (ms)")
    axes[1].set_ylabel(r"$\max\,\epsilon_{\infty,\mathrm{pk}}$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, outdir / "interaction_accuracy_runtime")
    plt.close(fig)


def completion_figure(data, outdir):
    summary = data["summary"]
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    for method in METHODS:
        rows = [r for r in summary if r["method"] == method]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["requested_tolerance"], reverse=True)
        x = np.array([r["requested_tolerance"] for r in rows])
        certified = np.array([r.get("certified_fraction", r.get("completion_fraction", 0.0)) for r in rows])
        capability = np.array([r.get("tested_capability_fraction", y) for r, y in zip(rows, certified)])
        ax.semilogx(x, certified, marker="o", label=method)
        if np.any(capability > certified):
            ax.semilogx(x, capability, marker="x", linestyle="--", alpha=0.6)
    ax.set_xlabel(r"$\epsilon_{\mathrm{req}}$")
    ax.set_ylabel("certified fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.invert_xaxis()
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save_figure(fig, outdir / "interaction_convergence_coverage")
    plt.close(fig)


def scaling_figure(data, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.1))
    specs = [
        ("fftlog", "N_F", r"$N_F$"),
        ("fftlog", "N_D", r"$N_D$"),
        ("fftlog", "N_p", r"$N_p$"),
        ("finite", "N_q", r"$N_q$"),
        ("finite", "N_D", r"$N_D$"),
        ("finite", "N_p", r"$N_p$"),
    ]
    for ax, (branch, axis, xlabel) in zip(axes.ravel(), specs):
        rows = [r for r in data["rows"] if r["branch"] == branch and r["axis"] == axis]
        x = np.array([r[axis] for r in rows], dtype=float)
        y = np.array([1e3 * r["median_seconds"] for r in rows], dtype=float)
        q1 = np.array([1e3 * r["q25_seconds"] for r in rows], dtype=float)
        q3 = np.array([1e3 * r["q75_seconds"] for r in rows], dtype=float)
        ax.loglog(x, y, marker="o")
        ax.fill_between(x, q1, q3, alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$t$ (ms)")
        ax.text(0.05, 0.92, "FFTLog" if branch == "fftlog" else "Simpson", transform=ax.transAxes, va="top")
    fig.tight_layout()
    save_figure(fig, outdir / "interaction_scaling")
    plt.close(fig)

    rows = [r for r in data["rows"] if r["branch"] == "finite" and r["axis"] == "s_q"]
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    x = np.array([r["N_qs"] for r in rows], dtype=float)
    y = np.array([1e3 * r["median_seconds"] for r in rows], dtype=float)
    ax.loglog(x, y, marker="o")
    ax.set_xlabel(r"$N_{q,s}$")
    ax.set_ylabel(r"$t$ (ms)")
    fig.tight_layout()
    save_figure(fig, outdir / "interaction_finite_refinement_scaling")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy", type=Path, default=Path("paper/benchmarks/results/interaction_benchmarks.json"))
    ap.add_argument("--scaling", type=Path, default=Path("paper/benchmarks/results/interaction_scaling.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("paper/benchmarks/results/figures"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.accuracy.exists():
        data = json.loads(args.accuracy.read_text())
        accuracy_figure(data, args.output_dir)
        completion_figure(data, args.output_dir)
    if args.scaling.exists():
        scaling_figure(json.loads(args.scaling.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
