#!/usr/bin/env python3
"""Generate publication figures from QUARTIC2D numerical benchmark results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RADIAL_METHODS = ("trapezoid", "simpson", "gl4", "gl8", "ogata")
INTERACTION_METHODS = ("trapezoid", "simpson", "gl4", "gl8", "ogata", "fftlog")
SUPPORTED_SCHEMA_VERSIONS = (4, 5, 6)
MARKERS = {
    "trapezoid": "o",
    "simpson": "s",
    "gl4": "^",
    "gl8": "v",
    "ogata": "D",
    "fftlog": "P",
}
LABELS = {
    "trapezoid": "Trapezoid",
    "simpson": "Simpson",
    "gl4": "GL4",
    "gl8": "GL8",
    "ogata": "Ogata",
    "fftlog": "FFTLog",
}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def _groups(rows, method, key="n_nodes"):
    selected = [row for row in rows if row["method"] == method]
    return [(value, [row for row in selected if row[key] == value]) for value in sorted({row[key] for row in selected})]


def _aggregate(rows, method, xkey, ykey, groupkey="n_nodes"):
    points = []
    for _, group in _groups(rows, method, groupkey):
        xvals = np.asarray([row[xkey] for row in group], dtype=float)
        yvals = np.asarray([row[ykey] for row in group], dtype=float)
        points.append({
            "x": float(np.median(xvals)),
            "xlo": float(np.quantile(xvals, 0.25)),
            "xhi": float(np.quantile(xvals, 0.75)),
            "y": float(np.median(yvals)),
            "ylo": float(np.quantile(yvals, 0.25)),
            "yhi": float(np.quantile(yvals, 0.75)),
        })
    return points


def _line_with_spread(ax, points, method, *, x_runtime=False):
    if not points:
        return
    x = np.asarray([p["x"] for p in points])
    y = np.asarray([p["y"] for p in points])
    ylo = np.asarray([p["ylo"] for p in points])
    yhi = np.asarray([p["yhi"] for p in points])
    line = ax.plot(x, y, marker=MARKERS[method], label=LABELS[method])[0]
    ax.fill_between(x, ylo, yhi, alpha=0.12, color=line.get_color())
    if x_runtime:
        xlo = np.asarray([p["xlo"] for p in points])
        xhi = np.asarray([p["xhi"] for p in points])
        ax.errorbar(
            x,
            y,
            xerr=np.vstack((x - xlo, xhi - x)),
            fmt="none",
            ecolor=line.get_color(),
            alpha=0.45,
            capsize=1.5,
            lw=0.8,
        )


def _format_log(ax, xlabel, ylabel, *, legend=True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    if legend:
        ax.legend(ncol=2)


def _selected_distance(row, selected):
    p = row["parameters"]
    if selected and all(p.get(key) == value for key, value in selected.items()):
        return 0.0
    if "subdivisions" in selected and "subdivisions" in p:
        return abs(np.log2(p["subdivisions"] / selected["subdivisions"]))
    if {"N", "h"}.issubset(selected) and {"N", "h"}.issubset(p):
        return (
            abs(np.log(max(p["N"], 1) / max(selected["N"], 1)))
            + abs(np.log(max(p["h"], 1.0e-300) / max(selected["h"], 1.0e-300)))
        )
    if {"n", "bias"}.issubset(selected) and {"n", "bias"}.issubset(p):
        return abs(np.log(max(p["n"], 1) / max(selected["n"], 1))) + abs(p["bias"] - selected["bias"])
    return np.inf


def _tolerance_plots(rows, methods, figdir: Path, prefix: str):
    # Requested tolerance versus achieved error.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in methods:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda r: r["requested_rtol"])
        if rr:
            ax.plot(
                [r["requested_rtol"] for r in rr],
                [r["validation_max_relative_l2_error"] for r in rr],
                marker=MARKERS[method],
                label=LABELS[method],
            )
    if rows:
        values = np.asarray([r["requested_rtol"] for r in rows])
        lo, hi = float(np.min(values)), float(np.max(values))
        ax.plot([lo, hi], [lo, hi], "--", label="achieved = requested")
    _format_log(ax, "Requested relative tolerance (rtol)", r"Worst held-out relative $L^2$ error")
    fig.tight_layout()
    _save(fig, figdir / f"{prefix}_requested_tolerance_vs_achieved_error")

    # Requested tolerance versus selected numerical work.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in methods:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda r: r["requested_rtol"])
        if rr:
            ax.plot(
                [r["requested_rtol"] for r in rr],
                [r["n_nodes"] for r in rr],
                marker=MARKERS[method],
                label=LABELS[method],
            )
    _format_log(ax, "Requested relative tolerance (rtol)", "Selected transform / quadrature nodes")
    fig.tight_layout()
    _save(fig, figdir / f"{prefix}_requested_tolerance_vs_nodes")

    # Requested tolerance versus measured runtime.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in methods:
        rr = sorted([r for r in rows if r["method"] == method], key=lambda r: r["requested_rtol"])
        if rr:
            ax.plot(
                [r["requested_rtol"] for r in rr],
                [r["runtime_median_seconds"] for r in rr],
                marker=MARKERS[method],
                label=LABELS[method],
            )
    _format_log(ax, "Requested relative tolerance (rtol)", "Median runtime [s]")
    fig.tight_layout()
    _save(fig, figdir / f"{prefix}_requested_tolerance_vs_runtime")


def make_plots(results_path, figdir):
    data = json.loads(Path(results_path).read_text())
    schema_version = data.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
        raise ValueError(
            f"plot_numerical_benchmarks.py supports benchmark schema_version(s): {supported}; "
            f"got {schema_version!r}"
        )

    figdir = Path(figdir)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "figure.dpi": 120,
    })

    radial = data["radial_validation_rows"]

    # Radial transform: discretization accuracy versus runtime.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in RADIAL_METHODS:
        _line_with_spread(
            ax,
            _aggregate(radial, method, "runtime_median_seconds", "discretization_relative_l2_error"),
            method,
            x_runtime=True,
        )
    _format_log(ax, "Median runtime per transform [s]", r"Relative $L^2$ discretization error")
    fig.tight_layout()
    _save(fig, figdir / "radial_runtime_vs_discretization_error")

    # Radial transform: accuracy versus common work-resolution coordinate.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in RADIAL_METHODS:
        _line_with_spread(ax, _aggregate(radial, method, "n_nodes", "discretization_relative_l2_error"), method)
    _format_log(ax, "Transform / quadrature nodes", r"Relative $L^2$ discretization error")
    fig.tight_layout()
    _save(fig, figdir / "radial_accuracy_vs_nodes")

    # Radial roundtrip: method error versus runtime with q-range floor shown.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in RADIAL_METHODS:
        _line_with_spread(
            ax,
            _aggregate(radial, method, "runtime_median_seconds", "roundtrip_relative_l2_error"),
            method,
            x_runtime=True,
        )
    floor = float(np.median([r["q_truncation_roundtrip_relative_l2_error"] for r in radial]))
    ax.axhline(floor, ls="--", lw=1.0, label="median q-range truncation floor")
    _format_log(ax, "Median runtime per transform [s]", r"Roundtrip relative $L^2$ error")
    fig.tight_layout()
    _save(fig, figdir / "radial_runtime_vs_roundtrip_error")

    # Radial error components at each calibrated parameter point (not additive).
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xpos = np.arange(len(RADIAL_METHODS))
    disc, total, trunc = [], [], []
    for method in RADIAL_METHODS:
        selected = data["radial_calibration"][method]["selected_parameters"]
        rows = [r for r in radial if r["method"] == method]
        best = min(_selected_distance(r, selected) for r in rows)
        chosen = [r for r in rows if np.isclose(_selected_distance(r, selected), best)]
        disc.append(np.median([r["discretization_relative_l2_error"] for r in chosen]))
        total.append(np.median([r["total_relative_l2_error"] for r in chosen]))
        trunc.append(np.median([r["truncation_relative_l2_error"] for r in chosen]))
    ax.semilogy(xpos, disc, "o-", label="discretization error")
    ax.semilogy(xpos, total, "s-", label="total error")
    ax.semilogy(xpos, trunc, "^-", label="finite-support truncation error")
    ax.set_xticks(xpos, [LABELS[m] for m in RADIAL_METHODS], rotation=25, ha="right")
    ax.set_ylabel(r"Median relative $L^2$ error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, figdir / "radial_error_components")

    _tolerance_plots(data["radial_tolerance_rows"], RADIAL_METHODS, figdir, "radial")

    interaction = data["interaction_validation_rows"]

    # Interaction transform: accuracy/runtime Pareto paths.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in INTERACTION_METHODS:
        _line_with_spread(
            ax,
            _aggregate(interaction, method, "runtime_median_seconds", "discretization_relative_l2_error"),
            method,
            x_runtime=True,
        )
    _format_log(ax, "Median runtime per interaction [s]", r"Relative $L^2$ discretization error")
    fig.tight_layout()
    _save(fig, figdir / "interaction_runtime_vs_discretization_error")

    # Interaction transform: common work-resolution coordinate.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in INTERACTION_METHODS:
        _line_with_spread(ax, _aggregate(interaction, method, "n_nodes", "discretization_relative_l2_error"), method)
    _format_log(ax, "Transform / quadrature nodes", r"Relative $L^2$ discretization error")
    fig.tight_layout()
    _save(fig, figdir / "interaction_accuracy_vs_nodes")

    _tolerance_plots(data["interaction_tolerance_rows"], INTERACTION_METHODS, figdir, "interaction")

    # Large-displacement stability: peak-normalized absolute error avoids the
    # meaningless blow-up of pointwise relative error after the exact value
    # decays toward zero.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in INTERACTION_METHODS:
        rr = sorted([r for r in data["large_delta_rows"] if r["method"] == method], key=lambda r: r["delta"])
        if rr:
            ax.plot(
                [r["delta"] for r in rr],
                [r["absolute_error_over_reference_peak"] for r in rr],
                marker=MARKERS[method],
                ms=3,
                label=LABELS[method],
            )
    _format_log(ax, r"Displacement $\Delta$", "Absolute error / reference peak")
    fig.tight_layout()
    _save(fig, figdir / "interaction_large_delta_accuracy")

    # Output-count scaling.
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for method in INTERACTION_METHODS:
        rr = sorted([r for r in data["scaling_rows"] if r["method"] == method], key=lambda r: r["n_output"])
        if rr:
            x = np.asarray([r["n_output"] for r in rr])
            y = np.asarray([r["runtime_median_seconds"] for r in rr])
            ylo = np.asarray([r["runtime_q25_seconds"] for r in rr])
            yhi = np.asarray([r["runtime_q75_seconds"] for r in rr])
            line = ax.plot(x, y, marker=MARKERS[method], label=LABELS[method])[0]
            ax.fill_between(x, ylo, yhi, alpha=0.12, color=line.get_color())
    _format_log(ax, "Number of requested output points", "Median runtime [s]")
    fig.tight_layout()
    _save(fig, figdir / "interaction_output_count_scaling")

    # PETAL2D-connected representative workload.
    petal = data.get("petal2d_rows", [])
    if petal:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for method in INTERACTION_METHODS:
            rr = [r for r in petal if r["method"] == method]
            if rr:
                ax.scatter(
                    [np.median([r["runtime_median_seconds"] for r in rr])],
                    [np.max([r["interaction_discretization_relative_l2_error"] for r in rr])],
                    marker=MARKERS[method],
                    s=45,
                    label=LABELS[method],
                )
        _format_log(ax, "Median runtime [s]", r"Relative $L^2$ interaction discretization error")
        fig.tight_layout()
        _save(fig, figdir / "petal2d_runtime_vs_accuracy")

        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        xpos = np.arange(len(INTERACTION_METHODS))
        interaction_error = []
        total_error = []
        first_stage = []
        q_truncation = []
        for method in INTERACTION_METHODS:
            row = next(r for r in petal if r["method"] == method)
            interaction_error.append(row["interaction_discretization_relative_l2_error"])
            total_error.append(row["total_relative_l2_error"])
            first_stage.append(row["first_stage_relative_l2_error"])
            q_truncation.append(row["q_truncation_relative_l2_error"])
        ax.semilogy(xpos, interaction_error, "o-", label="interaction-stage error")
        ax.semilogy(xpos, total_error, "s-", label="end-to-end total error")
        ax.semilogy(xpos, first_stage, "^-", label="first-stage error")
        ax.semilogy(xpos, q_truncation, "v-", label="q-range truncation error")
        ax.set_xticks(xpos, [LABELS[m] for m in INTERACTION_METHODS], rotation=25, ha="right")
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _save(fig, figdir / "petal2d_error_components")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--figdir", default=None)
    args = parser.parse_args()
    path = Path(args.results)
    make_plots(path, Path(args.figdir) if args.figdir else path.parent / "figures")


if __name__ == "__main__":
    main()
