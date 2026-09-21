#!/usr/bin/env python3
"""Fast screening benchmark for automatic-convergence methods.

This benchmark identifies methods that merit the full repeated automatic-
convergence performance benchmark.  It is a screening tool, not publication
performance evidence.  Each case is evaluated once, without warmups,
profiling, or repeated timing.

For HarmonicTransform, q support and q-grid density are fixed to values already
selected by the completed HarmonicTransform benchmark.  The screen therefore
measures only method-specific radial-quadrature convergence.  Finalists must be
rerun with run_autoconvergence_performance.py, which measures complete
parameter selection including q support and q-grid calibration.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from quartic2d import HarmonicTransform, Interaction
from quartic2d._experimental.ogata import (
    HarmonicTransform as OgataHarmonicTransform,
    Interaction as OgataInteraction,
)

try:
    from _common import environment_metadata
    from _interaction_suite import (
        DEFAULT_SUBDIVISIONS,
        FINITE_METHODS,
        build_fields_from_harmonic_results,
        kernel_registry,
        validation_cases,
    )
    from run_autoconvergence_performance import (
        DEFAULT_BIAS_VALUES,
        DEFAULT_N_VALUES,
        DEFAULT_OGATA_N_VALUES,
        DELTA_DOMAINS,
        HARMONIC_SUBDIVISIONS,
        SUPPORTED_HARMONIC_METHODS,
        SUPPORTED_INTERACTION_METHODS,
        _harmonic_rows,
        make_displacements,
        parse_csv,
    )
    from run_harmonic_transform import SyntheticDecomposition, workloads
except ImportError:  # pragma: no cover
    from paper.benchmarks._common import environment_metadata
    from paper.benchmarks._interaction_suite import (
        DEFAULT_SUBDIVISIONS,
        FINITE_METHODS,
        build_fields_from_harmonic_results,
        kernel_registry,
        validation_cases,
    )
    from paper.benchmarks.run_autoconvergence_performance import (
        DEFAULT_BIAS_VALUES,
        DEFAULT_N_VALUES,
        DEFAULT_OGATA_N_VALUES,
        DELTA_DOMAINS,
        HARMONIC_SUBDIVISIONS,
        SUPPORTED_HARMONIC_METHODS,
        SUPPORTED_INTERACTION_METHODS,
        _harmonic_rows,
        make_displacements,
        parse_csv,
    )
    from paper.benchmarks.run_harmonic_transform import SyntheticDecomposition, workloads


DEFAULT_HARMONIC_WORKLOAD = "gaussian_anisotropic_m2"
DEFAULT_INTERACTION_CASE = "nodal_rpa"
DEFAULT_TOLERANCE = 1.0e-4
DEFAULT_N_DELTA = 8


def time_once(function):
    start = time.perf_counter()
    value = function()
    return float(time.perf_counter() - start), value


def harmonic_class(method: str):
    return OgataHarmonicTransform if method == "ogata" else HarmonicTransform


def interaction_class(method: str):
    return OgataInteraction if method == "ogata" else Interaction


def run_harmonic_screen(
    *,
    result_path: Path,
    workload_name: str,
    methods: tuple[str, ...],
    tolerance: float,
    atol: float,
) -> list[dict]:
    data = json.loads(result_path.read_text())
    rows_by_workload, source_target = _harmonic_rows(data, tolerance)
    source_row = rows_by_workload.get(workload_name)
    if source_row is None:
        raise ValueError(
            f"no completed HarmonicTransform benchmark row for workload={workload_name!r} "
            f"at target={source_target:g}"
        )

    workload_map = {item.name: item for item in workloads()}
    decomposition = SyntheticDecomposition(workload_map[workload_name], int(source_row["n_r"]))
    q_max = float(source_row["q_support"]["q_max"])
    n_q = int(source_row["n_q_selected"])
    rows = []

    for method in methods:
        print(
            f"[screen:HarmonicTransform] {workload_name} | {method} | rtol={tolerance:.0e}",
            flush=True,
        )
        try:
            cls = harmonic_class(method)
            transform = cls(
                decomposition,
                q_max=q_max,
                n_q=n_q,
                method=method,
                interpolator="cubic",
                subdivisions=1,
                check=False,
            )
            if method == "ogata":
                convergence_seconds, convergence = time_once(
                    lambda: transform.converge(
                        rtol=tolerance,
                        atol=atol,
                        apply=False,
                    )
                )
            else:
                convergence_seconds, convergence = time_once(
                    lambda: transform.converge(
                        rtol=tolerance,
                        atol=atol,
                        subdivisions=HARMONIC_SUBDIVISIONS,
                        apply=False,
                    )
                )
            production_seconds = None
            if convergence.converged and convergence.selected_parameters:
                if method == "ogata":
                    production_kwargs = {
                        "N": int(convergence.selected_parameters["N"]),
                        "h": float(convergence.selected_parameters["h"]),
                    }
                else:
                    production_kwargs = {
                        "subdivisions": int(convergence.selected_parameters["subdivisions"])
                    }
                production_seconds, _ = time_once(
                    lambda: cls(
                        decomposition,
                        q_max=q_max,
                        n_q=n_q,
                        method=method,
                        interpolator="cubic",
                        check=False,
                        **production_kwargs,
                    )
                )
            rows.append(
                {
                    "stage": "HarmonicTransform",
                    "workload": workload_name,
                    "method": method,
                    "requested_tolerance": float(tolerance),
                    "status": "complete" if convergence.converged else "not_converged",
                    "fixed_q_max": q_max,
                    "fixed_n_q": n_q,
                    "selected_parameters": dict(convergence.selected_parameters),
                    "convergence_seconds": convergence_seconds,
                    "production_seconds": production_seconds,
                    "convergence_to_production": (
                        None
                        if production_seconds is None or production_seconds <= 0.0
                        else float(convergence_seconds / production_seconds)
                    ),
                    "convergence": convergence.to_dict(include_values=False),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "stage": "HarmonicTransform",
                    "workload": workload_name,
                    "method": method,
                    "requested_tolerance": float(tolerance),
                    "status": "failed",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def run_interaction_screen(
    *,
    harmonic_result_path: Path,
    case_name: str,
    methods: tuple[str, ...],
    tolerance: float,
    atol: float,
    delta_domains: tuple[str, ...],
    n_delta: int,
    upstream_target: float,
    upstream_method: str,
) -> list[dict]:
    cases = {item.name: item for item in validation_cases()}
    case = cases[case_name]
    fields, _metadata = build_fields_from_harmonic_results(
        harmonic_result_path,
        target=upstream_target,
        method=upstream_method,
        workload_names=(case.workload,),
    )
    field = fields[case.workload]
    kernel = kernel_registry()[case.kernel]
    rows = []

    for delta_domain in delta_domains:
        deltas = make_displacements(delta_domain, n_delta)
        for method in methods:
            cls = interaction_class(method)
            print(
                f"[screen:Interaction] {case.name} | delta={delta_domain} | "
                f"{method} | rtol={tolerance:.0e}",
                flush=True,
            )

            def calibrate():
                return cls.converge_parameters(
                    deltas,
                    field,
                    field,
                    kernel.U,
                    rtol=tolerance,
                    atol=atol,
                    method=method,
                    interpolator="cubic",
                    subdivisions=DEFAULT_SUBDIVISIONS,
                    n_values=DEFAULT_N_VALUES,
                    bias=-0.5,
                    bias_values=DEFAULT_BIAS_VALUES,
                    ogata_n_values=DEFAULT_OGATA_N_VALUES,
                    verbose=False,
                )

            try:
                convergence_seconds, convergence = time_once(calibrate)
                production_seconds = None
                if convergence.converged:
                    production_seconds, _ = time_once(
                        lambda: convergence.interaction(
                            deltas,
                            field,
                            field,
                            kernel.U,
                        )
                    )
                rows.append(
                    {
                        "stage": "Interaction",
                        "case": case.name,
                        "workload": case.workload,
                        "kernel": case.kernel,
                        "delta_domain": delta_domain,
                        "delta_min": float(DELTA_DOMAINS[delta_domain][0]),
                        "delta_max": float(DELTA_DOMAINS[delta_domain][1]),
                        "n_delta": int(n_delta),
                        "method": method,
                        "requested_tolerance": float(tolerance),
                        "status": "complete" if convergence.converged else "not_converged",
                        "selected_parameters": (
                            convergence.parameters
                            if convergence.converged
                            else {
                                "method": method,
                                "interpolator": "cubic",
                                **convergence.search.selected_parameters,
                            }
                        ),
                        "convergence_seconds": convergence_seconds,
                        "production_seconds": production_seconds,
                        "convergence_to_production": (
                            None
                            if production_seconds is None or production_seconds <= 0.0
                            else float(convergence_seconds / production_seconds)
                        ),
                        "convergence": convergence.to_dict(include_values=False),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "stage": "Interaction",
                        "case": case.name,
                        "delta_domain": delta_domain,
                        "method": method,
                        "requested_tolerance": float(tolerance),
                        "status": "failed",
                        "exception": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (row.get("stage"), row.get("method"), row.get("delta_domain"))
            for row in rows
        }
    )
    summary = []
    for stage, method, delta_domain in keys:
        selected = [
            row
            for row in rows
            if row.get("stage") == stage
            and row.get("method") == method
            and row.get("delta_domain") == delta_domain
        ]
        complete = [row for row in selected if row.get("status") == "complete"]
        convergence_times = [
            float(row["convergence_seconds"])
            for row in selected
            if row.get("convergence_seconds") is not None
        ]
        production_times = [
            float(row["production_seconds"])
            for row in complete
            if row.get("production_seconds") is not None
        ]
        ratios = [
            float(row["convergence_to_production"])
            for row in complete
            if row.get("convergence_to_production") is not None
        ]
        summary.append(
            {
                "stage": stage,
                "method": method,
                "delta_domain": delta_domain,
                "n_rows": len(selected),
                "n_converged": len(complete),
                "convergence_fraction": float(len(complete) / len(selected)) if selected else None,
                "median_convergence_seconds": (
                    None if not convergence_times else float(statistics.median(convergence_times))
                ),
                "median_production_seconds": (
                    None if not production_times else float(statistics.median(production_times))
                ),
                "median_convergence_to_production": (
                    None if not ratios else float(statistics.median(ratios))
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/benchmarks/results/autoconvergence_screen.json"),
    )
    parser.add_argument(
        "--harmonic-results",
        type=Path,
        default=Path("paper/benchmarks/results/harmonic_transform.json"),
    )
    parser.add_argument("--stages", default="harmonic,interaction")
    parser.add_argument("--harmonic-workload", default=DEFAULT_HARMONIC_WORKLOAD)
    parser.add_argument("--interaction-case", default=DEFAULT_INTERACTION_CASE)
    parser.add_argument("--methods", default=",".join(SUPPORTED_INTERACTION_METHODS))
    parser.add_argument("--include-ogata", action="store_true")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--upstream-target", type=float, default=1.0e-4)
    parser.add_argument("--upstream-method", default="simpson")
    parser.add_argument("--delta-domains", default="standard,large")
    parser.add_argument("--n-delta", type=int, default=DEFAULT_N_DELTA)
    args = parser.parse_args()

    stages = parse_csv(args.stages)
    unknown_stages = sorted(set(stages) - {"harmonic", "interaction"})
    if unknown_stages:
        raise ValueError(f"unknown stages: {unknown_stages}")
    if args.tolerance <= 0.0 or args.atol < 0.0:
        raise ValueError("tolerance must be positive and atol must be non-negative")
    if args.n_delta < 2:
        raise ValueError("n_delta must be at least 2")

    methods = parse_csv(args.methods)
    allowed_methods = set(SUPPORTED_INTERACTION_METHODS)
    if args.include_ogata:
        allowed_methods.add("ogata")
        if "ogata" not in methods:
            methods = methods + ("ogata",)
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise ValueError(f"unknown or unavailable methods: {unknown_methods}")

    harmonic_methods = tuple(
        method
        for method in methods
        if method in SUPPORTED_HARMONIC_METHODS or (args.include_ogata and method == "ogata")
    )
    delta_domains = parse_csv(args.delta_domains)
    unknown_domains = sorted(set(delta_domains) - set(DELTA_DOMAINS))
    if unknown_domains:
        raise ValueError(f"unknown delta domains: {unknown_domains}")

    workload_names = {item.name for item in workloads()}
    if args.harmonic_workload not in workload_names:
        raise ValueError(f"unknown harmonic workload: {args.harmonic_workload!r}")
    case_names = {item.name for item in validation_cases()}
    if args.interaction_case not in case_names:
        raise ValueError(f"unknown interaction case: {args.interaction_case!r}")

    rows = []
    if "harmonic" in stages:
        rows.extend(
            run_harmonic_screen(
                result_path=args.harmonic_results,
                workload_name=args.harmonic_workload,
                methods=harmonic_methods,
                tolerance=args.tolerance,
                atol=args.atol,
            )
        )
    if "interaction" in stages:
        rows.extend(
            run_interaction_screen(
                harmonic_result_path=args.harmonic_results,
                case_name=args.interaction_case,
                methods=methods,
                tolerance=args.tolerance,
                atol=args.atol,
                delta_domains=delta_domains,
                n_delta=args.n_delta,
                upstream_target=args.upstream_target,
                upstream_method=args.upstream_method,
            )
        )

    result = {
        "schema": 1,
        "benchmark": "Automatic-convergence method screen",
        "benchmark_class": "performance_screen",
        "publication_use": False,
        "scope": {
            "purpose": "identify methods for the full repeated automatic-convergence performance benchmark",
            "timing_policy": "one untimed setup followed by one convergence timing and one selected-parameter production timing; no warmups or repeats",
            "harmonic_policy": "q support and q-grid density are fixed from the completed HarmonicTransform benchmark; only radial-quadrature convergence is screened",
            "interaction_policy": "complete method-specific Interaction parameter convergence is timed on a reduced displacement set",
            "validation_policy": "no accuracy or reference-solution claim is made by this screen",
        },
        "environment": environment_metadata(),
        "settings": {
            "stages": list(stages),
            "harmonic_workload": args.harmonic_workload,
            "interaction_case": args.interaction_case,
            "methods": list(methods),
            "harmonic_methods": list(harmonic_methods),
            "include_ogata": bool(args.include_ogata),
            "tolerance": float(args.tolerance),
            "atol": float(args.atol),
            "delta_domains": {
                name: {"delta_min": DELTA_DOMAINS[name][0], "delta_max": DELTA_DOMAINS[name][1]}
                for name in delta_domains
            },
            "n_delta": int(args.n_delta),
            "upstream_target": float(args.upstream_target),
            "upstream_method": args.upstream_method,
        },
        "rows": rows,
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("\n=== automatic-convergence screen ===")
    for row in result["summary"]:
        domain = "" if row["delta_domain"] is None else f" delta={row['delta_domain']}"
        t_conv = row["median_convergence_seconds"]
        t_prod = row["median_production_seconds"]
        overhead = row["median_convergence_to_production"]
        print(
            f"{row['stage']:17s} {row['method']:10s}{domain:16s} "
            f"conv={row['n_converged']}/{row['n_rows']} "
            f"t_conv={'--' if t_conv is None else f'{t_conv:.6g} s'} "
            f"t_prod={'--' if t_prod is None else f'{t_prod:.6g} s'} "
            f"conv/prod={'--' if overhead is None else f'{overhead:.3g}x'}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
