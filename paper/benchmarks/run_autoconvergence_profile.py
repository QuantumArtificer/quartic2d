#!/usr/bin/env python3
"""Focused profiling of automatic-convergence routines.

The profiler isolates the principal convergence components used by the
supported finalist methods.  It is intended for implementation optimization,
not publication timing.  Timing results obtained under cProfile include
instrumentation overhead and must not be used as performance measurements.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path
from typing import Callable

import numpy as np

from quartic2d import HarmonicTransform, Interaction
from quartic2d._sampling import converge_q_sampling

try:
    from _common import environment_metadata
    from _interaction_suite import (
        DEFAULT_SUBDIVISIONS,
        build_fields_from_harmonic_results,
        kernel_registry,
        validation_cases,
    )
    from run_autoconvergence_performance import (
        DEFAULT_BIAS_VALUES,
        DEFAULT_N_VALUES,
        DELTA_DOMAINS,
        HARMONIC_SUBDIVISIONS,
        _harmonic_rows,
        _interaction_evaluation_count,
        _recorded_evaluation_seconds,
        make_displacements,
        parse_csv,
    )
    from run_harmonic_transform import SyntheticDecomposition, workloads
except ImportError:  # pragma: no cover
    from paper.benchmarks._common import environment_metadata
    from paper.benchmarks._interaction_suite import (
        DEFAULT_SUBDIVISIONS,
        build_fields_from_harmonic_results,
        kernel_registry,
        validation_cases,
    )
    from paper.benchmarks.run_autoconvergence_performance import (
        DEFAULT_BIAS_VALUES,
        DEFAULT_N_VALUES,
        DELTA_DOMAINS,
        HARMONIC_SUBDIVISIONS,
        _harmonic_rows,
        _interaction_evaluation_count,
        _recorded_evaluation_seconds,
        make_displacements,
        parse_csv,
    )
    from paper.benchmarks.run_harmonic_transform import SyntheticDecomposition, workloads


DEFAULT_HARMONIC_WORKLOAD = "gaussian_anisotropic_m2"
DEFAULT_INTERACTION_CASE = "nodal_rpa"
DEFAULT_METHODS = ("simpson", "gl4", "fftlog")
DEFAULT_TOLERANCE = 1.0e-4
DEFAULT_N_DELTA = 8


def _profile(function: Callable[[], object], path: Path, *, top_n: int) -> tuple[object, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(function)
    wall_seconds = float(time.perf_counter() - start)
    profiler.dump_stats(path)

    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, name), values in stats.stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = values
        entries.append(
            {
                "function": name,
                "file": str(Path(filename).name),
                "line": int(line),
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "self_seconds": float(self_seconds),
                "cumulative_seconds": float(cumulative_seconds),
            }
        )

    by_cumulative = sorted(
        entries,
        key=lambda item: item["cumulative_seconds"],
        reverse=True,
    )[: int(top_n)]
    by_self = sorted(
        entries,
        key=lambda item: item["self_seconds"],
        reverse=True,
    )[: int(top_n)]
    return result, {
        "wall_seconds_profiled": wall_seconds,
        "pstats": str(path),
        "top_by_cumulative_time": by_cumulative,
        "top_by_self_time": by_self,
    }


def _load_harmonic_workload(result_path: Path, workload_name: str, target: float):
    data = json.loads(result_path.read_text())
    rows_by_workload, source_target = _harmonic_rows(data, target)
    source_row = rows_by_workload.get(workload_name)
    if source_row is None:
        raise ValueError(
            f"no completed HarmonicTransform benchmark row for workload={workload_name!r} "
            f"at target={source_target:g}"
        )
    workload_map = {item.name: item for item in workloads()}
    decomposition = SyntheticDecomposition(workload_map[workload_name], int(source_row["n_r"]))
    return decomposition, source_row, source_target


def profile_harmonic_sampling(
    *,
    harmonic_result_path: Path,
    workload_name: str,
    tolerance: float,
    atol: float,
    q_tail_rtol: float,
    subdivisions: int,
    profile_dir: Path,
    top_n: int,
) -> dict:
    decomposition, source_row, source_target = _load_harmonic_workload(
        harmonic_result_path,
        workload_name,
        tolerance,
    )

    function = lambda: converge_q_sampling(
        decomposition,
        method="simpson",
        interpolator="cubic",
        subdivisions=int(subdivisions),
        q_tail_rtol=float(q_tail_rtol),
        interpolation_rtol=float(tolerance),
        interpolation_atol=float(atol),
    )
    result, profile = _profile(
        function,
        profile_dir / f"harmonic_sampling__{workload_name}__simpson.pstats",
        top_n=top_n,
    )
    return {
        "stage": "HarmonicTransform",
        "component": "q_sampling",
        "workload": workload_name,
        "method": "simpson",
        "requested_tolerance": float(tolerance),
        "q_tail_rtol": float(q_tail_rtol),
        "subdivisions": int(subdivisions),
        "input_benchmark_target": float(source_target),
        "input_n_r": int(source_row["n_r"]),
        "result": result.to_dict(),
        "profile": profile,
    }


def profile_harmonic_quadrature(
    *,
    harmonic_result_path: Path,
    workload_name: str,
    methods: tuple[str, ...],
    tolerance: float,
    atol: float,
    profile_dir: Path,
    top_n: int,
) -> list[dict]:
    decomposition, source_row, source_target = _load_harmonic_workload(
        harmonic_result_path,
        workload_name,
        tolerance,
    )
    q_max = float(source_row["q_support"]["q_max"])
    n_q = int(source_row["n_q_selected"])
    rows = []

    for method in methods:
        if method == "fftlog":
            continue
        transform = HarmonicTransform(
            decomposition,
            q_max=q_max,
            n_q=n_q,
            method=method,
            interpolator="cubic",
            subdivisions=1,
            check=False,
        )
        function = lambda transform=transform: transform.converge(
            rtol=float(tolerance),
            atol=float(atol),
            subdivisions=HARMONIC_SUBDIVISIONS,
            apply=False,
        )
        result, profile = _profile(
            function,
            profile_dir / f"harmonic_quadrature__{workload_name}__{method}.pstats",
            top_n=top_n,
        )
        rows.append(
            {
                "stage": "HarmonicTransform",
                "component": "radial_quadrature",
                "workload": workload_name,
                "method": method,
                "requested_tolerance": float(tolerance),
                "fixed_q_max": q_max,
                "fixed_n_q": n_q,
                "input_benchmark_target": float(source_target),
                "result": result.to_dict(include_values=False),
                "profile": profile,
            }
        )
    return rows


def profile_interaction(
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
    profile_dir: Path,
    top_n: int,
) -> list[dict]:
    cases = {item.name: item for item in validation_cases()}
    case = cases[case_name]
    fields, upstream_metadata = build_fields_from_harmonic_results(
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
            function = lambda method=method, deltas=deltas: Interaction.converge_parameters(
                deltas,
                field,
                field,
                kernel.U,
                rtol=float(tolerance),
                atol=float(atol),
                method=method,
                interpolator="cubic",
                subdivisions=DEFAULT_SUBDIVISIONS,
                n_values=DEFAULT_N_VALUES,
                bias=-0.5,
                bias_values=DEFAULT_BIAS_VALUES,
                verbose=False,
            )
            result, profile = _profile(
                function,
                profile_dir / f"interaction__{case_name}__{delta_domain}__{method}.pstats",
                top_n=top_n,
            )
            recorded_seconds = _recorded_evaluation_seconds(result.search)
            wall_seconds = float(profile["wall_seconds_profiled"])
            rows.append(
                {
                    "stage": "Interaction",
                    "component": "parameter_convergence",
                    "case": case.name,
                    "workload": case.workload,
                    "kernel": case.kernel,
                    "delta_domain": delta_domain,
                    "delta_min": float(DELTA_DOMAINS[delta_domain][0]),
                    "delta_max": float(DELTA_DOMAINS[delta_domain][1]),
                    "n_delta": int(n_delta),
                    "method": method,
                    "requested_tolerance": float(tolerance),
                    "converged": bool(result.converged),
                    "selected_parameters": (
                        result.parameters
                        if result.converged
                        else {
                            "method": method,
                            "interpolator": "cubic",
                            **result.search.selected_parameters,
                        }
                    ),
                    "search_work": {
                        "evaluation_count": _interaction_evaluation_count(result.search),
                        "recorded_evaluation_seconds": recorded_seconds,
                        "profiled_wall_minus_recorded_evaluations_seconds": (
                            None
                            if recorded_seconds is None
                            else float(max(0.0, wall_seconds - recorded_seconds))
                        ),
                    },
                    "convergence": result.to_dict(include_values=False),
                    "upstream": upstream_metadata,
                    "profile": profile,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/benchmarks/results/autoconvergence_profile.json"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("paper/benchmarks/results/profiles/autoconvergence"),
    )
    parser.add_argument(
        "--harmonic-results",
        type=Path,
        default=Path("paper/benchmarks/results/harmonic_transform.json"),
    )
    parser.add_argument("--harmonic-workload", default=DEFAULT_HARMONIC_WORKLOAD)
    parser.add_argument("--interaction-case", default=DEFAULT_INTERACTION_CASE)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--components", default="harmonic_sampling,harmonic_quadrature,interaction")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--q-tail-rtol", type=float, default=1.0e-3)
    parser.add_argument("--sampling-subdivisions", type=int, default=2)
    parser.add_argument("--upstream-target", type=float, default=1.0e-4)
    parser.add_argument("--upstream-method", default="simpson")
    parser.add_argument("--delta-domains", default="standard,large")
    parser.add_argument("--n-delta", type=int, default=DEFAULT_N_DELTA)
    parser.add_argument("--profile-top", type=int, default=40)
    args = parser.parse_args()

    methods = parse_csv(args.methods)
    unknown_methods = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown_methods:
        raise ValueError(f"unknown profiling methods: {unknown_methods}")
    components = parse_csv(args.components)
    allowed_components = {"harmonic_sampling", "harmonic_quadrature", "interaction"}
    unknown_components = sorted(set(components) - allowed_components)
    if unknown_components:
        raise ValueError(f"unknown profiling components: {unknown_components}")
    delta_domains = parse_csv(args.delta_domains)
    unknown_domains = sorted(set(delta_domains) - set(DELTA_DOMAINS))
    if unknown_domains:
        raise ValueError(f"unknown delta domains: {unknown_domains}")
    if args.tolerance <= 0.0 or args.atol < 0.0 or args.q_tail_rtol <= 0.0:
        raise ValueError("tolerances must be positive and atol must be non-negative")
    if args.sampling_subdivisions < 1:
        raise ValueError("sampling_subdivisions must be positive")
    if args.n_delta < 2:
        raise ValueError("n_delta must be at least 2")
    if args.profile_top < 1:
        raise ValueError("profile_top must be positive")

    rows = []
    if "harmonic_sampling" in components:
        print("[profile] HarmonicTransform q sampling | simpson", flush=True)
        rows.append(
            profile_harmonic_sampling(
                harmonic_result_path=args.harmonic_results,
                workload_name=args.harmonic_workload,
                tolerance=args.tolerance,
                atol=args.atol,
                q_tail_rtol=args.q_tail_rtol,
                subdivisions=args.sampling_subdivisions,
                profile_dir=args.profile_dir,
                top_n=args.profile_top,
            )
        )
    if "harmonic_quadrature" in components:
        harmonic_methods = tuple(method for method in methods if method in {"simpson", "gl4"})
        rows.extend(
            profile_harmonic_quadrature(
                harmonic_result_path=args.harmonic_results,
                workload_name=args.harmonic_workload,
                methods=harmonic_methods,
                tolerance=args.tolerance,
                atol=args.atol,
                profile_dir=args.profile_dir,
                top_n=args.profile_top,
            )
        )
    if "interaction" in components:
        print("[profile] Interaction parameter convergence", flush=True)
        rows.extend(
            profile_interaction(
                harmonic_result_path=args.harmonic_results,
                case_name=args.interaction_case,
                methods=methods,
                tolerance=args.tolerance,
                atol=args.atol,
                delta_domains=delta_domains,
                n_delta=args.n_delta,
                upstream_target=args.upstream_target,
                upstream_method=args.upstream_method,
                profile_dir=args.profile_dir,
                top_n=args.profile_top,
            )
        )

    result = {
        "schema": 1,
        "benchmark": "Automatic-convergence focused profile",
        "benchmark_class": "implementation_profile",
        "publication_use": False,
        "scope": {
            "purpose": "identify implementation bottlenecks in automatic parameter convergence",
            "timing_policy": "one cProfile-instrumented run per component; profiled wall times are not publication performance measurements",
            "harmonic_sampling_policy": "q support and q-grid interpolation are profiled separately using Simpson quadrature",
            "harmonic_quadrature_policy": "radial quadrature convergence is profiled at fixed q support and q-grid density",
            "interaction_policy": "complete method-specific parameter convergence is profiled on the reduced displacement set",
        },
        "environment": environment_metadata(),
        "settings": {
            "components": list(components),
            "harmonic_workload": args.harmonic_workload,
            "interaction_case": args.interaction_case,
            "methods": list(methods),
            "tolerance": float(args.tolerance),
            "atol": float(args.atol),
            "q_tail_rtol": float(args.q_tail_rtol),
            "sampling_subdivisions": int(args.sampling_subdivisions),
            "delta_domains": {
                name: {"delta_min": DELTA_DOMAINS[name][0], "delta_max": DELTA_DOMAINS[name][1]}
                for name in delta_domains
            },
            "n_delta": int(args.n_delta),
            "upstream_target": float(args.upstream_target),
            "upstream_method": args.upstream_method,
            "profile_top": int(args.profile_top),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")
    print(f"Profiles: {args.profile_dir}")


if __name__ == "__main__":
    main()
