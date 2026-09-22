#!/usr/bin/env python3
"""Automatic-convergence performance benchmark for QUARTIC2D.

The benchmark measures parameter-selection cost independently from numerical
validation.  For each tested workload it records convergence wall time,
production timing at the selected parameters, production timing at the lowest
and highest tested resolution candidates, the number of convergence
evaluations when recoverable from the convergence record, and optional cProfile
hot spots.

The highest tested resolution is a performance baseline only.  Its numerical
sufficiency must be established by a separate reference/validation benchmark.
"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from quartic2d import HarmonicTransform, Interaction

from benchmarks._common import environment_metadata
from benchmarks._interaction_suite import (
    DEFAULT_SUBDIVISIONS,
    DELTA_DOMAINS,
    FFTLOG_BIAS_VALUES_LARGE,
    FFTLOG_BIAS_VALUES_STANDARD,
    FFTLOG_N_VALUES,
    FINITE_METHODS,
    LARGE_DELTA_SUBDIVISIONS,
    build_fields_from_harmonic_results,
    kernel_registry,
    validation_cases,
)
from benchmarks.publication.harmonic_transform import SyntheticDecomposition, workloads

SUPPORTED_HARMONIC_METHODS = ("trapezoid", "simpson", "gl4", "gl8")
SUPPORTED_INTERACTION_METHODS = FINITE_METHODS + ("fftlog", "ogata")
DEFAULT_HARMONIC_WORKLOADS = (
    "gaussian_isotropic",
    "gaussian_anisotropic_m2",
    "oscillatory_exponential",
)
DEFAULT_INTERACTION_CASES = (
    "isotropic_coulomb",
    "anisotropic_rk_strong",
    "nodal_rpa",
    "complex_gate",
)
HARMONIC_SUBDIVISIONS = (1, 2, 4, 8, 16, 32)
DEFAULT_N_VALUES = FFTLOG_N_VALUES
DEFAULT_BIAS_VALUES = FFTLOG_BIAS_VALUES_STANDARD
DEFAULT_OGATA_N_VALUES = (
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
)
def parse_csv(text: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def timing_summary(samples: Iterable[float]) -> dict:
    values = [float(x) for x in samples]
    if not values:
        return {
            "median_seconds": None,
            "q25_seconds": None,
            "q75_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "samples_seconds": [],
        }
    ordered = np.asarray(values, dtype=float)
    return {
        "median_seconds": float(statistics.median(values)),
        "q25_seconds": float(np.quantile(ordered, 0.25)),
        "q75_seconds": float(np.quantile(ordered, 0.75)),
        "min_seconds": float(np.min(ordered)),
        "max_seconds": float(np.max(ordered)),
        "samples_seconds": values,
    }


def time_callable(
    function: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict, object | None]:
    for _ in range(int(warmups)):
        function()
    samples = []
    last = None
    for _ in range(int(repeats)):
        start = time.perf_counter()
        last = function()
        samples.append(time.perf_counter() - start)
    return timing_summary(samples), last


def profile_callable(
    function: Callable[[], object],
    path: Path,
    *,
    top_n: int,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    profiler.dump_stats(path)

    stats = pstats.Stats(profiler)
    entries = []
    for (filename, line, name), (primitive, total, self_time, cumulative, _callers) in stats.stats.items():
        entries.append(
            {
                "function": name,
                "file": str(Path(filename).name),
                "line": int(line),
                "primitive_calls": int(primitive),
                "total_calls": int(total),
                "self_seconds": float(self_time),
                "cumulative_seconds": float(cumulative),
            }
        )
    entries.sort(key=lambda item: item["cumulative_seconds"], reverse=True)
    return {
        "pstats": str(path),
        "top_by_cumulative_time": entries[: int(top_n)],
    }


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def median_seconds(timing: dict | None) -> float | None:
    if not timing:
        return None
    value = timing.get("median_seconds")
    return None if value is None else float(value)


def derived_timing_metrics(
    calibration: dict,
    production: dict | None,
    floor: dict | None,
    upper: dict | None,
) -> dict:
    t_cal = median_seconds(calibration)
    t_prod = median_seconds(production)
    t_floor = median_seconds(floor)
    t_upper = median_seconds(upper)
    total = None if t_cal is None or t_prod is None else t_cal + t_prod
    return {
        "calibration_to_selected_production": ratio(t_cal, t_prod),
        "calibration_to_timing_floor": ratio(t_cal, t_floor),
        "calibration_plus_production_to_upper_candidate": ratio(total, t_upper),
    }


def _harmonic_rows(data: dict, requested_target: float) -> tuple[dict[str, dict], float]:
    targets = sorted({float(row["target"]) for row in data.get("rows", [])})
    admissible = [target for target in targets if target <= float(requested_target)]
    if not admissible:
        raise ValueError(
            f"no HarmonicTransform input benchmark target is at least as strict as "
            f"{requested_target:g}"
        )
    source_target = max(admissible)
    out = {
        str(row["workload"]): row
        for row in data.get("rows", [])
        if np.isclose(float(row.get("target", np.nan)), source_target, rtol=0.0, atol=1.0e-15)
    }
    return out, source_target


def _resolve_names(requested: str, defaults: tuple[str, ...], available: Iterable[str]) -> tuple[str, ...]:
    available = tuple(available)
    if requested == "default":
        names = defaults
    elif requested == "all":
        names = available
    else:
        names = parse_csv(requested)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown benchmark names: {unknown}")
    return tuple(names)


def _interaction_evaluation_count(search) -> int | None:
    kind = str(search.metadata.get("kind", ""))
    if kind == "fixed_order_richardson":
        return len(search.steps)
    if kind == "verified_sequence":
        return len(search.steps)
    if kind == "fftlog_resolution_and_bias":
        count = 0
        for step in search.steps:
            nested = step.metadata.get("resolution_search", {})
            count += len(nested.get("steps", []))
        for window in search.metadata.get("bias_windows", []):
            if window.get("resolution_converged"):
                count += len(window.get("biases", []))
        return count
    if kind == "ogata_coupled_N_h":
        count = 0
        for item in search.metadata.get("per_h", []):
            count += len(item.get("n_search", {}).get("steps", []))
        return count
    return None


def _recorded_evaluation_seconds(search) -> float | None:
    if not search.steps:
        return None
    return float(sum(float(step.runtime_seconds) for step in search.steps))


def make_displacements(domain: str, n: int) -> np.ndarray:
    lower, upper = DELTA_DOMAINS[domain]
    magnitudes = np.geomspace(lower, upper, int(n))
    angles = np.linspace(0.0, 1.75 * np.pi, int(n), endpoint=True)
    return np.column_stack((magnitudes * np.cos(angles), magnitudes * np.sin(angles)))


def interaction_class(method: str):
    return Interaction


def interaction_parameters(
    method: str,
    *,
    level: str,
    delta_domain: str,
    selected: dict | None = None,
) -> dict:
    if level == "selected":
        if selected is None:
            raise ValueError("selected parameters are required")
        return dict(selected)
    if method in FINITE_METHODS:
        ladder = DEFAULT_SUBDIVISIONS if delta_domain == "standard" else LARGE_DELTA_SUBDIVISIONS
        if level == "upper" and delta_domain == "large":
            # The large-delta ladder is intentionally deep enough for
            # certification. Repeating its terminal point is not a useful
            # performance baseline and can dominate the benchmark runtime.
            return {}
        subdivisions = ladder[0] if level == "floor" else ladder[-1]
        return {"method": method, "interpolator": "cubic", "subdivisions": int(subdivisions)}
    if method == "fftlog":
        n = DEFAULT_N_VALUES[0] if level == "floor" else DEFAULT_N_VALUES[-1]
        bias = -0.5 if delta_domain == "standard" else 0.0
        return {"method": method, "interpolator": "cubic", "n": int(n), "bias": float(bias)}
    if method == "ogata":
        if level == "floor":
            return {
                "method": method,
                "interpolator": "cubic",
                "N": int(DEFAULT_OGATA_N_VALUES[0]),
                "h": 0.05,
            }
        return {}
    raise ValueError(f"unknown method {method!r}")


def run_harmonic(
    *,
    result_path: Path,
    output_profile_dir: Path,
    workload_names: tuple[str, ...],
    methods: tuple[str, ...],
    tolerances: tuple[float, ...],
    q_tail_rtol: float,
    calibration_warmups: int,
    calibration_repeats: int,
    production_warmups: int,
    production_repeats: int,
    profile: bool,
    profile_top: int,
) -> list[dict]:
    data = json.loads(result_path.read_text())
    rows_by_workload, source_target = _harmonic_rows(data, min(tolerances))
    workload_map = {item.name: item for item in workloads()}
    rows = []

    for workload_name in workload_names:
        source_row = rows_by_workload.get(workload_name)
        if source_row is None:
            raise ValueError(
                f"no completed HarmonicTransform benchmark row for workload={workload_name!r} "
                f"at target={source_target:g}"
            )
        decomposition = SyntheticDecomposition(workload_map[workload_name], int(source_row["n_r"]))
        for method in methods:
            for tolerance in tolerances:
                print(
                    f"[HarmonicTransform] {workload_name} | {method} | rtol={tolerance:.0e}",
                    flush=True,
                )

                def calibrate():
                    return HarmonicTransform.converge_parameters(
                        decomposition,
                        rtol=tolerance,
                        atol=1.0e-12,
                        q_tail_rtol=q_tail_rtol,
                        method=method,
                        interpolator="cubic",
                        subdivisions=HARMONIC_SUBDIVISIONS,
                        verbose=False,
                    )

                try:
                    calibration, convergence = time_callable(
                        calibrate,
                        warmups=calibration_warmups,
                        repeats=calibration_repeats,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "stage": "HarmonicTransform",
                            "workload": workload_name,
                            "method": method,
                            "requested_tolerance": float(tolerance),
                            "status": "calibration_failed",
                            "exception": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                parameters = convergence.parameters
                production, _ = time_callable(
                    lambda: convergence.transform(decomposition, check=False),
                    warmups=production_warmups,
                    repeats=production_repeats,
                )
                fixed = {
                    "q_max": float(parameters["q_max"]),
                    "n_q": int(parameters["n_q"]),
                    "method": method,
                    "interpolator": "cubic",
                    "check": False,
                }
                if "q_grid" in parameters:
                    fixed["q_grid"] = parameters["q_grid"]
                floor, _ = time_callable(
                    lambda: HarmonicTransform(decomposition, subdivisions=1, **fixed),
                    warmups=production_warmups,
                    repeats=production_repeats,
                )
                upper, _ = time_callable(
                    lambda: HarmonicTransform(
                        decomposition,
                        subdivisions=int(HARMONIC_SUBDIVISIONS[-1]),
                        **fixed,
                    ),
                    warmups=production_warmups,
                    repeats=production_repeats,
                )

                profile_result = None
                if profile:
                    profile_name = (
                        f"harmonic__{workload_name}__{method}__rtol_{tolerance:.0e}.pstats"
                        .replace("+", "")
                    )
                    profile_result = profile_callable(
                        calibrate,
                        output_profile_dir / profile_name,
                        top_n=profile_top,
                    )

                quadrature_seconds = float(
                    sum(step.runtime_seconds for step in convergence.quadrature.steps)
                )
                t_cal = median_seconds(calibration)
                rows.append(
                    {
                        "stage": "HarmonicTransform",
                        "workload": workload_name,
                        "category": source_row.get("category"),
                        "method": method,
                        "requested_tolerance": float(tolerance),
                        "q_tail_rtol": float(q_tail_rtol),
                        "input_benchmark_target": float(source_target),
                        "status": "complete",
                        "selected_parameters": parameters,
                        "calibration_timing": calibration,
                        "selected_production_timing": production,
                        "timing_floor": {
                            "definition": "selected q support and q grid with subdivisions=1",
                            "timing": floor,
                        },
                        "upper_candidate_timing": {
                            "definition": f"selected q support and q grid with subdivisions={HARMONIC_SUBDIVISIONS[-1]}",
                            "numerical_sufficiency": "not established by this performance benchmark",
                            "timing": upper,
                        },
                        "timing_metrics": derived_timing_metrics(
                            calibration,
                            production,
                            floor,
                            upper,
                        ),
                        "search_work": {
                            "quadrature_evaluations": int(len(convergence.quadrature.steps)),
                            "q_interpolation_levels": int(len(convergence.sampling.interpolation_steps)),
                            "recorded_quadrature_seconds": quadrature_seconds,
                            "calibration_minus_recorded_quadrature_seconds": (
                                None if t_cal is None else float(max(0.0, t_cal - quadrature_seconds))
                            ),
                        },
                        "convergence": convergence.to_dict(),
                        "profile": profile_result,
                    }
                )
    return rows


def run_interaction(
    *,
    harmonic_result_path: Path,
    output_profile_dir: Path,
    case_names: tuple[str, ...],
    methods: tuple[str, ...],
    tolerances: tuple[float, ...],
    delta_domains: tuple[str, ...],
    n_delta: int,
    upstream_target: float,
    upstream_method: str,
    calibration_warmups: int,
    calibration_repeats: int,
    production_warmups: int,
    production_repeats: int,
    profile: bool,
    profile_top: int,
) -> list[dict]:
    cases = {item.name: item for item in validation_cases()}
    selected_cases = [cases[name] for name in case_names]
    needed_workloads = tuple(dict.fromkeys(item.workload for item in selected_cases))
    fields, upstream_metadata = build_fields_from_harmonic_results(
        harmonic_result_path,
        target=upstream_target,
        method=upstream_method,
        workload_names=needed_workloads,
    )
    kernels = kernel_registry()
    rows = []

    for case in selected_cases:
        field = fields[case.workload]
        kernel = kernels[case.kernel]
        for delta_domain in delta_domains:
            deltas = make_displacements(delta_domain, n_delta)
            for method in methods:
                cls = interaction_class(method)
                for tolerance in tolerances:
                    print(
                        f"[Interaction] {case.name} | delta={delta_domain} | {method} | "
                        f"rtol={tolerance:.0e}",
                        flush=True,
                    )

                    finite_ladder = (
                        DEFAULT_SUBDIVISIONS
                        if delta_domain == "standard"
                        else LARGE_DELTA_SUBDIVISIONS
                    )
                    fftlog_bias = -0.5 if delta_domain == "standard" else 0.0
                    fftlog_bias_values = (
                        FFTLOG_BIAS_VALUES_STANDARD
                        if delta_domain == "standard"
                        else FFTLOG_BIAS_VALUES_LARGE
                    )

                    def calibrate():
                        return cls.converge_parameters(
                            deltas,
                            field,
                            field,
                            kernel.U,
                            rtol=tolerance,
                            atol=1.0e-12,
                            method=method,
                            interpolator="cubic",
                            subdivisions=finite_ladder,
                            n_values=DEFAULT_N_VALUES,
                            bias=fftlog_bias,
                            bias_values=fftlog_bias_values,
                            ogata_n_values=DEFAULT_OGATA_N_VALUES,
                            verbose=False,
                        )

                    try:
                        calibration, convergence = time_callable(
                            calibrate,
                            warmups=calibration_warmups,
                            repeats=calibration_repeats,
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "stage": "Interaction",
                                "case": case.name,
                                "delta_domain": delta_domain,
                                "method": method,
                                "requested_tolerance": float(tolerance),
                                "status": "calibration_failed",
                                "exception": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue

                    selected_parameters = (
                        convergence.parameters
                        if convergence.converged
                        else {
                            "method": method,
                            "interpolator": "cubic",
                            **convergence.search.selected_parameters,
                        }
                    )
                    production = None
                    if convergence.converged:
                        production, _ = time_callable(
                            lambda: convergence.interaction(
                                deltas,
                                field,
                                field,
                                kernel.U,
                            ),
                            warmups=production_warmups,
                            repeats=production_repeats,
                        )

                    floor_parameters = interaction_parameters(method, level="floor", delta_domain=delta_domain)
                    floor, _ = time_callable(
                        lambda: cls(deltas, field, field, kernel.U, **floor_parameters),
                        warmups=production_warmups,
                        repeats=production_repeats,
                    )

                    upper = None
                    upper_parameters = interaction_parameters(method, level="upper", delta_domain=delta_domain)
                    if upper_parameters:
                        upper, _ = time_callable(
                            lambda: cls(deltas, field, field, kernel.U, **upper_parameters),
                            warmups=production_warmups,
                            repeats=production_repeats,
                        )

                    profile_result = None
                    if profile:
                        profile_name = (
                            f"interaction__{case.name}__delta_{delta_domain}__{method}__"
                            f"rtol_{tolerance:.0e}.pstats"
                        ).replace("+", "")
                        profile_result = profile_callable(
                            calibrate,
                            output_profile_dir / profile_name,
                            top_n=profile_top,
                        )

                    recorded_seconds = _recorded_evaluation_seconds(convergence.search)
                    t_cal = median_seconds(calibration)
                    rows.append(
                        {
                            "stage": "Interaction",
                            "case": case.name,
                            "category": case.category,
                            "workload": case.workload,
                            "kernel": case.kernel,
                            "delta_domain": delta_domain,
                            "delta_min": float(DELTA_DOMAINS[delta_domain][0]),
                            "delta_max": float(DELTA_DOMAINS[delta_domain][1]),
                            "n_delta": int(n_delta),
                            "method": method,
                            "requested_tolerance": float(tolerance),
                            "status": "complete" if convergence.converged else "not_converged",
                            "selected_parameters": selected_parameters,
                            "calibration_timing": calibration,
                            "selected_production_timing": production,
                            "timing_floor": {
                                "definition": "lowest tested production-resolution candidate",
                                "parameters": floor_parameters,
                                "timing": floor,
                            },
                            "upper_candidate_timing": (
                                None
                                if upper is None
                                else {
                                    "definition": "highest tested production-resolution candidate at the preferred auxiliary parameter",
                                    "parameters": upper_parameters,
                                    "numerical_sufficiency": "not established by this performance benchmark",
                                    "timing": upper,
                                }
                            ),
                            "timing_metrics": derived_timing_metrics(
                                calibration,
                                production,
                                floor,
                                upper,
                            ),
                            "search_work": {
                                "evaluation_count": _interaction_evaluation_count(convergence.search),
                                "recorded_evaluation_seconds": recorded_seconds,
                                "calibration_minus_recorded_evaluation_seconds": (
                                    None
                                    if t_cal is None or recorded_seconds is None
                                    else float(max(0.0, t_cal - recorded_seconds))
                                ),
                            },
                            "upstream_field": upstream_metadata[case.workload],
                            "convergence": convergence.to_dict(include_values=False),
                            "profile": profile_result,
                        }
                    )
    return rows


def method_summary(rows: list[dict]) -> list[dict]:
    keys = sorted(
        {
            (row.get("stage"), row.get("method"), row.get("delta_domain"))
            for row in rows
            if row.get("status") in {"complete", "not_converged"}
        }
    )
    out = []
    for stage, method, delta_domain in keys:
        selected = [
            row
            for row in rows
            if row.get("stage") == stage
            and row.get("method") == method
            and row.get("delta_domain") == delta_domain
        ]
        complete = [row for row in selected if row.get("status") == "complete"]
        calibration = [
            median_seconds(row.get("calibration_timing"))
            for row in selected
            if median_seconds(row.get("calibration_timing")) is not None
        ]
        production = [
            median_seconds(row.get("selected_production_timing"))
            for row in complete
            if median_seconds(row.get("selected_production_timing")) is not None
        ]
        overhead = [
            row.get("timing_metrics", {}).get("calibration_to_selected_production")
            for row in complete
        ]
        overhead = [float(x) for x in overhead if x is not None]
        out.append(
            {
                "stage": stage,
                "method": method,
                "delta_domain": delta_domain,
                "n_rows": len(selected),
                "n_converged": len(complete),
                "convergence_fraction": float(len(complete) / len(selected)) if selected else None,
                "median_calibration_seconds": (
                    None if not calibration else float(statistics.median(calibration))
                ),
                "median_selected_production_seconds": (
                    None if not production else float(statistics.median(production))
                ),
                "median_calibration_to_selected_production": (
                    None if not overhead else float(statistics.median(overhead))
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/publication/autoconvergence_performance.json"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("benchmarks/results/development/profiles/autoconvergence"),
    )
    parser.add_argument("--stages", default="harmonic,interaction")
    parser.add_argument(
        "--harmonic-results",
        type=Path,
        default=Path("benchmarks/results/publication/harmonic_transform.json"),
    )
    parser.add_argument("--harmonic-workloads", default="default")
    parser.add_argument("--interaction-cases", default="default")
    parser.add_argument("--methods", default="simpson,gl4,fftlog")
    parser.add_argument("--include-ogata", action="store_true")
    parser.add_argument("--tolerances", default="1e-4")
    parser.add_argument("--q-tail-rtol", type=float, default=1.0e-3)
    parser.add_argument("--upstream-target", type=float, default=1.0e-4)
    parser.add_argument("--upstream-method", default="simpson")
    parser.add_argument("--delta-domains", default="standard,large")
    parser.add_argument("--n-delta", type=int, default=32)
    parser.add_argument("--calibration-warmups", type=int, default=0)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--production-warmups", type=int, default=2)
    parser.add_argument("--production-repeats", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-top", type=int, default=25)
    args = parser.parse_args()

    stages = parse_csv(args.stages)
    unknown_stages = sorted(set(stages) - {"harmonic", "interaction"})
    if unknown_stages:
        raise ValueError(f"unknown stages: {unknown_stages}")

    tolerances = parse_csv(args.tolerances, float)
    if not tolerances or any(value <= 0.0 for value in tolerances):
        raise ValueError("tolerances must contain positive values")

    methods = parse_csv(args.methods)
    allowed_methods = set(SUPPORTED_INTERACTION_METHODS)
    if args.include_ogata:
        allowed_methods.add("ogata")
        if "ogata" not in methods:
            methods = methods + ("ogata",)
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise ValueError(f"unknown or unavailable methods: {unknown_methods}")

    harmonic_methods = tuple(method for method in methods if method in SUPPORTED_HARMONIC_METHODS)
    delta_domains = parse_csv(args.delta_domains)
    unknown_domains = sorted(set(delta_domains) - set(DELTA_DOMAINS))
    if unknown_domains:
        raise ValueError(f"unknown delta domains: {unknown_domains}")

    if args.calibration_repeats < 1 or args.production_repeats < 1:
        raise ValueError("repeat counts must be positive")
    if args.calibration_warmups < 0 or args.production_warmups < 0:
        raise ValueError("warmup counts must be non-negative")
    if args.n_delta < 2:
        raise ValueError("n_delta must be at least 2")

    workload_names_available = tuple(item.name for item in workloads() if not item.stress)
    harmonic_workloads = _resolve_names(
        args.harmonic_workloads,
        DEFAULT_HARMONIC_WORKLOADS,
        workload_names_available,
    )
    interaction_case_names_available = tuple(item.name for item in validation_cases())
    interaction_cases = _resolve_names(
        args.interaction_cases,
        DEFAULT_INTERACTION_CASES,
        interaction_case_names_available,
    )

    rows = []
    if "harmonic" in stages:
        rows.extend(
            run_harmonic(
                result_path=args.harmonic_results,
                output_profile_dir=args.profile_dir,
                workload_names=harmonic_workloads,
                methods=harmonic_methods,
                tolerances=tolerances,
                q_tail_rtol=args.q_tail_rtol,
                calibration_warmups=args.calibration_warmups,
                calibration_repeats=args.calibration_repeats,
                production_warmups=args.production_warmups,
                production_repeats=args.production_repeats,
                profile=args.profile,
                profile_top=args.profile_top,
            )
        )

    if "interaction" in stages:
        rows.extend(
            run_interaction(
                harmonic_result_path=args.harmonic_results,
                output_profile_dir=args.profile_dir,
                case_names=interaction_cases,
                methods=methods,
                tolerances=tolerances,
                delta_domains=delta_domains,
                n_delta=args.n_delta,
                upstream_target=args.upstream_target,
                upstream_method=args.upstream_method,
                calibration_warmups=args.calibration_warmups,
                calibration_repeats=args.calibration_repeats,
                production_warmups=args.production_warmups,
                production_repeats=args.production_repeats,
                profile=args.profile,
                profile_top=args.profile_top,
            )
        )

    result = {
        "schema": 1,
        "benchmark": "Automatic-convergence performance and profiling",
        "benchmark_class": "performance",
        "scope": {
            "calibration_timing": "wall time of automatic numerical-parameter selection",
            "selected_production_timing": "production evaluation using the selected parameters without rerunning convergence",
            "timing_floor": "production timing at the lowest tested numerical-resolution candidate for the same workload",
            "upper_candidate_timing": "production timing at the highest tested resolution candidate; numerical sufficiency is external to this performance benchmark",
            "profiling": "cProfile is executed in a separate untimed convergence run and does not contribute to reported timing samples",
            "validation_policy": "no accuracy or reference-solution claim is made by this benchmark",
            "delta_regime_policy": "standard and large-delta regimes use distinct, predeclared convergence search boxes; large delta extends finite-rule resolution and centers the FFTLog bias search near zero, while Ogata is the public oscillatory specialist",
        },
        "environment": environment_metadata(),
        "settings": {
            "stages": list(stages),
            "harmonic_workloads": list(harmonic_workloads),
            "interaction_cases": list(interaction_cases),
            "methods": list(methods),
            "harmonic_methods": list(harmonic_methods),
            "include_ogata": bool(args.include_ogata),
            "tolerances": [float(value) for value in tolerances],
            "q_tail_rtol": float(args.q_tail_rtol),
            "delta_domains": {
                name: {"delta_min": DELTA_DOMAINS[name][0], "delta_max": DELTA_DOMAINS[name][1]}
                for name in delta_domains
            },
            "n_delta": int(args.n_delta),
            "finite_subdivisions_standard": list(DEFAULT_SUBDIVISIONS),
            "finite_subdivisions_large": list(LARGE_DELTA_SUBDIVISIONS),
            "fftlog_n_values": list(DEFAULT_N_VALUES),
            "fftlog_bias_values_standard": list(FFTLOG_BIAS_VALUES_STANDARD),
            "fftlog_bias_values_large": list(FFTLOG_BIAS_VALUES_LARGE),
            "fftlog_preferred_bias_standard": -0.5,
            "fftlog_preferred_bias_large": 0.0,
            "calibration_warmups": int(args.calibration_warmups),
            "calibration_repeats": int(args.calibration_repeats),
            "production_warmups": int(args.production_warmups),
            "production_repeats": int(args.production_repeats),
            "profile": bool(args.profile),
            "profile_top": int(args.profile_top),
        },
        "rows": rows,
        "summary": method_summary(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("\n=== automatic-convergence performance ===")
    for row in result["summary"]:
        domain = "" if row["delta_domain"] is None else f" delta={row['delta_domain']}"
        cal = row["median_calibration_seconds"]
        prod = row["median_selected_production_seconds"]
        overhead = row["median_calibration_to_selected_production"]
        cal_text = "--" if cal is None else f"{cal:.6g} s"
        prod_text = "--" if prod is None else f"{prod:.6g} s"
        overhead_text = "--" if overhead is None else f"{overhead:.3g}x"
        print(
            f"{row['stage']:17s} {row['method']:10s}{domain:16s} "
            f"conv={row['n_converged']}/{row['n_rows']} "
            f"t_conv={cal_text} t_prod={prod_text} conv/prod={overhead_text}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
