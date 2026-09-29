#!/usr/bin/env python3
"""Production timing for independently qualified fixed Interaction parameters.

This benchmark intentionally performs no convergence search and no accuracy
reference calculation.  It consumes a completed Interaction convergence JSON,
selects configurations that already passed that benchmark's independent
reference, and measures repeated production cost with those parameters frozen.

Automatically certified rows use their selected production parameters.
For conservative finite-rule refusals, the minimum tested subdivision count
shown by the independent oracle to satisfy the target is used.  Other
conservative refusals are included only when their terminal configuration
itself passed the independent reference.  Rows with no demonstrated
reference-passing configuration are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import numpy as np

from benchmarks._common import environment_metadata, timed_call, write_json
from benchmarks._interaction_suite import (
    DELTA_DOMAINS,
    build_fields_from_harmonic_results,
    interaction_from_parameters,
    kernel_registry,
    validation_cases,
)


def parse_csv(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def make_displacements(domain: str, n: int) -> np.ndarray:
    lower, upper = DELTA_DOMAINS[domain]
    magnitudes = np.geomspace(lower, upper, int(n))
    angles = np.linspace(0.0, 1.75 * np.pi, int(n), endpoint=True)
    return np.column_stack((magnitudes * np.cos(angles), magnitudes * np.sin(angles)))


def _qualified_configuration(row: dict) -> tuple[str, dict, dict] | None:
    """Return qualification kind, fixed parameters, and reference error."""
    if bool(row.get("automatic_converged")) and bool(row.get("reference_pass")):
        error = row.get("selected_reference_error") or {}
        if not bool(error.get("passed")):
            return None
        return "automatic_certificate", dict(row.get("selected_parameters") or {}), dict(error)

    oracle = row.get("oracle") or {}
    minimum = oracle.get("minimum_tested_subdivisions")
    if minimum is not None:
        for item in oracle.get("history", []):
            if int(item.get("subdivisions", -1)) != int(minimum):
                continue
            if not bool(item.get("passed")):
                return None
            error = {
                "relative_l2": float(item["relative_l2"]),
                "relative_peak": float(item["relative_peak"]),
                "passed": True,
            }
            return (
                "independent_oracle_qualification",
                {"subdivisions": int(minimum)},
                error,
            )

    terminal = row.get("terminal_reference_error") or {}
    if bool(terminal.get("passed")):
        parameters = dict(row.get("selected_parameters") or {})
        if not parameters:
            return None
        return "independent_terminal_qualification", parameters, dict(terminal)
    return None


def _normalize_parameters(method: str, parameters: dict) -> dict:
    out = {"method": str(method), "interpolator": "cubic"}
    out.update(parameters)
    return out


def _summary(rows: list[dict]) -> dict:
    methods = sorted({row["method"] for row in rows})
    by_method = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        medians = [row["production_timing"]["median_seconds"] for row in selected]
        by_method.append(
            {
                "method": method,
                "n_qualified_rows": len(selected),
                "n_automatic": sum(row["qualification_kind"] == "automatic_certificate" for row in selected),
                "n_independently_qualified_after_refusal": sum(
                    row["qualification_kind"] != "automatic_certificate"
                    for row in selected
                ),
                "n_oracle_qualified_after_refusal": sum(
                    row["qualification_kind"] == "independent_oracle_qualification"
                    for row in selected
                ),
                "n_terminal_qualified_after_refusal": sum(
                    row["qualification_kind"] == "independent_terminal_qualification"
                    for row in selected
                ),
                "median_fixed_production_seconds": (
                    None if not medians else float(statistics.median(medians))
                ),
            }
        )

    by_case = []
    for case in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == case]
        fastest = min(selected, key=lambda row: row["production_timing"]["median_seconds"])
        fastest_time = float(fastest["production_timing"]["median_seconds"])
        entries = []
        for row in sorted(selected, key=lambda item: item["production_timing"]["median_seconds"]):
            timing = float(row["production_timing"]["median_seconds"])
            entries.append(
                {
                    "method": row["method"],
                    "qualification_kind": row["qualification_kind"],
                    "median_seconds": timing,
                    "relative_to_fastest": timing / fastest_time,
                }
            )
        by_case.append(
            {
                "case": case,
                "fastest_qualified_method": fastest["method"],
                "fastest_median_seconds": fastest_time,
                "methods": entries,
            }
        )
    return {"by_method": by_method, "by_case": by_case}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/fixed_configuration_performance_large_1e-3.json"),
    )
    parser.add_argument(
        "--qualification-results",
        type=Path,
        default=Path("benchmarks/results/interaction_convergence_large_1e-3.json"),
    )
    parser.add_argument(
        "--harmonic-results",
        type=Path,
        default=Path("benchmarks/results/harmonic_transform.json"),
    )
    parser.add_argument("--cases", default="default")
    parser.add_argument("--methods", default="simpson,gl4,fftlog,ogata")
    parser.add_argument("--delta-domain", default="large", choices=tuple(DELTA_DOMAINS))
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--upstream-target", type=float, default=1.0e-4)
    parser.add_argument("--upstream-method", default="simpson")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    source = json.loads(args.qualification_results.read_text())
    source_summary = source.get("summary", {})
    if not bool(source_summary.get("validation_passed")):
        raise ValueError("qualification results are not a passed validation dataset")
    source_git = source.get("environment", {}).get("git", {})

    methods = set(parse_csv(args.methods))
    available_cases = {item.name: item for item in validation_cases()}
    if args.cases == "default":
        case_names = tuple(
            dict.fromkeys(
                str(row["case"])
                for row in source.get("rows", [])
                if row.get("delta_domain") == args.delta_domain
                and row.get("method") in methods
                and np.isclose(float(row.get("requested_tolerance", np.nan)), args.tolerance)
            )
        )
    else:
        case_names = parse_csv(args.cases)
    unknown_cases = sorted(set(case_names) - set(available_cases))
    if unknown_cases:
        raise ValueError(f"unknown cases: {unknown_cases}")

    source_rows = [
        row
        for row in source.get("rows", [])
        if row.get("delta_domain") == args.delta_domain
        and row.get("case") in case_names
        and row.get("method") in methods
        and np.isclose(float(row.get("requested_tolerance", np.nan)), args.tolerance)
    ]
    if not source_rows:
        raise ValueError("qualification results contain no matching rows")

    n_by_domain = source.get("settings", {}).get("n_delta_by_domain", {})
    n_delta = int(n_by_domain.get(args.delta_domain, source.get("settings", {}).get("n_delta", 32)))
    deltas = make_displacements(args.delta_domain, n_delta)

    selected_case_names = tuple(dict.fromkeys(row["case"] for row in source_rows))
    needed_workloads = tuple(
        dict.fromkeys(available_cases[name].workload for name in selected_case_names)
    )
    fields, upstream_metadata = build_fields_from_harmonic_results(
        args.harmonic_results,
        target=args.upstream_target,
        method=args.upstream_method,
        workload_names=needed_workloads,
    )
    kernels = kernel_registry()

    rows: list[dict] = []
    for source_row in source_rows:
        qualified = _qualified_configuration(source_row)
        if qualified is None:
            continue
        qualification_kind, parameters, reference_error = qualified
        method = str(source_row["method"])
        case = available_cases[str(source_row["case"])]
        field = fields[case.workload]
        kernel = kernels[case.kernel]
        fixed_parameters = _normalize_parameters(method, parameters)

        print(
            f"[fixed production] {case.name} | {method} | "
            f"qualification={qualification_kind}",
            flush=True,
        )

        evaluate = partial(
            interaction_from_parameters,
            deltas,
            field,
            field,
            kernel,
            method,
            fixed_parameters,
        )
        value, timing = timed_call(evaluate, warmups=args.warmups, repeats=args.repeats)
        if value is None or not np.all(np.isfinite(value.V)):
            raise RuntimeError(f"non-finite fixed-configuration output for {case.name}/{method}")

        rows.append(
            {
                "case": case.name,
                "category": case.category,
                "workload": case.workload,
                "kernel": case.kernel,
                "delta_domain": args.delta_domain,
                "delta_min": float(DELTA_DOMAINS[args.delta_domain][0]),
                "delta_max": float(DELTA_DOMAINS[args.delta_domain][1]),
                "n_delta": n_delta,
                "method": method,
                "requested_tolerance": float(args.tolerance),
                "qualification_kind": qualification_kind,
                "source_classification": source_row.get("classification"),
                "fixed_parameters": fixed_parameters,
                "qualification_reference_error": reference_error,
                "production_timing": timing,
                "output_checksum": float(np.sum(np.abs(value.V))),
                "upstream_field": upstream_metadata[case.workload],
            }
        )

    if not rows:
        raise RuntimeError("no independently qualified configurations were found")

    result = {
        "schema": 1,
        "benchmark": "Interaction fixed-configuration production timing after independent qualification",
        "benchmark_class": "performance",
        "scope": {
            "accuracy_source": str(args.qualification_results),
            "qualification_rule": (
                "time automatically certified reference-passing parameters; for conservative "
                "finite-rule refusals use the minimum oracle-proven passing subdivision count; "
                "otherwise use terminal parameters only when the independent reference already passed"
            ),
            "timing_policy": (
                "no convergence search or external reference is executed in this benchmark; "
                "only fixed-parameter production evaluations are timed"
            ),
            "exclusion_policy": (
                "configurations without an independently demonstrated reference pass in the "
                "qualification benchmark are excluded"
            ),
        },
        "environment": environment_metadata(),
        "qualification_source": {
            "path": str(args.qualification_results),
            "schema": source.get("schema"),
            "benchmark": source.get("benchmark"),
            "git": source_git,
            "validation_passed": bool(source_summary.get("validation_passed")),
        },
        "settings": {
            "qualification_results": str(args.qualification_results),
            "harmonic_results": str(args.harmonic_results),
            "cases": list(case_names),
            "methods": sorted(methods),
            "delta_domain": args.delta_domain,
            "requested_tolerance": float(args.tolerance),
            "n_delta": n_delta,
            "upstream_target": float(args.upstream_target),
            "upstream_method": args.upstream_method,
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
        },
        "rows": rows,
        "summary": _summary(rows),
    }
    write_json(args.output, result)

    print("\n=== fixed-configuration production timing ===")
    for case in result["summary"]["by_case"]:
        methods_text = ", ".join(
            f"{item['method']}={1e3 * item['median_seconds']:.3g} ms"
            for item in case["methods"]
        )
        print(f"{case['case']:24s} {methods_text}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
