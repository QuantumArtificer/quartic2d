#!/usr/bin/env python3
"""Validate automatic Interaction convergence against an independent reference.

The convergence routine sees only the fixed momentum-space inputs and its own
successive numerical approximations.  The high-order direct reference is used
*afterward* as an oracle to detect false convergence and, for fixed-order finite
rules, to check whether the selected subdivision count is the cheapest tested
setting that actually meets the requested tolerance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quartic2d import Interaction

try:
    from _common import environment_metadata
    from _interaction_suite import (
        ALL_METHODS,
        DEFAULT_SUBDIVISIONS,
        FINITE_METHODS,
        build_fields_from_harmonic_results,
        direct_reference,
        displacement_grid,
        interaction_class,
        interaction_from_parameters,
        kernel_registry,
        relative_l2,
        relative_peak,
        validation_cases,
    )
except ImportError:  # pragma: no cover
    from paper.benchmarks._common import environment_metadata
    from paper.benchmarks._interaction_suite import (
        ALL_METHODS,
        DEFAULT_SUBDIVISIONS,
        FINITE_METHODS,
        build_fields_from_harmonic_results,
        direct_reference,
        displacement_grid,
        interaction_class,
        interaction_from_parameters,
        kernel_registry,
        relative_l2,
        relative_peak,
        validation_cases,
    )



def parse_csv(text, cast=str):
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def build_interaction(deltas, field, kernel, method, parameters):
    return interaction_from_parameters(deltas, field, field, kernel, method, parameters)


def finite_oracle(method, deltas, field, kernel, reference, tolerance, subdivisions):
    history = []
    chosen = None
    for s in subdivisions:
        value = np.asarray(
            build_interaction(
                deltas,
                field,
                kernel,
                method,
                {"subdivisions": int(s)},
            ).V,
            dtype=np.complex128,
        )
        e2 = relative_l2(value, reference)
        ep = relative_peak(value, reference)
        passed = bool(e2 <= tolerance and ep <= tolerance)
        history.append(
            {
                "subdivisions": int(s),
                "relative_l2": float(e2),
                "relative_peak": float(ep),
                "passed": passed,
            }
        )
        if passed and chosen is None:
            chosen = int(s)
            break
    return chosen, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("paper/benchmarks/results/interaction_auto_convergence.json"),
    )
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--harmonic-results", type=Path, default=Path("paper/benchmarks/results/harmonic_transform.json"))
    ap.add_argument("--upstream-target", type=float, default=1.0e-4)
    ap.add_argument("--upstream-method", default="simpson")
    ap.add_argument("--tolerances", default="1e-3,1e-4,1e-5")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--reference-order", type=int, default=24)
    ap.add_argument("--n-delta", type=int, default=32)
    args = ap.parse_args()

    methods = parse_csv(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    tolerances = parse_csv(args.tolerances, float)

    all_cases = validation_cases()
    needed_workloads = tuple(dict.fromkeys(case.workload for case in all_cases))
    fields, upstream_metadata = build_fields_from_harmonic_results(
        args.harmonic_results,
        target=args.upstream_target,
        method=args.upstream_method,
        workload_names=needed_workloads,
    )
    ks = kernel_registry()
    if args.cases != "all":
        wanted = set(parse_csv(args.cases))
        all_cases = [case for case in all_cases if case.name in wanted]
    deltas = displacement_grid(args.n_delta)

    references = {}
    reference_stability = {}
    for case in all_cases:
        field = fields[case.workload]
        kernel = ks[case.kernel]
        low = direct_reference(field, field, kernel, deltas, order=args.reference_order)
        high = direct_reference(field, field, kernel, deltas, order=2 * args.reference_order)
        references[case.name] = high
        reference_stability[case.name] = {
            "relative_l2_change": relative_l2(low, high),
            "relative_peak_change": relative_peak(low, high),
            "order_low": int(args.reference_order),
            "order_high": int(2 * args.reference_order),
        }

    rows = []
    false_positives = 0
    finite_misses = 0
    finite_nonminimal = 0

    for case in all_cases:
        field = fields[case.workload]
        kernel = ks[case.kernel]
        reference = references[case.name]
        for method in methods:
            for tolerance in tolerances:
                print(f"[{case.name}] {method} tol={tolerance:.0e}", flush=True)
                t0 = time.perf_counter()
                result = interaction_class(method).converge_parameters(
                    deltas,
                    field,
                    field,
                    kernel.U,
                    method=method,
                    rtol=tolerance,
                    atol=1.0e-12,
                    subdivisions=DEFAULT_SUBDIVISIONS,
                    verbose=False,
                )
                calibration_seconds = time.perf_counter() - t0

                selected_error = None
                reference_pass = None
                if result.converged:
                    interaction = result.interaction(deltas, field, field, kernel.U)
                    e2 = relative_l2(interaction.V, reference)
                    ep = relative_peak(interaction.V, reference)
                    reference_pass = bool(e2 <= tolerance and ep <= tolerance)
                    selected_error = {
                        "relative_l2": float(e2),
                        "relative_peak": float(ep),
                    }
                    if not reference_pass:
                        false_positives += 1

                oracle = None
                classification = "converged" if result.converged else "conservative_rejection"
                if method in FINITE_METHODS:
                    oracle_s, oracle_history = finite_oracle(
                        method,
                        deltas,
                        field,
                        kernel,
                        reference,
                        tolerance,
                        DEFAULT_SUBDIVISIONS,
                    )
                    selected_s = (
                        int(result.search.selected_parameters["subdivisions"])
                        if result.converged and "subdivisions" in result.search.selected_parameters
                        else None
                    )
                    oracle = {
                        "minimum_tested_subdivisions": oracle_s,
                        "history": oracle_history,
                        "selected_is_minimum_tested": bool(
                            selected_s is not None and oracle_s is not None and selected_s == oracle_s
                        ),
                    }
                    if oracle_s is not None and not result.converged:
                        finite_misses += 1
                        classification = "missed_convergence"
                    elif result.converged and oracle_s is not None and selected_s != oracle_s:
                        finite_nonminimal += 1
                        classification = "converged_nonminimal"
                    elif result.converged and reference_pass:
                        classification = "converged_minimal"

                rows.append(
                    {
                        "case": case.name,
                        "category": case.category,
                        "field": case.workload,
                        "kernel": case.kernel,
                        "method": method,
                        "requested_tolerance": float(tolerance),
                        "automatic_converged": bool(result.converged),
                        "selected_parameters": result.parameters if result.converged else result.search.selected_parameters,
                        "selected_reference_error": selected_error,
                        "reference_pass": reference_pass,
                        "classification": classification,
                        "calibration_seconds": float(calibration_seconds),
                        "oracle": oracle,
                        "convergence": result.to_dict(include_values=False),
                    }
                )

    result = {
        "schema": 1,
        "benchmark": "Interaction automatic-convergence validation",
        "scope": {
            "automatic_search_reference_access": "none; convergence uses only successive Interaction evaluations",
            "validation_oracle": "high-order direct q quadrature of the same fixed momentum-space interpolants and finite q support",
            "upstream_error_policy": "external to Interaction convergence and excluded from validation error",
            "finite_rule_validation": "automatic selection is compared with the minimum tested subdivision count satisfying both relative L2 and peak-normalized error targets",
            "fftlog_validation": "no false-positive convergence is allowed; conservative rejection is acceptable when bias robustness cannot be established",
        },
        "environment": environment_metadata(),
        "settings": {
            "methods": list(methods),
            "tolerances": [float(x) for x in tolerances],
            "subdivisions": list(DEFAULT_SUBDIVISIONS),
            "reference_order": int(args.reference_order),
            "n_delta": int(args.n_delta),
            "harmonic_results": str(args.harmonic_results),
            "upstream_target": float(args.upstream_target),
            "upstream_method": args.upstream_method,
        },
        "fixed_fields": upstream_metadata,
        "reference_stability": reference_stability,
        "rows": rows,
        "summary": {
            "n_rows": len(rows),
            "false_positive_convergence": int(false_positives),
            "finite_missed_convergence": int(finite_misses),
            "finite_nonminimal_selection": int(finite_nonminimal),
            "validation_passed": bool(
                false_positives == 0 and finite_misses == 0 and finite_nonminimal == 0
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("\n=== automatic convergence validation ===")
    print(f"false positives       : {false_positives}")
    print(f"finite missed         : {finite_misses}")
    print(f"finite non-minimal    : {finite_nonminimal}")
    print(f"validation            : {'PASS' if result['summary']['validation_passed'] else 'FAIL'}")
    print(f"Saved: {args.output}")
    if false_positives or finite_misses or finite_nonminimal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
