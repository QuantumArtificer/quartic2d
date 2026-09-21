#!/usr/bin/env python3
"""Validate automatic Interaction convergence across displacement regimes.

This benchmark is the correctness counterpart to
``run_autoconvergence_performance.py``.  It validates standard and large
separations against independent direct references, reports finite-rule
selection efficiency, and explicitly separates FFTLog backend capability from
FFTLog auto-selector coverage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks._common import environment_metadata
from benchmarks._interaction_suite import (
    ALL_METHODS,
    DEFAULT_SUBDIVISIONS,
    DELTA_DOMAINS,
    FFTLOG_BIAS_VALUES_LARGE,
    FFTLOG_BIAS_VALUES_STANDARD,
    FFTLOG_N_VALUES,
    FINITE_METHODS,
    LARGE_DELTA_SUBDIVISIONS,
    build_fields_from_harmonic_results,
    direct_reference,
    displacement_grid,
    interaction_class,
    interaction_from_parameters,
    kernel_registry,
    relative_l2,
    relative_peak,
    stable_reference_pair,
    validation_cases,
)

DEFAULT_INTERACTION_CASES = (
    "isotropic_coulomb",
    "anisotropic_rk_strong",
    "nodal_rpa",
    "complex_gate",
)

DEFAULT_OGATA_N_VALUES = (
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288,
)


def parse_csv(text, cast=str):
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def build_interaction(deltas, field, kernel, method, parameters):
    return interaction_from_parameters(deltas, field, field, kernel, method, parameters)


def error_record(value, reference, tolerance):
    value = np.asarray(value, dtype=np.complex128)
    e2 = relative_l2(value, reference)
    ep = relative_peak(value, reference)
    return {
        "relative_l2": float(e2),
        "relative_peak": float(ep),
        "passed": bool(e2 <= tolerance and ep <= tolerance),
    }


def finite_oracle(method, deltas, field, kernel, reference, tolerance, subdivisions):
    history = []
    chosen = None
    for s in subdivisions:
        value = build_interaction(
            deltas, field, kernel, method, {"subdivisions": int(s)}
        ).V
        error = error_record(value, reference, tolerance)
        history.append({"subdivisions": int(s), **error})
        if error["passed"] and chosen is None:
            chosen = int(s)
            break
    return chosen, history


def fftlog_capability_sweep(
    deltas,
    field,
    kernel,
    reference,
    tolerance,
    *,
    n_values,
    bias_values,
):
    """Oracle-test the fixed FFTLog search box without changing certification."""
    history = []
    best = None
    n_passing = 0
    for bias in bias_values:
        for n in n_values:
            value = build_interaction(
                deltas,
                field,
                kernel,
                "fftlog",
                {"n": int(n), "bias": float(bias)},
            ).V
            error = error_record(value, reference, tolerance)
            row = {"n": int(n), "bias": float(bias), **error}
            history.append(row)
            if error["passed"]:
                n_passing += 1
            score = max(error["relative_l2"], error["relative_peak"])
            if best is None or score < best[0]:
                best = (score, row)
    return {
        "n_tested": len(history),
        "n_reference_passing": int(n_passing),
        "capability_found": bool(n_passing > 0),
        "best_tested": None if best is None else best[1],
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/publication/interaction_convergence.json"),
    )
    ap.add_argument("--methods", default="simpson,gl4,fftlog,ogata")
    ap.add_argument(
        "--harmonic-results",
        type=Path,
        default=Path("benchmarks/results/publication/harmonic_transform.json"),
    )
    ap.add_argument("--upstream-target", type=float, default=1.0e-4)
    ap.add_argument("--upstream-method", default="simpson")
    ap.add_argument("--tolerances", default="1e-4")
    ap.add_argument("--cases", default="default")
    ap.add_argument("--delta-domains", default="standard,large")
    ap.add_argument("--reference-order", type=int, default=24)
    ap.add_argument("--reference-budget-fraction", type=float, default=0.1)
    ap.add_argument("--large-reference-coarse-order", type=int, default=8)
    ap.add_argument("--large-reference-coarse-phase-step", type=float, default=float(np.pi))
    ap.add_argument("--large-reference-max-levels", type=int, default=4)
    ap.add_argument("--n-delta", type=int, default=32)
    ap.add_argument(
        "--subdivisions-standard",
        default=",".join(str(x) for x in DEFAULT_SUBDIVISIONS),
    )
    ap.add_argument(
        "--subdivisions-large",
        default=",".join(str(x) for x in LARGE_DELTA_SUBDIVISIONS),
    )
    ap.add_argument(
        "--fftlog-n-values",
        default=",".join(str(x) for x in FFTLOG_N_VALUES),
    )
    ap.add_argument(
        "--fftlog-bias-values-standard",
        default=",".join(str(x) for x in FFTLOG_BIAS_VALUES_STANDARD),
    )
    ap.add_argument(
        "--fftlog-bias-values-large",
        default=",".join(str(x) for x in FFTLOG_BIAS_VALUES_LARGE),
    )
    ap.add_argument("--fftlog-preferred-bias-standard", type=float, default=-0.5)
    ap.add_argument("--fftlog-preferred-bias-large", type=float, default=0.0)
    ap.add_argument("--no-fftlog-capability-sweep", action="store_true")
    ap.add_argument("--ogata-hstart", type=float, default=0.05)
    ap.add_argument("--ogata-hdecrement", type=float, default=2.0)
    ap.add_argument("--ogata-maxiter", type=int, default=20)
    ap.add_argument(
        "--ogata-n-values",
        default=",".join(str(x) for x in DEFAULT_OGATA_N_VALUES),
    )
    args = ap.parse_args()

    methods = parse_csv(args.methods)
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    tolerances = parse_csv(args.tolerances, float)
    domains = parse_csv(args.delta_domains)
    unknown_domains = sorted(set(domains) - set(DELTA_DOMAINS))
    if unknown_domains:
        raise ValueError(f"unknown delta domains: {unknown_domains}")
    subdivisions_by_domain = {
        "standard": parse_csv(args.subdivisions_standard, int),
        "large": parse_csv(args.subdivisions_large, int),
    }
    fftlog_n_values = parse_csv(args.fftlog_n_values, int)
    fftlog_bias_values = {
        "standard": parse_csv(args.fftlog_bias_values_standard, float),
        "large": parse_csv(args.fftlog_bias_values_large, float),
    }
    fftlog_preferred_bias = {
        "standard": float(args.fftlog_preferred_bias_standard),
        "large": float(args.fftlog_preferred_bias_large),
    }
    ogata_n_values = parse_csv(args.ogata_n_values, int)

    all_available_cases = validation_cases()
    if args.cases == "default":
        wanted = set(DEFAULT_INTERACTION_CASES)
        all_cases = [case for case in all_available_cases if case.name in wanted]
    elif args.cases == "all":
        all_cases = all_available_cases
    else:
        wanted = set(parse_csv(args.cases))
        all_cases = [case for case in all_available_cases if case.name in wanted]
        missing = sorted(wanted - {case.name for case in all_cases})
        if missing:
            raise ValueError(f"unknown interaction cases: {missing}")
    needed_workloads = tuple(dict.fromkeys(case.workload for case in all_cases))
    fields, upstream_metadata = build_fields_from_harmonic_results(
        args.harmonic_results,
        target=args.upstream_target,
        method=args.upstream_method,
        workload_names=needed_workloads,
    )
    kernels = kernel_registry()

    reference_limit = float(min(tolerances) * args.reference_budget_fraction)
    references = {}
    reference_stability = {}
    for domain in domains:
        deltas = displacement_grid(args.n_delta, domain=domain)
        for case in all_cases:
            field = fields[case.workload]
            kernel = kernels[case.kernel]
            print(f"[reference] {domain} | {case.name}", flush=True)
            low, high, reference_metadata = stable_reference_pair(
                field,
                field,
                kernel,
                deltas,
                domain=domain,
                reference_order=args.reference_order,
                reference_limit=reference_limit,
                large_coarse_order=args.large_reference_coarse_order,
                large_coarse_phase_step=args.large_reference_coarse_phase_step,
                large_max_levels=args.large_reference_max_levels,
            )
            e2 = float(reference_metadata["relative_l2_change"])
            ep = float(reference_metadata["relative_peak_change"])
            passed = bool(reference_metadata["converged"])
            key = (domain, case.name)
            references[key] = high
            reference_stability[f"{domain}::{case.name}"] = {
                **reference_metadata,
                "required_max_change": reference_limit,
                "passed": passed,
            }
            print(f"  dL2={e2:.3e} dPk={ep:.3e} {'PASS' if passed else 'FAIL'}", flush=True)
    unstable = [name for name, item in reference_stability.items() if not item["passed"]]
    if unstable:
        raise RuntimeError("interaction reference stability failed for: " + ", ".join(unstable))

    rows = []
    false_positives = 0
    finite_misses = 0
    finite_nonminimal = 0

    for domain in domains:
        deltas = displacement_grid(args.n_delta, domain=domain)
        subdivisions = subdivisions_by_domain[domain]
        biases = fftlog_bias_values[domain]
        preferred_bias = fftlog_preferred_bias[domain]
        for case in all_cases:
            field = fields[case.workload]
            kernel = kernels[case.kernel]
            reference = references[(domain, case.name)]
            for method in methods:
                for tolerance in tolerances:
                    print(
                        f"[{domain}] {case.name} | {method} tol={tolerance:.0e}",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    result = interaction_class(method).converge_parameters(
                        deltas,
                        field,
                        field,
                        kernel.U,
                        method=method,
                        rtol=tolerance,
                        atol=1.0e-12,
                        subdivisions=subdivisions,
                        n_values=fftlog_n_values,
                        bias=preferred_bias,
                        bias_values=biases,
                        hstart=args.ogata_hstart,
                        hdecrement=args.ogata_hdecrement,
                        maxiter=args.ogata_maxiter,
                        ogata_n_values=ogata_n_values,
                        verbose=False,
                    )
                    calibration_seconds = time.perf_counter() - t0

                    selected_parameters = (
                        result.parameters if result.converged else result.search.selected_parameters
                    )
                    selected_error = None
                    reference_pass = None
                    if result.converged:
                        interaction = result.interaction(deltas, field, field, kernel.U)
                        selected_error = error_record(interaction.V, reference, tolerance)
                        reference_pass = bool(selected_error["passed"])
                        if not reference_pass:
                            false_positives += 1

                    terminal_error = None
                    if not result.converged and selected_parameters:
                        try:
                            terminal = build_interaction(
                                deltas, field, kernel, method, selected_parameters
                            )
                            terminal_error = error_record(terminal.V, reference, tolerance)
                        except Exception:
                            terminal_error = None

                    oracle = None
                    capability = None
                    classification = "converged" if result.converged else "conservative_rejection"
                    if method in FINITE_METHODS:
                        oracle_s, oracle_history = finite_oracle(
                            method,
                            deltas,
                            field,
                            kernel,
                            reference,
                            tolerance,
                            subdivisions,
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
                    elif method == "fftlog" and not args.no_fftlog_capability_sweep:
                        capability = fftlog_capability_sweep(
                            deltas,
                            field,
                            kernel,
                            reference,
                            tolerance,
                            n_values=fftlog_n_values,
                            bias_values=biases,
                        )
                        if not result.converged and capability["capability_found"]:
                            classification = "selector_miss_backend_capable"
                        elif result.converged and reference_pass:
                            classification = "converged_reference_pass"
                        elif not capability["capability_found"]:
                            classification = "tested_box_not_capable"
                    elif method == "ogata":
                        if result.converged and reference_pass:
                            classification = "converged_reference_pass"
                        elif terminal_error is not None and terminal_error["passed"]:
                            classification = "conservative_rejection_terminal_pass"

                    rows.append(
                        {
                            "delta_domain": domain,
                            "delta_min": float(DELTA_DOMAINS[domain][0]),
                            "delta_max": float(DELTA_DOMAINS[domain][1]),
                            "case": case.name,
                            "category": case.category,
                            "field": case.workload,
                            "kernel": case.kernel,
                            "method": method,
                            "requested_tolerance": float(tolerance),
                            "automatic_converged": bool(result.converged),
                            "selected_parameters": selected_parameters,
                            "selected_reference_error": selected_error,
                            "reference_pass": reference_pass,
                            "terminal_reference_error": terminal_error,
                            "classification": classification,
                            "calibration_seconds": float(calibration_seconds),
                            "oracle": oracle,
                            "capability": capability,
                            "convergence": result.to_dict(include_values=False),
                        }
                    )

    finite_rows = [row for row in rows if row["method"] in FINITE_METHODS]
    finite_oracle_rows = [
        row for row in finite_rows
        if row.get("oracle") is not None
        and row["oracle"].get("minimum_tested_subdivisions") is not None
    ]
    finite_minimal = [row for row in finite_oracle_rows if row["oracle"]["selected_is_minimum_tested"]]
    finite_nonminimal_rows = [
        row for row in finite_oracle_rows
        if row["automatic_converged"] and not row["oracle"]["selected_is_minimum_tested"]
    ]

    method_domain_summary = []
    for domain in domains:
        for method in methods:
            selected = [row for row in rows if row["delta_domain"] == domain and row["method"] == method]
            if not selected:
                continue
            certified = [row for row in selected if row["automatic_converged"]]
            passed = [row for row in certified if row.get("reference_pass") is True]
            capability_found = [
                row for row in selected
                if row.get("capability") is not None and row["capability"].get("capability_found")
            ]
            method_domain_summary.append(
                {
                    "delta_domain": domain,
                    "method": method,
                    "n_rows": len(selected),
                    "n_automatic_converged": len(certified),
                    "n_certified_reference_pass": len(passed),
                    "n_false_positive": sum(row.get("reference_pass") is False for row in certified),
                    "n_backend_capability_found": len(capability_found),
                    "n_selector_miss_backend_capable": sum(
                        row["classification"] == "selector_miss_backend_capable" for row in selected
                    ),
                }
            )

    coverage = []
    for domain in domains:
        for case in all_cases:
            for tolerance in tolerances:
                selected = [
                    row for row in rows
                    if row["delta_domain"] == domain
                    and row["case"] == case.name
                    and np.isclose(row["requested_tolerance"], tolerance)
                ]
                passing = [
                    row["method"] for row in selected
                    if row["automatic_converged"] and row.get("reference_pass") is True
                ]
                coverage.append(
                    {
                        "delta_domain": domain,
                        "case": case.name,
                        "requested_tolerance": float(tolerance),
                        "passing_methods": passing,
                        "covered": bool(passing),
                    }
                )

    # A conservative rejection is not a wrong numerical answer.  Treat
    # reference-passing terminal finite-rule values as selector-efficiency
    # misses, while correctness is reserved for false-positive certificates.
    correctness_passed = bool(false_positives == 0)
    coverage_passed = bool(coverage and all(item["covered"] for item in coverage))
    validation_passed = bool(correctness_passed and coverage_passed)

    result = {
        "schema": 3,
        "benchmark": "Interaction automatic-convergence validation across delta regimes",
        "scope": {
            "automatic_search_reference_access": "none; convergence uses only successive Interaction evaluations",
            "standard_reference": "high-order direct q quadrature of the same fixed momentum-space interpolants",
            "large_delta_reference": "independent phase-resolved composite Gauss-Legendre quadrature with delta*dq controlled explicitly; coarse/fine reference agreement is required before backend validation",
            "upstream_error_policy": "external to Interaction convergence and excluded from validation error",
            "finite_rule_validation": "automatic selection is compared with the minimum tested subdivision count satisfying both relative L2 and peak-normalized error targets; conservative missed certificates are reported as selector-efficiency misses, not correctness failures",
            "fftlog_capability": "the complete benchmark search box is oracle-tested separately from the automatic certificate so non-monotone n/bias behavior cannot be misreported as backend incapability",
            "ogata_policy": "Ogata remains benchmark/private but is validated on the same large-delta references as public methods",
            "coverage_policy": "every case, tolerance, and delta regime must have at least one automatically certified method that independently passes the requested tolerance",
        },
        "environment": environment_metadata(),
        "settings": {
            "methods": list(methods),
            "tolerances": [float(x) for x in tolerances],
            "delta_domains": {
                name: {"delta_min": DELTA_DOMAINS[name][0], "delta_max": DELTA_DOMAINS[name][1]}
                for name in domains
            },
            "subdivisions_by_domain": {name: list(subdivisions_by_domain[name]) for name in domains},
            "fftlog_n_values": list(fftlog_n_values),
            "fftlog_bias_values_by_domain": {name: list(fftlog_bias_values[name]) for name in domains},
            "fftlog_preferred_bias_by_domain": fftlog_preferred_bias,
            "fftlog_capability_sweep": not args.no_fftlog_capability_sweep,
            "ogata_hstart": float(args.ogata_hstart),
            "ogata_hdecrement": float(args.ogata_hdecrement),
            "ogata_maxiter": int(args.ogata_maxiter),
            "ogata_n_values": list(ogata_n_values),
            "reference_order": int(args.reference_order),
            "reference_budget_fraction": float(args.reference_budget_fraction),
            "large_reference_coarse_order": int(args.large_reference_coarse_order),
            "large_reference_coarse_phase_step": float(args.large_reference_coarse_phase_step),
            "large_reference_max_levels": int(args.large_reference_max_levels),
            "n_delta": int(args.n_delta),
            "harmonic_results": str(args.harmonic_results),
            "upstream_target": float(args.upstream_target),
            "upstream_method": args.upstream_method,
        },
        "fixed_fields": upstream_metadata,
        "reference_stability": reference_stability,
        "rows": rows,
        "coverage": coverage,
        "summary": {
            "n_rows": len(rows),
            "false_positive_convergence": int(false_positives),
            "finite_missed_convergence": int(finite_misses),
            "correctness_passed": correctness_passed,
            "coverage_passed": coverage_passed,
            "validation_passed": validation_passed,
            "finite_selection_efficiency": {
                "n_oracle_resolvable": len(finite_oracle_rows),
                "n_exact_minimal": len(finite_minimal),
                "n_conservative_nonminimal": len(finite_nonminimal_rows),
                "exact_minimal_fraction": (
                    float(len(finite_minimal) / len(finite_oracle_rows)) if finite_oracle_rows else None
                ),
            },
            "by_method_and_domain": method_domain_summary,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("\n=== automatic convergence validation across delta regimes ===")
    print(f"false positives       : {false_positives}")
    print(f"finite missed         : {finite_misses}")
    print(f"correctness           : {'PASS' if correctness_passed else 'FAIL'}")
    print(f"method coverage       : {'PASS' if coverage_passed else 'FAIL'}")
    for item in method_domain_summary:
        print(
            f"  {item['delta_domain']:8s} {item['method']:10s} "
            f"cert={item['n_certified_reference_pass']}/{item['n_rows']} "
            f"fp={item['n_false_positive']} "
            f"cap={item['n_backend_capability_found']} "
            f"selector-miss={item['n_selector_miss_backend_capable']}"
        )
    uncovered = [item for item in coverage if not item["covered"]]
    if uncovered:
        print("uncovered cases:")
        for item in uncovered:
            print(
                f"  {item['delta_domain']} | {item['case']} | "
                f"rtol={item['requested_tolerance']:.0e}"
            )
    print(f"validation            : {'PASS' if validation_passed else 'FAIL'}")
    print(f"Saved: {args.output}")
    if not validation_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
