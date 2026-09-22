#!/usr/bin/env python3
"""Interaction accuracy/runtime benchmark with upstream benchmark continuity.

The fixed momentum-space fields are reconstructed from a completed
``harmonic_transform.json`` using the same analytic workload definitions and
its actual selected numerical parameters.  Interaction convergence is then
performed independently for each backend and requested tolerance.  A high-order
direct quadrature of the *same fixed q-space interpolants on the same support*
is evaluated only afterward as an oracle.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from quartic2d import Interaction

from benchmarks._common import environment_metadata
from benchmarks._interaction_suite import (
    ALL_METHODS,
    DEFAULT_SUBDIVISIONS,
    PRIMARY_WORKLOADS,
    benchmark_cases,
    build_fields_from_harmonic_results,
    direct_reference,
    displacement_grid,
    interaction_class,
    interaction_from_parameters,
    kernel_registry,
    relative_l2,
    relative_peak,
)

def parse_csv(text, cast=str):
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def timed(fn, warmups=2, repeats=5):
    for _ in range(int(warmups)):
        fn()
    samples = []
    value = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=float)
    return value, {
        "median_seconds": float(statistics.median(samples)),
        "q25_seconds": float(np.quantile(arr, 0.25)),
        "q75_seconds": float(np.quantile(arr, 0.75)),
        "min_seconds": float(np.min(arr)),
        "max_seconds": float(np.max(arr)),
        "samples_seconds": [float(x) for x in samples],
    }


def _parameter_signature(parameters):
    keep = {k: parameters[k] for k in ("subdivisions", "n", "bias", "N", "h") if k in parameters}
    return json.dumps(keep, sort_keys=True)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["requested_tolerance"])].append(row)

    summary = []
    for (method, tol), items in sorted(groups.items(), key=lambda x: (ALL_METHODS.index(x[0][0]), -x[0][1])):
        converged = [r for r in items if r["automatic_converged"]]
        passed = [r for r in converged if r["reference_pass"]]
        rejected = [r for r in items if not r["automatic_converged"]]
        conservative_pass = [r for r in rejected if r.get("terminal_reference_pass") is True]
        terminal_fail = [r for r in rejected if r.get("terminal_reference_pass") is False]
        terminal_unknown = [r for r in rejected if r.get("terminal_reference_pass") is None]
        timed_rows = [r for r in passed if r.get("production_timing")]
        signatures = Counter(_parameter_signature(r["selected_parameters"]) for r in converged)
        summary.append(
            {
                "method": method,
                "requested_tolerance": float(tol),
                "n_cases": len(items),
                "n_automatic_converged": len(converged),
                "n_reference_pass": len(passed),
                "n_false_positive": int(sum(bool(r["automatic_converged"] and r["reference_pass"] is False) for r in items)),
                "n_conservative_rejection_oracle_pass": len(conservative_pass),
                "n_rejected_terminal_oracle_fail": len(terminal_fail),
                "n_rejected_terminal_oracle_unavailable": len(terminal_unknown),
                "certified_fraction": float(len(passed) / len(items)) if items else 0.0,
                "tested_capability_fraction": float((len(passed) + len(conservative_pass)) / len(items)) if items else 0.0,
                "worst_relative_l2": None if not passed else float(max(r["selected_reference_error"]["relative_l2"] for r in passed)),
                "worst_relative_peak": None if not passed else float(max(r["selected_reference_error"]["relative_peak"] for r in passed)),
                "median_production_seconds": None if not timed_rows else float(np.median([r["production_timing"]["median_seconds"] for r in timed_rows])),
                "max_production_seconds": None if not timed_rows else float(max(r["production_timing"]["median_seconds"] for r in timed_rows)),
                "median_calibration_seconds": float(np.median([r["calibration_seconds"] for r in items])),
                "selected_parameter_counts": [
                    {"parameters": json.loads(sig), "count": int(count)}
                    for sig, count in signatures.most_common()
                ],
            }
        )
    return summary


def summarize_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["group"], row["method"], row["requested_tolerance"])].append(row)
    out = []
    for (group, method, tol), items in sorted(groups.items()):
        passed = [r for r in items if r["automatic_converged"] and r["reference_pass"]]
        out.append(
            {
                "group": group,
                "method": method,
                "requested_tolerance": float(tol),
                "n_cases": len(items),
                "n_reference_pass": len(passed),
                "completion_fraction": float(len(passed) / len(items)) if items else 0.0,
                "worst_relative_l2": None if not passed else float(max(r["selected_reference_error"]["relative_l2"] for r in passed)),
                "worst_relative_peak": None if not passed else float(max(r["selected_reference_error"]["relative_peak"] for r in passed)),
                "median_production_seconds": None if not passed else float(np.median([r["production_timing"]["median_seconds"] for r in passed])),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("benchmarks/results/interaction_accuracy.json"))
    ap.add_argument(
        "--harmonic-results",
        type=Path,
        default=Path("benchmarks/results/harmonic_transform.json"),
        help="completed HarmonicTransform benchmark used to reconstruct the fixed q-space fields",
    )
    ap.add_argument("--upstream-target", type=float, default=1.0e-4)
    ap.add_argument("--upstream-method", default="simpson")
    ap.add_argument("--groups", default="continuity,kernels,hard,control")
    ap.add_argument("--cases", default="all", help="optional comma-separated case names after group construction")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--tolerances", default="1e-3,1e-4,1e-5")
    ap.add_argument("--n-delta", type=int, default=32)
    ap.add_argument("--reference-order", type=int, default=24)
    ap.add_argument(
        "--reference-budget-fraction",
        type=float,
        default=0.1,
        help="maximum low/high oracle change as a fraction of the tightest requested tolerance",
    )
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--atol", type=float, default=1.0e-12)
    ap.add_argument("--ogata-hstart", type=float, default=0.05)
    ap.add_argument("--ogata-hdecrement", type=float, default=2.0)
    ap.add_argument("--ogata-maxiter", type=int, default=20)
    ap.add_argument(
        "--ogata-n-values",
        default=(
            "64,128,256,512,1024,2048,4096,8192,16384,32768,"
            "65536,131072,262144,524288"
        ),
        help="comma-separated Ogata N ladder used by the coupled convergence search",
    )
    args = ap.parse_args()

    methods = parse_csv(args.methods)
    unknown_methods = sorted(set(methods) - set(ALL_METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")
    tolerances = parse_csv(args.tolerances, float)
    ogata_n_values = parse_csv(args.ogata_n_values, int)
    groups = parse_csv(args.groups)
    cases = benchmark_cases(groups)
    if args.cases != "all":
        wanted_cases = set(parse_csv(args.cases))
        cases = [case for case in cases if case.name in wanted_cases]
        missing = sorted(wanted_cases - {case.name for case in cases})
        if missing:
            raise ValueError(f"unknown or excluded interaction cases: {missing}")
    needed_workloads = tuple(dict.fromkeys(case.workload for case in cases))

    fields, upstream_metadata = build_fields_from_harmonic_results(
        args.harmonic_results,
        target=args.upstream_target,
        method=args.upstream_method,
        workload_names=needed_workloads,
    )
    kernels = kernel_registry()
    deltas = displacement_grid(args.n_delta)

    references = {}
    reference_stability = {}
    reference_limit = float(min(tolerances) * args.reference_budget_fraction)
    print(f"[reference] {len(cases)} interaction cases", flush=True)
    for case in cases:
        field = fields[case.workload]
        kernel = kernels[case.kernel]
        low = direct_reference(field, field, kernel, deltas, order=args.reference_order)
        high = direct_reference(field, field, kernel, deltas, order=2 * args.reference_order)
        references[case.name] = high
        reference_l2_change = relative_l2(low, high)
        reference_peak_change = relative_peak(low, high)
        reference_stability[case.name] = {
            "workload": case.workload,
            "kernel": case.kernel,
            "order_low": int(args.reference_order),
            "order_high": int(2 * args.reference_order),
            "relative_l2_change": reference_l2_change,
            "relative_peak_change": reference_peak_change,
            "required_max_change": reference_limit,
            "passed": bool(reference_l2_change <= reference_limit and reference_peak_change <= reference_limit),
        }
        print(
            f"  {case.name}: dL2={reference_stability[case.name]['relative_l2_change']:.3e} "
            f"dPk={reference_stability[case.name]['relative_peak_change']:.3e} "
            f"{'PASS' if reference_stability[case.name]['passed'] else 'FAIL'}",
            flush=True,
        )

    unstable_references = [name for name, item in reference_stability.items() if not item["passed"]]
    if unstable_references:
        raise RuntimeError(
            "interaction oracle stability failed for: " + ", ".join(unstable_references)
        )

    rows = []
    false_positives = 0
    for case in cases:
        field = fields[case.workload]
        kernel = kernels[case.kernel]
        reference = references[case.name]
        for method in methods:
            for tolerance in tolerances:
                print(f"[{case.group}] {case.workload} x {case.kernel} | {method} tol={tolerance:.0e}", flush=True)
                t0 = time.perf_counter()
                convergence = Interaction.converge_parameters(
                    deltas,
                    field,
                    field,
                    kernel.U,
                    method=method,
                    rtol=tolerance,
                    atol=args.atol,
                    subdivisions=DEFAULT_SUBDIVISIONS,
                    hstart=args.ogata_hstart,
                    hdecrement=args.ogata_hdecrement,
                    maxiter=args.ogata_maxiter,
                    ogata_n_values=ogata_n_values,
                    verbose=False,
                )
                calibration_seconds = time.perf_counter() - t0

                selected_parameters = (
                    convergence.parameters if convergence.converged else convergence.search.selected_parameters
                )
                selected_error = None
                reference_pass = None
                production_timing = None
                failure_status = None

                if convergence.converged:
                    def build_selected():
                        return interaction_from_parameters(
                            deltas,
                            field,
                            field,
                            kernel,
                            method,
                            selected_parameters,
                        )

                    interaction, production_timing = timed(
                        build_selected,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                    value = np.asarray(interaction.V, dtype=np.complex128)
                    e2 = relative_l2(value, reference)
                    ep = relative_peak(value, reference)
                    reference_pass = bool(e2 <= tolerance and ep <= tolerance)
                    selected_error = {"relative_l2": float(e2), "relative_peak": float(ep)}
                    if not reference_pass:
                        false_positives += 1
                        failure_status = "automatic_false_positive"
                else:
                    failure_status = convergence.search.metadata.get("status", "automatic_convergence_not_established")

                # Oracle-test the terminal tested point for rejected searches.
                # This diagnostic never changes the automatic convergence certificate;
                # it only distinguishes conservative rejection from actual failure at
                # the tested parameter limit.
                terminal_error = None
                terminal_pass = None
                if not convergence.converged and selected_parameters:
                    try:
                        terminal_interaction = interaction_from_parameters(
                            deltas,
                            field,
                            field,
                            kernel,
                            method,
                            selected_parameters,
                        )
                        terminal_value = np.asarray(terminal_interaction.V, dtype=np.complex128)
                        terminal_l2 = relative_l2(terminal_value, reference)
                        terminal_peak = relative_peak(terminal_value, reference)
                        terminal_error = {
                            "relative_l2": float(terminal_l2),
                            "relative_peak": float(terminal_peak),
                        }
                        terminal_pass = bool(terminal_l2 <= tolerance and terminal_peak <= tolerance)
                    except (KeyError, TypeError, ValueError):
                        terminal_error = None
                        terminal_pass = None

                rows.append(
                    {
                        "case": case.name,
                        "group": case.group,
                        "category": case.category,
                        "workload": case.workload,
                        "kernel": case.kernel,
                        "kernel_family": kernel.family,
                        "kernel_parameters": kernel.parameters,
                        "method": method,
                        "requested_tolerance": float(tolerance),
                        "automatic_converged": bool(convergence.converged),
                        "selected_parameters": selected_parameters,
                        "selected_reference_error": selected_error,
                        "reference_pass": reference_pass,
                        "failure_status": failure_status,
                        "terminal_reference_error": terminal_error,
                        "terminal_reference_pass": terminal_pass,
                        "calibration_seconds": float(calibration_seconds),
                        "production_timing": production_timing,
                        "convergence": convergence.to_dict(include_values=False),
                    }
                )

    summary = summarize(rows)
    result = {
        "schema": 3,
        "benchmark": "Interaction accuracy, automatic convergence, and runtime",
        "scope": {
            "continuity": "fixed q-space fields are reconstructed from the same workload definitions and the actual selected parameters in the preceding HarmonicTransform benchmark",
            "interaction_accuracy_reference": "high-order direct q quadrature of the same fixed HarmonicTransform interpolants on the same finite q support; the oracle alone splits at analytically known kernel nonanalyticities such as q=2 kF",
            "upstream_error_policy": "radial-input representation, HarmonicTransform in-domain error, and omitted q support are recorded as upstream metadata and excluded from Interaction error",
            "automatic_convergence_reference_access": "none; automatic convergence sees only successive Interaction evaluations",
            "oracle_policy": "the direct reference is evaluated independently and only after parameter selection; both relative L2 and peak-normalized errors must satisfy the requested tolerance",
            "rejection_diagnostic": "automatic rejections with usable terminal parameters are also oracle-tested at the terminal tested point; this diagnostic never changes the automatic convergence certificate",
            "timed_operation": "production Interaction construction/evaluation using parameters selected by automatic convergence",
            "excluded_from_production_timing": "upstream HarmonicTransform construction; automatic convergence/calibration; direct reference; plotting",
        },
        "environment": environment_metadata(),
        "settings": {
            "harmonic_results": str(args.harmonic_results),
            "upstream_target": float(args.upstream_target),
            "upstream_method": args.upstream_method,
            "groups": list(groups),
            "case_filter": args.cases,
            "methods": list(methods),
            "requested_tolerances": [float(x) for x in tolerances],
            "subdivision_candidates": list(DEFAULT_SUBDIVISIONS),
            "n_delta": int(args.n_delta),
            "delta_min": float(np.min(np.linalg.norm(deltas, axis=1))),
            "delta_max": float(np.max(np.linalg.norm(deltas, axis=1))),
            "reference_order": int(args.reference_order),
            "reference_budget_fraction": float(args.reference_budget_fraction),
            "reference_required_max_change": reference_limit,
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "atol": float(args.atol),
            "ogata_hstart": float(args.ogata_hstart),
            "ogata_hdecrement": float(args.ogata_hdecrement),
            "ogata_maxiter": int(args.ogata_maxiter),
            "ogata_n_values": [int(x) for x in ogata_n_values],
        },
        "shared_workloads": {
            "primary": list(PRIMARY_WORKLOADS),
            "used": list(needed_workloads),
            "provenance": upstream_metadata,
        },
        "kernels": {
            name: {
                "family": kernel.family,
                "category": kernel.category,
                "parameters": kernel.parameters,
                "literature_note": kernel.literature_note,
                "oracle_breakpoints": [float(x) for x in kernel.breakpoints],
            }
            for name, kernel in kernels.items()
        },
        "cases": [case.__dict__ for case in cases],
        "reference_stability": reference_stability,
        "rows": rows,
        "summary": summary,
        "group_summary": summarize_groups(rows),
        "validation": {
            "false_positive_convergence": int(false_positives),
            "conservative_rejections_oracle_pass": int(sum(r.get("terminal_reference_pass") is True for r in rows if not r["automatic_converged"])),
            "rejections_terminal_oracle_fail": int(sum(r.get("terminal_reference_pass") is False for r in rows if not r["automatic_converged"])),
            "rejections_terminal_oracle_unavailable": int(sum(r.get("terminal_reference_pass") is None for r in rows if not r["automatic_converged"])),
            "passed": bool(false_positives == 0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print("\n=== interaction benchmark summary ===")
    for item in summary:
        runtime = item["median_production_seconds"]
        runtime_text = "--" if runtime is None else f"{1e3 * runtime:.2f} ms"
        l2 = item["worst_relative_l2"]
        peak = item["worst_relative_peak"]
        print(
            f"{item['method']:10s} tol={item['requested_tolerance']:.0e} "
            f"cert={item['n_reference_pass']:2d}/{item['n_cases']:2d} "
            f"rej-pass={item['n_conservative_rejection_oracle_pass']:2d} "
            f"rej-fail={item['n_rejected_terminal_oracle_fail']:2d} "
            f"maxL2={'--' if l2 is None else f'{l2:.2e}'} "
            f"maxPk={'--' if peak is None else f'{peak:.2e}'} t50={runtime_text}"
        )
    print(f"false positives: {false_positives}")
    print(f"Saved: {args.output}")
    if false_positives:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
