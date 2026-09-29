#!/usr/bin/env python3
"""Cross-stage validation of adaptive HarmonicTransform -> Interaction.

This benchmark complements the stage-isolated HarmonicTransform and Interaction
benchmarks.  It keeps the validated radial input for each analytic workload
fixed, constructs the momentum-space field with the public automatic adaptive
HarmonicTransform calibration, then runs Interaction automatic convergence on
that sparse field.  The final interaction is compared with the independent
high-order direct-q reference built from the canonical dense HarmonicTransform
benchmark field.

The comparison therefore detects errors introduced by composing adaptive
q-support/q-grid selection with the downstream interaction integration, while
leaving the stage-isolated validation datasets unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from benchmarks._common import environment_metadata
from benchmarks._interaction_suite import (
    DEFAULT_SUBDIVISIONS,
    FFTLOG_BIAS_VALUES_LARGE,
    FFTLOG_BIAS_VALUES_STANDARD,
    FFTLOG_N_VALUES,
    LARGE_DELTA_SUBDIVISIONS,
    benchmark_cases,
    build_fields_from_harmonic_results,
    displacement_grid,
    interaction_class,
    kernel_registry,
    relative_l2,
    relative_peak,
    stable_reference_pair,
)
from benchmarks.harmonic_transform import SyntheticDecomposition, workloads
from quartic2d import HarmonicTransform


def _write(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--harmonic-results",
        default="benchmarks/results/harmonic_transform.json",
        help="canonical dense HarmonicTransform benchmark used as the cross-stage reference",
    )
    ap.add_argument("--upstream-target", type=float, default=1.0e-4)
    ap.add_argument("--upstream-method", default="simpson")
    ap.add_argument("--harmonic-rtol", type=float, default=1.0e-4)
    ap.add_argument("--q-tail-rtol", type=float, default=1.0e-3)
    ap.add_argument("--interaction-rtol", type=float, default=1.0e-4)
    ap.add_argument("--atol", type=float, default=1.0e-12)
    ap.add_argument("--methods", default="simpson,gl4")
    ap.add_argument("--groups", default="continuity,kernels,hard,control")
    ap.add_argument("--cases", default="")
    ap.add_argument("--delta-domain", choices=("standard", "large"), default="standard")
    ap.add_argument("--n-delta", type=int, default=32)
    ap.add_argument("--reference-order", type=int, default=24)
    ap.add_argument("--reference-budget-fraction", type=float, default=0.1)
    ap.add_argument("--large-reference-coarse-order", type=int, default=8)
    ap.add_argument("--large-reference-coarse-phase-step", type=float, default=float(np.pi))
    ap.add_argument("--large-reference-max-levels", type=int, default=4)
    ap.add_argument(
        "--output",
        default="benchmarks/results/cross_stage_1e-4.json",
    )
    args = ap.parse_args()

    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    groups = tuple(x.strip() for x in args.groups.split(",") if x.strip())
    cases = benchmark_cases(groups)
    if args.cases:
        wanted = {x.strip() for x in args.cases.split(",") if x.strip()}
        cases = [case for case in cases if case.name in wanted]
        missing = sorted(wanted - {case.name for case in cases})
        if missing:
            raise ValueError(f"unknown or group-excluded case names: {missing}")
    if not cases:
        raise ValueError("no interaction cases selected")

    workload_names = tuple(dict.fromkeys(case.workload for case in cases))
    dense_fields, upstream = build_fields_from_harmonic_results(
        args.harmonic_results,
        target=args.upstream_target,
        method=args.upstream_method,
        workload_names=workload_names,
    )
    workload_map = {item.name: item for item in workloads()}

    adaptive_fields = {}
    adaptive_metadata = {}
    for name in workload_names:
        decomposition = SyntheticDecomposition(workload_map[name], int(upstream[name]["n_r"]))
        convergence = HarmonicTransform.converge_parameters(
            decomposition,
            rtol=args.harmonic_rtol,
            atol=args.atol,
            q_tail_rtol=args.q_tail_rtol,
            method=args.upstream_method,
            verbose=False,
        )
        if not convergence.converged:
            raise RuntimeError(f"adaptive HarmonicTransform did not converge for {name!r}")
        adaptive_fields[name] = convergence.transform(decomposition, check=False)
        adaptive_metadata[name] = {
            "n_r": int(upstream[name]["n_r"]),
            "q_max": float(convergence.sampling.q_max),
            "n_q": int(convergence.sampling.n_q),
            "grid_kind": str(convergence.sampling.grid_kind),
            "parameters": convergence.parameters,
            "sampling": convergence.sampling.to_dict(),
            "quadrature": convergence.quadrature.to_dict(),
        }
        print(
            f"[adaptive field] {name}: qmax={convergence.sampling.q_max:.6g} "
            f"nq={convergence.sampling.n_q}",
            flush=True,
        )

    kernels = kernel_registry()
    deltas = displacement_grid(args.n_delta, domain=args.delta_domain)
    reference_limit = float(args.interaction_rtol * args.reference_budget_fraction)
    output = Path(args.output)
    result = {
        "schema": 1,
        "benchmark": "Adaptive HarmonicTransform -> Interaction cross-stage validation",
        "scope": {
            "purpose": (
                "validate composition of automatic adaptive q sampling with downstream "
                "Interaction automatic convergence"
            ),
            "adaptive_input": (
                "public HarmonicTransform.converge_parameters using the same radial input size "
                "qualified by the canonical HarmonicTransform benchmark"
            ),
            "reference": (
                "high-order direct q quadrature of the canonical dense HarmonicTransform benchmark "
                "field for the same analytic radial input"
            ),
            "interpretation": (
                "reported error includes the effect of adaptive q support/grid plus the selected "
                "Interaction backend, relative to the canonical dense-field reference"
            ),
        },
        "environment": environment_metadata(),
        "settings": {
            "harmonic_results": str(args.harmonic_results),
            "upstream_target": args.upstream_target,
            "upstream_method": args.upstream_method,
            "harmonic_rtol": args.harmonic_rtol,
            "q_tail_rtol": args.q_tail_rtol,
            "interaction_rtol": args.interaction_rtol,
            "atol": args.atol,
            "methods": list(methods),
            "groups": list(groups),
            "delta_domain": args.delta_domain,
            "n_delta": args.n_delta,
            "reference_order": args.reference_order,
            "reference_budget_fraction": args.reference_budget_fraction,
            "reference_required_max_change": reference_limit,
            "large_reference_coarse_order": args.large_reference_coarse_order,
            "large_reference_coarse_phase_step": args.large_reference_coarse_phase_step,
            "large_reference_max_levels": args.large_reference_max_levels,
        },
        "adaptive_fields": adaptive_metadata,
        "reference_stability": {},
        "rows": [],
        "summary": [],
        "complete": False,
    }
    _write(output, result)

    for index, case in enumerate(cases, start=1):
        kernel = kernels[case.kernel]
        _, dense_reference, reference_metadata = stable_reference_pair(
            dense_fields[case.workload],
            dense_fields[case.workload],
            kernel,
            deltas,
            domain=args.delta_domain,
            reference_order=args.reference_order,
            reference_limit=reference_limit,
            large_coarse_order=args.large_reference_coarse_order,
            large_coarse_phase_step=args.large_reference_coarse_phase_step,
            large_max_levels=args.large_reference_max_levels,
        )
        result["reference_stability"][case.name] = {
            **reference_metadata,
            "required_max_change": reference_limit,
            "passed": bool(reference_metadata["converged"]),
        }
        if not reference_metadata["converged"]:
            _write(output, result)
            raise RuntimeError(f"end-to-end reference did not stabilize for {case.name!r}")

        field = adaptive_fields[case.workload]
        for method in methods:
            print(f"[{index:02d}/{len(cases):02d}] {case.name} | {method}", flush=True)
            convergence = interaction_class(method).converge_parameters(
                deltas,
                field,
                field,
                kernel.U,
                rtol=args.interaction_rtol,
                atol=args.atol,
                method=method,
                subdivisions=(
                    LARGE_DELTA_SUBDIVISIONS
                    if args.delta_domain == "large"
                    else DEFAULT_SUBDIVISIONS
                ),
                n_values=FFTLOG_N_VALUES,
                bias=(0.0 if args.delta_domain == "large" else -0.5),
                bias_values=(
                    FFTLOG_BIAS_VALUES_LARGE
                    if args.delta_domain == "large"
                    else FFTLOG_BIAS_VALUES_STANDARD
                ),
                verbose=False,
            )
            boundary = convergence.search.metadata.get("q_boundary_robustness", {})
            if convergence.converged:
                classification = "certified"
            elif boundary.get("status") == "upstream_q_boundary_not_robust":
                classification = "safe_refusal_upstream_q_boundary"
            else:
                classification = "safe_refusal_backend"
            row = {
                "delta_domain": args.delta_domain,
                "case": case.name,
                "group": case.group,
                "workload": case.workload,
                "kernel": case.kernel,
                "method": method,
                "automatic_converged": bool(convergence.converged),
                "selected_parameters": (
                    convergence.parameters
                    if convergence.converged
                    else convergence.search.selected_parameters
                ),
                "classification": classification,
                "q_boundary_robustness": boundary,
                "convergence": convergence.to_dict(include_values=False),
            }
            if convergence.converged:
                interaction = convergence.interaction(deltas, field, field, kernel.U)
                l2 = relative_l2(interaction.V, dense_reference)
                peak = relative_peak(interaction.V, dense_reference)
                row.update(
                    {
                        "relative_l2": l2,
                        "relative_peak": peak,
                        "reference_pass": bool(
                            l2 <= args.interaction_rtol and peak <= args.interaction_rtol
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "relative_l2": None,
                        "relative_peak": None,
                        "reference_pass": False,
                    }
                )
            result["rows"].append(row)
            _write(output, result)

    summaries = []
    false_positives = 0
    for method in methods:
        rows = [row for row in result["rows"] if row["method"] == method]
        converged = [row for row in rows if row["automatic_converged"]]
        passing = [row for row in converged if row["reference_pass"]]
        false_positive_rows = [row for row in converged if not row["reference_pass"]]
        false_positives += len(false_positive_rows)
        method_summary = {
            "method": method,
            "n_cases": len(rows),
            "n_converged": len(converged),
            "n_reference_pass": len(passing),
            "n_false_positive": len(false_positive_rows),
            "n_upstream_boundary_refusal": sum(
                row["classification"] == "safe_refusal_upstream_q_boundary" for row in rows
            ),
            "n_backend_refusal": sum(
                row["classification"] == "safe_refusal_backend" for row in rows
            ),
            "worst_relative_l2": max(
                (row["relative_l2"] for row in converged), default=None
            ),
            "worst_relative_peak": max(
                (row["relative_peak"] for row in converged), default=None
            ),
        }
        summaries.append(method_summary)

    coverage = []
    for case in cases:
        rows = [row for row in result["rows"] if row["case"] == case.name]
        passing = [
            row for row in rows
            if row["automatic_converged"] and row["reference_pass"]
        ]
        coverage.append(
            {
                "case": case.name,
                "covered": bool(passing),
                "certifying_methods": [row["method"] for row in passing],
                "upstream_boundary_refusal_methods": [
                    row["method"]
                    for row in rows
                    if row["classification"] == "safe_refusal_upstream_q_boundary"
                ],
            }
        )

    coverage_complete = all(item["covered"] for item in coverage)
    safe_coverage = all(
        item["covered"] or bool(item["upstream_boundary_refusal_methods"])
        for item in coverage
    )
    # Cross-stage correctness is a safety test: any automatic certificate must
    # pass the independent dense-field reference. Specialist backends may
    # safely refuse. If no backend covers a case, the case is acceptable only
    # when the production q-boundary guard explicitly identifies the upstream
    # representation as unsupported for this observable.
    passed = false_positives == 0 and safe_coverage
    result["summary"] = summaries
    result["coverage"] = coverage
    result["false_positives"] = int(false_positives)
    result["coverage_complete"] = bool(coverage_complete)
    result["safe_coverage"] = bool(safe_coverage)
    result["validation_passed"] = bool(passed)
    result["complete"] = True
    _write(output, result)

    print(f"\n=== adaptive pipeline validation ({args.delta_domain}) ===")
    for item in summaries:
        l2 = "--" if item["worst_relative_l2"] is None else f"{item['worst_relative_l2']:.3e}"
        pk = "--" if item["worst_relative_peak"] is None else f"{item['worst_relative_peak']:.3e}"
        print(
            f"{item['method']:10s} conv={item['n_converged']:2d}/{item['n_cases']:2d} "
            f"pass={item['n_reference_pass']:2d}/{item['n_cases']:2d} "
            f"fp={item['n_false_positive']} q-boundary-refusal={item['n_upstream_boundary_refusal']} "
            f"maxL2={l2} maxPk={pk}"
        )
    print(f"false positives: {false_positives}")
    print(f"coverage       : {'PASS' if coverage_complete else 'INCOMPLETE'}")
    print(f"safe coverage  : {'PASS' if safe_coverage else 'FAIL'}")
    print(f"validation     : {'PASS' if passed else 'FAIL'}")
    print(f"Saved: {output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
