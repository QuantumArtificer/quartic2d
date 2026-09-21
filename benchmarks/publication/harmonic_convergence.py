#!/usr/bin/env python3
"""Publication-grade validation of HarmonicTransform automatic convergence.

This runner tests the *public* ``HarmonicTransform.converge_parameters`` selector
against analytic transforms over the same diverse workload family used for
reference-accuracy qualification.  It is intentionally distinct from
``harmonic_transform.py``: the latter establishes controlled fixed-parameter
accuracy/tolerance response, while this benchmark establishes whether the
automatic selector certifies safe parameters without access to the analytic
reference.

Resolvable smooth, oscillatory, cusp, algebraic, nodal, complex, and higher-m
profiles must certify at the requested tolerance.  Deliberately discontinuous
slow-tail profiles must refuse q-support certification rather than emit a false
certificate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quartic2d import HarmonicTransform
from benchmarks._common import environment_metadata
from benchmarks.publication.harmonic_transform import (
    SyntheticDecomposition,
    q_tail_l2_analytic,
    signed_exact,
    workloads,
)


VALIDATED_NR = {
    "gaussian_isotropic": 160,
    "gaussian_odd_pair": 160,
    "gaussian_anisotropic_m2": 160,
    "gaussian_high_order_m4": 160,
    "nodal_mixed": 256,
    "complex_mixed": 160,
    "exponential_cusp": 1536,
    "oscillatory_exponential": 2048,
    "algebraic_mixed": 3072,
}


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def aggregate_error(field, workload, q_max: float, n_check: int) -> tuple[float, float, list[dict]]:
    q = np.linspace(0.0, float(q_max), int(n_check))
    numerical = []
    exact = []
    per_mode = []
    for mode in workload.modes:
        ref = np.asarray(signed_exact(mode, q), dtype=np.complex128)
        val = np.asarray(field(mode.m, q), dtype=np.complex128)
        numerical.append(val.ravel())
        exact.append(ref.ravel())
        norm = max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
        peak = max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
        per_mode.append(
            {
                "m": int(mode.m),
                "relative_l2": float(np.linalg.norm(val - ref) / norm),
                "relative_peak": float(np.max(np.abs(val - ref)) / peak),
                "q_tail_relative_l2": float(q_tail_l2_analytic(mode, q_max)),
            }
        )
    num = np.concatenate(numerical)
    ref = np.concatenate(exact)
    norm = max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
    peak = max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
    return (
        float(np.linalg.norm(num - ref) / norm),
        float(np.max(np.abs(num - ref)) / peak),
        per_mode,
    )


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("benchmarks/results/publication/harmonic_convergence.json"))
    ap.add_argument("--methods", default="simpson,gl4")
    ap.add_argument("--rtol", type=float, default=1.0e-4)
    ap.add_argument("--atol", type=float, default=1.0e-12)
    ap.add_argument("--q-tail-rtol", type=float, default=1.0e-3)
    ap.add_argument("--n-check", type=int, default=4001)
    ap.add_argument("--include-stress", action="store_true")
    args = ap.parse_args()

    methods = parse_csv(args.methods)
    workload_map = {item.name: item for item in workloads()}
    rows: list[dict] = []

    result = {
        "schema": 1,
        "benchmark": "HarmonicTransform automatic-convergence validation",
        "scope": {
            "automatic_search_reference_access": "none; converge_parameters uses only numerical self-convergence diagnostics",
            "external_reference": "analytic Hankel transforms of the benchmark radial harmonics evaluated only after automatic parameter selection",
            "accuracy_metric": "global relative L2 error over all retained harmonics; peak-normalized absolute error is reported as a diagnostic",
            "q_support_metric": "analytic omitted-tail relative L2 per harmonic",
            "stress_policy": "discontinuous slow-tail profiles must refuse q-support certification rather than return an unsupported certificate",
        },
        "environment": environment_metadata(),
        "settings": {
            "methods": list(methods),
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "q_tail_rtol": float(args.q_tail_rtol),
            "n_check": int(args.n_check),
            "validated_n_r": dict(VALIDATED_NR),
            "include_stress": bool(args.include_stress),
        },
        "rows": rows,
        "summary": {},
        "complete": False,
    }
    write(args.output, result)

    false_positives = 0
    missed = 0
    for name, n_r in VALIDATED_NR.items():
        workload = workload_map[name]
        decomposition = SyntheticDecomposition(workload, n_r)
        for method in methods:
            print(f"[{name}] {method}", flush=True)
            t0 = time.perf_counter()
            try:
                convergence = HarmonicTransform.converge_parameters(
                    decomposition,
                    rtol=args.rtol,
                    atol=args.atol,
                    q_tail_rtol=args.q_tail_rtol,
                    method=method,
                    verbose=False,
                )
                elapsed = time.perf_counter() - t0
            except Exception as exc:
                rows.append(
                    {
                        "workload": name,
                        "category": workload.category,
                        "method": method,
                        "n_r": int(n_r),
                        "automatic_converged": False,
                        "classification": "unexpected_exception",
                        "exception": repr(exc),
                        "calibration_seconds": float(time.perf_counter() - t0),
                    }
                )
                missed += 1
                write(args.output, result)
                continue

            if not convergence.converged:
                rows.append(
                    {
                        "workload": name,
                        "category": workload.category,
                        "method": method,
                        "n_r": int(n_r),
                        "automatic_converged": False,
                        "classification": "missed_convergence",
                        "calibration_seconds": float(elapsed),
                        "convergence": convergence.to_dict(include_values=False),
                    }
                )
                missed += 1
                write(args.output, result)
                continue

            field = convergence.transform(decomposition, check=False)
            l2, peak, per_mode = aggregate_error(
                field, workload, convergence.sampling.q_max, args.n_check
            )
            worst_tail = max(item["q_tail_relative_l2"] for item in per_mode)
            reference_pass = bool(l2 <= args.rtol and worst_tail <= args.q_tail_rtol)
            if not reference_pass:
                false_positives += 1
            rows.append(
                {
                    "workload": name,
                    "category": workload.category,
                    "method": method,
                    "n_r": int(n_r),
                    "automatic_converged": True,
                    "selected_parameters": convergence.parameters,
                    "q_max": float(convergence.sampling.q_max),
                    "n_q": int(convergence.sampling.n_q),
                    "grid_kind": convergence.sampling.grid_kind,
                    "calibration_seconds": float(elapsed),
                    "external_reference_error": {
                        "relative_l2": l2,
                        "relative_peak": peak,
                        "worst_q_tail_relative_l2": float(worst_tail),
                        "per_mode": per_mode,
                        "passed": reference_pass,
                    },
                    "classification": (
                        "certified_reference_pass" if reference_pass else "automatic_false_positive"
                    ),
                    "convergence": convergence.to_dict(include_values=False),
                }
            )
            write(args.output, result)

    stress_expected = []
    if args.include_stress:
        for name in ("top_hat_stress", "annulus_stress"):
            workload = workload_map[name]
            decomposition = SyntheticDecomposition(workload, 256)
            for method in methods:
                print(f"[{name}] {method} expected refusal", flush=True)
                t0 = time.perf_counter()
                refused = False
                exception_text = None
                convergence_dict = None
                try:
                    convergence = HarmonicTransform.converge_parameters(
                        decomposition,
                        rtol=args.rtol,
                        atol=args.atol,
                        q_tail_rtol=args.q_tail_rtol,
                        method=method,
                        verbose=False,
                    )
                    convergence_dict = convergence.to_dict(include_values=False)
                except RuntimeError as exc:
                    exception_text = str(exc)
                    refused = "Could not certify q-space support" in exception_text
                elapsed = time.perf_counter() - t0
                stress_expected.append(refused)
                rows.append(
                    {
                        "workload": name,
                        "category": workload.category,
                        "method": method,
                        "n_r": 256,
                        "expected_outcome": "q_support_refusal",
                        "refused_as_expected": bool(refused),
                        "classification": "expected_refusal" if refused else "stress_refusal_failed",
                        "exception": exception_text,
                        "convergence": convergence_dict,
                        "calibration_seconds": float(elapsed),
                    }
                )
                write(args.output, result)

    certified_rows = [row for row in rows if row.get("automatic_converged") is True]
    passing_rows = [row for row in certified_rows if row.get("external_reference_error", {}).get("passed") is True]
    regular_rows = [row for row in rows if row.get("workload") in VALIDATED_NR]
    validation_passed = bool(
        false_positives == 0
        and missed == 0
        and len(passing_rows) == len(VALIDATED_NR) * len(methods)
        and (not args.include_stress or all(stress_expected))
    )
    result["summary"] = {
        "n_regular_rows": len(regular_rows),
        "n_certified_reference_pass": len(passing_rows),
        "false_positive_certificates": int(false_positives),
        "missed_convergence": int(missed),
        "stress_refusals_expected": int(len(stress_expected)),
        "stress_refusals_observed": int(sum(stress_expected)),
        "worst_certified_relative_l2": max(
            (row["external_reference_error"]["relative_l2"] for row in passing_rows),
            default=None,
        ),
        "worst_certified_relative_peak": max(
            (row["external_reference_error"]["relative_peak"] for row in passing_rows),
            default=None,
        ),
        "worst_certified_q_tail_relative_l2": max(
            (row["external_reference_error"]["worst_q_tail_relative_l2"] for row in passing_rows),
            default=None,
        ),
        "validation_passed": validation_passed,
    }
    result["validation_passed"] = validation_passed
    result["complete"] = True
    write(args.output, result)

    print("\n=== HarmonicTransform automatic-convergence validation ===")
    print(f"certified/reference pass : {len(passing_rows)}/{len(VALIDATED_NR) * len(methods)}")
    print(f"false positives          : {false_positives}")
    print(f"missed convergence       : {missed}")
    if args.include_stress:
        print(f"expected stress refusals : {sum(stress_expected)}/{len(stress_expected)}")
    print(f"validation               : {'PASS' if validation_passed else 'FAIL'}")
    print(f"Saved: {args.output}")
    raise SystemExit(0 if validation_passed else 2)


if __name__ == "__main__":
    main()
