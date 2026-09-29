#!/usr/bin/env python3
"""True dependency-stack validation: PETAL2D -> QUARTIC2D -> Interaction.

Unlike ``cross_stage.py``, which starts from analytically supplied radial
harmonics to isolate QUARTIC2D composition, this benchmark starts from sampled
2D functions.  PETAL2D performs the polar harmonic decomposition, QUARTIC2D
selects an adaptive Hankel representation, and Interaction selects its own
numerical parameters.  Final interactions are compared with independently
stabilized references constructed from analytic momentum-space form factors.

The benchmark therefore measures cumulative error through the actual public
PETAL2D + QUARTIC2D workflow while retaining stage-resolved diagnostics so that
upstream decomposition error is not silently attributed to downstream
quadrature.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from petal2d import PolarDecomposition

from benchmarks._common import environment_metadata
from benchmarks._interaction_suite import (
    DEFAULT_SUBDIVISIONS,
    FFTLOG_BIAS_VALUES_LARGE,
    FFTLOG_BIAS_VALUES_STANDARD,
    FFTLOG_N_VALUES,
    LARGE_DELTA_SUBDIVISIONS,
    displacement_grid,
    kernel_registry,
    relative_l2,
    relative_peak,
    stable_reference_pair,
)
from quartic2d import HarmonicTransform, Interaction

ArrayFn = Callable[[np.ndarray], np.ndarray]
DensityFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PipelineCase:
    name: str
    category: str
    kernel: str
    density: DensityFn
    radial_coefficients: dict[int, ArrayFn]
    form_factors: dict[int, ArrayFn]

    @property
    def modes(self) -> tuple[int, ...]:
        return tuple(sorted(int(m) for m in self.radial_coefficients))


class AnalyticField:
    """Minimal exact momentum-space field used only by the independent oracle."""

    def __init__(self, case: PipelineCase, q_max: float, n_q: int = 129):
        self.q = np.linspace(0.0, float(q_max), int(n_q))
        self.m_values = np.asarray(case.modes, dtype=int)
        self._functions = case.form_factors

    def __call__(self, m: int, q):
        try:
            fn = self._functions[int(m)]
        except KeyError as exc:  # pragma: no cover - benchmark programming error
            raise KeyError(f"analytic field has no harmonic m={m}") from exc
        return np.asarray(fn(np.asarray(q, dtype=float)), dtype=np.complex128)


def _gaussian0(r):
    r = np.asarray(r, dtype=float)
    return np.exp(-r * r) / np.pi


def _gaussian0_q(q):
    q = np.asarray(q, dtype=float)
    return np.exp(-q * q / 4.0) / (2.0 * np.pi)


def _gaussian_density(x, y):
    return np.exp(-(x * x + y * y)) / np.pi


def _anisotropic_density(x, y, anisotropy: float = 0.35):
    r2 = x * x + y * y
    return np.exp(-r2) * (1.0 + anisotropy * (x * x - y * y)) / np.pi


def _anisotropic_m2(r, anisotropy: float = 0.35):
    r = np.asarray(r, dtype=float)
    return anisotropy * r * r * np.exp(-r * r) / (2.0 * np.pi)


def _anisotropic_m2_q(q, anisotropy: float = 0.35):
    q = np.asarray(q, dtype=float)
    return anisotropy * q * q * np.exp(-q * q / 4.0) / (16.0 * np.pi)


def _nodal_density(x, y):
    r2 = x * x + y * y
    return (1.0 - r2) * np.exp(-r2) / np.pi


def _nodal0(r):
    r = np.asarray(r, dtype=float)
    return (1.0 - r * r) * np.exp(-r * r) / np.pi


def _nodal0_q(q):
    q = np.asarray(q, dtype=float)
    return q * q * np.exp(-q * q / 4.0) / (8.0 * np.pi)


def pipeline_cases() -> dict[str, PipelineCase]:
    return {
        "gaussian_coulomb": PipelineCase(
            "gaussian_coulomb",
            "isotropic Gaussian / bare Coulomb",
            "coulomb",
            _gaussian_density,
            {0: _gaussian0},
            {0: _gaussian0_q},
        ),
        "anisotropic_rk_strong": PipelineCase(
            "anisotropic_rk_strong",
            "anisotropic Gaussian / strong Rytova-Keldysh screening",
            "rk_r0_10",
            _anisotropic_density,
            {-2: _anisotropic_m2, 0: _gaussian0, 2: _anisotropic_m2},
            {-2: _anisotropic_m2_q, 0: _gaussian0_q, 2: _anisotropic_m2_q},
        ),
        "nodal_rpa": PipelineCase(
            "nodal_rpa",
            "nodal Gaussian / 2DEG RPA cusp",
            "rpa_2deg_kf1_qtf1",
            _nodal_density,
            {0: _nodal0},
            {0: _nodal0_q},
        ),
    }


def _write(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _aggregate_mode_error(numerical: dict[int, np.ndarray], exact: dict[int, np.ndarray]):
    num = np.concatenate([np.ravel(numerical[m]) for m in sorted(exact)])
    ref = np.concatenate([np.ravel(exact[m]) for m in sorted(exact)])
    diff = num - ref
    norm = max(float(np.linalg.norm(ref)), np.finfo(float).tiny)
    peak = max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
    return {
        "relative_l2": float(np.linalg.norm(diff) / norm),
        "relative_peak": float(np.max(np.abs(diff)) / peak),
    }


def _petal_metrics(case: PipelineCase, decomposition: PolarDecomposition) -> dict:
    r = np.asarray(decomposition.r, dtype=float)
    retained = tuple(sorted(int(m) for m in decomposition.m_sorted))
    numerical = {}
    exact = {}
    per_mode = {}
    for m in case.modes:
        ref = np.asarray(case.radial_coefficients[m](r), dtype=np.complex128)
        val = np.asarray(decomposition[m], dtype=np.complex128)
        numerical[m] = val
        exact[m] = ref
        l2 = np.linalg.norm(val - ref) / max(np.linalg.norm(ref), np.finfo(float).tiny)
        peak = np.max(np.abs(val - ref)) / max(np.max(np.abs(ref)), np.finfo(float).tiny)
        per_mode[str(m)] = {"relative_l2": float(l2), "relative_peak": float(peak)}
    aggregate = _aggregate_mode_error(numerical, exact)
    return {
        "expected_modes": list(case.modes),
        "retained_modes": list(retained),
        "mode_set_matches": retained == case.modes,
        "per_mode": per_mode,
        **aggregate,
    }


def _harmonic_metrics(case: PipelineCase, field: HarmonicTransform, n_check: int = 4001) -> dict:
    q = np.linspace(0.0, float(field.q[-1]), int(n_check))
    numerical = {m: np.asarray(field(m, q), dtype=np.complex128) for m in case.modes}
    exact = {m: np.asarray(case.form_factors[m](q), dtype=np.complex128) for m in case.modes}
    aggregate = _aggregate_mode_error(numerical, exact)
    per_mode = {}
    for m in case.modes:
        ref = exact[m]
        val = numerical[m]
        per_mode[str(m)] = {
            "relative_l2": float(
                np.linalg.norm(val - ref) / max(np.linalg.norm(ref), np.finfo(float).tiny)
            ),
            "relative_peak": float(
                np.max(np.abs(val - ref)) / max(np.max(np.abs(ref)), np.finfo(float).tiny)
            ),
        }
    return {
        "q_max": float(field.q[-1]),
        "n_q": int(field.q.size),
        "per_mode": per_mode,
        **aggregate,
    }


def _analytic_reference(
    case: PipelineCase,
    kernel,
    deltas: np.ndarray,
    *,
    domain: str,
    q_max: float,
    q_expansion_factor: float,
    reference_order: int,
    reference_limit: float,
    large_coarse_order: int,
    large_coarse_phase_step: float,
    large_max_levels: int,
):
    fields = [
        AnalyticField(case, q_max),
        AnalyticField(case, q_max * q_expansion_factor),
    ]
    references = []
    quadrature_meta = []
    for field in fields:
        _, high, metadata = stable_reference_pair(
            field,
            field,
            kernel,
            deltas,
            domain=domain,
            reference_order=reference_order,
            reference_limit=reference_limit,
            large_coarse_order=large_coarse_order,
            large_coarse_phase_step=large_coarse_phase_step,
            large_max_levels=large_max_levels,
        )
        references.append(high)
        quadrature_meta.append(metadata)

    q_l2 = relative_l2(references[0], references[1])
    q_peak = relative_peak(references[0], references[1])
    converged = bool(
        all(meta["converged"] for meta in quadrature_meta)
        and q_l2 <= reference_limit
        and q_peak <= reference_limit
    )
    metadata = {
        "kind": "analytic form factors + independently stabilized q quadrature",
        "base_q_max": float(q_max),
        "expanded_q_max": float(q_max * q_expansion_factor),
        "q_cutoff_relative_l2_change": float(q_l2),
        "q_cutoff_relative_peak_change": float(q_peak),
        "quadrature": quadrature_meta,
        "required_max_change": float(reference_limit),
        "converged": converged,
    }
    return references[1], metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="gaussian_coulomb,anisotropic_rk_strong,nodal_rpa")
    ap.add_argument("--delta-domain", choices=("standard", "large"), default="standard")
    ap.add_argument("--methods", default="gl4")
    ap.add_argument("--n-delta", type=int, default=32)
    ap.add_argument("--xy-max", type=float, default=6.0)
    ap.add_argument("--nxy", type=int, default=181)
    ap.add_argument("--nr", type=int, default=256)
    ap.add_argument("--ntheta", type=int, default=256)
    ap.add_argument("--rmax", type=float, default=5.5)
    ap.add_argument("--petal-recon-rtol", type=float, default=1.0e-5)
    ap.add_argument("--petal-profile-rtol", type=float, default=1.0e-4)
    ap.add_argument("--harmonic-rtol", type=float, default=1.0e-4)
    ap.add_argument("--q-tail-rtol", type=float, default=1.0e-3)
    ap.add_argument("--interaction-rtol", type=float, default=1.0e-4)
    ap.add_argument("--atol", type=float, default=1.0e-12)
    ap.add_argument("--reference-order", type=int, default=24)
    ap.add_argument("--reference-budget-fraction", type=float, default=0.1)
    ap.add_argument("--reference-q-max", type=float, default=16.0)
    ap.add_argument("--reference-q-expansion-factor", type=float, default=1.25)
    ap.add_argument("--large-reference-coarse-order", type=int, default=8)
    ap.add_argument("--large-reference-coarse-phase-step", type=float, default=float(np.pi))
    ap.add_argument("--large-reference-max-levels", type=int, default=4)
    ap.add_argument(
        "--output",
        default="benchmarks/results/end_to_end_standard.json",
    )
    args = ap.parse_args()

    registry = pipeline_cases()
    requested = tuple(x.strip() for x in args.cases.split(",") if x.strip())
    unknown = sorted(set(requested) - set(registry))
    if unknown:
        raise ValueError(f"unknown end-to-end cases: {unknown}")
    cases = [registry[name] for name in requested]
    methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if not methods:
        raise ValueError("at least one Interaction method is required")

    x = np.linspace(-args.xy_max, args.xy_max, args.nxy)
    y = np.linspace(-args.xy_max, args.xy_max, args.nxy)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    deltas = displacement_grid(args.n_delta, domain=args.delta_domain)
    kernels = kernel_registry()
    reference_limit = float(args.interaction_rtol * args.reference_budget_fraction)
    output = Path(args.output)

    result = {
        "schema": 1,
        "benchmark": "PETAL2D -> HarmonicTransform -> Interaction end-to-end validation",
        "scope": {
            "input": "analytic 2D functions sampled on a Cartesian grid, then decomposed from the sampled array by petal2d.PolarDecomposition",
            "pipeline": "PETAL2D PolarDecomposition -> QUARTIC2D HarmonicTransform automatic convergence -> Interaction automatic convergence",
            "reference": "analytic momentum-space harmonics integrated with independently stabilized direct q quadrature",
            "error_policy": "PETAL2D radial-profile error, transformed-field error, and final interaction error are all recorded separately; final interaction error is cumulative across the full pipeline",
        },
        "environment": environment_metadata(),
        "settings": {
            "cases": list(requested),
            "delta_domain": args.delta_domain,
            "methods": list(methods),
            "n_delta": args.n_delta,
            "xy_max": args.xy_max,
            "nxy": args.nxy,
            "nr": args.nr,
            "ntheta": args.ntheta,
            "rmax": args.rmax,
            "petal_recon_rtol": args.petal_recon_rtol,
            "petal_recon_err_tol_percent": 100.0 * args.petal_recon_rtol,
            "petal_profile_rtol": args.petal_profile_rtol,
            "harmonic_rtol": args.harmonic_rtol,
            "q_tail_rtol": args.q_tail_rtol,
            "interaction_rtol": args.interaction_rtol,
            "atol": args.atol,
            "reference_order": args.reference_order,
            "reference_budget_fraction": args.reference_budget_fraction,
            "reference_q_max": args.reference_q_max,
            "reference_q_expansion_factor": args.reference_q_expansion_factor,
            "large_reference_coarse_order": args.large_reference_coarse_order,
            "large_reference_coarse_phase_step": args.large_reference_coarse_phase_step,
            "large_reference_max_levels": args.large_reference_max_levels,
        },
        "cases": [],
        "summary": {},
        "complete": False,
    }
    _write(output, result)

    all_rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases):02d}] PETAL2D {case.name}", flush=True)
        sampled_density = np.asarray(case.density(x_mesh, y_mesh))
        decomposition = PolarDecomposition(
            sampled_density,
            x,
            y,
            Nr=args.nr,
            Ntheta=args.ntheta,
            rmax=args.rmax,
            recon_err_tol=100.0 * args.petal_recon_rtol,
            radial_power_tail_fraction=1.0e-10,
            radial_relative_amplitude_threshold=1.0e-7,
            origin=(0.0, 0.0),
            interp_method="cubic",
        )
        petal = _petal_metrics(case, decomposition)

        harmonic_convergence = HarmonicTransform.converge_parameters(
            decomposition,
            rtol=args.harmonic_rtol,
            atol=args.atol,
            q_tail_rtol=args.q_tail_rtol,
            method="simpson",
            verbose=False,
        )
        if not harmonic_convergence.converged:
            raise RuntimeError(f"HarmonicTransform did not converge for end-to-end case {case.name!r}")
        field = harmonic_convergence.transform(decomposition, check=False)
        harmonic = _harmonic_metrics(case, field)

        kernel = kernels[case.kernel]
        reference, reference_meta = _analytic_reference(
            case,
            kernel,
            deltas,
            domain=args.delta_domain,
            q_max=args.reference_q_max,
            q_expansion_factor=args.reference_q_expansion_factor,
            reference_order=args.reference_order,
            reference_limit=reference_limit,
            large_coarse_order=args.large_reference_coarse_order,
            large_coarse_phase_step=args.large_reference_coarse_phase_step,
            large_max_levels=args.large_reference_max_levels,
        )
        if not reference_meta["converged"]:
            raise RuntimeError(f"analytic interaction reference did not stabilize for {case.name!r}")

        case_row = {
            "name": case.name,
            "category": case.category,
            "kernel": case.kernel,
            "petal2d": petal,
            "harmonic_transform": {
                "convergence": harmonic_convergence.to_dict(),
                "analytic_error": harmonic,
            },
            "reference_stability": reference_meta,
            "interactions": [],
        }

        for method in methods:
            print(f"    Interaction {method}", flush=True)
            convergence = Interaction.converge_parameters(
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
            row = {
                "method": method,
                "automatic_converged": bool(convergence.converged),
                "selected_parameters": convergence.parameters,
                "convergence": convergence.to_dict(include_values=False),
                "relative_l2": None,
                "relative_peak": None,
                "reference_pass": False,
            }
            if convergence.converged:
                interaction = convergence.interaction(deltas, field, field, kernel.U)
                l2 = relative_l2(interaction.V, reference)
                peak = relative_peak(interaction.V, reference)
                row.update(
                    {
                        "relative_l2": float(l2),
                        "relative_peak": float(peak),
                        "reference_pass": bool(
                            l2 <= args.interaction_rtol and peak <= args.interaction_rtol
                        ),
                    }
                )
            case_row["interactions"].append(row)
            all_rows.append((case, petal, harmonic, row))

        result["cases"].append(case_row)
        _write(output, result)

    petal_pass = all(
        petal["mode_set_matches"]
        and petal["relative_l2"] <= args.petal_profile_rtol
        and petal["relative_peak"] <= args.petal_profile_rtol
        for _, petal, _, _ in all_rows
    )
    harmonic_pass = all(
        harmonic["relative_l2"] <= args.harmonic_rtol
        and harmonic["relative_peak"] <= args.harmonic_rtol
        for _, _, harmonic, _ in all_rows
    )
    interaction_pass = all(
        row["automatic_converged"] and row["reference_pass"]
        for _, _, _, row in all_rows
    )
    result["summary"] = {
        "n_cases": len(cases),
        "n_interaction_rows": len(all_rows),
        "petal2d_stage_pass": bool(petal_pass),
        "harmonic_stage_pass": bool(harmonic_pass),
        "interaction_stage_pass": bool(interaction_pass),
        "worst_petal_relative_l2": max(p["relative_l2"] for _, p, _, _ in all_rows),
        "worst_petal_relative_peak": max(p["relative_peak"] for _, p, _, _ in all_rows),
        "worst_harmonic_relative_l2": max(h["relative_l2"] for _, _, h, _ in all_rows),
        "worst_harmonic_relative_peak": max(h["relative_peak"] for _, _, h, _ in all_rows),
        "worst_interaction_relative_l2": max(
            (row["relative_l2"] for _, _, _, row in all_rows if row["relative_l2"] is not None),
            default=None,
        ),
        "worst_interaction_relative_peak": max(
            (row["relative_peak"] for _, _, _, row in all_rows if row["relative_peak"] is not None),
            default=None,
        ),
    }
    passed = bool(petal_pass and harmonic_pass and interaction_pass)
    result["validation_passed"] = passed
    result["complete"] = True
    _write(output, result)

    print(f"\n=== PETAL2D -> QUARTIC2D end-to-end ({args.delta_domain}) ===")
    print(
        f"PETAL2D   : {'PASS' if petal_pass else 'FAIL'} "
        f"maxL2={result['summary']['worst_petal_relative_l2']:.3e} "
        f"maxPk={result['summary']['worst_petal_relative_peak']:.3e}"
    )
    print(
        f"Harmonic  : {'PASS' if harmonic_pass else 'FAIL'} "
        f"maxL2={result['summary']['worst_harmonic_relative_l2']:.3e} "
        f"maxPk={result['summary']['worst_harmonic_relative_peak']:.3e}"
    )
    print(
        f"Interaction: {'PASS' if interaction_pass else 'FAIL'} "
        f"maxL2={result['summary']['worst_interaction_relative_l2']:.3e} "
        f"maxPk={result['summary']['worst_interaction_relative_peak']:.3e}"
    )
    print(f"validation: {'PASS' if passed else 'FAIL'}")
    print(f"Saved: {output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
