#!/usr/bin/env python3
"""Benchmark automatic q support, q sampling, and q-space cubic interpolation.

This benchmark validates the Step-2 sampling theory independently of the radial
quadrature benchmark:

* q-space cubic interpolation: natural versus not-a-knot boundary conditions;
* Parseval-tail q_max criterion propagated through representative interactions;
* the normal one-pass HarmonicTransform defaults and the opt-in convergence helper on the analytic anisotropic PETAL2D workload.

The connected workload uses the exact radial harmonic profiles corresponding to
the PETAL2D benchmark density.  This isolates QUARTIC2D's Step-2 numerics while
retaining the same radial grids, cutoff rules, harmonics, and form factors.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad_vec, simpson
from scipy.interpolate import CubicSpline
from scipy.special import j1, jv, kv

import quartic2d
from quartic2d import HarmonicTransform, Interaction
from quartic2d._diagnostics import default_q_grid, inspect_transform

SCHEMA_VERSION = 1


def gaussian_m0_r(r):
    r = np.asarray(r)
    return np.exp(-r * r)


def gaussian_m0_q(q):
    q = np.asarray(q)
    return 0.5 * np.exp(-q * q / 4.0)


def gaussian_m2_r(r):
    r = np.asarray(r)
    return r**2 * np.exp(-r * r)


def gaussian_m2_q(q):
    q = np.asarray(q)
    return q**2 * np.exp(-q * q / 4.0) / 8.0


def gaussian_m4_r(r):
    r = np.asarray(r)
    return r**4 * np.exp(-r * r)


def gaussian_m4_q(q):
    q = np.asarray(q)
    return q**4 * np.exp(-q * q / 4.0) / 32.0


def nodal_r(r):
    r = np.asarray(r)
    return (1.0 - r * r) * np.exp(-r * r)


def nodal_q(q):
    q = np.asarray(q)
    return q * q * np.exp(-q * q / 4.0) / 8.0


def exponential_r(r):
    return np.exp(-np.asarray(r))


def exponential_q(q):
    q = np.asarray(q)
    return (1.0 + q * q) ** -1.5


def algebraic_r3_r(r):
    r = np.asarray(r)
    return (1.0 + r * r) ** -1.5


def algebraic_r3_q(q):
    return np.exp(-np.asarray(q))


def algebraic_r4_r(r):
    r = np.asarray(r)
    return (1.0 + r * r) ** -2.0


def algebraic_r4_q(q):
    q = np.asarray(q, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(q == 0.0, 0.5, 0.5 * q * kv(1, q))


def algebraic_m2_r(r):
    r = np.asarray(r)
    return r**2 / (1.0 + r * r) ** 3.5


def algebraic_m2_q(q):
    q = np.asarray(q)
    return q**2 * np.exp(-q) / 15.0


def oscillatory_exponential_r(r, a=0.35, omega=5.0):
    r = np.asarray(r)
    return np.exp(-a * r) * np.cos(omega * r)


def oscillatory_exponential_q(q, a=0.35, omega=5.0):
    q = np.asarray(q, dtype=float)
    s = a - 1j * omega
    return np.real(s / (s * s + q * q) ** 1.5)


def top_hat_r(r, R=2.0):
    return (np.asarray(r) <= R).astype(float)


def top_hat_q(q, R=2.0):
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(q == 0.0, 0.5 * R * R, R * j1(R * q) / q)


def annulus_r(r, R1=1.0, R2=2.5):
    r = np.asarray(r)
    return ((r >= R1) & (r <= R2)).astype(float)


def annulus_q(q, R1=1.0, R2=2.5):
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            q == 0.0,
            0.5 * (R2 * R2 - R1 * R1),
            (R2 * j1(R2 * q) - R1 * j1(R1 * q)) / q,
        )


def radial_cases():
    return [
        ("gaussian_m0", 0, 6.0, gaussian_m0_r, gaussian_m0_q),
        ("exponential_m0", 0, 24.0, exponential_r, exponential_q),
        ("algebraic_r3_m0", 0, 100.0, algebraic_r3_r, algebraic_r3_q),
        ("oscillatory_exponential_m0", 0, 40.0, oscillatory_exponential_r, oscillatory_exponential_q),
        ("gaussian_m2", 2, 6.0, gaussian_m2_r, gaussian_m2_q),
        ("gaussian_m4", 4, 6.0, gaussian_m4_r, gaussian_m4_q),
        ("nodal_gaussian_m0", 0, 7.0, nodal_r, nodal_q),
        ("algebraic_r4_m0", 0, 60.0, algebraic_r4_r, algebraic_r4_q),
        ("algebraic_m2", 2, 60.0, algebraic_m2_r, algebraic_m2_q),
        ("top_hat_m0", 0, 3.0, top_hat_r, top_hat_q),
        ("annulus_m0", 0, 4.0, annulus_r, annulus_q),
    ]


def support_radius(r, profile, *, tail_fraction=1.0e-8, amplitude_fraction=1.0e-6):
    values = np.asarray(profile(r))
    power = r * np.abs(values) ** 2
    total = simpson(power, x=r)
    power_radius = float(r[-1])
    for i in range(2, r.size):
        inside = simpson(power[: i + 1], x=r[: i + 1])
        if max(total - inside, 0.0) <= tail_fraction * total:
            power_radius = float(r[i])
            break
    threshold = amplitude_fraction * float(np.max(np.abs(values)))
    indices = np.flatnonzero(np.abs(values) >= threshold)
    amplitude_radius = float(r[indices[-1]]) if indices.size else 0.0
    return max(power_radius, amplitude_radius)


def interpolation_benchmark():
    rows = []
    for name, _m, r_max, radial, exact in radial_cases():
        r = np.linspace(0.0, r_max, 513)
        R = support_radius(r, radial)
        # The q interval is deliberately broad enough to include the useful
        # transformed support of each analytic case.
        q_max = {
            "gaussian_m0": 7.0,
            "exponential_m0": 32.0,
            "algebraic_r3_m0": 8.5,
            "oscillatory_exponential_m0": 16.0,
            "gaussian_m2": 7.0,
            "gaussian_m4": 7.7,
            "nodal_gaussian_m0": 7.0,
            "algebraic_r4_m0": 9.5,
            "algebraic_m2": 13.0,
            "top_hat_m0": 20.0,
            "annulus_m0": 20.0,
        }[name]
        q_ref = np.linspace(0.0, q_max, 20001)
        reference = np.asarray(exact(q_ref))
        norm = max(float(np.linalg.norm(reference)), 1.0e-300)
        peak = max(float(np.max(np.abs(reference))), 1.0e-300)
        for oversampling in (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
            n_q = max(4, 1 + math.ceil(oversampling * q_max * R / np.pi))
            q = np.linspace(0.0, q_max, n_q)
            values = exact(q)
            for bc_type in ("natural", "not-a-knot"):
                spline = CubicSpline(q, values, bc_type=bc_type, extrapolate=False)
                approx = spline(q_ref)
                rows.append(
                    {
                        "case": name,
                        "support_radius": R,
                        "q_max": q_max,
                        "oversampling": oversampling,
                        "n_q": n_q,
                        "bc_type": bc_type,
                        "relative_l2_error": float(np.linalg.norm(approx - reference) / norm),
                        "relative_linf_error": float(np.max(np.abs(approx - reference)) / peak),
                    }
                )
    return rows


def fast_default_diagnostic_benchmark():
    """Check one-pass defaults and cheap fault detectors on analytic profiles."""
    rows = []
    for name, m, r_max, radial, exact in radial_cases():
        r = np.linspace(0.0, r_max, 513)
        rho = np.asarray(radial(r))
        R = support_radius(r, radial)

        class Decomposition:
            def __init__(self):
                self.r = r
                self.rho = {int(m): rho}
                self.m_sorted = [int(m)]
                self.cutoff_radius = {int(m): float(R)}
                self.power_fracs = {int(m): 1.0}

            def __getitem__(self, key):
                return self.rho[int(key)]

        dec = Decomposition()
        q_max, n_q, info = default_q_grid(dec)
        q = np.linspace(0.0, q_max, n_q)
        F = np.asarray(exact(q))
        diagnostic = inspect_transform(dec, q, {int(m): F})

        # Exact omitted-tail norm, used only by the benchmark to judge the
        # cheap indicators.  The library fault detector itself never computes
        # this expensive reference integral.
        q_ref = np.linspace(0.0, info["q_ceiling"], 200001)
        ref = np.asarray(exact(q_ref))
        density = q_ref * np.abs(ref) ** 2
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(q_ref))]
        )
        total = float(cumulative[-1])
        idx = int(np.searchsorted(q_ref, q_max))
        omitted_l2 = (
            0.0
            if total <= 0.0
            else float(np.sqrt(max(total - cumulative[min(idx, q_ref.size - 1)], 0.0) / total))
        )
        rows.append(
            {
                "case": name,
                "q_max": float(q_max),
                "n_q": int(n_q),
                "q_ceiling": float(info["q_ceiling"]),
                "healthy": bool(diagnostic.healthy),
                "issue_codes": [issue.code for issue in diagnostic.issues],
                "exact_omitted_q_l2": omitted_l2,
                "boundary_amplitude_ratio": float(
                    diagnostic.per_harmonic[int(m)]["boundary_amplitude_ratio"]
                ),
                "boundary_power_fraction": float(
                    diagnostic.per_harmonic[int(m)]["boundary_power_fraction"]
                ),
            }
        )
    return rows


def _ff_gaussian_m0(q):
    q = np.asarray(q, dtype=float)
    return 0.5 * np.exp(-q * q / 4.0)


def _ff_gaussian_m2(q):
    q = np.asarray(q, dtype=float)
    return 0.08 * q * q * np.exp(-q * q / 4.0)


def _ff_nodal(q):
    q = np.asarray(q, dtype=float)
    return q * q * np.exp(-q * q / 4.0) / 8.0


def _ff_algebraic(q):
    return np.exp(-np.asarray(q, dtype=float))


def qU_coulomb(q):
    return 2.0 * np.pi * np.ones_like(np.asarray(q, dtype=float))


def qU_keldysh(r0):
    return lambda q: 2.0 * np.pi / (1.0 + r0 * np.asarray(q, dtype=float))


def qU_screened(q):
    q = np.asarray(q, dtype=float)
    return 2.0 * np.pi * q / np.sqrt(q * q + 1.0)


def qU_gaussian(q):
    q = np.asarray(q, dtype=float)
    return q * np.exp(-(0.5 * q) ** 2)


def qU_lorentzian(q):
    q = np.asarray(q, dtype=float)
    return q / (1.0 + (q / 2.0) ** 2)


def interaction_cases():
    iso = {0: _ff_gaussian_m0}
    aniso = {0: _ff_gaussian_m0, 2: _ff_gaussian_m2, -2: _ff_gaussian_m2}
    return [
        ("gaussian_coulomb", iso, qU_coulomb, 20.0),
        ("anisotropic_keldysh_r0_1", aniso, qU_keldysh(1.0), 20.0),
        ("algebraic_gaussian_kernel", {0: _ff_algebraic}, qU_gaussian, 40.0),
        ("anisotropic_keldysh_r0_0p1", aniso, qU_keldysh(0.1), 20.0),
        ("anisotropic_keldysh_r0_10", aniso, qU_keldysh(10.0), 20.0),
        ("nodal_screened", {0: _ff_nodal}, qU_screened, 20.0),
        ("algebraic_lorentzian", {0: _ff_algebraic}, qU_lorentzian, 40.0),
    ]


def required_q_from_form_factors(form_factors, q_max, tail_rtol):
    q = np.linspace(0.0, q_max, 200001)
    required = []
    for func in form_factors.values():
        density = q * np.abs(func(q)) ** 2
        cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(q))]
        )
        total = float(cumulative[-1])
        tail = np.sqrt(np.maximum(0.0, 1.0 - cumulative / total))
        indices = np.flatnonzero(tail <= tail_rtol)
        required.append(float(q[indices[0]]) if indices.size else float(q_max))
    return max(required)


def interaction_reference(form_factors, qU, deltas, q_max):
    total = np.zeros(deltas.size, dtype=np.complex128)
    for m, fm in form_factors.items():
        for mp, fp in form_factors.items():
            order = abs(int(m) - int(mp))

            def integrand(q, fm=fm, fp=fp, order=order):
                return qU(q) * fm(q) * np.conj(fp(q)) * jv(order, deltas * q)

            values, _ = quad_vec(
                integrand, 0.0, q_max, epsabs=1.0e-12, epsrel=1.0e-11, norm="2"
            )
            total += np.asarray(values)
    return 2.0 * np.pi * total


def q_tail_interaction_benchmark():
    rows = []
    deltas = np.geomspace(1.0e-2, 1.0e2, 32)
    for tolerance in (1.0e-2, 1.0e-3, 1.0e-4):
        for name, form_factors, qU, q_reference in interaction_cases():
            q_required = required_q_from_form_factors(form_factors, q_reference, tolerance)
            reference = interaction_reference(form_factors, qU, deltas, q_reference)
            truncated = interaction_reference(form_factors, qU, deltas, q_required)
            rows.append(
                {
                    "case": name,
                    "q_tail_rtol": tolerance,
                    "q_reference": q_reference,
                    "q_required": q_required,
                    "q_fraction": q_required / q_reference,
                    "interaction_truncation_relative_l2_error": float(
                        np.linalg.norm(truncated - reference) / np.linalg.norm(reference)
                    ),
                }
            )
    return rows


class AnalyticDecomposition:
    """PETAL2D-compatible analytic decomposition for the connected workload."""

    def __init__(self, anisotropy=0.35):
        self.anisotropy = float(anisotropy)
        self.r = np.linspace(0.0, 6.0, 256)
        rho0 = np.exp(-self.r * self.r) / np.pi
        rho2 = self.anisotropy * self.r**2 * np.exp(-self.r * self.r) / (2.0 * np.pi)
        self.rho = {0: rho0, 2: rho2, -2: rho2.copy()}
        self.m_sorted = [0, 2, -2]
        R0 = support_radius(self.r, lambda r: np.exp(-r * r) / np.pi)
        R2 = support_radius(
            self.r,
            lambda r: self.anisotropy * r**2 * np.exp(-r * r) / (2.0 * np.pi),
        )
        self.cutoff_radius = {0: R0, 2: R2, -2: R2}
        p0 = simpson(self.r * np.abs(rho0) ** 2, x=self.r)
        p2 = simpson(self.r * np.abs(rho2) ** 2, x=self.r)
        total = p0 + 2.0 * p2
        self.power_fracs = {0: p0 / total, 2: p2 / total, -2: p2 / total}

    def __getitem__(self, m):
        return self.rho[int(m)]


def anisotropic_exact_form_factor(m, q, anisotropy=0.35):
    q = np.asarray(q, dtype=float)
    envelope = np.exp(-q * q / 4.0)
    if int(m) == 0:
        return envelope / (2.0 * np.pi)
    return anisotropy * q * q * envelope / (16.0 * np.pi)


def connected_reference(deltas, q_max=40.0, n_q=65537):
    q = np.linspace(0.0, q_max, n_q)
    form = {m: anisotropic_exact_form_factor(m, q) for m in (0, 2, -2)}
    total = np.zeros(deltas.size, dtype=np.complex128)
    for m in form:
        for mp in form:
            base = 2.0 * np.pi * form[m] * np.conj(form[mp])
            total += simpson(
                jv(abs(m - mp), np.outer(deltas, q)) * base[None, :],
                x=q,
                axis=1,
            )
    return 2.0 * np.pi * total


def connected_benchmark():
    decomposition = AnalyticDecomposition()
    default = HarmonicTransform(decomposition, check=False)
    converged_result = HarmonicTransform.converge_parameters(
        decomposition, verbose=False
    )
    converged = converged_result.transform(decomposition, check=False)
    baseline = HarmonicTransform(
        decomposition,
        q_max=12.0,
        n_q=512,
        method="simpson",
        interpolator="cubic",
        subdivisions=4,
        check=False,
    )
    deltas = np.geomspace(1.0e-2, 1.0e2, 32)
    xy = np.column_stack((deltas, np.zeros_like(deltas)))

    def U(q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(q > 0.0, 2.0 * np.pi / q, np.inf)

    reference = connected_reference(deltas)
    rows = []
    for name, field in (
        ("default_fast_path", default),
        ("converged", converged),
        ("legacy_q12_n512", baseline),
    ):
        values = Interaction(
            xy,
            field,
            field,
            U,
            method="simpson",
            interpolator="cubic",
            subdivisions=16,
        ).V
        rows.append(
            {
                "configuration": name,
                "q_max": float(field.q[-1]),
                "n_q": int(field.q.size),
                "radial_subdivisions": int(
                    field.quadrature_convergence.selected_parameters["subdivisions"]
                )
                if field.quadrature_convergence is not None
                else 4,
                "fast_diagnostics_healthy": bool(field.diagnostics.healthy),
                "interaction_relative_l2_error": float(
                    np.linalg.norm(values - reference) / np.linalg.norm(reference)
                ),
            }
        )
    return {
        "rows": rows,
        "sampling_convergence": converged_result.sampling.to_dict(),
        "quadrature_convergence": converged_result.quadrature.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/development/q_sampling.json"),
    )
    args = parser.parse_args()
    result = {
        "schema_version": SCHEMA_VERSION,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "quartic2d": quartic2d.__version__,
        },
        "theory": {
            "q_ceiling": "pi / max(diff(r))",
            "n_q_scaling": "1 + ceil(s * q_max * R_max / pi)",
            "q_tail_default_rtol": 1.0e-3,
            "q_interpolation_default_rtol": 1.0e-4,
        },
        "interpolation_rows": interpolation_benchmark(),
        "fast_default_diagnostic_rows": fast_default_diagnostic_benchmark(),
        "q_tail_interaction_rows": q_tail_interaction_benchmark(),
        "connected": connected_benchmark(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
