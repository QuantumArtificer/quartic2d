#!/usr/bin/env python3
"""Run QUARTIC2D numerical-method validation and performance benchmarks.

The benchmark is intentionally split into the two numerical problems solved by
QUARTIC2D:

1. radial Hankel transforms of sampled PETAL2D harmonics;
2. interaction Hankel transforms as functions of displacement.

Calibration cases select numerical resolution with the same convergence API
exposed to users. Disjoint validation cases measure achieved accuracy and
runtime. Finite-domain truncation errors are reported separately from numerical
discretization errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

# Match PETAL2D's single-process benchmark policy unless the launcher already
# requested a different thread count.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import hankel
import numpy as np
import petal2d
import quartic2d
import scipy
from scipy.integrate import quad_vec, simpson
from scipy.special import j1, jv, kv

from quartic2d import HarmonicTransform, Interaction
from quartic2d._experimental.ogata import (
    Interaction as OgataInteraction,
    hankel_transform_sampled as ogata_hankel_transform_sampled,
)
from quartic2d._convergence import (
    ConvergenceResult,
    ConvergenceStep,
    converge_ogata,
    converge_sequence,
    relative_l2_error,
    relative_linf_error,
)
from quartic2d._numerics import Interpolator1D, hankel_transform_sampled

RADIAL_METHODS = ("trapezoid", "simpson", "gl4", "gl8", "ogata")
INTERACTION_METHODS = ("trapezoid", "simpson", "gl4", "gl8", "ogata", "fftlog")
FINITE_METHODS = ("trapezoid", "simpson", "gl4", "gl8")
SCHEMA_VERSION = 6


# -----------------------------------------------------------------------------
# Radial-transform reference cases
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RadialCase:
    name: str
    category: str
    nu: int
    r_max: float
    q_max: float
    radial: Callable[[np.ndarray], np.ndarray]
    exact: Callable[[np.ndarray], np.ndarray]
    split: str
    points: tuple[float, ...] = ()


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


def radial_cases() -> list[RadialCase]:
    # Calibration and validation are disjoint. The suite follows the pattern
    # used by numerical-transform comparison papers: smooth, slow-tail,
    # oscillatory, higher-order, nodal, and discontinuous profiles.
    return [
        RadialCase("gaussian_m0", "smooth_localized", 0, 6.0, 12.0, gaussian_m0_r, gaussian_m0_q, "calibration"),
        RadialCase("exponential_m0", "exponential", 0, 24.0, 12.0, exponential_r, exponential_q, "calibration"),
        RadialCase("algebraic_r3_m0", "algebraic", 0, 100.0, 8.0, algebraic_r3_r, algebraic_r3_q, "calibration"),
        RadialCase("oscillatory_exponential_m0", "oscillatory", 0, 40.0, 12.0, oscillatory_exponential_r, oscillatory_exponential_q, "validation"),
        RadialCase("gaussian_m2", "smooth_high_order", 2, 6.0, 12.0, gaussian_m2_r, gaussian_m2_q, "validation"),
        RadialCase("gaussian_m4", "smooth_high_order", 4, 6.0, 12.0, gaussian_m4_r, gaussian_m4_q, "validation"),
        RadialCase("nodal_gaussian_m0", "nodal", 0, 7.0, 12.0, nodal_r, nodal_q, "validation"),
        RadialCase("algebraic_r4_m0", "algebraic", 0, 60.0, 8.0, algebraic_r4_r, algebraic_r4_q, "validation"),
        RadialCase("algebraic_m2", "algebraic_high_order", 2, 60.0, 10.0, algebraic_m2_r, algebraic_m2_q, "validation"),
        RadialCase("top_hat_m0", "discontinuous", 0, 3.0, 20.0, top_hat_r, top_hat_q, "validation", (2.0,)),
        RadialCase("annulus_m0", "discontinuous", 0, 4.0, 20.0, annulus_r, annulus_q, "validation", (1.0, 2.5)),
    ]


# -----------------------------------------------------------------------------
# Interaction-transform reference cases
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelCase:
    name: str
    category: str
    U: Callable[[np.ndarray], np.ndarray]
    qU: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class InteractionCase:
    name: str
    category: str
    form_factors: dict[int, Callable[[np.ndarray], np.ndarray]]
    kernel: KernelCase
    q_max: float
    split: str


class AnalyticHarmonics:
    """Sampled harmonic field with an analytic evaluator for benchmark use."""

    def __init__(self, functions: dict[int, Callable], q_max: float, n_q: int):
        self._functions = {int(m): f for m, f in functions.items()}
        self._m_values = np.array(list(self._functions), dtype=int)
        self.q = np.linspace(0.0, q_max, int(n_q))
        self.F_q = {m: np.asarray(self._functions[m](self.q), dtype=np.complex128) for m in self._m_values}

    @property
    def m_values(self):
        return self._m_values.copy()

    def __call__(self, m, q):
        return self._functions[int(m)](np.asarray(q, dtype=float))


def kernel_cases() -> dict[str, KernelCase]:
    def coulomb(q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(q > 0.0, 2.0 * np.pi / q, np.inf)

    def keldysh(r0):
        def U(q):
            q = np.asarray(q, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(q > 0.0, 2.0 * np.pi / (q * (1.0 + r0 * q)), np.inf)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / (1.0 + r0 * q)

        return U, qU

    def screened(kappa):
        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / (q + kappa)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * q / (q + kappa)

        return U, qU

    def gaussian_q(sigma):
        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * np.exp(-0.5 * (sigma * q) ** 2)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return q * U(q)

        return U, qU

    def lorentzian(q0):
        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / (1.0 + (q / q0) ** 2)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return q * U(q)

        return U, qU

    U01, qU01 = keldysh(0.1)
    U1, qU1 = keldysh(1.0)
    U10, qU10 = keldysh(10.0)
    Us, qUs = screened(1.0)
    Ug, qUg = gaussian_q(0.5)
    Ul, qUl = lorentzian(2.0)
    return {
        "coulomb": KernelCase("coulomb", "singular", coulomb, lambda q: 2.0 * np.pi * np.ones_like(np.asarray(q, dtype=float))),
        "keldysh_r0_0p1": KernelCase("keldysh_r0_0p1", "singular_screened", U01, qU01),
        "keldysh_r0_1": KernelCase("keldysh_r0_1", "singular_screened", U1, qU1),
        "keldysh_r0_10": KernelCase("keldysh_r0_10", "singular_screened", U10, qU10),
        "screened_kappa_1": KernelCase("screened_kappa_1", "regular_screened", Us, qUs),
        "gaussian_sigma_0p5": KernelCase("gaussian_sigma_0p5", "localized", Ug, qUg),
        "lorentzian_q0_2": KernelCase("lorentzian_q0_2", "algebraic", Ul, qUl),
    }


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


def interaction_cases() -> list[InteractionCase]:
    kernels = kernel_cases()
    iso = {0: _ff_gaussian_m0}
    aniso = {0: _ff_gaussian_m0, 2: _ff_gaussian_m2, -2: _ff_gaussian_m2}
    nodal = {0: _ff_nodal}
    algebraic = {0: _ff_algebraic}
    return [
        InteractionCase("gaussian_coulomb", "smooth_singular", iso, kernels["coulomb"], 20.0, "calibration"),
        InteractionCase("anisotropic_keldysh_r0_1", "multi_harmonic", aniso, kernels["keldysh_r0_1"], 20.0, "calibration"),
        InteractionCase("algebraic_gaussian_kernel", "slow_tail_localized_kernel", algebraic, kernels["gaussian_sigma_0p5"], 40.0, "calibration"),
        InteractionCase("anisotropic_keldysh_r0_0p1", "multi_harmonic", aniso, kernels["keldysh_r0_0p1"], 20.0, "validation"),
        InteractionCase("anisotropic_keldysh_r0_10", "multi_harmonic", aniso, kernels["keldysh_r0_10"], 20.0, "validation"),
        InteractionCase("nodal_screened", "nodal", nodal, kernels["screened_kappa_1"], 20.0, "validation"),
        InteractionCase("algebraic_lorentzian", "slow_tail", algebraic, kernels["lorentzian_q0_2"], 40.0, "validation"),
    ]


def interaction_reference(case: InteractionCase, deltas: np.ndarray, q_max: float | None = None) -> np.ndarray:
    deltas = np.asarray(deltas, dtype=float)
    q_max = float(case.q_max if q_max is None else q_max)
    total = np.zeros(deltas.size, dtype=np.complex128)
    m_values = tuple(int(m) for m in case.form_factors)
    for m in m_values:
        for mp in m_values:
            order = abs(m - mp)

            def integrand(q, m=m, mp=mp, order=order):
                return (
                    case.kernel.qU(q)
                    * case.form_factors[m](q)
                    * np.conj(case.form_factors[mp](q))
                    * jv(order, deltas * q)
                )

            values, _ = quad_vec(
                integrand, 0.0, q_max, epsabs=1.0e-12, epsrel=1.0e-11, norm="2"
            )
            total += np.asarray(values)
    return 2.0 * np.pi * total


def build_interaction(case: InteractionCase, deltas: np.ndarray, method: str, params: dict, n_q: int) -> Interaction:
    field = AnalyticHarmonics(case.form_factors, case.q_max, n_q)
    deltas = np.asarray(deltas, dtype=float)
    xy = np.column_stack((deltas, np.zeros_like(deltas)))
    interaction_type = OgataInteraction if method == "ogata" else Interaction
    return interaction_type(
        xy,
        field,
        field,
        case.kernel.U,
        method=method,
        interpolator="cubic",
        N=int(params.get("N", 1024)),
        h=params.get("h"),
        n=int(params.get("n", 512)),
        bias=float(params.get("bias", -0.5)),
        subdivisions=int(params.get("subdivisions", 1)),
    )


def interaction_eval(case: InteractionCase, deltas: np.ndarray, method: str, params: dict, n_q: int) -> np.ndarray:
    return np.asarray(build_interaction(case, deltas, method, params, n_q).V, dtype=np.complex128)


# -----------------------------------------------------------------------------
# Common utilities
# -----------------------------------------------------------------------------

def environment_metadata() -> dict:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "hankel": getattr(hankel, "__version__", None),
        "petal2d": getattr(petal2d, "__version__", None),
        "quartic2d": getattr(quartic2d, "__version__", None),
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def timed_call(func, *, warmups: int, repeats: int):
    for _ in range(max(0, int(warmups))):
        func()
    samples = []
    result = None
    for _ in range(max(1, int(repeats))):
        t0 = perf_counter()
        result = func()
        samples.append(perf_counter() - t0)
    arr = np.asarray(samples, dtype=float)
    return result, {
        "runtime_median_seconds": float(np.median(arr)),
        "runtime_q25_seconds": float(np.quantile(arr, 0.25)),
        "runtime_q75_seconds": float(np.quantile(arr, 0.75)),
        "runtime_samples_seconds": arr.tolist(),
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row if k != "parameters"})
    if any("parameters" in row for row in rows):
        keys.append("parameters")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            if "parameters" in item:
                item["parameters"] = json.dumps(item["parameters"], sort_keys=True)
            writer.writerow(item)


def sampled_grid(case: RadialCase, nr: int):
    r = np.linspace(0.0, case.r_max, int(nr))
    return r, np.asarray(case.radial(r), dtype=float)


def same_domain_reference(case: RadialCase, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)

    def integrand(r):
        return r * case.radial(r) * jv(case.nu, q * r)

    values, _ = quad_vec(
        integrand,
        0.0,
        case.r_max,
        epsabs=1.0e-12,
        epsrel=1.0e-11,
        points=list(case.points) or None,
        norm="2",
    )
    return np.asarray(values)


def q_truncation_roundtrip_reference(case: RadialCase, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)

    def integrand(q):
        return q * case.exact(q) * jv(case.nu, r * q)

    values, _ = quad_vec(
        integrand, 0.0, case.q_max, epsabs=1.0e-12, epsrel=1.0e-11, norm="2"
    )
    return np.asarray(values)


def eval_radial_method(r, values, q, nu, method, params, *, interpolator="cubic"):
    transform = ogata_hankel_transform_sampled if method == "ogata" else hankel_transform_sampled
    return transform(
        r,
        values,
        q,
        nu,
        method=method,
        interpolator=interpolator,
        ogata_N=int(params.get("N", 1024)),
        ogata_h=params.get("h"),
        subdivisions=int(params.get("subdivisions", 1)),
    )


def n_nodes(method: str, params: dict, n_input: int) -> int:
    if method == "gl4":
        return 4 * (n_input - 1) * int(params["subdivisions"])
    if method == "gl8":
        return 8 * (n_input - 1) * int(params["subdivisions"])
    if method in ("simpson", "trapezoid"):
        return (n_input - 1) * int(params["subdivisions"]) + 1
    if method == "ogata":
        return int(params["N"])
    if method == "fftlog":
        return int(params["n"])
    raise ValueError(method)


def _unique_parameter_path(rows):
    unique = []
    seen = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def radial_parameter_path(method: str, selected: dict, *, quick: bool):
    if method in FINITE_METHODS:
        values = (1, 2, 4, 8) if quick else (1, 2, 4, 8, 16)
        return [{"subdivisions": int(v)} for v in values]
    rows = []
    if {"N", "h"}.issubset(selected):
        rows.append({"N": int(selected["N"]), "h": float(selected["h"])})
    h_values = (0.004, 0.002, 0.001) if quick else (0.008, 0.004, 0.002, 0.001, 0.0005)
    rows.extend({"h": float(h), "N": int(np.ceil(np.pi / h))} for h in h_values)
    return _unique_parameter_path(rows)


def interaction_parameter_path(method: str, selected: dict, *, quick: bool):
    if method in FINITE_METHODS:
        values = (1, 2, 4, 8) if quick else (1, 2, 4, 8, 16)
        return [{"subdivisions": int(v)} for v in values]
    if method == "ogata":
        rows = []
        if {"N", "h"}.issubset(selected):
            rows.append({"N": int(selected["N"]), "h": float(selected["h"])})
        h_values = (0.004, 0.002, 0.001) if quick else (0.008, 0.004, 0.002, 0.001, 0.0005)
        rows.extend({"h": float(h), "N": int(np.ceil(np.pi / h))} for h in h_values)
        return _unique_parameter_path(rows)
    values = (128, 256, 512, 1024) if quick else (128, 256, 512, 1024, 2048)
    rows = [{"n": int(n), "bias": float(selected["bias"])} for n in values]
    if {"n", "bias"}.issubset(selected):
        rows.insert(0, {"n": int(selected["n"]), "bias": float(selected["bias"])})
    return _unique_parameter_path(rows)


def normalized_radial_calibration_vector(cases_all, nr, q_by_case, method, params):
    chunks = []
    for case in cases_all:
        if case.split != "calibration":
            continue
        r, y = sampled_grid(case, nr)
        values = eval_radial_method(r, y, q_by_case[case.name], case.nu, method, params)
        norm = max(float(np.linalg.norm(values)), 1.0e-300)
        chunks.append(values / norm)
    return np.concatenate(chunks)


def calibrate_radial(method, cases_all, nr, q_by_case, *, rtol, atol, quick):
    if method == "ogata":
        per_case = []
        for case in cases_all:
            if case.split != "calibration":
                continue
            r, y = sampled_grid(case, nr)
            interp = Interpolator1D(r, y, "cubic")
            qpos = q_by_case[case.name]
            qpos = qpos[qpos > 0.0]
            stride = max(1, qpos.size // 8)
            K = np.unique(np.concatenate((qpos[::stride], qpos[-1:])))
            conv = converge_ogata(
                interp,
                case.nu,
                K,
                rtol=rtol,
                atol=atol,
                hstart=0.05,
                hdecrement=2.0,
                maxiter=15,
            )
            per_case.append({"case": case.name, "result": conv.to_dict()})
        successful = [
            row for row in per_case
            if row["result"]["resolution_converged"]
            and {"N", "h"}.issubset(row["result"]["selected_parameters"])
        ]
        if successful:
            h = min(float(row["result"]["selected_parameters"]["h"]) for row in successful)
            N = max(int(row["result"]["selected_parameters"]["N"]) for row in successful)
            selected = {"N": N, "h": h}
        else:
            selected = {}
        all_converged = bool(per_case) and len(successful) == len(per_case)
        return ConvergenceResult(
            method="ogata",
            rtol=rtol,
            atol=atol,
            selected_parameters=selected,
            selected_values=np.empty(0, dtype=np.complex128),
            steps=[ConvergenceStep(selected, 0.0, converged=all_converged, metadata={"source": "hankel.get_h"})],
            resolution_converged=all_converged,
            metadata={
                "kind": "hankel.get_h_across_radial_calibration_cases",
                "cases": per_case,
                "n_converged_cases": len(successful),
                "n_calibration_cases": len(per_case),
            },
        )

    subdivisions = (1, 2, 4, 8) if quick else (1, 2, 4, 8, 16, 32)

    def evaluator(params):
        return normalized_radial_calibration_vector(cases_all, nr, q_by_case, method, params)

    return converge_sequence(
        method,
        [{"subdivisions": int(v)} for v in subdivisions],
        evaluator,
        rtol=rtol,
        atol=atol,
    )


def roundtrip_radial(r, y, q, nu, method, params):
    F = eval_radial_method(r, y, q, nu, method, params)
    reconstructed = eval_radial_method(q, F, r, nu, method, params)
    return relative_l2_error(reconstructed, y)


def select_interaction_bias(cases_all, deltas, n_q, biases, n):
    rows = []
    refs = {case.name: interaction_reference(case, deltas) for case in cases_all if case.split == "calibration"}
    for bias in biases:
        errors = []
        for case in cases_all:
            if case.split != "calibration":
                continue
            values = interaction_eval(case, deltas, "fftlog", {"n": int(n), "bias": float(bias)}, n_q)
            errors.append(relative_l2_error(values, refs[case.name]))
        rows.append({
            "bias": float(bias),
            "n": int(n),
            "median_relative_l2_error": float(np.median(errors)),
            "max_relative_l2_error": float(np.max(errors)),
        })
    best = min(rows, key=lambda row: (row["max_relative_l2_error"], row["median_relative_l2_error"]))
    return float(best["bias"]), rows


def calibrate_interaction(method, cases_all, deltas, n_q, *, rtol, atol, quick):
    bias = None
    bias_rows = None
    if method == "fftlog":
        biases = (-0.5, -0.25, 0.0) if quick else (-0.75, -0.5, -0.25, 0.0, 0.25)
        bias, bias_rows = select_interaction_bias(cases_all, deltas, n_q, biases, n=1024)

    case_results = []
    for case in cases_all:
        if case.split != "calibration":
            continue
        if method == "fftlog":
            initial = {"n": 128, "bias": bias}
        elif method == "ogata":
            initial = {"N": 256, "h": 0.004}
        else:
            initial = {"subdivisions": 1}
        obj = build_interaction(case, deltas, method, initial, n_q)
        if method == "fftlog":
            conv = obj.converge(
                rtol=rtol,
                atol=atol,
                n_values=(128, 256, 512, 1024) if quick else (128, 256, 512, 1024, 2048),
                apply=False,
            )
        elif method == "ogata":
            conv = obj.converge(
                rtol=rtol,
                atol=atol,
                hstart=0.05,
                hdecrement=2.0,
                maxiter=15,
                apply=False,
            )
        else:
            conv = obj.converge(
                rtol=rtol,
                atol=atol,
                subdivisions=(1, 2, 4, 8) if quick else (1, 2, 4, 8, 16),
                apply=False,
            )
        case_results.append({"case": case.name, "result": conv.to_dict()})

    if method in FINITE_METHODS:
        subdivisions = max(int(row["result"]["selected_parameters"]["subdivisions"]) for row in case_results)
        selected = {"subdivisions": subdivisions}
    elif method == "ogata":
        successful = [
            row for row in case_results
            if row["result"]["resolution_converged"]
            and {"N", "h"}.issubset(row["result"]["selected_parameters"])
        ]
        selected = (
            {
                "N": max(int(row["result"]["selected_parameters"]["N"]) for row in successful),
                "h": min(float(row["result"]["selected_parameters"]["h"]) for row in successful),
            }
            if successful else {}
        )
    else:
        selected = {
            "n": max(int(row["result"]["selected_parameters"]["n"]) for row in case_results),
            "bias": float(bias),
        }

    metadata = {"kind": "interaction_calibration", "cases": case_results}
    if bias_rows is not None:
        metadata["bias_selection"] = {
            "criterion": "minimum worst-case calibration discretization error at fixed n",
            "rows": bias_rows,
        }
    return ConvergenceResult(
        method=method,
        rtol=rtol,
        atol=atol,
        selected_parameters=selected,
        selected_values=np.empty(0, dtype=np.complex128),
        steps=[ConvergenceStep(selected, 0.0, converged=True)],
        resolution_converged=bool(selected) and all(row["result"]["resolution_converged"] for row in case_results),
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# PETAL2D-connected representative workload
# -----------------------------------------------------------------------------

def _anisotropic_density(x, y, anisotropy=0.35):
    r2 = x * x + y * y
    return np.exp(-r2) * (1.0 + anisotropy * (x * x - y * y)) / np.pi


def _anisotropic_exact_form_factor(m: int, q, anisotropy=0.35):
    q = np.asarray(q, dtype=float)
    envelope = np.exp(-q * q / 4.0)
    if int(m) == 0:
        return envelope / (2.0 * np.pi)
    if abs(int(m)) == 2:
        return anisotropy * q * q * envelope / (16.0 * np.pi)
    return np.zeros_like(q)


def _interaction_reference_on_grid(form_factor, m_values, deltas, kernel: KernelCase, *, q_max: float, n_q: int):
    q = np.linspace(0.0, float(q_max), int(n_q))
    deltas = np.asarray(deltas, dtype=float)
    qU = np.asarray(kernel.qU(q), dtype=np.complex128)
    total = np.zeros(deltas.size, dtype=np.complex128)
    F = {int(m): np.asarray(form_factor(int(m), q), dtype=np.complex128) for m in m_values}
    for m in m_values:
        for mp in m_values:
            order = abs(int(m) - int(mp))
            base = qU * F[int(m)] * np.conj(F[int(mp)])
            total += simpson(jv(order, np.outer(deltas, q)) * base[None, :], x=q, axis=1)
    return 2.0 * np.pi * total


def prepare_petal2d_workload(*, quick: bool, subdivisions: int):
    nxy = 129 if quick else 181
    x = np.linspace(-6.0, 6.0, nxy)
    y = np.linspace(-6.0, 6.0, nxy)
    cdh = petal2d.PolarDecomposition(
        _anisotropic_density,
        x,
        y,
        Nr=160 if quick else 256,
        Ntheta=256,
        origin=(0.0, 0.0),
        recon_err_tol=1.0e-4,
        radial_power_tail_fraction=1.0e-8,
        radial_relative_amplitude_threshold=1.0e-6,
    )
    retained = tuple(sorted(int(m) for m in cdh.m_sorted))
    if retained != (-2, 0, 2):
        raise RuntimeError(f"PETAL2D benchmark expected retained harmonics (-2, 0, 2), got {retained}.")
    field = HarmonicTransform(
        cdh,
        q_max=12.0,
        n_q=256 if quick else 512,
        interpolator="cubic",
        method="gl4",
        subdivisions=int(subdivisions),
    )
    return cdh, field


def petal2d_interaction_eval(method, params, deltas, kernel: KernelCase, field):
    deltas = np.asarray(deltas, dtype=float)
    xy = np.column_stack((deltas, np.zeros_like(deltas)))
    interaction_type = OgataInteraction if method == "ogata" else Interaction
    return np.asarray(
        interaction_type(
            xy,
            field,
            field,
            kernel.U,
            method=method,
            interpolator="cubic",
            N=int(params.get("N", 1024)),
            h=params.get("h"),
            n=int(params.get("n", 512)),
            bias=float(params.get("bias", -0.5)),
            subdivisions=int(params.get("subdivisions", 1)),
        ).V,
        dtype=np.complex128,
    )


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------


def migrate_result_to_schema6(result):
    """Migrate schema-5 benchmark results without discarding valid work."""
    if result.get("schema_version") == SCHEMA_VERSION:
        return result
    if result.get("schema_version") != 5:
        raise RuntimeError(
            f"Cannot resume schema {result.get('schema_version')} with benchmark schema {SCHEMA_VERSION}. "
            "Use a new output directory."
        )

    result["schema_version"] = SCHEMA_VERSION
    result.setdefault("methodology", {})["error_component_policy"] = (
        "reported stage/component errors use different reference calculations and are not additive; "
        "cancellation between stages is possible"
    )

    # Schema 6 extends the quick FFTLog convergence path through n=1024 and
    # selects bias at n=1024. Recompute only FFTLog-derived sections.
    result.setdefault("interaction_calibration", {}).pop("fftlog", None)
    result["interaction_validation_rows"] = [
        row for row in result.get("interaction_validation_rows", []) if row.get("method") != "fftlog"
    ]
    result["interaction_tolerance_rows"] = [
        row for row in result.get("interaction_tolerance_rows", []) if row.get("method") != "fftlog"
    ]
    result["large_delta_rows"] = [
        row for row in result.get("large_delta_rows", []) if row.get("method") != "fftlog"
    ]
    result["scaling_rows"] = [
        row for row in result.get("scaling_rows", []) if row.get("method") != "fftlog"
    ]
    result["petal2d_rows"] = [
        row for row in result.get("petal2d_rows", []) if row.get("method") != "fftlog"
    ]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", "--warmup", dest="warmups", type=int, default=2)
    parser.add_argument("--cpu", type=int, default=None, help="Optional Linux CPU affinity for reproducible timings.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results/paper_numerics")
    args = parser.parse_args()

    quick = bool(args.quick)
    repeats = int(args.repeats if args.repeats is not None else (5 if quick else 9))
    warmups = int(args.warmups)
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be >= 1 and warmups must be >= 0")
    if args.cpu is not None:
        try:
            os.sched_setaffinity(0, {int(args.cpu)})
        except AttributeError as exc:
            raise RuntimeError("--cpu affinity requires os.sched_setaffinity") from exc

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"

    if args.resume and result_path.exists():
        result = migrate_result_to_schema6(json.loads(result_path.read_text()))
        save_json(result_path, result)
    else:
        result = {
            "schema_version": SCHEMA_VERSION,
            "quick": quick,
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "timing": {
                "warmups": warmups,
                "repeats": repeats,
                "statistic": "median",
                "spread": "interquartile_range",
                "cpu_affinity": args.cpu,
            },
            "environment": environment_metadata(),
            "methodology": {
                "error_definitions": {
                    "discretization_relative_l2_error": "||numerical-same_domain_reference||_2 / ||same_domain_reference||_2",
                    "truncation_relative_l2_error": "||same_domain_reference-infinite_reference||_2 / ||infinite_reference||_2",
                    "total_relative_l2_error": "||numerical-infinite_reference||_2 / ||infinite_reference||_2",
                    "roundtrip_relative_l2_error": "||f_roundtrip-f_sampled||_2 / ||f_sampled||_2",
                    "interaction_discretization_relative_l2_error": "||H_numerical-H_same_domain_reference||_2 / ||H_same_domain_reference||_2",
                    "interaction_truncation_relative_l2_error": "||H_same_domain_reference-H_infinite_reference||_2 / ||H_infinite_reference||_2",
                    "interaction_total_relative_l2_error": "||H_numerical-H_infinite_reference||_2 / ||H_infinite_reference||_2",
                    "absolute_error_over_reference_peak": "|H_numerical-H_reference| / max_delta |H_reference|",
                },
                "parameter_semantics": {
                    "rtol": "requested relative convergence tolerance",
                    "atol": "requested absolute convergence tolerance",
                    "subdivisions": "equal subdivisions of each original sampled interval before finite composite quadrature",
                    "N": "Ogata node count",
                    "h": "Ogata resolution parameter",
                    "nu": "Bessel/Hankel order",
                    "K": "representative transform scale(s) supplied to hankel.get_h",
                    "n": "FFTLog sequence length",
                    "bias": "FFTLog power-law bias/tilt; conditioning parameter, not resolution parameter",
                    "offset": "FFTLog logarithmic-grid offset selected internally by scipy.fft.fhtoffset",
                    "n_nodes": "cross-method number of transform/integration nodes",
                    "n_output": "number of requested transform output points",
                },
                "timing_policy": "perf_counter wall-clock timings after warmups; median and interquartile range reported, matching PETAL2D benchmark conventions",
                "calibration_policy": "parameters selected only on calibration cases; achieved errors reported on disjoint validation cases",
                "error_component_policy": "reported stage/component errors use different reference calculations and are not additive; cancellation between stages is possible",
            },
            "literature_design": {
                "principles": [
                    "analytic or independently integrated reference problems",
                    "multiple function classes because Hankel-method performance is problem dependent",
                    "separate discretization and finite-domain truncation errors",
                    "requested-tolerance versus achieved-error sweeps",
                    "repeated warmed timings with robust summary statistics",
                    "runtime scaling with output count",
                    "representative PETAL2D-connected workload",
                    "Ogata get_h calibration restricted to smooth/non-oscillatory calibration profiles, matching upstream documented assumptions; oscillatory profiles are held out for validation",
                ],
                "references": [
                    "Murray & Poulin, JOSS 2019, doi:10.21105/joss.01397",
                    "Hickstein et al., Rev. Sci. Instrum. 2019, doi:10.1063/1.5092635",
                    "Barnett, Magland & af Klinteberg, SIAM J. Sci. Comput. 2019, doi:10.1137/18M120885X",
                    "Hamilton, MNRAS 2000, FFTLog appendix",
                    "Cree & Bones, Computers & Mathematics with Applications 1993, doi:10.1016/0898-1221(93)90081-6",
                    "Virtanen et al., Nature Methods 2020, doi:10.1038/s41592-019-0686-2",
                ],
            },
            "radial_calibration": {},
            "radial_reference_cases": {},
            "radial_validation_rows": [],
            "radial_tolerance_rows": [],
            "interaction_calibration": {},
            "interaction_validation_rows": [],
            "interaction_tolerance_rows": [],
            "large_delta_rows": [],
            "scaling_rows": [],
            "petal2d_rows": [],
        }

    # ---------------- radial-transform benchmark ----------------
    rcases = radial_cases()
    nr = 257 if quick else 513
    nq = 129 if quick else 257
    q_by_case = {case.name: np.linspace(0.0, case.q_max, nq) for case in rcases}
    domain_refs = {}
    for case in rcases:
        q = q_by_case[case.name]
        domain_ref = same_domain_reference(case, q)
        exact = np.asarray(case.exact(q))
        domain_refs[case.name] = domain_ref
        if case.name not in result["radial_reference_cases"]:
            r, y = sampled_grid(case, nr)
            q_recon = q_truncation_roundtrip_reference(case, r)
            result["radial_reference_cases"][case.name] = {
                "category": case.category,
                "nu": case.nu,
                "split": case.split,
                "r_max": case.r_max,
                "q_max": case.q_max,
                "radial_truncation_relative_l2_error": relative_l2_error(domain_ref, exact),
                "q_truncation_roundtrip_relative_l2_error": relative_l2_error(q_recon, y),
            }
    save_json(result_path, result)

    radial_selected = {}
    for method in RADIAL_METHODS:
        if method in result["radial_calibration"]:
            radial_selected[method] = result["radial_calibration"][method]["selected_parameters"]
            print(f"[radial calibration] {method} (resume)", flush=True)
            continue
        print(f"[radial calibration] {method}", flush=True)
        conv = calibrate_radial(method, rcases, nr, q_by_case, rtol=args.rtol, atol=args.atol, quick=quick)
        result["radial_calibration"][method] = conv.to_dict()
        radial_selected[method] = conv.selected_parameters
        save_json(result_path, result)

    done = {
        (row["method"], row["case"], json.dumps(row["parameters"], sort_keys=True))
        for row in result["radial_validation_rows"]
    }
    for method in RADIAL_METHODS:
        for params in radial_parameter_path(method, radial_selected[method], quick=quick):
            for case in rcases:
                if case.split != "validation":
                    continue
                key = (method, case.name, json.dumps(params, sort_keys=True))
                if key in done:
                    continue
                r, y = sampled_grid(case, nr)
                q = q_by_case[case.name]
                domain_ref = domain_refs[case.name]
                exact = np.asarray(case.exact(q))
                F, timing = timed_call(
                    lambda: eval_radial_method(r, y, q, case.nu, method, params),
                    warmups=warmups,
                    repeats=repeats,
                )
                result["radial_validation_rows"].append({
                    "method": method,
                    "case": case.name,
                    "category": case.category,
                    "parameters": params,
                    "n_nodes": n_nodes(method, params, nr),
                    **timing,
                    "discretization_relative_l2_error": relative_l2_error(F, domain_ref),
                    "discretization_relative_linf_error": relative_linf_error(F, domain_ref),
                    "truncation_relative_l2_error": relative_l2_error(domain_ref, exact),
                    "total_relative_l2_error": relative_l2_error(F, exact),
                    "total_relative_linf_error": relative_linf_error(F, exact),
                    "roundtrip_relative_l2_error": roundtrip_radial(r, y, q, case.nu, method, params),
                    "q_truncation_roundtrip_relative_l2_error": result["radial_reference_cases"][case.name]["q_truncation_roundtrip_relative_l2_error"],
                })
                save_json(result_path, result)
        print(f"[radial validation] {method}", flush=True)

    requested_rtols = (1.0e-2, 1.0e-3, 1.0e-4) if quick else (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
    done_tol = {(row["method"], row["requested_rtol"]) for row in result["radial_tolerance_rows"]}
    for requested_rtol in requested_rtols:
        for method in RADIAL_METHODS:
            if (method, requested_rtol) in done_tol:
                continue
            conv = calibrate_radial(method, rcases, nr, q_by_case, rtol=requested_rtol, atol=args.atol, quick=quick)
            params = conv.selected_parameters
            errors = []
            runtimes = []
            for case in rcases:
                if case.split != "validation":
                    continue
                r, y = sampled_grid(case, nr)
                q = q_by_case[case.name]
                F, timing = timed_call(
                    lambda: eval_radial_method(r, y, q, case.nu, method, params),
                    warmups=warmups,
                    repeats=max(1, min(repeats, 3)),
                )
                errors.append(relative_l2_error(F, domain_refs[case.name]))
                runtimes.append(timing["runtime_median_seconds"])
            result["radial_tolerance_rows"].append({
                "method": method,
                "requested_rtol": float(requested_rtol),
                "resolution_converged": bool(conv.resolution_converged),
                "parameters": params,
                "n_nodes": n_nodes(method, params, nr),
                "validation_median_relative_l2_error": float(np.median(errors)),
                "validation_max_relative_l2_error": float(np.max(errors)),
                "runtime_median_seconds": float(np.median(runtimes)),
            })
            save_json(result_path, result)
        print(f"[radial tolerance] rtol={requested_rtol:g}", flush=True)

    # ---------------- interaction-transform benchmark ----------------
    icases = interaction_cases()
    interaction_deltas = np.geomspace(1.0e-2, 1.0e2 if quick else 1.0e3, 16 if quick else 28)
    interaction_nq = 513 if quick else 1025

    interaction_selected = {}
    for method in INTERACTION_METHODS:
        if method in result["interaction_calibration"]:
            interaction_selected[method] = result["interaction_calibration"][method]["selected_parameters"]
            print(f"[interaction calibration] {method} (resume)", flush=True)
            continue
        print(f"[interaction calibration] {method}", flush=True)
        conv = calibrate_interaction(
            method,
            icases,
            interaction_deltas,
            interaction_nq,
            rtol=args.rtol,
            atol=args.atol,
            quick=quick,
        )
        result["interaction_calibration"][method] = conv.to_dict()
        interaction_selected[method] = conv.selected_parameters
        save_json(result_path, result)

    irefs = {
        case.name: interaction_reference(case, interaction_deltas, q_max=case.q_max)
        for case in icases if case.split == "validation"
    }
    irefs_infinite = {
        case.name: interaction_reference(case, interaction_deltas, q_max=np.inf)
        for case in icases if case.split == "validation"
    }
    done_i = {
        (row["method"], row["case"], json.dumps(row["parameters"], sort_keys=True))
        for row in result["interaction_validation_rows"]
    }
    for method in INTERACTION_METHODS:
        for params in interaction_parameter_path(method, interaction_selected[method], quick=quick):
            for case in icases:
                if case.split != "validation":
                    continue
                key = (method, case.name, json.dumps(params, sort_keys=True))
                if key in done_i:
                    continue
                values, timing = timed_call(
                    lambda: interaction_eval(case, interaction_deltas, method, params, interaction_nq),
                    warmups=warmups,
                    repeats=max(1, min(repeats, 5)),
                )
                reference = irefs[case.name]
                infinite_reference = irefs_infinite[case.name]
                result["interaction_validation_rows"].append({
                    "method": method,
                    "case": case.name,
                    "category": case.category,
                    "kernel": case.kernel.name,
                    "parameters": params,
                    "n_nodes": n_nodes(method, params, interaction_nq),
                    **timing,
                    "discretization_relative_l2_error": relative_l2_error(values, reference),
                    "discretization_relative_linf_error": relative_linf_error(values, reference),
                    "truncation_relative_l2_error": relative_l2_error(reference, infinite_reference),
                    "total_relative_l2_error": relative_l2_error(values, infinite_reference),
                    "total_relative_linf_error": relative_linf_error(values, infinite_reference),
                })
                save_json(result_path, result)
        print(f"[interaction validation] {method}", flush=True)

    done_itol = {(row["method"], row["requested_rtol"]) for row in result["interaction_tolerance_rows"]}
    for requested_rtol in requested_rtols:
        for method in INTERACTION_METHODS:
            if (method, requested_rtol) in done_itol:
                continue
            conv = calibrate_interaction(
                method,
                icases,
                interaction_deltas,
                interaction_nq,
                rtol=requested_rtol,
                atol=args.atol,
                quick=quick,
            )
            params = conv.selected_parameters
            errors = []
            runtimes = []
            for case in icases:
                if case.split != "validation":
                    continue
                reference = irefs[case.name]
                values, timing = timed_call(
                    lambda: interaction_eval(case, interaction_deltas, method, params, interaction_nq),
                    warmups=warmups,
                    repeats=max(1, min(repeats, 3)),
                )
                errors.append(relative_l2_error(values, reference))
                runtimes.append(timing["runtime_median_seconds"])
            result["interaction_tolerance_rows"].append({
                "method": method,
                "requested_rtol": float(requested_rtol),
                "resolution_converged": bool(conv.resolution_converged),
                "parameters": params,
                "n_nodes": n_nodes(method, params, interaction_nq),
                "validation_median_relative_l2_error": float(np.median(errors)),
                "validation_max_relative_l2_error": float(np.max(errors)),
                "runtime_median_seconds": float(np.median(runtimes)),
            })
            save_json(result_path, result)
        print(f"[interaction tolerance] rtol={requested_rtol:g}", flush=True)

    # Large displacement: actual Interaction path with G(q)=exp(-q), whose exact
    # Hankel transform is (1 + Delta^2)^(-3/2). Peak-normalized absolute error
    # remains meaningful after the exact result becomes tiny.
    large_case = InteractionCase(
        "large_delta_exponential",
        "large_delta",
        {0: lambda q: np.exp(-0.5 * np.asarray(q, dtype=float))},
        KernelCase(
            "unit",
            "regular",
            lambda q: np.ones_like(np.asarray(q, dtype=float)),
            lambda q: np.asarray(q, dtype=float),
        ),
        40.0,
        "validation",
    )
    deltas = np.geomspace(1.0e-2, 1.0e3 if quick else 1.0e4, 25 if quick else 49)
    exact_H = (1.0 + deltas * deltas) ** -1.5
    peak = float(np.max(np.abs(exact_H)))
    done_large_delta = {row["method"] for row in result["large_delta_rows"]}
    for method in INTERACTION_METHODS:
        if method in done_large_delta:
            continue
        params = interaction_selected[method]
        values, timing = timed_call(
            lambda: interaction_eval(large_case, deltas, method, params, interaction_nq),
            warmups=warmups,
            repeats=repeats,
        )
        H = values / (2.0 * np.pi)
        for delta, value, reference in zip(deltas, H, exact_H):
            result["large_delta_rows"].append({
                "method": method,
                "delta": float(delta),
                "parameters": params,
                **{k: v for k, v in timing.items() if k != "runtime_samples_seconds"},
                "absolute_error_over_reference_peak": float(abs(value - reference) / peak),
                "pointwise_relative_error": float(abs(value - reference) / abs(reference)) if abs(reference) > 1.0e-12 * peak else None,
                "reference_over_peak": float(abs(reference) / peak),
            })
        save_json(result_path, result)
        print(f"[large delta] {method}", flush=True)

    # Scaling in number of interaction outputs using the same actual Interaction path.
    counts = (16, 64, 256) if quick else (16, 64, 256, 1024, 2048)
    done_scaling = {(row["method"], row["n_output"]) for row in result["scaling_rows"]}
    scaling_case = InteractionCase(
        "output_scaling",
        "scaling",
        {0: lambda q: np.exp(-0.5 * np.asarray(q, dtype=float))},
        KernelCase(
            "unit",
            "regular",
            lambda q: np.ones_like(np.asarray(q, dtype=float)),
            lambda q: np.asarray(q, dtype=float),
        ),
        40.0,
        "validation",
    )
    for count in counts:
        deltas = np.geomspace(1.0e-2, 1.0e3, count)
        for method in INTERACTION_METHODS:
            if (method, count) in done_scaling:
                continue
            params = interaction_selected[method]
            _, timing = timed_call(
                lambda: interaction_eval(scaling_case, deltas, method, params, interaction_nq),
                warmups=warmups,
                repeats=repeats,
            )
            result["scaling_rows"].append({
                "method": method,
                "n_output": int(count),
                "parameters": params,
                **timing,
            })
            save_json(result_path, result)
        print(f"[scaling] n_output={count}", flush=True)

    # Representative PETAL2D-connected anisotropic workload. First-stage GL4
    # settings are fixed across all interaction methods so the comparison isolates
    # the second transform while exercising the real software pipeline.
    done_petal = {row["method"] for row in result.get("petal2d_rows", [])}
    if len(done_petal) < len(INTERACTION_METHODS):
        radial_gl4 = radial_selected["gl4"]
        _, field = prepare_petal2d_workload(
            quick=quick,
            subdivisions=int(radial_gl4.get("subdivisions", 1)),
        )
        deltas = np.geomspace(1.0e-2, 1.0e2, 20 if quick else 32)
        kernel = kernel_cases()["coulomb"]
        ref_nq = 8193 if quick else 16385
        same_domain = _interaction_reference_on_grid(
            lambda m, q: field(m, q),
            tuple(int(m) for m in field.m_values),
            deltas,
            kernel,
            q_max=12.0,
            n_q=ref_nq,
        )
        analytic_same_domain = _interaction_reference_on_grid(
            _anisotropic_exact_form_factor,
            (0, 2, -2),
            deltas,
            kernel,
            q_max=12.0,
            n_q=ref_nq,
        )
        analytic_reference = _interaction_reference_on_grid(
            _anisotropic_exact_form_factor,
            (0, 2, -2),
            deltas,
            kernel,
            q_max=40.0,
            n_q=32769,
        )
        for method in INTERACTION_METHODS:
            if method in done_petal:
                continue
            params = interaction_selected[method]
            values, timing = timed_call(
                lambda: petal2d_interaction_eval(method, params, deltas, kernel, field),
                warmups=warmups,
                repeats=max(1, min(repeats, 5)),
            )
            result["petal2d_rows"].append({
                "method": method,
                "kernel": kernel.name,
                "parameters": params,
                "m_vals": [int(m) for m in field.m_values],
                "first_stage_quadrature": "gl4",
                "first_stage_subdivisions": int(radial_gl4.get("subdivisions", 1)),
                **timing,
                "interaction_discretization_relative_l2_error": relative_l2_error(values, same_domain),
                "first_stage_relative_l2_error": relative_l2_error(same_domain, analytic_same_domain),
                "q_truncation_relative_l2_error": relative_l2_error(analytic_same_domain, analytic_reference),
                "total_relative_l2_error": relative_l2_error(values, analytic_reference),
            })
            save_json(result_path, result)
        print("[petal2d] anisotropic interaction", flush=True)

    write_csv(out / "radial_validation.csv", result["radial_validation_rows"])
    write_csv(out / "radial_tolerance.csv", result["radial_tolerance_rows"])
    write_csv(out / "interaction_validation.csv", result["interaction_validation_rows"])
    write_csv(out / "interaction_tolerance.csv", result["interaction_tolerance_rows"])
    write_csv(out / "large_delta.csv", result["large_delta_rows"])
    write_csv(out / "scaling.csv", result["scaling_rows"])
    write_csv(out / "petal2d.csv", result["petal2d_rows"])

    from plot_numerical_benchmarks import make_plots

    make_plots(result_path, out / "figures")
    print(result_path)


if __name__ == "__main__":
    main()
