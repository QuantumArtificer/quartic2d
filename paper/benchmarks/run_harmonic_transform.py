#!/usr/bin/env python3
"""Paper benchmark: end-to-end multi-harmonic ``HarmonicTransform``.

This is manuscript evidence, not a package CI test. It times the public
``HarmonicTransform`` constructor and keeps q-range truncation, q-grid
interpolation, radial-input representation, and radial-quadrature errors
separate.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import quad, quad_vec, simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import j1, jv, kv

from quartic2d import HarmonicTransform
from quartic2d._numerics import Interpolator1D, composite_gauss_nodes_weights


@dataclass(frozen=True)
class Mode:
    m: int
    radial: Callable[[np.ndarray], np.ndarray]
    exact: Callable[[np.ndarray], np.ndarray]
    amplitude: complex = 1.0
    cutoff: float | None = None


@dataclass(frozen=True)
class Workload:
    name: str
    category: str
    r_max: float
    modes: tuple[Mode, ...]
    stress: bool = False


class SyntheticDecomposition:
    """Minimal PETAL2D-compatible decomposition for transform benchmarks."""

    def __init__(self, workload: Workload, n_r: int):
        self.r = np.linspace(0.0, workload.r_max, int(n_r))
        self.m_sorted = [mode.m for mode in workload.modes]
        self.rho = {
            mode.m: np.asarray(mode.amplitude * mode.radial(self.r), dtype=np.complex128)
            for mode in workload.modes
        }
        self.cutoff_radius = {
            mode.m: float(workload.r_max if mode.cutoff is None else mode.cutoff)
            for mode in workload.modes
        }
        # Only used by optional diagnostics/convergence helpers, not by the
        # explicit check=False production timing path.
        powers = {}
        for mode in workload.modes:
            vals = self.rho[mode.m]
            powers[mode.m] = float(simpson(np.abs(vals) ** 2 * self.r, x=self.r))
        total = max(sum(powers.values()), np.finfo(float).tiny)
        self.power_fracs = {m: p / total for m, p in powers.items()}

    def __getitem__(self, m):
        return self.rho[int(m)]


# -----------------------------------------------------------------------------
# Analytic radial profiles and their infinite-domain Hankel transforms
# -----------------------------------------------------------------------------

def gaussian_m0_r(r):
    r = np.asarray(r)
    return np.exp(-r * r)


def gaussian_m0_q(q):
    q = np.asarray(q)
    return 0.5 * np.exp(-q * q / 4.0)


def gaussian_m1_r(r):
    r = np.asarray(r)
    return r * np.exp(-r * r)


def gaussian_m1_q(q):
    q = np.asarray(q)
    return 0.25 * q * np.exp(-q * q / 4.0)


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


def oscillatory_exponential_r(r, a=0.35, omega=5.0):
    r = np.asarray(r)
    return np.exp(-a * r) * np.cos(omega * r)


def oscillatory_exponential_q(q, a=0.35, omega=5.0):
    q = np.asarray(q, dtype=float)
    s = a - 1j * omega
    return np.real(s / (s * s + q * q) ** 1.5)


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


def signed_exact(mode: Mode, q):
    # HarmonicTransform uses J_|m| and applies J_-n=(-1)^n J_n.
    sign = (-1.0) ** abs(mode.m) if mode.m < 0 else 1.0
    return sign * mode.amplitude * np.asarray(mode.exact(q), dtype=np.complex128)


def workloads() -> list[Workload]:
    return [
        Workload(
            "gaussian_isotropic",
            "smooth / one harmonic",
            6.0,
            (Mode(0, gaussian_m0_r, gaussian_m0_q),),
        ),
        Workload(
            "gaussian_odd_pair",
            "smooth / odd conjugate pair",
            6.0,
            (
                Mode(1, gaussian_m1_r, gaussian_m1_q, 0.5),
                Mode(-1, gaussian_m1_r, gaussian_m1_q, 0.5),
            ),
        ),
        Workload(
            "gaussian_anisotropic_m2",
            "smooth / three harmonics",
            6.0,
            (
                Mode(0, gaussian_m0_r, gaussian_m0_q, 1.0),
                Mode(2, gaussian_m2_r, gaussian_m2_q, 0.30),
                Mode(-2, gaussian_m2_r, gaussian_m2_q, 0.30),
            ),
        ),
        Workload(
            "gaussian_high_order_m4",
            "smooth / five harmonics",
            6.0,
            (
                Mode(0, gaussian_m0_r, gaussian_m0_q, 1.0),
                Mode(2, gaussian_m2_r, gaussian_m2_q, 0.30),
                Mode(-2, gaussian_m2_r, gaussian_m2_q, 0.30),
                Mode(4, gaussian_m4_r, gaussian_m4_q, 0.05),
                Mode(-4, gaussian_m4_r, gaussian_m4_q, 0.05),
            ),
        ),
        Workload(
            "nodal_mixed",
            "nodal / three harmonics",
            7.0,
            (
                Mode(0, nodal_r, nodal_q, 1.0),
                Mode(2, gaussian_m2_r, gaussian_m2_q, 0.20, cutoff=6.0),
                Mode(-2, gaussian_m2_r, gaussian_m2_q, 0.20, cutoff=6.0),
            ),
        ),
        Workload(
            "complex_mixed",
            "complex / mixed parity",
            6.0,
            (
                Mode(0, gaussian_m0_r, gaussian_m0_q, 1.0 + 0.0j),
                Mode(1, gaussian_m1_r, gaussian_m1_q, 0.35 * np.exp(0.4j)),
                Mode(-2, gaussian_m2_r, gaussian_m2_q, 0.20 * np.exp(-0.7j)),
            ),
        ),
        Workload(
            "exponential_cusp",
            "localized cusp / one harmonic",
            24.0,
            (Mode(0, exponential_r, exponential_q),),
        ),
        Workload(
            "oscillatory_exponential",
            "oscillatory localized / one harmonic",
            40.0,
            (Mode(0, oscillatory_exponential_r, oscillatory_exponential_q),),
        ),
        Workload(
            "algebraic_mixed",
            "algebraic / three harmonics",
            60.0,
            (
                Mode(0, algebraic_r4_r, algebraic_r4_q, 1.0),
                Mode(2, algebraic_m2_r, algebraic_m2_q, 0.50),
                Mode(-2, algebraic_m2_r, algebraic_m2_q, 0.50),
            ),
        ),
        Workload(
            "top_hat_stress",
            "discontinuous stress test",
            3.0,
            (Mode(0, top_hat_r, top_hat_q, cutoff=2.0),),
            stress=True,
        ),
        Workload(
            "annulus_stress",
            "discontinuous stress test",
            4.0,
            (Mode(0, annulus_r, annulus_q, cutoff=2.5),),
            stress=True,
        ),
    ]


# -----------------------------------------------------------------------------
# Error/support helpers
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Reference, input-qualification, and convergence helpers
# -----------------------------------------------------------------------------

from functools import lru_cache
from numpy.polynomial.legendre import leggauss

try:
    from _common import environment_metadata
except ImportError:  # pragma: no cover
    from paper.benchmarks._common import environment_metadata


def radial_norm_sq(mode: Mode, upper=np.inf) -> float:
    f = lambda r: r * abs(mode.amplitude * mode.radial(r)) ** 2
    return float(quad(f, 0.0, upper, epsabs=1e-13, epsrel=1e-11, limit=1000)[0])


def q_tail_l2_analytic(mode: Mode, q0: float) -> float:
    """Analytic-profile q-tail, used only to seed the sampled-input search."""
    total = radial_norm_sq(mode)
    if total == 0.0:
        return 0.0
    fn = lambda q: q * abs(signed_exact(mode, q)) ** 2
    tail = float(
        quad(
            fn,
            q0,
            np.inf,
            epsabs=max(1e-15, total * 1e-13),
            epsrel=1e-10,
            limit=2000,
        )[0]
    )
    return math.sqrt(max(tail, 0.0) / total)


def required_q_max_analytic(mode: Mode, target: float) -> float:
    """Return an analytic q-support estimate without imposing an input ceiling."""
    if target <= 0.0:
        raise ValueError("target must be positive")
    hi = 1.0
    while q_tail_l2_analytic(mode, hi) > target:
        hi *= 2.0
        if hi > 1.0e5:
            raise RuntimeError(f"Could not bracket analytic q support for m={mode.m}.")
    return float(
        brentq(
            lambda q: q_tail_l2_analytic(mode, q) - target,
            0.0,
            hi,
            xtol=1e-8,
            rtol=1e-10,
            maxiter=200,
        )
    )


def _sampled_mode_objects(decomp, workload: Workload):
    """Build the exact numerical input objects seen by Quartic2D.

    The benchmark reference is the Hankel transform of these cubic radial
    interpolants over their supplied cutoffs.  Consequently PETAL2D sampling,
    domain, and cutoff errors are not part of the Quartic2D accuracy metric.
    """
    out = {}
    r_all = np.asarray(decomp.r, dtype=float)
    for mode in workload.modes:
        cutoff = float(decomp.cutoff_radius[mode.m])
        mask = r_all <= cutoff + 16.0 * np.finfo(float).eps * max(1.0, abs(cutoff))
        r = r_all[mask]
        y = np.asarray(decomp[mode.m])[mask]
        if r.size < 2:
            raise RuntimeError(f"Insufficient radial samples for m={mode.m}.")
        upper = float(r[-1])
        interp = Interpolator1D(r, y, "cubic")
        order = abs(int(mode.m))
        sign = (-1.0) ** order if mode.m < 0 else 1.0
        # For a cubic spline, r*|s(r)|^2 is a degree-7 polynomial on each
        # interval. Four-point Gauss-Legendre is therefore exact on every
        # spline panel up to floating-point roundoff.
        z_norm, w_norm = composite_gauss_nodes_weights(r, 4)
        total_norm = float(np.sum(w_norm * z_norm * np.abs(interp(z_norm)) ** 2))
        out[int(mode.m)] = {
            "mode": mode,
            "r": r,
            "upper": upper,
            "interp": interp,
            "order": order,
            "sign": sign,
            "total_norm": total_norm,
        }
    return out


def evaluate_sampled_reference(sampled, q):
    """High-accuracy reference transform of the exact sampled radial input."""
    q = np.asarray(q, dtype=float)
    out = {}
    for m, obj in sampled.items():
        if q.size == 0:
            out[m] = np.empty(0, dtype=np.complex128)
            continue
        interp = obj["interp"]
        order = obj["order"]
        upper = obj["upper"]
        vals, _ = quad_vec(
            lambda r: r * interp(r) * jv(order, q * r),
            0.0,
            upper,
            epsabs=1e-12,
            epsrel=1e-10,
            norm="2",
        )
        out[m] = obj["sign"] * np.asarray(vals, dtype=np.complex128)
    return out


def _radial_input_metrics(decomp, workload: Workload):
    """External input qualification metrics, evaluated in radial space.

    Parseval makes the weighted-radial L2 error equal to the full Hankel-space
    L2 error.  These values qualify the synthetic PETAL2D-compatible input but
    are never added to Quartic2D's benchmark error.
    """
    sampled = _sampled_mode_objects(decomp, workload)
    rows = []
    worst = 0.0
    for m, obj in sampled.items():
        mode = obj["mode"]
        interp = obj["interp"]
        upper = obj["upper"]
        exact_total = radial_norm_sq(mode)
        # This is an external input-qualification calculation, not part of
        # the Quartic2D error. Composite GL8 is applied panel-by-panel on the
        # supplied radial mesh so narrow interpolation defects are not skipped.
        z_in, w_in = composite_gauss_nodes_weights(obj["r"], 8)
        exact_in = mode.amplitude * mode.radial(z_in)
        interior = float(
            np.sum(w_in * z_in * np.abs(interp(z_in) - exact_in) ** 2)
        )
        tail = float(
            quad(
                lambda r: r * abs(mode.amplitude * mode.radial(r)) ** 2,
                upper,
                np.inf,
                epsabs=max(1e-15, exact_total * 1e-13),
                epsrel=1e-10,
                limit=2000,
            )[0]
        )
        interp_eps = math.sqrt(max(interior, 0.0) / exact_total) if exact_total > 0 else 0.0
        tail_eps = math.sqrt(max(tail, 0.0) / exact_total) if exact_total > 0 else 0.0
        total_eps = math.sqrt(max(interior + tail, 0.0) / exact_total) if exact_total > 0 else 0.0
        worst = max(worst, total_eps)
        rows.append(
            {
                "m": int(m),
                "effective_cutoff": upper,
                "sampling_interpolation_relative_l2": interp_eps,
                "finite_domain_tail_relative_l2": tail_eps,
                "total_input_representation_relative_l2": total_eps,
            }
        )
    return sampled, rows, float(worst)


@lru_cache(maxsize=None)
def _legendre_rule(n: int):
    x, w = leggauss(int(n))
    return np.asarray(x, dtype=float), np.asarray(w, dtype=float)


def q_reference_rule(q_max: float, n: int):
    x, w = _legendre_rule(int(n))
    return 0.5 * q_max * (x + 1.0), 0.5 * q_max * w


def _q_error_metrics(q, w, diff, ref, total_norm):
    num = float(np.sum(w * q * np.abs(diff) ** 2))
    inband = float(np.sum(w * q * np.abs(ref) ** 2))
    total_eps = math.sqrt(max(num, 0.0) / total_norm) if total_norm > 0.0 else 0.0
    inband_eps = math.sqrt(max(num, 0.0) / inband) if inband > 0.0 else 0.0
    peak = float(np.max(np.abs(ref))) if np.size(ref) else 0.0
    peak_eps = float(np.max(np.abs(diff)) / peak) if peak > 0.0 else float(np.max(np.abs(diff)))
    return total_eps, inband_eps, peak_eps, inband


def sampled_q_reference_bundle(sampled, q_max: float, order: int):
    """Independent in-domain reference for the exact sampled radial input.

    This object is used only to measure q-grid interpolation and radial
    quadrature errors inside [0, q_max].  It deliberately does not estimate
    omitted q-space weight: high-q artifacts of a finite sampled spline belong
    to the external input representation, not to Quartic2D's transform error.
    """
    q, w = q_reference_rule(q_max, int(order))
    ref = evaluate_sampled_reference(sampled, q)
    modes = {}
    for m, values in ref.items():
        total = float(sampled[m]["total_norm"])
        inband = float(np.sum(w * q * np.abs(values) ** 2))
        modes[m] = {
            "total_norm": total,
            "inband_norm": inband,
        }
    return {"q": q, "w": w, "reference": ref, "modes": modes, "order": int(order)}


def analytic_q_support_bundle(workload: Workload, q_max: float):
    """Known-reference q-support metrics for the analytic benchmark profiles."""
    modes = {}
    for mode in workload.modes:
        modes[int(mode.m)] = {
            "tail_relative_l2": float(q_tail_l2_analytic(mode, q_max)),
        }
    return {"q_max": float(q_max), "modes": modes}


def select_analytic_qmax(workload: Workload, tail_target: float, *, safety_fraction: float = 0.95):
    """Choose q_max from the known analytic benchmark transform.

    The small safety margin avoids placing the selected q_max exactly on the
    allocated tail-error boundary, while keeping the selected domain close to
    the minimum required support.
    """
    if not (0.0 < safety_fraction <= 1.0):
        raise ValueError("safety_fraction must lie in (0, 1].")
    selection_target = float(tail_target) * float(safety_fraction)
    rows = []
    q_values = []
    for mode in workload.modes:
        q_m = required_q_max_analytic(mode, selection_target)
        rows.append({
            "m": int(mode.m),
            "q_max_required": float(q_m),
            "tail_relative_l2_at_q_max": float(q_tail_l2_analytic(mode, q_m)),
        })
        q_values.append(float(q_m))
    q_max = float(max(q_values))
    bundle = analytic_q_support_bundle(workload, q_max)
    return q_max, bundle, rows, selection_target

def qualify_input(
    workload: Workload,
    *,
    n_r_candidates,
    input_rtol: float,
    q_required: float,
    q_ceiling_fraction: float,
):
    """Find the first tested radial input satisfying external preconditions.

    Input representation quality and radial Nyquist support are external to
    Quartic2D.  They are recorded as benchmark preconditions and are never
    included in Quartic2D accuracy metrics.
    """
    history = []
    for n_r in n_r_candidates:
        decomp = SyntheticDecomposition(workload, int(n_r))
        dr = float(decomp.r[1] - decomp.r[0])
        q_ceiling = math.pi / dr
        sampled, input_modes, worst_input = _radial_input_metrics(decomp, workload)
        q_limit = q_ceiling_fraction * q_ceiling
        candidate = {
            "n_r": int(n_r),
            "r_max": float(workload.r_max),
            "dr": dr,
            "q_ceiling": q_ceiling,
            "q_limit": q_limit,
            "worst_input_representation_relative_l2": worst_input,
            "input_representation_pass": bool(worst_input <= input_rtol),
            "q_support_precondition_pass": bool(q_required <= q_limit),
        }
        history.append(candidate)
        if worst_input > input_rtol or q_required > q_limit:
            continue
        return {
            "decomposition": decomp,
            "sampled": sampled,
            "input_modes": input_modes,
            "worst_input": worst_input,
            "history": history,
            "dr": dr,
            "q_ceiling": q_ceiling,
            "q_limit": q_limit,
        }, history
    return None, history

def select_nq_sampled(
    sampled,
    q_max: float,
    q_bundle,
    candidates,
    *,
    l2_target: float,
    peak_target: float,
):
    """Converge uniform q sampling against the independent sampled-input reference."""
    q_test = q_bundle["q"]
    w_test = q_bundle["w"]
    ref_test = q_bundle["reference"]
    history = []
    for n_q in candidates:
        q_nodes = np.linspace(0.0, q_max, int(n_q))
        ref_nodes = evaluate_sampled_reference(sampled, q_nodes)
        modes = []
        for m, y_nodes in ref_nodes.items():
            spline = CubicSpline(q_nodes, y_nodes, bc_type="not-a-knot", extrapolate=False)
            pred = spline(q_test)
            diff = pred - ref_test[m]
            eps_total, eps_inband, eps_peak, _ = _q_error_metrics(
                q_test,
                w_test,
                diff,
                ref_test[m],
                sampled[m]["total_norm"],
            )
            modes.append(
                {
                    "m": int(m),
                    "relative_l2_total_norm": eps_total,
                    "relative_l2_inband_norm": eps_inband,
                    "relative_max_peak": eps_peak,
                }
            )
        worst_l2 = max(x["relative_l2_total_norm"] for x in modes)
        worst_peak = max(x["relative_max_peak"] for x in modes)
        row = {
            "n_q": int(n_q),
            "worst_relative_l2_total_norm": worst_l2,
            "worst_relative_max_peak": worst_peak,
            "modes": modes,
        }
        history.append(row)
        if worst_l2 <= l2_target and worst_peak <= peak_target:
            return int(n_q), ref_nodes, history
    return None, None, history


def _node_quadrature_errors(ht, sampled, q_nodes, ref_nodes):
    rows = []
    for m, ref in ref_nodes.items():
        diff = np.asarray(ht[m]) - ref
        num = float(np.real(simpson(q_nodes * np.abs(diff) ** 2, x=q_nodes)))
        eps_total = math.sqrt(max(num, 0.0) / sampled[m]["total_norm"])
        peak = float(np.max(np.abs(ref)))
        eps_peak = float(np.max(np.abs(diff)) / peak) if peak > 0.0 else float(np.max(np.abs(diff)))
        rows.append(
            {
                "m": int(m),
                "relative_l2_total_norm": eps_total,
                "relative_max_peak": eps_peak,
            }
        )
    return rows


def _in_domain_errors(ht, sampled, q_bundle):
    rows = []
    q = q_bundle["q"]
    w = q_bundle["w"]
    for m, ref in q_bundle["reference"].items():
        pred = ht(m, q)
        diff = pred - ref
        eps_total, eps_inband, eps_peak, _ = _q_error_metrics(
            q, w, diff, ref, sampled[m]["total_norm"]
        )
        rows.append(
            {
                "m": int(m),
                "relative_l2_total_norm": eps_total,
                "relative_l2_inband_norm": eps_inband,
                "relative_max_peak": eps_peak,
            }
        )
    return rows


def select_subdivisions(
    decomp,
    sampled,
    q_max,
    n_q,
    method,
    candidates,
    ref_nodes,
    q_bundle,
    *,
    in_domain_target: float,
    peak_target: float,
):
    history = []
    q_nodes = np.linspace(0.0, q_max, int(n_q))
    selected = None
    for subdivisions in candidates:
        t0 = time.perf_counter()
        ht = HarmonicTransform(
            decomp,
            q_max=q_max,
            n_q=n_q,
            method=method,
            interpolator="cubic",
            subdivisions=int(subdivisions),
            check=False,
        )
        dt = time.perf_counter() - t0
        node_rows = _node_quadrature_errors(ht, sampled, q_nodes, ref_nodes)
        in_rows = _in_domain_errors(ht, sampled, q_bundle)
        worst_in = max(x["relative_l2_total_norm"] for x in in_rows)
        worst_peak = max(x["relative_max_peak"] for x in in_rows)
        history.append(
            {
                "subdivisions": int(subdivisions),
                "single_run_seconds": dt,
                "quadrature_node_errors": node_rows,
                "in_domain_errors": in_rows,
                "worst_in_domain_relative_l2_total_norm": worst_in,
                "worst_in_domain_relative_max_peak": worst_peak,
            }
        )
        if worst_in <= in_domain_target and worst_peak <= peak_target:
            selected = int(subdivisions)
            break
    return selected, history


def time_constructor(decomp, *, q_max, n_q, method, subdivisions, warmups, repeats):
    kwargs = dict(
        q_max=q_max,
        n_q=n_q,
        method=method,
        interpolator="cubic",
        subdivisions=subdivisions,
        check=False,
    )
    for _ in range(int(warmups)):
        HarmonicTransform(decomp, **kwargs)
    times = []
    last = None
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        last = HarmonicTransform(decomp, **kwargs)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=float)
    return last, {
        "median_seconds": float(statistics.median(times)),
        "q25_seconds": float(np.quantile(arr, 0.25)),
        "q75_seconds": float(np.quantile(arr, 0.75)),
        "min_seconds": float(min(times)),
        "max_seconds": float(max(times)),
        "times_seconds": [float(x) for x in times],
    }


def _final_method_metrics(ht, sampled, q_bundle, support_bundle):
    in_rows = _in_domain_errors(ht, sampled, q_bundle)
    by_m = {x["m"]: x for x in in_rows}
    rows = []
    for m, in_row in by_m.items():
        support = float(support_bundle["modes"][int(m)]["tail_relative_l2"])
        rows.append(
            {
                "m": int(m),
                "q_support_tail_relative_l2": support,
                "in_domain_relative_l2_total_norm": float(in_row["relative_l2_total_norm"]),
                "in_domain_relative_l2_inband_norm": float(in_row["relative_l2_inband_norm"]),
                "in_domain_relative_max_peak": float(in_row["relative_max_peak"]),
            }
        )
    return rows

def run(args):
    tols = [float(x) for x in args.tolerances.split(",")]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    n_r_candidates = [int(x) for x in args.nr_candidates.split(",") if x.strip()]
    nq_candidates = [int(x) for x in args.nq_candidates.split(",") if x.strip()]
    sub_candidates = [int(x) for x in args.subdivisions.split(",") if x.strip()]

    all_workloads = workloads()
    selected_workloads = all_workloads if args.include_stress else [w for w in all_workloads if not w.stress]
    if args.workloads:
        wanted = {x.strip() for x in args.workloads.split(",") if x.strip()}
        selected_workloads = [w for w in selected_workloads if w.name in wanted]

    result = {
        "schema": 3,
        "benchmark": "HarmonicTransform end-to-end",
        "scope": {
            "quartic2d_error_reference": (
                "in-domain error is measured against the high-accuracy Hankel transform of the same "
                "sampled cubic radial input supplied to Quartic2D"
            ),
            "q_support_reference": (
                "q-space truncation support is evaluated from the known analytic transform of each "
                "benchmark profile; sampled-input high-q artifacts are excluded as external input effects"
            ),
            "petal2d_input_error_policy": (
                "external input representation is qualification metadata only and is excluded from Quartic2D accuracy"
            ),
            "timed_operation": (
                "HarmonicTransform(decomposition, q_max=..., n_q=..., method=..., subdivisions=..., check=False)"
            ),
            "excluded_from_timing": (
                "input construction and qualification; analytic/reference calculations; convergence search; diagnostics; plotting"
            ),
        },
        "theory": {
            "q_ceiling": "pi / dr",
            "parseval_input_qualification": (
                "weighted radial L2 input error equals full Hankel-space L2 input error"
            ),
            "accuracy_policy": (
                "q-support and in-domain transform errors are reported separately and must each satisfy "
                "their allocated error budgets; external input representation error is excluded"
            ),
        },
        "settings": {
            "requested_tolerances": tols,
            "methods": methods,
            "n_r_candidates": n_r_candidates,
            "n_q_candidates": nq_candidates,
            "subdivision_candidates": sub_candidates,
            "input_rtol_fraction": args.input_rtol_fraction,
            "q_ceiling_fraction": args.q_ceiling_fraction,
            "reference_q_order": args.reference_q_order,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "environment": environment_metadata(),
        "rows": [],
    }

    sqrt2 = math.sqrt(2.0)
    for workload in selected_workloads:
        print(f"\n[{workload.name}] {workload.category}", flush=True)
        for tol in tols:
            tail_target = tol / sqrt2
            in_domain_target = tol / sqrt2
            grid_target = in_domain_target / 2.0
            peak_grid_target = tol / 2.0
            input_rtol = args.input_rtol_fraction * tol

            row = {
                "workload": workload.name,
                "category": workload.category,
                "stress": workload.stress,
                "n_modes": len(workload.modes),
                "m_values": [int(m.m) for m in workload.modes],
                "target": tol,
                "error_budget": {
                    "q_tail_relative_l2": tail_target,
                    "in_domain_relative_l2_total_norm": in_domain_target,
                    "q_grid_relative_l2_total_norm": grid_target,
                    "q_grid_relative_max_peak": peak_grid_target,
                    "final_in_domain_relative_max_peak": tol,
                    "external_input_qualification_relative_l2": input_rtol,
                },
            }

            try:
                q_max, support_bundle, support_modes, support_selection_target = select_analytic_qmax(
                    workload, tail_target
                )
            except Exception as exc:
                row["status"] = "analytic_q_support_reference_failed"
                row["reference_error"] = repr(exc)
                result["rows"].append(row)
                print(f"  tol={tol:g}: analytic q-support reference FAILED: {exc}", flush=True)
                continue
            row["q_support"] = {
                "q_max": q_max,
                "allocated_tail_relative_l2": tail_target,
                "selection_tail_relative_l2": support_selection_target,
                "modes": [
                    {
                        "m": int(m),
                        "tail_relative_l2": float(v["tail_relative_l2"]),
                    }
                    for m, v in support_bundle["modes"].items()
                ],
                "per_mode_requirements": support_modes,
            }

            qualified, input_history = qualify_input(
                workload,
                n_r_candidates=n_r_candidates,
                input_rtol=input_rtol,
                q_required=q_max,
                q_ceiling_fraction=args.q_ceiling_fraction,
            )
            row["input_qualification_history"] = input_history
            if qualified is None:
                row["status"] = "input_precondition_not_met"
                result["rows"].append(row)
                print(
                    f"  tol={tol:g}: external input precondition not met by tested N_r candidates",
                    flush=True,
                )
                continue

            decomp = qualified["decomposition"]
            sampled = qualified["sampled"]
            q_bundle = sampled_q_reference_bundle(sampled, q_max, args.reference_q_order)
            n_r = int(decomp.r.size)
            row["n_r"] = n_r
            row["r_max"] = float(workload.r_max)
            row["dr"] = qualified["dr"]
            row["q_ceiling"] = qualified["q_ceiling"]
            row["q_max_required"] = q_max
            row["input_requirements"] = {
                "interpretation": (
                    "minimum tested uniform radial input satisfying the external representation threshold "
                    "and radial-Nyquist margin required by the analytic q-support benchmark"
                ),
                "n_r": n_r,
                "r_max": float(workload.r_max),
                "dr": qualified["dr"],
                "q_ceiling": qualified["q_ceiling"],
                "q_ceiling_fraction": args.q_ceiling_fraction,
                "qualification_relative_l2_threshold": input_rtol,
                "worst_input_representation_relative_l2": qualified["worst_input"],
                "modes": qualified["input_modes"],
            }
            n_q, ref_nodes, nq_history = select_nq_sampled(
                sampled,
                q_max,
                q_bundle,
                nq_candidates,
                l2_target=grid_target,
                peak_target=peak_grid_target,
            )
            row["n_q_selected"] = n_q
            row["n_q_history"] = nq_history
            if n_q is None:
                row["status"] = "q_grid_search_limit_reached"
                row["q_grid_search_limit"] = int(max(nq_candidates))
                result["rows"].append(row)
                print(
                    f"  tol={tol:g}: q-grid search reached Nq={max(nq_candidates)} before the allocated budget",
                    flush=True,
                )
                continue

            row["methods"] = {}
            completed = 0
            for method in methods:
                selected_sub, sub_history = select_subdivisions(
                    decomp,
                    sampled,
                    q_max,
                    n_q,
                    method,
                    sub_candidates,
                    ref_nodes,
                    q_bundle,
                    in_domain_target=in_domain_target,
                    peak_target=tol,
                )
                mr = {
                    "status": "quadrature_not_converged" if selected_sub is None else "candidate_selected",
                    "subdivisions_selected": selected_sub,
                    "subdivision_history": sub_history,
                }
                if selected_sub is not None:
                    ht, timing = time_constructor(
                        decomp,
                        q_max=q_max,
                        n_q=n_q,
                        method=method,
                        subdivisions=selected_sub,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )
                    final_modes = _final_method_metrics(ht, sampled, q_bundle, support_bundle)
                    worst_support = max(x["q_support_tail_relative_l2"] for x in final_modes)
                    worst_in = max(x["in_domain_relative_l2_total_norm"] for x in final_modes)
                    worst_peak = max(x["in_domain_relative_max_peak"] for x in final_modes)
                    passed = (
                        worst_support <= tail_target
                        and worst_in <= in_domain_target
                        and worst_peak <= tol
                    )
                    mr.update(
                        {
                            "status": "complete" if passed else "verification_failed",
                            "timing": timing,
                            "final_modes": final_modes,
                            "worst_q_support_tail_relative_l2": worst_support,
                            "worst_in_domain_relative_l2_total_norm": worst_in,
                            "worst_in_domain_relative_max_peak": worst_peak,
                        }
                    )
                    completed += int(passed)
                row["methods"][method] = mr

            row["status"] = "complete" if completed else "quartic2d_tolerance_not_reached"
            result["rows"].append(row)
            pieces = []
            for method in methods:
                mr = row["methods"][method]
                if mr.get("status") != "complete":
                    pieces.append(f"{method}: {mr.get('status')}")
                else:
                    pieces.append(
                        f"{method}: Nr={n_r}, nq={n_q}, sub={mr['subdivisions_selected']}, "
                        f"{1e3*mr['timing']['median_seconds']:.1f} ms, "
                        f"eps_in={mr['worst_in_domain_relative_l2_total_norm']:.2e}, "
                        f"eps_tail={mr['worst_q_support_tail_relative_l2']:.2e}"
                    )
            print(f"  tol={tol:g}: qmax={q_max:.4g}; " + "; ".join(pieces), flush=True)

    return result


def summarize(result):
    print("\n=== HarmonicTransform end-to-end summary ===")
    print("In-domain Quartic2D error is referenced to the same sampled radial input supplied to the constructor.")
    print("q-support is evaluated from the known analytic benchmark transform.")
    print("PETAL2D/input representation is an external qualification and is not included in Quartic2D error.\n")
    header = (
        f"{'workload':28s} {'tol':>8s} {'Nr':>5s} {'qmax':>8s} {'nq':>5s} "
        f"{'method':>8s} {'sub':>4s} {'ms':>9s} {'eps_in':>10s} {'eps_tail':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        if row.get("status") not in ("complete", "quartic2d_tolerance_not_reached"):
            print(
                f"{row['workload']:28s} {row['target']:8.0e} {'--':>5s} {'--':>8s} {'--':>5s} "
                f"{'--':>8s} {'--':>4s} {'--':>9s} {row.get('status','failed'):>10s} {'--':>10s}"
            )
            continue
        for method, mr in row.get("methods", {}).items():
            if mr.get("status") != "complete":
                print(
                    f"{row['workload']:28s} {row['target']:8.0e} {row['n_r']:5d} "
                    f"{row['q_max_required']:8.3g} {row['n_q_selected']:5d} {method:>8s} "
                    f"{'--':>4s} {'--':>9s} {mr.get('status','failed'):>10s} {'--':>10s}"
                )
            else:
                print(
                    f"{row['workload']:28s} {row['target']:8.0e} {row['n_r']:5d} "
                    f"{row['q_max_required']:8.3g} {row['n_q_selected']:5d} {method:>8s} "
                    f"{mr['subdivisions_selected']:4d} {1e3*mr['timing']['median_seconds']:9.2f} "
                    f"{mr['worst_in_domain_relative_l2_total_norm']:10.2e} "
                    f"{mr['worst_q_support_tail_relative_l2']:10.2e}"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="harmonic_transform_benchmark.json")
    ap.add_argument("--tolerances", default="1e-3,1e-4,1e-5")
    ap.add_argument(
        "--nr-candidates",
        default="160,256,384,512,768,1024,1536,2048,3072,4096",
        help="tested radial-input sizes; the first externally qualified size is used",
    )
    ap.add_argument("--methods", default="simpson,gl4")
    ap.add_argument("--nq-candidates", default="16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096")
    ap.add_argument("--subdivisions", default="1,2,4,8,16")
    ap.add_argument("--input-rtol-fraction", type=float, default=0.10)
    ap.add_argument("--q-ceiling-fraction", type=float, default=0.80)
    ap.add_argument("--reference-q-order", type=int, default=128)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--include-stress", action="store_true")
    ap.add_argument("--workloads", default="", help="comma-separated workload names; empty means all")
    args = ap.parse_args()

    if not (0.0 < args.input_rtol_fraction < 1.0):
        ap.error("--input-rtol-fraction must be between 0 and 1")
    if not (0.0 < args.q_ceiling_fraction < 1.0):
        ap.error("--q-ceiling-fraction must be between 0 and 1")

    result = run(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summarize(result)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
