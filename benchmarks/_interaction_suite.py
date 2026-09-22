"""Shared interaction-benchmark inputs, kernels, and reference helpers.

The interaction benchmarks reuse the *same* analytic workload definitions used
by ``run_harmonic_transform.py``.  Fixed momentum-space fields are reconstructed
from a completed HarmonicTransform benchmark JSON so the two benchmark layers
share the actual selected radial/q resolutions rather than merely similar test
functions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jv

from quartic2d import HarmonicTransform, Interaction
from quartic2d._numerics import common_grid

from benchmarks.publication.harmonic_transform import SyntheticDecomposition, workloads

FINITE_METHODS = ("trapezoid", "simpson", "gl4", "gl8")
ALL_METHODS = FINITE_METHODS + ("ogata", "fftlog")
DEFAULT_SUBDIVISIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
LARGE_DELTA_SUBDIVISIONS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
)
FFTLOG_N_VALUES = (128, 256, 512, 1024, 2048, 4096)
FFTLOG_N_VALUES_DEEP = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
FFTLOG_BIAS_VALUES_STANDARD = (-0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35)
FFTLOG_BIAS_VALUES_LARGE = (-0.20, -0.10, 0.00, 0.10, 0.20)
DELTA_DOMAINS = {
    "standard": (1.0e-2, 1.0e2),
    "large": (1.0e2, 1.0e4),
}
PRIMARY_WORKLOADS = (
    "gaussian_isotropic",
    "gaussian_odd_pair",
    "gaussian_anisotropic_m2",
    "gaussian_high_order_m4",
    "nodal_mixed",
    "complex_mixed",
    "exponential_cusp",
    "oscillatory_exponential",
    "algebraic_mixed",
)
REPRESENTATIVE_WORKLOADS = (
    "gaussian_isotropic",
    "gaussian_anisotropic_m2",
    "nodal_mixed",
    "complex_mixed",
)
HARD_WORKLOADS = (
    "exponential_cusp",
    "oscillatory_exponential",
    "algebraic_mixed",
)


@dataclass(frozen=True)
class Kernel:
    name: str
    family: str
    category: str
    parameters: dict
    U: Callable[[np.ndarray], np.ndarray]
    qU: Callable[[np.ndarray], np.ndarray]
    literature_note: str
    breakpoints: tuple[float, ...] = ()


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    category: str
    workload: str
    kernel: str


def _safe_divide_by_q(numerator, q, limit):
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q, dtype=float)
    positive = q > 0.0
    out[positive] = np.asarray(numerator(q[positive]), dtype=float) / q[positive]
    out[~positive] = float(limit)
    return out


def kernel_registry() -> dict[str, Kernel]:
    """Literature-motivated 2D interaction kernels plus one numerical control.

    Overall electrostatic prefactors are normalized to ``2*pi`` because the
    benchmark uses relative errors.  The physically relevant q dependence is
    retained.
    """

    def coulomb_U(q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(q > 0.0, 2.0 * np.pi / q, np.inf)

    def coulomb_qU(q):
        return 2.0 * np.pi * np.ones_like(np.asarray(q, dtype=float))

    def rk(name, r0):
        r0 = float(r0)

        def U(q):
            q = np.asarray(q, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(q > 0.0, 2.0 * np.pi / (q * (1.0 + r0 * q)), np.inf)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / (1.0 + r0 * q)

        return Kernel(
            name,
            "rytova_keldysh",
            "2D polarizability screening",
            {"r0": r0},
            U,
            qU,
            "Rytova-Keldysh form U(q) proportional to 1/[q(1+r0 q)].",
        )

    def dual_gate(name, d):
        d = float(d)

        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * _safe_divide_by_q(lambda x: np.tanh(d * x), q, d)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * np.tanh(d * q)

        return Kernel(
            name,
            "dual_gate",
            "dual-metal-gate screening",
            {"d": d},
            U,
            qU,
            "Dual-gate screened Coulomb form U(q) proportional to tanh(q d)/q.",
        )

    def thomas_fermi(name, q_tf):
        q_tf = float(q_tf)

        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / (q + q_tf)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * q / (q + q_tf)

        return Kernel(
            name,
            "thomas_fermi",
            "static carrier screening",
            {"q_tf": q_tf},
            U,
            qU,
            "2D Thomas-Fermi screened Coulomb form U(q) proportional to 1/(q+qTF).",
        )

    def yukawa(name, kappa):
        kappa = float(kappa)

        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / np.sqrt(q * q + kappa * kappa)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * q / np.sqrt(q * q + kappa * kappa)

        return Kernel(
            name,
            "yukawa",
            "Debye/Yukawa screened Coulomb interaction",
            {"kappa": kappa},
            U,
            qU,
            "Particles confined to 2D with a 3D Yukawa interaction exp(-kappa r)/r: U(q) proportional to 1/sqrt(q^2+kappa^2).",
        )

    def helmholtz_yukawa(name, kappa):
        kappa = float(kappa)

        def U(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi / (q * q + kappa * kappa)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * q / (q * q + kappa * kappa)

        return Kernel(
            name,
            "helmholtz_yukawa_2d",
            "strictly-2D Helmholtz/Yukawa Green function",
            {"kappa": kappa},
            U,
            qU,
            "Strictly-2D screened Green function K0(kappa r): U(q) proportional to 1/(q^2+kappa^2).",
        )

    def single_gate(name, d):
        d = float(d)

        def numerator(q):
            q = np.asarray(q, dtype=float)
            return -np.expm1(-2.0 * d * q)

        def U(q):
            return 2.0 * np.pi * _safe_divide_by_q(numerator, q, 2.0 * d)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * numerator(q)

        return Kernel(
            name,
            "single_gate",
            "single-metal-gate image-charge screening",
            {"d": d},
            U,
            qU,
            "Single-gate image-charge form U(q) proportional to [1-exp(-2 q d)]/q.",
        )

    def interlayer(name, d):
        d = float(d)

        def U(q):
            q = np.asarray(q, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(q > 0.0, 2.0 * np.pi * np.exp(-d * q) / q, np.inf)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * np.exp(-d * q)

        return Kernel(
            name,
            "interlayer_coulomb",
            "Coulomb interaction between parallel 2D layers",
            {"d": d},
            U,
            qU,
            "Interlayer Coulomb form U(q) proportional to exp(-q d)/q.",
        )

    def gated_interlayer(name, d_layer, d_gate):
        d_layer = float(d_layer)
        d_gate = float(d_gate)

        def numerator(q):
            q = np.asarray(q, dtype=float)
            return np.exp(-d_layer * q) * np.tanh(d_gate * q)

        def U(q):
            return 2.0 * np.pi * _safe_divide_by_q(numerator, q, d_gate)

        def qU(q):
            q = np.asarray(q, dtype=float)
            return 2.0 * np.pi * numerator(q)

        return Kernel(
            name,
            "gated_interlayer",
            "interlayer Coulomb interaction with symmetric gate screening",
            {"d_layer": d_layer, "d_gate": d_gate},
            U,
            qU,
            "Screened interlayer model U(q) proportional to exp(-q d_layer) tanh(q d_gate)/q.",
        )

    def static_2deg_rpa(name, k_f, q_tf):
        k_f = float(k_f)
        q_tf = float(q_tf)
        q_c = 2.0 * k_f

        def polarizability_factor(q):
            q = np.asarray(q, dtype=float)
            out = np.ones_like(q)
            above = q > q_c
            if np.any(above):
                ratio = q_c / q[above]
                out[above] = 1.0 - np.sqrt(np.maximum(0.0, 1.0 - ratio * ratio))
            return out

        def U(q):
            q = np.asarray(q, dtype=float)
            screening = q_tf * polarizability_factor(q)
            return 2.0 * np.pi / (q + screening)

        def qU(q):
            q = np.asarray(q, dtype=float)
            screening = q_tf * polarizability_factor(q)
            return 2.0 * np.pi * q / (q + screening)

        return Kernel(
            name,
            "static_2deg_rpa",
            "zero-temperature 2DEG RPA/Stern screening",
            {"k_f": k_f, "q_tf": q_tf, "q_2kf": q_c},
            U,
            qU,
            "Static T=0 2DEG RPA kernel with the Stern polarizability cusp at q=2 kF.",
            breakpoints=(q_c,),
        )

    def gaussian_U(q):
        q = np.asarray(q, dtype=float)
        return 2.0 * np.pi * np.exp(-0.125 * q * q)

    def gaussian_qU(q):
        q = np.asarray(q, dtype=float)
        return q * gaussian_U(q)

    return {
        "coulomb": Kernel(
            "coulomb",
            "coulomb",
            "bare 2D Coulomb",
            {},
            coulomb_U,
            coulomb_qU,
            "Bare 2D Fourier-space Coulomb form U(q) proportional to 1/q.",
        ),
        "rk_r0_0p1": rk("rk_r0_0p1", 0.1),
        "rk_r0_1": rk("rk_r0_1", 1.0),
        "rk_r0_10": rk("rk_r0_10", 10.0),
        "dual_gate_d1": dual_gate("dual_gate_d1", 1.0),
        "single_gate_d1": single_gate("single_gate_d1", 1.0),
        "tf_q1": thomas_fermi("tf_q1", 1.0),
        "yukawa_k0p1": yukawa("yukawa_k0p1", 0.1),
        "yukawa_k1": yukawa("yukawa_k1", 1.0),
        "yukawa_k10": yukawa("yukawa_k10", 10.0),
        "helmholtz_yukawa_k1": helmholtz_yukawa("helmholtz_yukawa_k1", 1.0),
        "interlayer_d1": interlayer("interlayer_d1", 1.0),
        "gated_interlayer_d1_D1": gated_interlayer("gated_interlayer_d1_D1", 1.0, 1.0),
        "rpa_2deg_kf1_qtf1": static_2deg_rpa("rpa_2deg_kf1_qtf1", 1.0, 1.0),
        "gaussian_control": Kernel(
            "gaussian_control",
            "numerical_control",
            "localized smooth numerical control",
            {"exponent": 0.125},
            gaussian_U,
            gaussian_qU,
            "Smooth non-singular control kernel; not presented as a material model.",
        ),
    }


def benchmark_cases(groups: Iterable[str] = ("continuity", "kernels", "hard", "control")) -> list[Case]:
    groups = tuple(groups)
    allowed = {"continuity", "kernels", "hard", "control"}
    unknown = sorted(set(groups) - allowed)
    if unknown:
        raise ValueError(f"unknown interaction benchmark groups: {unknown}")

    out: list[Case] = []
    if "continuity" in groups:
        for workload in PRIMARY_WORKLOADS:
            out.append(
                Case(
                    f"continuity__{workload}__rk_r0_1",
                    "continuity",
                    "same workload as HarmonicTransform benchmark / intermediate RK screening",
                    workload,
                    "rk_r0_1",
                )
            )

    if "kernels" in groups:
        kernel_names = (
            "coulomb",
            "rk_r0_0p1",
            "rk_r0_10",
            "tf_q1",
            "yukawa_k0p1",
            "yukawa_k1",
            "yukawa_k10",
            "helmholtz_yukawa_k1",
            "single_gate_d1",
            "dual_gate_d1",
            "interlayer_d1",
            "gated_interlayer_d1_D1",
            "rpa_2deg_kf1_qtf1",
        )
        for workload in REPRESENTATIVE_WORKLOADS:
            for kernel in kernel_names:
                out.append(
                    Case(
                        f"kernels__{workload}__{kernel}",
                        "kernels",
                        "representative workload / literature-motivated kernel sweep",
                        workload,
                        kernel,
                    )
                )

    if "hard" in groups:
        for workload in HARD_WORKLOADS:
            for kernel in ("coulomb", "dual_gate_d1", "interlayer_d1", "rpa_2deg_kf1_qtf1"):
                out.append(
                    Case(
                        f"hard__{workload}__{kernel}",
                        "hard",
                        "difficult radial input / long-range or gate-screened interaction",
                        workload,
                        kernel,
                    )
                )

    if "control" in groups:
        out.append(
            Case(
                "control__complex_mixed__gaussian",
                "control",
                "smooth localized numerical control",
                "complex_mixed",
                "gaussian_control",
            )
        )

    # Deduplicate by the actual mathematical pair while preserving the first
    # group assignment.  This avoids re-running a pair that appears in two
    # conceptual slices of the benchmark matrix.
    seen = set()
    unique = []
    for case in out:
        key = (case.workload, case.kernel)
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)
    return unique


def validation_cases() -> list[Case]:
    """Compact cases retained for automatic-convergence regression validation."""
    return [
        Case("isotropic_coulomb", "validation", "one harmonic / singular kernel", "gaussian_isotropic", "coulomb"),
        Case("anisotropic_rk_weak", "validation", "three harmonics / weak RK screening", "gaussian_anisotropic_m2", "rk_r0_0p1"),
        Case("anisotropic_rk_strong", "validation", "three harmonics / strong RK screening", "gaussian_anisotropic_m2", "rk_r0_10"),
        Case("nodal_tf", "validation", "nodal / Thomas-Fermi screened", "nodal_mixed", "tf_q1"),
        Case("complex_gate", "validation", "complex mixed parity / dual-gate screened", "complex_mixed", "dual_gate_d1"),
        Case("isotropic_yukawa", "validation", "one harmonic / Yukawa screened", "gaussian_isotropic", "yukawa_k1"),
        Case("anisotropic_interlayer", "validation", "three harmonics / interlayer Coulomb", "gaussian_anisotropic_m2", "interlayer_d1"),
        Case("nodal_rpa", "validation", "nodal / 2DEG RPA cusp", "nodal_mixed", "rpa_2deg_kf1_qtf1"),
    ]


def _find_harmonic_row(data: dict, workload: str, target: float) -> dict:
    matches = [
        row
        for row in data.get("rows", [])
        if row.get("workload") == workload and np.isclose(float(row.get("target", np.nan)), float(target), rtol=0.0, atol=1e-15)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one HarmonicTransform row for workload={workload!r}, target={target:g}; found {len(matches)}"
        )
    return matches[0]


def build_fields_from_harmonic_results(
    result_path: str | Path,
    *,
    target: float = 1.0e-4,
    method: str = "simpson",
    workload_names: Iterable[str] = PRIMARY_WORKLOADS,
):
    """Rebuild fixed q-space fields from a completed HarmonicTransform benchmark."""
    result_path = Path(result_path)
    data = json.loads(result_path.read_text())
    if int(data.get("schema", 0)) < 3:
        raise ValueError("interaction continuity benchmark requires HarmonicTransform benchmark schema >= 3")

    workload_map = {w.name: w for w in workloads()}
    fields = {}
    metadata = {}
    for name in workload_names:
        if name not in workload_map:
            raise KeyError(f"unknown shared workload {name!r}")
        row = _find_harmonic_row(data, name, target)
        if row.get("status") != "complete":
            raise ValueError(
                f"HarmonicTransform workload {name!r} is not complete at target={target:g}: {row.get('status')}"
            )
        method_row = row.get("methods", {}).get(method)
        if not method_row or method_row.get("status") != "complete":
            raise ValueError(
                f"HarmonicTransform method {method!r} is not complete for workload {name!r} at target={target:g}"
            )

        n_r = int(row["n_r"])
        q_max = float(row["q_support"]["q_max"])
        n_q = int(row["n_q_selected"])
        subdivisions = int(method_row["subdivisions_selected"])
        decomposition = SyntheticDecomposition(workload_map[name], n_r)
        field = HarmonicTransform(
            decomposition,
            q_max=q_max,
            n_q=n_q,
            method=method,
            interpolator="cubic",
            subdivisions=subdivisions,
            check=False,
        )
        fields[name] = field
        metadata[name] = {
            "source_benchmark": str(result_path),
            "source_schema": int(data.get("schema", 0)),
            "source_target": float(target),
            "workload": name,
            "category": row.get("category"),
            "n_r": n_r,
            "r_max": float(row["r_max"]),
            "dr": float(row["dr"]),
            "q_max": q_max,
            "n_q": n_q,
            "method": method,
            "subdivisions": subdivisions,
            "m_values": [int(m) for m in field.m_values],
            "upstream_input_representation_l2": float(row["input_requirements"]["worst_input_representation_relative_l2"]),
            "upstream_q_support_tail_l2": float(method_row["worst_q_support_tail_relative_l2"]),
            "upstream_in_domain_l2": float(method_row["worst_in_domain_relative_l2_total_norm"]),
            "upstream_in_domain_peak": float(method_row["worst_in_domain_relative_max_peak"]),
        }
    return fields, metadata


def displacement_grid(n: int = 32, domain: str = "standard"):
    """Return the deterministic displacement grid for a named benchmark regime."""
    if domain not in DELTA_DOMAINS:
        raise ValueError(f"unknown delta domain {domain!r}; expected one of {sorted(DELTA_DOMAINS)}")
    lower, upper = DELTA_DOMAINS[domain]
    mags = np.geomspace(float(lower), float(upper), int(n))
    angles = np.linspace(0.0, 1.75 * np.pi, int(n), endpoint=True)
    return np.column_stack((mags * np.cos(angles), mags * np.sin(angles)))


def _phase_resolved_legendre(grid, magnitude, order, max_phase_step):
    """Composite Gauss-Legendre nodes with explicit Bessel-phase resolution.

    The reference grid is refined so that ``magnitude * dq`` never exceeds
    ``max_phase_step`` on a subpanel.  This makes the direct reference suitable
    for large separations without relying on any production Interaction backend.
    """
    grid = np.asarray(grid, dtype=float)
    magnitude = float(magnitude)
    max_phase_step = float(max_phase_step)
    if not np.isfinite(max_phase_step) or max_phase_step <= 0.0:
        raise ValueError("max_phase_step must be finite and positive.")
    x, w = leggauss(int(order))
    left_parts = []
    right_parts = []
    for left, right in zip(grid[:-1], grid[1:]):
        count = max(1, int(np.ceil(magnitude * (right - left) / max_phase_step)))
        edges = np.linspace(left, right, count + 1)
        left_parts.append(edges[:-1])
        right_parts.append(edges[1:])
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    mid = 0.5 * (left + right)
    half = 0.5 * (right - left)
    q = (mid[:, None] + half[:, None] * x[None, :]).reshape(-1)
    weights = (half[:, None] * w[None, :]).reshape(-1)
    return q, weights


def phase_resolved_direct_reference(
    field1,
    field2,
    kernel: Kernel,
    deltas: np.ndarray,
    *,
    order: int = 12,
    max_phase_step: float = np.pi / 2.0,
):
    """Independent direct reference that remains resolved at large ``delta``.

    Unlike the ordinary fixed-q reference, each displacement is integrated on
    its own phase-resolved composite Gauss-Legendre grid.  Refinement is based
    only on the Bessel phase ``delta * q`` and therefore does not use Simpson,
    GL4, FFTLog, or Ogata convergence results.
    """
    deltas = np.asarray(deltas, dtype=float)
    mags = np.linalg.norm(deltas, axis=1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    modes1 = tuple(int(m) for m in field1.m_values)
    modes2 = tuple(int(m) for m in field2.m_values)
    grid = np.asarray(common_grid(field1.q, field2.q), dtype=float)
    interior_breakpoints = [
        float(x) for x in kernel.breakpoints if float(grid[0]) < float(x) < float(grid[-1])
    ]
    if interior_breakpoints:
        grid = np.unique(np.concatenate((grid, np.asarray(interior_breakpoints, dtype=float))))

    total = np.zeros(mags.size, dtype=np.complex128)
    for index, (magnitude, angle) in enumerate(zip(mags, angles)):
        q, weights = _phase_resolved_legendre(
            grid, magnitude, int(order), float(max_phase_step)
        )
        kernel_values = kernel.qU(q)
        values1 = {m: field1(m, q) for m in modes1}
        values2 = {m: field2(m, q) for m in modes2}
        value = 0.0j
        for m in modes1:
            for mp in modes2:
                base = weights * kernel_values * values1[m] * np.conj(values2[mp])
                radial = np.sum(jv(abs(m - mp), magnitude * q) * base)
                value += 2.0 * np.pi * np.exp(1j * (m - mp) * angle) * radial
        total[index] = value
    return total


def _composite_legendre(grid, order):
    grid = np.asarray(grid, dtype=float)
    x, w = leggauss(int(order))
    left = grid[:-1, None]
    right = grid[1:, None]
    mid = 0.5 * (left + right)
    half = 0.5 * (right - left)
    q = (mid + half * x[None, :]).reshape(-1)
    weights = (half * w[None, :]).reshape(-1)
    return q, weights


def direct_reference(field1, field2, kernel: Kernel, deltas: np.ndarray, order: int = 24):
    mags = np.linalg.norm(deltas, axis=1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    modes1 = tuple(int(m) for m in field1.m_values)
    modes2 = tuple(int(m) for m in field2.m_values)
    total = np.zeros(mags.size, dtype=np.complex128)
    grid = np.asarray(common_grid(field1.q, field2.q), dtype=float)
    interior_breakpoints = [
        float(x) for x in kernel.breakpoints if float(grid[0]) < float(x) < float(grid[-1])
    ]
    if interior_breakpoints:
        # The oracle may split at analytically known kernel nonanalyticities
        # (for example the 2DEG Stern cusp at q=2 kF).  Production methods do
        # not receive this split; it only prevents reference error from being
        # mistaken for backend error.
        grid = np.unique(np.concatenate((grid, np.asarray(interior_breakpoints, dtype=float))))
    q, weights = _composite_legendre(grid, int(order))
    for m in modes1:
        for mp in modes2:
            base = weights * kernel.qU(q) * field1(m, q) * np.conj(field2(mp, q))
            radial = jv(abs(m - mp), np.outer(mags, q)) @ base
            phase = np.exp(1j * (m - mp) * angles)
            total += 2.0 * np.pi * phase * radial
    return total


def stable_reference_pair(
    field1,
    field2,
    kernel: Kernel,
    deltas: np.ndarray,
    *,
    domain: str,
    reference_order: int = 24,
    reference_limit: float = 1.0e-5,
    large_coarse_order: int = 8,
    large_coarse_phase_step: float = np.pi,
    large_max_levels: int = 4,
):
    """Return an independently stabilized Interaction reference pair.

    Standard displacements use nested direct composite Gauss--Legendre orders.
    Large displacements refine both quadrature order and the explicit Bessel
    phase resolution until the reference itself satisfies ``reference_limit``.
    """
    if domain not in DELTA_DOMAINS:
        raise ValueError(f"unknown delta domain {domain!r}")

    if domain == "standard":
        low = direct_reference(field1, field2, kernel, deltas, order=int(reference_order))
        high = direct_reference(field1, field2, kernel, deltas, order=2 * int(reference_order))
        e2 = relative_l2(low, high)
        ep = relative_peak(low, high)
        metadata = {
            "kind": "fixed-grid composite Gauss-Legendre",
            "levels": [
                {"order": int(reference_order)},
                {"order": int(2 * reference_order)},
            ],
            "relative_l2_change": float(e2),
            "relative_peak_change": float(ep),
            "converged": bool(e2 <= reference_limit and ep <= reference_limit),
        }
        return low, high, metadata

    history = []
    previous = None
    previous_spec = None
    last_pair = None
    for level in range(int(large_max_levels)):
        order = int(large_coarse_order + 4 * level)
        phase_step = float(large_coarse_phase_step / (2.0 ** level))
        current = phase_resolved_direct_reference(
            field1,
            field2,
            kernel,
            deltas,
            order=order,
            max_phase_step=phase_step,
        )
        entry = {
            "level": int(level),
            "order": order,
            "max_phase_step": phase_step,
        }
        if previous is not None:
            last_pair = (previous, current)
            e2 = relative_l2(previous, current)
            ep = relative_peak(previous, current)
            entry.update(
                {
                    "relative_l2_change": float(e2),
                    "relative_peak_change": float(ep),
                    "passed": bool(e2 <= reference_limit and ep <= reference_limit),
                }
            )
            history.append(entry)
            if entry["passed"]:
                return previous, current, {
                    "kind": "adaptive phase-resolved composite Gauss-Legendre",
                    "levels": history,
                    "relative_l2_change": float(e2),
                    "relative_peak_change": float(ep),
                    "converged": True,
                    "coarse_spec": previous_spec,
                    "fine_spec": {"order": order, "max_phase_step": phase_step},
                }
        else:
            history.append(entry)
        previous = current
        previous_spec = {"order": order, "max_phase_step": phase_step}

    if len(history) < 2 or last_pair is None:
        raise RuntimeError("large-delta reference did not produce a refinement pair")
    low, high = last_pair
    last = history[-1]
    return low, high, {
        "kind": "adaptive phase-resolved composite Gauss-Legendre",
        "levels": history,
        "relative_l2_change": float(last.get("relative_l2_change", np.inf)),
        "relative_peak_change": float(last.get("relative_peak_change", np.inf)),
        "converged": False,
        "coarse_spec": previous_spec,
        "fine_spec": previous_spec,
    }


def relative_l2(value, reference):
    denom = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(value - reference) / denom)


def relative_peak(value, reference):
    denom = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    return float(np.max(np.abs(value - reference)) / denom)


def interaction_class(method: str):
    return Interaction


def interaction_from_parameters(deltas, field1, field2, kernel, method, parameters):
    return interaction_class(method)(
        deltas,
        field1,
        field2,
        kernel.U,
        method=method,
        interpolator="cubic",
        n=int(parameters.get("n", 512)),
        bias=float(parameters.get("bias", -0.5)),
        subdivisions=int(parameters.get("subdivisions", 1)),
        N=int(parameters.get("N", 2048)),
        h=parameters.get("h", None),
    )
