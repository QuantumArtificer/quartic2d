#!/usr/bin/env python3
"""Generate QUARTIC2D manuscript figures and numerical tables.

The main figures follow conventions used in peer-reviewed computational-physics and
numerical-method papers: direct numerical/reference comparisons with residual panels,
independent error versus requested numerical threshold, estimator/reference
comparisons, execution time versus achieved accuracy, and log-log scaling against
source-derived work measures. Detailed parameter maps, memory, cross-stage, and end-to-end audit
plots are reserved for the supplement.

Numerical benchmark results are read from the consolidated publication JSON in ``benchmarks/results``.
Representative line curves in Figures 1 and 2 are reconstructed from the same
analytic benchmark definitions and frozen numerical parameters used by the
publication benchmarks; no benchmark accuracy value is inferred from those line
curves. Numerical claims and tables are populated directly from benchmark JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import CubicSpline
from scipy.special import jv

from benchmarks._results import load_publication_results
from quartic2d._plot_style import (
    METHOD_LABELS,
    METHOD_ORDER,
    METHOD_STYLES,
    STATUS_LABELS,
    STATUS_STYLES,
)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

MAIN_WIDTH = 7.0
METHOD_LABEL = dict(METHOD_LABELS)
METHOD_MARKER = {method: METHOD_STYLES[method]["marker"] for method in METHOD_ORDER}
METHOD_LINESTYLE = {method: METHOD_STYLES[method]["linestyle"] for method in METHOD_ORDER}
CASE_LABEL = {
    "isotropic_coulomb": "isotropic Coulomb",
    "anisotropic_rk_strong": "anisotropic RK",
    "complex_gate": "complex dual-gate",
    "nodal_rpa": "nodal RPA",
}

CROSS_CASE_ID = {
    "isotropic_coulomb": "kernels__gaussian_isotropic__coulomb",
    "anisotropic_rk_strong": "kernels__gaussian_anisotropic_m2__rk_r0_10",
    "nodal_rpa": "kernels__nodal_mixed__rpa_2deg_kf1_qtf1",
    "complex_gate": "kernels__complex_mixed__dual_gate_d1",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def method_colors() -> dict[str, str]:
    """Return the package-wide semantic method colors."""
    return {method: METHOD_STYLES[method]["color"] for method in METHOD_ORDER}


COLORS = method_colors()


def finish_axis(ax, *, grid: bool = False) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if grid:
        ax.grid(True, which="major", alpha=0.18, linewidth=0.5)


def boxed_legend(ax, *args, **kwargs):
    """Draw an opaque framed legend so legend markers cannot read as data."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 1.0)
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("edgecolor", "black")
    leg = ax.legend(*args, **kwargs)
    leg.set_zorder(20)
    return leg


def panel_label(ax, label: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=9.5,
            ha="left", va="bottom")


def save_figure(fig, outdir: Path, stem: str, manifest: dict, sources: list[str], note: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    manifest[stem] = {"sources": sources, "note": note}


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def eps_ref(error: dict | None) -> float | None:
    if not error:
        return None
    vals = [error.get("relative_l2"), error.get("relative_peak")]
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return max(vals) if vals else None


def method_sort(values) -> list[str]:
    values = set(values)
    return [m for m in METHOD_ORDER if m in values] + sorted(values - set(METHOD_ORDER))


def selector_status(record: dict) -> str:
    """Map a convergence record to a presentation-level selector outcome."""
    if record.get("automatic_converged"):
        return (
            "accepted_reference_pass"
            if record.get("reference_pass")
            else "accepted_reference_fail"
        )
    classification = record.get("classification")
    if classification in {
        "selector_miss_backend_capable",
        "conservative_rejection_terminal_pass",
    }:
        return "qualified_after_refusal"
    if classification == "tested_box_not_capable":
        return "tested_box_not_capable"
    return "conservative_refusal"


def fmt_e(value) -> str:
    return "--" if value is None or not np.isfinite(float(value)) else f"{float(value):.3e}"


def fmt_ms(value) -> str:
    return "--" if value is None or not np.isfinite(float(value)) else f"{1e3*float(value):.3f}"


# -----------------------------------------------------------------------------
# Independent representative numerical curves for Figs. 1 and 2
# -----------------------------------------------------------------------------


def subdivide_grid(x: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    factor = int(factor)
    if factor == 1:
        return x.copy()
    offs = np.arange(factor, dtype=float) / factor
    panels = x[:-1, None] + np.diff(x)[:, None] * offs[None, :]
    return np.concatenate((panels.reshape(-1), x[-1:]))


def composite_gl_nodes_weights(x: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    nodes, weights = leggauss(int(order))
    a, b = x[:-1], x[1:]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    z = mid[:, None] + half[:, None] * nodes[None, :]
    w = half[:, None] * weights[None, :]
    return z.reshape(-1), w.reshape(-1)


def interp_complex(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y); z = np.asarray(z, float)
    re = CubicSpline(x, y.real, bc_type="not-a-knot", extrapolate=False)(z)
    if np.iscomplexobj(y):
        im = CubicSpline(x, y.imag, bc_type="not-a-knot", extrapolate=False)(z)
        out = re + 1j * im
    else:
        out = re
    return np.where(np.isfinite(out), out, 0.0)


def radial_hankel(r, f, q, order, *, method="simpson", subdivisions=1):
    r = np.asarray(r, float); f = np.asarray(f); q = np.asarray(q, float)
    grid = subdivide_grid(r, subdivisions)
    vals = interp_complex(r, f, grid)
    if method in ("simpson", "trapezoid"):
        base = grid * vals
        mat = jv(abs(int(order)), np.outer(q, grid)) * base[None, :]
        integ = simpson if method == "simpson" else trapezoid
        out = integ(mat, x=grid, axis=1)
    else:
        gl_order = 4 if method == "gl4" else 8
        z, w = composite_gl_nodes_weights(grid, gl_order)
        base = w * z * interp_complex(r, f, z)
        out = jv(abs(int(order)), np.outer(q, z)) @ base
    if int(order) < 0 and abs(int(order)) % 2:
        out = -out
    return out


def kernel_q(name: str, q):
    q = np.asarray(q, dtype=float)
    if name == "coulomb":
        with np.errstate(divide="ignore"):
            return np.where(q > 0, 2*np.pi/q, np.inf)
    if name == "rk_r0_10":
        with np.errstate(divide="ignore"):
            return np.where(q > 0, 2*np.pi/(q*(1+10*q)), np.inf)
    if name == "dual_gate_d1":
        out = np.empty_like(q)
        mask = q > 0
        out[mask] = 2*np.pi*np.tanh(q[mask])/q[mask]
        out[~mask] = 2*np.pi
        return out
    if name == "rpa_2deg_kf1_qtf1":
        p = np.ones_like(q)
        mask = q > 2.0
        ratio = 2.0 / q[mask]
        p[mask] = 1.0 - np.sqrt(np.maximum(0.0, 1.0-ratio*ratio))
        return 2*np.pi/(q+p)
    raise KeyError(name)


def q_times_kernel_zero(name: str) -> float:
    if name in ("coulomb", "rk_r0_10"):
        return 2*np.pi
    if name in ("dual_gate_d1", "rpa_2deg_kf1_qtf1"):
        return 0.0
    raise KeyError(name)


def interaction_from_sampled_field(q_grid, F, modes, deltas, kernel, *, method="gl4", subdivisions=1,
                                   reference=False):
    """Interaction along the +x direction from sampled F_m(q)."""
    q_grid = np.asarray(q_grid, float)
    deltas = np.asarray(deltas, float)
    if reference:
        # The canonical standard-domain reference uses high-order direct q
        # quadrature of the same fixed momentum-space interpolants.  For the
        # displayed standard-domain curves, 24-point Gauss-Legendre integration
        # on each native q interval matches that reference construction while
        # avoiding an unnecessary dense-panel visualization calculation.
        q, w = composite_gl_nodes_weights(q_grid, 24)
    elif method in ("gl4", "gl8"):
        panel = subdivide_grid(q_grid, subdivisions)
        q, w = composite_gl_nodes_weights(panel, 4 if method == "gl4" else 8)
    else:
        q = subdivide_grid(q_grid, subdivisions)
        w = None

    U = kernel_q(kernel, q)
    positive = q > 0
    result = np.zeros(deltas.size, dtype=np.complex128)
    for m in modes:
        fm = interp_complex(q_grid, F[m], q)
        for mp in modes:
            fp = interp_complex(q_grid, F[mp], q)
            n = int(m - mp)
            order = abs(n)
            bessel_parity = -1.0 if n > 0 and n % 2 else 1.0
            if w is not None:
                base = w * q * fm * np.conj(fp) * U
                pair = jv(order, np.outer(deltas, q)) @ base
            else:
                base = np.zeros(q.size, dtype=np.complex128)
                base[positive] = q[positive] * fm[positive] * np.conj(fp[positive]) * U[positive]
                if np.isclose(q[0], 0.0) and order == 0:
                    base[0] = fm[0] * np.conj(fp[0]) * q_times_kernel_zero(kernel)
                mat = jv(order, np.outer(deltas, q)) * base[None, :]
                pair = simpson(mat, x=q, axis=1) if method == "simpson" else trapezoid(mat, x=q, axis=1)
            # delta is chosen along x, hence exp[i(m-m')phi_delta] = 1.
            # The radial evaluation uses J_|m-m'|, so retain the parity
            # required by the exact signed order J_{m'-m}.
            result += 2*np.pi*bessel_parity*pair
    return np.real_if_close(result).real


# End-to-end analytic benchmark profiles.
def e2e_profiles(case: str):
    if case == "gaussian_coulomb":
        return {
            "modes": (0,), "rmax": 5.5,
            "radial": {0: lambda r: np.exp(-r*r)/np.pi},
            "exact": {0: lambda q: np.exp(-q*q/4)/(2*np.pi)},
        }
    if case == "anisotropic_rk_strong":
        a = 0.35
        return {
            "modes": (-2,0,2), "rmax": 5.5,
            "radial": {
                0: lambda r: np.exp(-r*r)/np.pi,
                -2: lambda r: a*r*r*np.exp(-r*r)/(2*np.pi),
                2: lambda r: a*r*r*np.exp(-r*r)/(2*np.pi),
            },
            "exact": {
                0: lambda q: np.exp(-q*q/4)/(2*np.pi),
                -2: lambda q: a*q*q*np.exp(-q*q/4)/(16*np.pi),
                2: lambda q: a*q*q*np.exp(-q*q/4)/(16*np.pi),
            },
        }
    if case == "nodal_rpa":
        return {
            "modes": (0,), "rmax": 5.5,
            "radial": {0: lambda r: (1-r*r)*np.exp(-r*r)/np.pi},
            "exact": {0: lambda q: q*q*np.exp(-q*q/4)/(8*np.pi)},
        }
    raise KeyError(case)


# Broad fixed-field benchmark profiles.
def broad_profiles(workload: str):
    if workload == "gaussian_isotropic":
        return {
            "modes": (0,), "rmax": 6.0,
            "radial": {0: lambda r: np.exp(-r*r)},
            "exact": {0: lambda q: 0.5*np.exp(-q*q/4)},
        }
    if workload == "gaussian_anisotropic_m2":
        return {
            "modes": (-2,0,2), "rmax": 6.0,
            "radial": {
                0: lambda r: np.exp(-r*r),
                -2: lambda r: 0.30*r*r*np.exp(-r*r),
                2: lambda r: 0.30*r*r*np.exp(-r*r),
            },
            "exact": {
                0: lambda q: 0.5*np.exp(-q*q/4),
                -2: lambda q: 0.30*q*q*np.exp(-q*q/4)/8,
                2: lambda q: 0.30*q*q*np.exp(-q*q/4)/8,
            },
        }
    if workload == "nodal_mixed":
        return {
            "modes": (-2,0,2), "rmax": 7.0,
            "radial": {
                0: lambda r: (1-r*r)*np.exp(-r*r),
                -2: lambda r: 0.20*r*r*np.exp(-r*r),
                2: lambda r: 0.20*r*r*np.exp(-r*r),
            },
            "exact": {
                0: lambda q: q*q*np.exp(-q*q/4)/8,
                -2: lambda q: 0.20*q*q*np.exp(-q*q/4)/8,
                2: lambda q: 0.20*q*q*np.exp(-q*q/4)/8,
            },
        }
    raise KeyError(workload)


def reconstruct_broad_field(harmonic_data: dict, workload: str, target=1e-4, method="simpson"):
    row = next(r for r in harmonic_data["rows"] if r["workload"] == workload and float(r["target"]) == float(target))
    prof = broad_profiles(workload)
    r = np.linspace(0, row["r_max"], int(row["n_r"]))
    q = np.linspace(0, row["q_max_required"], int(row["n_q_selected"]))
    sub = int(row["methods"][method]["subdivisions_selected"])
    F = {}
    for m in prof["modes"]:
        rr = r
        ff = prof["radial"][m](rr)
        F[m] = radial_hankel(rr, ff, q, m, method=method, subdivisions=sub)
    return row, prof, r, q, F


# -----------------------------------------------------------------------------
# Main Figure 1: method and representative calculation
# -----------------------------------------------------------------------------


def fig1_workflow(data, outdir, manifest):
    e2e = data["end_to_end_standard"]
    case = next(c for c in e2e["cases"] if c["name"] == "anisotropic_rk_strong")
    prof = e2e_profiles("anisotropic_rk_strong")
    params = case["harmonic_transform"]["convergence"]["parameters"]
    q = np.asarray(params["q_grid"], float)
    r = np.linspace(0, e2e["settings"]["rmax"], int(e2e["settings"]["nr"]))
    sub_h = int(params["subdivisions"])
    F = {m: radial_hankel(r, prof["radial"][m](r), q, m,
                          method=params["method"], subdivisions=sub_h)
         for m in prof["modes"]}
    deltas = np.logspace(-2, 2, 120)
    sub_i = int(case["interactions"][0]["selected_parameters"]["subdivisions"])
    V = interaction_from_sampled_field(q, F, prof["modes"], deltas, "rk_r0_10",
                                       method="gl4", subdivisions=sub_i)
    Vref = interaction_from_sampled_field(q, F, prof["modes"], deltas, "rk_r0_10",
                                          method="gl4", subdivisions=sub_i, reference=True)

    # No global workflow banner: panel order itself carries the method sequence.
    # This removes the unused top band and keeps panel labels clear of other text.
    fig = plt.figure(figsize=(MAIN_WIDTH, 4.45))
    gs = GridSpec(2, 2, figure=fig, wspace=0.31, hspace=0.34)
    # Keep panel (a) centered as a composite image+colorbar block instead of
    # letting colorbar layout push the image against one side of the grid cell.
    gsa = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0,0],
                                  width_ratios=(1.0, 0.045), wspace=0.08)
    axa = fig.add_subplot(gsa[0,0]); caxa = fig.add_subplot(gsa[0,1])
    axb = fig.add_subplot(gs[0,1])
    axc = fig.add_subplot(gs[1,0]); axd = fig.add_subplot(gs[1,1])

    xy = np.linspace(-3.0, 3.0, 251)
    X,Y = np.meshgrid(xy,xy,indexing="xy")
    rho = np.exp(-(X*X+Y*Y))*(1+0.35*(X*X-Y*Y))/np.pi
    im = axa.imshow(rho, extent=[xy[0],xy[-1],xy[0],xy[-1]], origin="lower",
                    aspect="equal", cmap="viridis")
    axa.set_xlabel(r"$x$"); axa.set_ylabel(r"$y$"); axa.set_title("transition field", pad=3)
    fig.colorbar(im, cax=caxa)
    panel_label(axa,"(a)",x=-0.14,y=1.04)

    axb.plot(r, prof["radial"][0](r), label=r"$m=0$")
    axb.plot(r, prof["radial"][2](r), label=r"$m=\pm2$")
    axb.set_xlabel(r"$r$"); axb.set_ylabel(r"$\rho_m(r)$"); axb.set_title("radial harmonics", pad=3)
    boxed_legend(axb, loc="upper right")
    finish_axis(axb); panel_label(axb,"(b)",x=-0.14,y=1.04)

    qdense = np.linspace(0, q[-1], 600)
    for m, lab in [(0,r"$m=0$"),(2,r"$m=\pm2$")]:
        exact = prof["exact"][m](qdense)
        num = interp_complex(q, F[m], qdense)
        line = axc.plot(qdense, exact, label=f"{lab} analytic")[0]
        axc.plot(qdense, num, linestyle="--", color=line.get_color(), label=f"{lab} numerical")
    axc.set_xlabel(r"$q$"); axc.set_ylabel(r"$F_m(q)$"); axc.set_title("Hankel transform", pad=3)
    # q >~ 4 is essentially empty for both displayed modes, so the lower-right
    # legend does not cover the numerical/reference curves.
    boxed_legend(axc, loc="upper right", fontsize=6.25)
    finish_axis(axc); panel_label(axc,"(c)",x=-0.14,y=1.04)

    scale = max(np.max(np.abs(Vref)), np.finfo(float).tiny)
    axd.plot(deltas, Vref/scale, label="independent reference")
    axd.plot(deltas, V/scale, "--", label="QUARTIC2D")
    axd.set_xscale("log")
    axd.set_xlabel(r"$\delta$"); axd.set_ylabel(r"$V(\delta)/\max|V_{\rm ref}|$"); axd.set_title("interaction", pad=3)
    boxed_legend(axd, loc="lower left")
    finish_axis(axd); panel_label(axd,"(d)",x=-0.14,y=1.04)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.945, bottom=0.105)
    save_figure(fig, outdir, "figure01_method_workflow", manifest,
                ["pipeline.json :: sections.end_to_end_standard_delta"],
                "Representative anisotropic RK calculation using the frozen end-to-end benchmark parameters.")


# -----------------------------------------------------------------------------
# Main Figure 2: direct numerical/reference accuracy with residual panels
# -----------------------------------------------------------------------------


def stacked_curve_panel(fig, subspec, x, ref, num, *, xlabel, ylabel, xscale="linear", title=None):
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=subspec, height_ratios=[3.1,1.0], hspace=0.06)
    ax = fig.add_subplot(sub[0]); er = fig.add_subplot(sub[1], sharex=ax)
    ax.plot(x, ref, label="reference")
    ax.plot(x, num, "--", label="numerical")
    peak = max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
    residual = np.abs(np.asarray(num)-np.asarray(ref))/peak
    residual = np.maximum(residual, 1e-16)
    er.plot(x, residual)
    ax.set_ylabel(ylabel); er.set_ylabel("error\n/ peak")
    er.set_xlabel(xlabel); er.set_yscale("log")
    if xscale == "log": ax.set_xscale("log"); er.set_xscale("log")
    ax.tick_params(labelbottom=False)
    boxed_legend(ax, loc="best")
    if title: ax.set_title(title, pad=2)
    finish_axis(ax); finish_axis(er, grid=True)
    return ax, er


def fig2_direct_accuracy(data, outdir, manifest):
    harmonic = data["harmonic_transform"]
    conv = data["interaction_convergence"]
    fig = plt.figure(figsize=(MAIN_WIDTH, 6.0))
    gs = GridSpec(2,2,figure=fig,wspace=0.34,hspace=0.36)

    # (a) Gaussian m=0 harmonic transform.
    _row, prof, _r, q, F = reconstruct_broad_field(harmonic,"gaussian_isotropic",1e-4,"simpson")
    qd = np.linspace(0,q[-1],700)
    ref = prof["exact"][0](qd); num=interp_complex(q,F[0],qd).real
    ax,_ = stacked_curve_panel(fig,gs[0,0],qd,ref,num,xlabel=r"$q$",ylabel=r"$F_0(q)$",title="Gaussian harmonic transform")
    panel_label(ax,"(a)")

    # (b) Anisotropic m=2 harmonic transform.
    _row2, prof2, _r2, q2, F2 = reconstruct_broad_field(harmonic,"gaussian_anisotropic_m2",1e-4,"simpson")
    qd2=np.linspace(0,q2[-1],700)
    ref2=prof2["exact"][2](qd2); num2=interp_complex(q2,F2[2],qd2).real
    ax,_=stacked_curve_panel(fig,gs[0,1],qd2,ref2,num2,xlabel=r"$q$",ylabel=r"$F_2(q)$",title="Anisotropic harmonic transform")
    panel_label(ax,"(b)")

    # (c) Smooth screened interaction: anisotropic RK.
    rconv = next(r for r in conv["rows"] if r["delta_domain"]=="standard" and r["case"]=="anisotropic_rk_strong" and r["method"]=="gl4")
    sub = int(rconv["selected_parameters"]["subdivisions"])
    deltas=np.logspace(-2,2,150)
    V=interaction_from_sampled_field(q2,F2,prof2["modes"],deltas,"rk_r0_10",method="gl4",subdivisions=sub)
    Vref=interaction_from_sampled_field(q2,F2,prof2["modes"],deltas,"rk_r0_10",method="gl4",subdivisions=sub,reference=True)
    scale=max(np.max(np.abs(Vref)),np.finfo(float).tiny)
    ax,_=stacked_curve_panel(fig,gs[1,0],deltas,Vref/scale,V/scale,xlabel=r"$\delta$",ylabel=r"$V/\max|V_{\rm ref}|$",xscale="log",title="Anisotropic RK interaction")
    panel_label(ax,"(c)")

    # (d) Nodal RPA cusp.
    _rown, profn, _rn, qn, Fn = reconstruct_broad_field(harmonic,"nodal_mixed",1e-4,"simpson")
    rconvn = next(r for r in conv["rows"] if r["delta_domain"]=="standard" and r["case"]=="nodal_rpa" and r["method"]=="gl4")
    subn=int(rconvn["selected_parameters"]["subdivisions"])
    Vn=interaction_from_sampled_field(qn,Fn,profn["modes"],deltas,"rpa_2deg_kf1_qtf1",method="gl4",subdivisions=subn)
    Vnref=interaction_from_sampled_field(qn,Fn,profn["modes"],deltas,"rpa_2deg_kf1_qtf1",method="gl4",subdivisions=subn,reference=True)
    scale=max(np.max(np.abs(Vnref)),np.finfo(float).tiny)
    ax,_=stacked_curve_panel(fig,gs[1,1],deltas,Vnref/scale,Vn/scale,xlabel=r"$\delta$",ylabel=r"$V/\max|V_{\rm ref}|$",xscale="log",title="Nodal RPA interaction")
    panel_label(ax,"(d)")

    save_figure(fig,outdir,"figure02_direct_accuracy",manifest,
                ["harmonic.json :: sections.reference_accuracy_and_tolerance_response","interaction_convergence.json :: sections.primary_tolerance"],
                "Direct numerical/reference curves with peak-normalized absolute residuals, following the comparison format used in numerical transform papers.")


# -----------------------------------------------------------------------------
# Main Figure 3: automatic error control
# -----------------------------------------------------------------------------


def estimate_selected_l2(row: dict) -> float | None:
    if row.get("method") not in ("simpson","gl4") or not row.get("automatic_converged"):
        return None
    selected = (row.get("selected_parameters") or {}).get("subdivisions")
    if selected is None: return None
    search=(row.get("convergence") or {}).get("search") or {}
    candidates=[]
    candidates.extend(((search.get("metadata") or {}).get("last_verification") or {}).get("candidate_checks") or [])
    for step in search.get("steps") or []:
        candidates.extend((step.get("metadata") or {}).get("candidate_checks") or [])
    vals=[]
    for c in candidates:
        if (c.get("parameters") or {}).get("subdivisions")==selected and c.get("estimated_relative_l2_error") is not None:
            vals.append(float(c["estimated_relative_l2_error"]))
    return vals[-1] if vals else None


def fig3_error_control(data, outdir, manifest):
    broad = data["interaction_accuracy"]
    conv = data["interaction_convergence"]
    fig, axs = plt.subplots(2, 2, figsize=(MAIN_WIDTH, 5.15),
                            gridspec_kw={"wspace": 0.34, "hspace": 0.42})

    # (a) Requested selector threshold versus independent reference error.
    ax = axs[0, 0]
    methods = method_sort(r["method"] for r in broad["rows"])
    offsets = {m: 10 ** ((i - (len(methods) - 1) / 2) * 0.035)
               for i, m in enumerate(methods)}
    ymax = 0.0
    for m in methods:
        rr = [r for r in broad["rows"] if r["method"] == m and r.get("reference_pass")]
        x, y = [], []
        for r in rr:
            e = eps_ref(r.get("selected_reference_error"))
            if e and e > 0:
                x.append(float(r["requested_tolerance"]) * offsets[m])
                y.append(e)
                ymax = max(ymax, e)
        if x:
            ax.scatter(x, y, s=10, alpha=0.34, marker=METHOD_MARKER[m],
                       label=METHOD_LABEL[m], color=COLORS[m], linewidths=0.3)
            for tol in sorted({float(r["requested_tolerance"]) for r in rr}):
                vals = np.array([
                    eps_ref(r.get("selected_reference_error"))
                    for r in rr
                    if float(r["requested_tolerance"]) == tol
                    and eps_ref(r.get("selected_reference_error"))
                ], float)
                if vals.size:
                    med = float(np.median(vals))
                    ymax = max(ymax, med)
                    ax.scatter([tol * offsets[m]], [med], s=31, marker=METHOD_MARKER[m],
                               color=COLORS[m], edgecolors="black", linewidths=0.45, zorder=4)
    guide = np.logspace(-5.2, -2.7, 100)
    ax.plot(guide, guide, color="0.35", linewidth=0.85, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"requested $\mathrm{rtol}$")
    ax.set_ylabel(r"independent $\epsilon_{\rm ref}$")
    ax.set_ylim(top=max(2.0e-2, ymax * 18.0))
    boxed_legend(ax, loc="upper left", fontsize=5.7, ncol=2,
                 borderpad=0.3, columnspacing=0.8, handlelength=1.4)
    finish_axis(ax, grid=True)
    panel_label(ax, "(a)")

    # (b) Finite-rule estimate versus independent error.
    ax = axs[0, 1]
    allv = []
    for m in ("simpson", "gl4"):
        xxv, yyv = [], []
        for r in conv["rows"]:
            if r["method"] != m or not r.get("reference_pass"):
                continue
            est = estimate_selected_l2(r)
            actual = (r.get("selected_reference_error") or {}).get("relative_l2")
            if est is not None and actual is not None and est > 0 and float(actual) > 0:
                xxv.append(est)
                yyv.append(float(actual))
                allv.extend([est, float(actual)])
        ax.scatter(xxv, yyv, marker=METHOD_MARKER[m], color=COLORS[m],
                   label=METHOD_LABEL[m], s=24)
    lo = min(allv) * 0.7
    hi = max(allv) * 1.4
    ax.plot([lo, hi], [lo, hi], color="0.35", linewidth=0.85, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"estimated relative $L^2$ change")
    ax.set_ylabel(r"independent relative $L^2$ error")
    boxed_legend(ax, loc="upper left", fontsize=6.2)
    finish_axis(ax, grid=True)
    panel_label(ax, "(b)")

    # (c,d) Case-level selector outcomes, separated by displacement domain.
    cases = ("isotropic_coulomb", "anisotropic_rk_strong", "complex_gate", "nodal_rpa")
    methods = ("simpson", "gl4", "fftlog", "ogata")
    for ax, domain, label in zip(axs[1], ("standard", "large"), ("(c)", "(d)")):
        for i, case in enumerate(cases):
            for j, method in enumerate(methods):
                record = next(r for r in conv["rows"]
                              if r["delta_domain"] == domain
                              and r["case"] == case and r["method"] == method)
                status = selector_status(record)
                style = STATUS_STYLES[status]
                kwargs = {"marker": style["marker"], "s": 37, "linewidths": 1.0, "zorder": 3}
                if style.get("fillstyle") == "none" and style["marker"] != "x":
                    kwargs.update(facecolors="none", edgecolors=style["color"])
                else:
                    kwargs.update(color=style["color"])
                ax.scatter(j, i, **kwargs)
        ax.set_xticks(range(len(methods)), [METHOD_LABEL[m] for m in methods], rotation=25, ha="right")
        ax.set_xlim(-0.5, len(methods) - 0.5)
        ax.set_ylim(len(cases) - 0.5, -0.5)
        ax.set_title("standard displacement" if domain == "standard" else "large displacement", pad=3)
        ax.grid(True, color="0.90", linewidth=0.55)
        ax.set_axisbelow(True)
        panel_label(ax, label)
    axs[1, 0].set_yticks(range(len(cases)), [CASE_LABEL[c] for c in cases])
    axs[1, 1].set_yticks(range(len(cases)), [])

    status_handles = []
    for status in ("accepted_reference_pass", "qualified_after_refusal",
                   "conservative_refusal", "tested_box_not_capable"):
        style = STATUS_STYLES[status]
        kwargs = {"marker": style["marker"], "linestyle": "none",
                  "label": STATUS_LABELS[status], "color": style["color"]}
        if style.get("fillstyle") == "none" and style["marker"] != "x":
            kwargs.update(markerfacecolor="none", markeredgecolor=style["color"])
        status_handles.append(Line2D([0], [0], **kwargs))
    fig.legend(status_handles, [h.get_label() for h in status_handles],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005),
               frameon=False, fontsize=5.8, columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.96)
    save_figure(
        fig, outdir, "figure03_automatic_error_control", manifest,
        ["interaction_accuracy.json :: sections.broad_accuracy_and_timing",
         "interaction_convergence.json :: sections.primary_tolerance"],
        "Requested self-convergence threshold versus independent error, finite-rule estimator validation, and case-level selector outcomes in standard and large-displacement domains.",
    )

def standard_accuracy_cost_points(accuracy, matrix, case_name):
    points=[]
    for r in accuracy["rows"]:
        if r["case"]!=case_name or not r.get("reference_pass") or not r.get("production_timing"): continue
        e=eps_ref(r.get("selected_reference_error"));t=float(r["production_timing"]["median_seconds"])
        if e and e>0:
            points.append({"method":r["method"],"tol":float(r["requested_tolerance"]),"error":e,"time":t,"kind":"broad"})
    # Add public finite one-off 1e-4 methods not in the three-tolerance sweep.
    existing={(p["method"],p["tol"]) for p in points}
    for r in matrix["rows"]:
        if r["case"]!=case_name or not r.get("reference_pass") or not r.get("production_timing"): continue
        key=(r["method"],float(r["requested_tolerance"]))
        if key in existing: continue
        e=eps_ref(r.get("selected_reference_error"));t=float(r["production_timing"]["median_seconds"])
        if e and e>0:
            points.append({"method":r["method"],"tol":float(r["requested_tolerance"]),"error":e,"time":t,"kind":"matrix"})
    return points


def accuracy_cost_panel(ax, points, title, *, label_points=False):
    for m in method_sort(p["method"] for p in points):
        rr=sorted([p for p in points if p["method"]==m],key=lambda p:p["error"],reverse=True)
        x=[p["error"] for p in rr];y=[1e3*p["time"] for p in rr]
        if len(rr)>1:
            ax.plot(x,y,linestyle=METHOD_LINESTYLE[m],marker=METHOD_MARKER[m],color=COLORS[m],label=METHOD_LABEL[m])
        else:
            ax.scatter(x,y,marker=METHOD_MARKER[m],color=COLORS[m],label=METHOD_LABEL[m])
        if label_points:
            for p,xx,yy in zip(rr,x,y):
                ax.annotate(METHOD_LABEL[m],(xx,yy),xytext=(3,3),textcoords="offset points",fontsize=6.2)
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel(r"achieved $\epsilon_{\rm ref}$");ax.set_ylabel(r"production time (ms)")
    ax.set_title(title,pad=3);finish_axis(ax,grid=True)


def fig4_accuracy_cost(data,outdir,manifest):
    acc=data["interaction_accuracy"];mat=data["interaction_method_matrix_1e-4"];fixed=data["fixed_performance_large_1e-3"]
    fig,axs=plt.subplots(2,2,figsize=(MAIN_WIDTH,5.25),gridspec_kw={"wspace":0.34,"hspace":0.42})
    cases=[
        ("kernels__gaussian_isotropic__coulomb","standard: isotropic Coulomb"),
        ("kernels__nodal_mixed__rpa_2deg_kf1_qtf1","standard: nodal RPA"),
    ]
    for i,(case,title) in enumerate(cases):
        pts=standard_accuracy_cost_points(acc,mat,case)
        accuracy_cost_panel(axs[0,i],pts,title)
        panel_label(axs[0,i],f"({chr(97+i)})")
    for j,(case,title) in enumerate([("isotropic_coulomb",r"large $\delta$: isotropic Coulomb"),("complex_gate",r"large $\delta$: complex dual-gate")]):
        pts=[]
        for r in fixed["rows"]:
            if r["case"]!=case: continue
            e=eps_ref(r["qualification_reference_error"]);t=float(r["production_timing"]["median_seconds"])
            pts.append({"method":r["method"],"tol":1e-3,"error":e,"time":t,"kind":r["qualification_kind"]})
        accuracy_cost_panel(axs[1,j],pts,title,label_points=False)
        panel_label(axs[1,j],f"({chr(99+j)})")
        axs[1,j].axvline(1e-3,color="0.45",linewidth=0.8,linestyle=":")
    # Collect all methods from all panels in canonical order.
    hd={}
    for ax in axs.flat:
        for h,l in zip(*ax.get_legend_handles_labels()): hd[l]=h
    ordered=[METHOD_LABEL[m] for m in METHOD_ORDER if METHOD_LABEL[m] in hd]
    fig.legend([hd[l] for l in ordered],ordered,loc="lower center",ncol=len(ordered),bbox_to_anchor=(0.5,0.0))
    fig.subplots_adjust(bottom=0.12)
    save_figure(fig,outdir,"figure04_accuracy_vs_cost",manifest,
                ["interaction_accuracy.json :: sections.broad_accuracy_and_timing","interaction_accuracy.json :: sections.primary_tolerance_method_matrix","performance.json :: sections.qualified_fixed_configuration_large_delta_practical_tolerance"],
                "Execution time versus achieved independent-reference accuracy, using per-workload comparisons rather than aggregate method rankings.")


# -----------------------------------------------------------------------------
# Main Figure 5: scaling and reusable calibration
# -----------------------------------------------------------------------------


def fig5_scaling_reuse(data,outdir,manifest):
    hs=data["harmonic_scaling"];ins=data["interaction_scaling"];perf=data["autoconvergence_performance_large_1e-3"];fixed=data["fixed_performance_large_1e-3"]
    fig,axs=plt.subplots(2,2,figsize=(MAIN_WIDTH,5.2),gridspec_kw={"wspace":0.35,"hspace":0.40})

    # (a) HarmonicTransform work collapse.
    ax=axs[0,0]
    for axis,marker in [("N_q","o"),("N_r","s"),("N_m","^"),("s_r","D")]:
        rr=[r for r in hs["rows"] if r["axis"]==axis]
        ax.scatter([r["work_units"] for r in rr],[r["median_seconds"] for r in rr],marker=marker,label=axis,s=20)
    x=np.logspace(math.log10(min(r["work_units"] for r in hs["rows"])),math.log10(max(r["work_units"] for r in hs["rows"])),100)
    c=np.median([r["median_seconds"]/r["work_units"] for r in hs["rows"]])
    ax.plot(x,c*x,color="0.2",linestyle="--",label=r"$\propto N_mN_qN_s$")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(r"$N_mN_qN_s$");ax.set_ylabel("runtime (s)")
    boxed_legend(ax, loc="lower right", ncol=2, fontsize=6.4, columnspacing=0.9, handlelength=1.7)
    finish_axis(ax,grid=True);panel_label(ax,"(a)")

    # (b) finite Interaction work collapse.
    ax=axs[0,1]
    rr=[r for r in ins["rows"] if r["branch"]=="finite"]
    byaxis=defaultdict(list)
    for r in rr:
        nqs=int(r.get("N_qs", int(r["s_q"])*(int(r["N_q"])-1)+1))
        work=int(r["N_p"])*int(r["N_D"])*nqs
        byaxis[r["axis"]].append((work,float(r["median_seconds"])))
    for axis,marker in [("N_q","o"),("N_D","s"),("N_p","^"),("s_q","D")]:
        pts=byaxis.get(axis,[])
        ax.scatter([p[0] for p in pts],[p[1] for p in pts],marker=marker,label=axis,s=20)
    works=[p[0] for pts in byaxis.values() for p in pts];times=[p[1] for pts in byaxis.values() for p in pts]
    x=np.logspace(math.log10(min(works)),math.log10(max(works)),100);c=np.median(np.array(times)/np.array(works))
    ax.plot(x,c*x,color="0.2",linestyle="--",label=r"$\propto N_pN_DN_{q,s}$")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(r"$N_pN_DN_{q,s}$");ax.set_ylabel("runtime (s)")
    boxed_legend(ax, loc="lower right", ncol=2, fontsize=6.4, columnspacing=0.9, handlelength=1.7)
    finish_axis(ax,grid=True);panel_label(ax,"(b)")

    # (c) FFTLog theoretical work variable; measured finite-range behavior left as data.
    ax=axs[1,0]
    rr=[r for r in ins["rows"] if r["branch"]=="fftlog"]
    byaxis=defaultdict(list)
    for r in rr:
        work=int(r["N_p"])*(float(r["N_F"])*math.log2(float(r["N_F"]))+float(r["N_D"]))
        byaxis[r["axis"]].append((work,float(r["median_seconds"])))
    for axis,marker in [("N_F","o"),("N_D","s"),("N_p","^")]:
        pts=byaxis.get(axis,[]); ax.scatter([p[0] for p in pts],[p[1] for p in pts],marker=marker,label=axis,s=20)
    works=[p[0] for pts in byaxis.values() for p in pts];times=[p[1] for pts in byaxis.values() for p in pts]
    x=np.logspace(math.log10(min(works)),math.log10(max(works)),100)
    # The dashed line is a proportional guide, deliberately not a fitted asymptote.
    c=np.median(np.array(times)/np.array(works));ax.plot(x,c*x,color="0.2",linestyle="--",label="linear work guide")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(r"$N_p[N_F\log_2N_F+N_D]$");ax.set_ylabel("runtime (s)")
    boxed_legend(ax, loc="upper right", ncol=2, fontsize=6.4, columnspacing=0.9, handlelength=1.7)
    finish_axis(ax,grid=True);panel_label(ax,"(c)")

    # (d) amortized repeated-use costs for two concrete workload families.
    ax=axs[1,1]
    M=np.unique(np.rint(np.logspace(0,3,200)).astype(int))
    scenarios=[("isotropic_coulomb",("fftlog","gl4")),("complex_gate",("ogata","gl4"))]
    for case,methods in scenarios:
        for m in methods:
            pr=next((r for r in perf["rows"] if r.get("case")==case and r["method"]==m and r.get("status")=="complete"),None)
            fr=next((r for r in fixed["rows"] if r["case"]==case and r["method"]==m and r["qualification_kind"]=="automatic_certificate"),None)
            if pr is None or fr is None: continue
            tcal=float(pr["calibration_timing"]["median_seconds"]);tprod=float(fr["production_timing"]["median_seconds"])
            y=1e3*(tprod+tcal/M)
            short_case = "Coulomb" if case == "isotropic_coulomb" else "dual-gate"
            label=f"{short_case} - {METHOD_LABEL[m]}"
            ax.plot(M,y,label=label,color=COLORS[m],linestyle="-" if case=="isotropic_coulomb" else "--")
    # Mark the measured GL4/Ogata crossover for the complex workload when it lies in range.
    try:
        p_og=next(r for r in perf["rows"] if r.get("case")=="complex_gate" and r["method"]=="ogata" and r.get("status")=="complete")
        p_gl=next(r for r in perf["rows"] if r.get("case")=="complex_gate" and r["method"]=="gl4" and r.get("status")=="complete")
        f_og=next(r for r in fixed["rows"] if r["case"]=="complex_gate" and r["method"]=="ogata" and r["qualification_kind"]=="automatic_certificate")
        f_gl=next(r for r in fixed["rows"] if r["case"]=="complex_gate" and r["method"]=="gl4" and r["qualification_kind"]=="automatic_certificate")
        c_og=float(p_og["calibration_timing"]["median_seconds"]); c_gl=float(p_gl["calibration_timing"]["median_seconds"])
        t_og=float(f_og["production_timing"]["median_seconds"]); t_gl=float(f_gl["production_timing"]["median_seconds"])
        if t_gl > t_og and c_og > c_gl:
            mcross=(c_og-c_gl)/(t_gl-t_og)
            if M.min() <= mcross <= M.max():
                ax.axvline(mcross,color="0.35",linestyle=":",linewidth=0.8)
                ax.text(mcross*1.08,0.06,fr"$M\simeq{mcross:.0f}$",transform=ax.get_xaxis_transform(),fontsize=6.5,rotation=90,va="bottom")
    except StopIteration:
        pass
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(r"repeated evaluations $M$");ax.set_ylabel(r"$t_{\rm eff}$ (ms/evaluation)")
    # Reserve a modest upper band and use a compact one-column legend wholly
    # inside the axes; long two-column labels previously exceeded the panel.
    ax.set_ylim(top=ax.get_ylim()[1]*3.0)
    boxed_legend(ax, loc="upper right", fontsize=5.55, ncol=1,
                 borderpad=0.35, handlelength=1.6)
    finish_axis(ax,grid=True);panel_label(ax,"(d)")

    save_figure(fig,outdir,"figure05_scaling_and_reuse",manifest,
                ["scaling.json :: sections.harmonic_transform","scaling.json :: sections.interaction","performance.json :: sections.automatic_convergence_cost_large_delta_practical_tolerance","performance.json :: sections.qualified_fixed_configuration_large_delta_practical_tolerance"],
                "Source-derived work scaling and measured calibration amortization. The FFTLog dashed line is a work guide, not a claimed fitted asymptote.")


# -----------------------------------------------------------------------------
# Supplementary figures
# -----------------------------------------------------------------------------


def figS1_harmonic_validation(data, outdir, manifest):
    d = data["harmonic_transform"]
    fig, axs = plt.subplots(1, 3, figsize=(MAIN_WIDTH, 3.25), sharey=True,
                            gridspec_kw={"wspace": 0.12})
    tolerances = sorted(d["summary"]["requested_tolerances"], reverse=True)
    workload_order = []
    for row in d["rows"]:
        if row["stress"]:
            continue
        if row["workload"] not in workload_order:
            workload_order.append(row["workload"] )
    labels = [name.replace("gaussian_", "").replace("_", " ") for name in workload_order]
    y = np.arange(len(workload_order), dtype=float)
    offsets = {"simpson": -0.10, "gl4": 0.10}
    for ax, tol, label in zip(axs, tolerances, ("(a)", "(b)", "(c)")):
        rows = [r for r in d["rows"] if float(r["target"]) == float(tol) and not r["stress"]]
        for method in ("simpson", "gl4"):
            xvals, yvals = [], []
            for i, workload in enumerate(workload_order):
                row = next(r for r in rows if r["workload"] == workload)
                result = (row.get("methods") or {}).get(method, {})
                if result.get("status") != "complete":
                    continue
                err = max(float(result["worst_in_domain_relative_l2_total_norm"]),
                          float(result["worst_in_domain_relative_max_peak"]))
                xvals.append(err)
                yvals.append(i + offsets[method])
            ax.scatter(xvals, yvals, marker=METHOD_MARKER[method], color=COLORS[method],
                       s=25, label=METHOD_LABEL[method], zorder=3)
        ax.axvline(float(tol), color="0.35", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"independent $\epsilon_{\rm ref}$")
        ax.set_title(fr"requested $\mathrm{{rtol}}={tol:.0e}$", pad=3)
        finish_axis(ax, grid=True)
        panel_label(ax, label)
    axs[0].set_yticks(y, labels)
    axs[0].invert_yaxis()
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    handles = [Line2D([0], [0], color=COLORS[m], marker=METHOD_MARKER[m],
                      linestyle="none", label=METHOD_LABEL[m]) for m in ("simpson", "gl4")]
    handles.append(Line2D([0], [0], color="0.35", linestyle=":", label="requested level"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=6.2)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.20, right=0.99)
    save_figure(
        fig, outdir, "figureS01_harmonic_validation", manifest,
        ["harmonic.json :: sections.reference_accuracy_and_tolerance_response"],
        "Per-workload HarmonicTransform independent-reference accuracy at three requested self-convergence levels.",
    )

def figS2_interaction_distributions(data, outdir, manifest):
    d = data["interaction_accuracy"]
    fig, axs = plt.subplots(1, 3, figsize=(MAIN_WIDTH, 2.75), sharey=True,
                            gridspec_kw={"wspace": 0.18})
    tols = sorted({float(r["requested_tolerance"]) for r in d["rows"]}, reverse=True)
    methods = method_sort(r["method"] for r in d["rows"])

    for ax, tol, label in zip(axs, tols, ("(a)", "(b)", "(c)")):
        tol_rows = [r for r in d["rows"] if float(r["requested_tolerance"]) == tol]
        case_order = []
        for r in tol_rows:
            if r["case"] not in case_order:
                case_order.append(r["case"])
        total_cases = len(case_order)
        for method in methods:
            errors = []
            for r in tol_rows:
                if r["method"] != method or not r.get("reference_pass"):
                    continue
                value = eps_ref(r.get("selected_reference_error"))
                if value is not None and np.isfinite(value) and value > 0:
                    errors.append(float(value))
            errors = np.sort(np.asarray(errors, dtype=float))
            if errors.size:
                cumulative = np.arange(1, errors.size + 1, dtype=float) / total_cases
                ax.step(errors, cumulative, where="post", color=COLORS[method],
                        linestyle=METHOD_LINESTYLE[method], label=METHOD_LABEL[method])
        ax.axvline(tol, color="0.35", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel(r"independent $\epsilon_{\rm ref}$")
        ax.set_title(fr"requested $\mathrm{{rtol}}={tol:.0e}$", pad=3)
        finish_axis(ax, grid=True)
        panel_label(ax, label)
    axs[0].set_ylabel("fraction of 74 cases reference-qualified")
    handles = [Line2D([0], [0], color=COLORS[m], linestyle=METHOD_LINESTYLE[m], label=METHOD_LABEL[m])
               for m in methods]
    handles.append(Line2D([0], [0], color="0.35", linestyle=":", label="requested level"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, fontsize=5.9, frameon=False, columnspacing=0.9, handlelength=1.6)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.08, right=0.99, wspace=0.20)
    save_figure(
        fig, outdir, "figureS02_interaction_accuracy_distributions", manifest,
        ["interaction_accuracy.json :: sections.broad_accuracy_and_timing"],
        "Cumulative independent-reference accuracy and coverage over the 74-case fixed-field suite. Curves plateau below one when a method has fewer reference-qualified cases in the declared search.",
    )

def fftlog_heatmap(ax,row,title):
    hist=(row.get("capability") or {}).get("history") or []
    ns=sorted({int(h["n"]) for h in hist});bs=sorted({float(h["bias"]) for h in hist})
    z=np.full((len(ns),len(bs)),np.nan)
    for h in hist:
        i=ns.index(int(h["n"]));j=bs.index(float(h["bias"]));z[i,j]=math.log10(max(float(h["relative_l2"]),float(h["relative_peak"])))
    im=ax.imshow(z,origin="lower",aspect="auto",interpolation="nearest")
    ax.set_xticks(np.arange(len(bs)),[f"{b:g}" for b in bs]);ax.set_yticks(np.arange(len(ns)),[str(n) for n in ns])
    ax.set_xlabel(r"$q_{\rm bias}$");ax.set_ylabel(r"$N$");ax.set_title(title,pad=3)
    return im


def figS3_numerical_method_parameters(data,outdir,manifest):
    d=data["interaction_convergence"]
    fig,axs=plt.subplots(2,2,figsize=(MAIN_WIDTH,5.0),gridspec_kw={"wspace":0.38,"hspace":0.42})
    r1=next(r for r in d["rows"] if r["delta_domain"]=="standard" and r["case"]=="nodal_rpa" and r["method"]=="fftlog")
    r2=next(r for r in d["rows"] if r["delta_domain"]=="large" and r["case"]=="complex_gate" and r["method"]=="fftlog")
    im1=fftlog_heatmap(axs[0,0],r1,"FFTLog: standard nodal RPA");panel_label(axs[0,0],"(a)")
    im2=fftlog_heatmap(axs[0,1],r2,r"FFTLog: large-$\delta$ complex gate");panel_label(axs[0,1],"(b)")
    # shared colorbar range would be misleading because hostile case spans huge errors; keep explicit bars.
    fig.colorbar(im1,ax=axs[0,0],fraction=.046,pad=.04,label=r"$\log_{10}\epsilon_{\rm ref}$")
    fig.colorbar(im2,ax=axs[0,1],fraction=.046,pad=.04,label=r"$\log_{10}\epsilon_{\rm ref}$")

    for ax,case,label in [(axs[1,0],"complex_gate","(c)"),(axs[1,1],"nodal_rpa","(d)")]:
        r=next(r for r in d["rows"] if r["delta_domain"]=="large" and r["case"]==case and r["method"]=="ogata")
        per_h=((r.get("convergence") or {}).get("search") or {}).get("metadata",{}).get("per_h",[])
        hs=[];Ns=[];changes=[]
        for p in per_h:
            sel=(p.get("n_search") or {}).get("selected_parameters") or {}
            if "N" not in sel: continue
            hs.append(float(p["h"]));Ns.append(int(sel["N"]));ch=p.get("relative_l2_change");changes.append(np.nan if ch is None else float(ch))
        ax.plot(hs,Ns,marker="o",label=r"selected $N$")
        ax.set_xscale("log");ax.set_yscale("log");ax.invert_xaxis();ax.set_xlabel(r"Ogata $h$");ax.set_ylabel(r"selected $N$");ax.set_title(f"Ogata: large-$\\delta$ {CASE_LABEL[case]}",pad=3)
        finish_axis(ax,grid=True);panel_label(ax,label)
    save_figure(fig,outdir,"figureS03_numerical_method_parameter_sensitivity",manifest,["interaction_convergence.json :: sections.primary_tolerance"],"FFTLog bounded capability maps and Ogata coupled h-N refinement paths for specialist numerical methods.")


def figS4_cross_stage_and_refusal(data, outdir, manifest):
    std = data["cross_stage_standard"]
    large = data["cross_stage_large"]
    fig, axs = plt.subplots(1, 3, figsize=(MAIN_WIDTH, 2.75),
                            gridspec_kw={"wspace": 0.38})

    ax = axs[0]
    for method in ("simpson", "gl4"):
        vals = np.sort(np.array([
            max(float(r["relative_l2"]), float(r["relative_peak"]))
            for r in std["rows"] if r["method"] == method and r.get("reference_pass")
        ], float))
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, where="post", label=METHOD_LABEL[method],
                color=COLORS[method], linestyle=METHOD_LINESTYLE[method])
    ax.axvline(1e-4, color="0.35", linestyle=":", linewidth=.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"cross-stage $\epsilon_{\rm ref}$")
    ax.set_ylabel("fraction of cases")
    finish_axis(ax, grid=True)
    panel_label(ax, "(a)")

    ax = axs[1]
    cases = ["isotropic_coulomb", "anisotropic_rk_strong", "nodal_rpa", "complex_gate"]
    x = np.arange(len(cases))
    for method in ("simpson", "gl4", "fftlog", "ogata"):
        yy, xx = [], []
        for i, case in enumerate(cases):
            record = next(r for r in large["rows"]
                          if r["method"] == method and r["case"] == CROSS_CASE_ID[case])
            if record.get("reference_pass"):
                xx.append(i)
                yy.append(max(float(record["relative_l2"]), float(record["relative_peak"])))
        if xx:
            ax.scatter(xx, yy, marker=METHOD_MARKER[method], color=COLORS[method], s=22)
    ax.axhline(1e-4, color="0.35", linestyle=":", linewidth=.8)
    ax.set_yscale("log")
    ax.set_xticks(x, [CASE_LABEL[c] for c in cases], rotation=35, ha="right", fontsize=6.4)
    ax.set_ylabel(r"qualified $\epsilon_{\rm ref}$")
    finish_axis(ax, grid=True)
    panel_label(ax, "(b)")

    ax = axs[2]
    vals, labs = [], []
    for case in cases:
        record = next(r for r in large["rows"]
                      if r["method"] == "gl4" and r["case"] == CROSS_CASE_ID[case])
        qb = record.get("q_boundary_robustness") or {}
        value = qb.get("relative_l2_change")
        if value is None and qb.get("status") == "upstream_q_boundary_not_robust":
            meta = ((record.get("convergence") or {}).get("search") or {}).get("metadata", {}).get("q_boundary_robustness", {})
            value = meta.get("relative_l2_change")
        if value is not None:
            vals.append(float(value))
            labs.append(CASE_LABEL[case])
    ax.bar(np.arange(len(vals)), vals, width=.65, color="0.65")
    ax.axhline(5e-5, color="0.35", linestyle=":", linewidth=.8)
    ax.text(0.03, 5e-5 * 1.10, "boundary budget", transform=ax.get_yaxis_transform(),
            fontsize=6.2, color="0.35", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(vals)), labs, rotation=35, ha="right", fontsize=6.4)
    ax.set_ylabel(r"q-boundary relative $L^2$ change")
    finish_axis(ax, grid=True)
    panel_label(ax, "(c)")

    method_handles = [Line2D([0], [0], color=COLORS[m], linestyle=METHOD_LINESTYLE[m],
                             marker=METHOD_MARKER[m], label=METHOD_LABEL[m])
                      for m in ("simpson", "gl4", "fftlog", "ogata")]
    fig.legend(method_handles, [h.get_label() for h in method_handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
               frameon=False, fontsize=5.9, columnspacing=0.9)
    fig.subplots_adjust(top=0.80, bottom=.28)
    save_figure(
        fig, outdir, "figureS04_cross_stage_and_safe_refusal", manifest,
        ["pipeline.json :: sections.cross_stage_standard_delta",
         "pipeline.json :: sections.cross_stage_large_delta"],
        "Cross-stage independent-reference accuracy and the explicit large-displacement upstream q-boundary safeguard.",
    )

def figS5_end_to_end(data, outdir, manifest):
    fig, axs = plt.subplots(1, 2, figsize=(MAIN_WIDTH, 2.85), sharey=True,
                            gridspec_kw={"wspace": .15})
    stage_styles = {
        "PETAL2D": {"color": "#0072B2", "hatch": ""},
        "HarmonicTransform": {"color": "#E69F00", "hatch": "//"},
        "Interaction": {"color": "#009E73", "hatch": ".."},
    }
    for ax, key, label, title in [
        (axs[0], "end_to_end_standard", "(a)", r"standard $\delta$"),
        (axs[1], "end_to_end_large", "(b)", r"large $\delta$"),
    ]:
        d = data[key]
        cases = d["cases"]
        x = np.arange(len(cases))
        width = .23
        stages = [
            ("PETAL2D", [c["petal2d"]["relative_l2"] for c in cases]),
            ("HarmonicTransform", [c["harmonic_transform"]["analytic_error"]["relative_l2"] for c in cases]),
            ("Interaction", [c["interactions"][0]["relative_l2"] for c in cases]),
        ]
        for i, (name, vals) in enumerate(stages):
            style = stage_styles[name]
            ax.bar(x + (i - 1) * width, vals, width=width, color=style["color"],
                   hatch=style["hatch"], label=name)
        ax.axhline(1e-4, color="0.35", linestyle=":", linewidth=.8)
        ax.set_yscale("log")
        ax.set_xticks(x, [c["name"].replace("_", " ") for c in cases],
                      rotation=28, ha="right")
        ax.set_title(title)
        finish_axis(ax, grid=True)
        panel_label(ax, label)
    axs[0].set_ylabel(r"stage relative $L^2$ error")
    handles = [mpl.patches.Patch(facecolor=stage_styles[name]["color"],
                                 hatch=stage_styles[name]["hatch"], label=name)
               for name in stage_styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=6.4)
    fig.subplots_adjust(top=0.78, bottom=.25)
    save_figure(
        fig, outdir, "figureS05_end_to_end_stage_errors", manifest,
        ["pipeline.json :: sections.end_to_end_standard_delta",
         "pipeline.json :: sections.end_to_end_large_delta"],
        "Stage-resolved errors through PETAL2D, HarmonicTransform, and Interaction.",
    )

def figS6_detailed_scaling(data,outdir,manifest):
    hs=data["harmonic_scaling"];ins=data["interaction_scaling"]
    fig,axs=plt.subplots(2,4,figsize=(MAIN_WIDTH,4.7),gridspec_kw={"wspace":.30,"hspace":.42})
    for ax,axis,label,xlabel in zip(axs[0],("N_q","N_r","N_m","s_r"),("(a)","(b)","(c)","(d)"),(r"$N_q$",r"$N_r$",r"$N_m$",r"$s_r$")):
        rr=[r for r in hs["rows"] if r["axis"]==axis];x=np.array([r["value"] for r in rr],float);y=np.array([r["median_seconds"] for r in rr],float);q1=np.array([r["q25_seconds"] for r in rr]);q3=np.array([r["q75_seconds"] for r in rr])
        ax.errorbar(x,y,yerr=np.vstack((y-q1,q3-y)),marker="o",capsize=2);ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(xlabel);finish_axis(ax,grid=True);panel_label(ax,label,x=-.18)
    configs=[("finite","N_q",r"finite $N_q$"),("finite","N_D",r"finite $N_D$"),("fftlog","N_F",r"FFTLog $N_F$"),("fftlog","N_D",r"FFTLog $N_D$")]
    for ax,(branch,axis,title),label in zip(axs[1],configs,("(e)","(f)","(g)","(h)")):
        rr=[r for r in ins["rows"] if r["branch"]==branch and r["axis"]==axis];x=np.array([r["value"] for r in rr],float);y=np.array([r["median_seconds"] for r in rr]);q1=np.array([r["q25_seconds"] for r in rr]);q3=np.array([r["q75_seconds"] for r in rr])
        ax.errorbar(x,y,yerr=np.vstack((y-q1,q3-y)),marker="o",capsize=2);ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(title);finish_axis(ax,grid=True);panel_label(ax,label,x=-.18)
    axs[0,0].set_ylabel("runtime (s)"); axs[1,0].set_ylabel("runtime (s)")
    save_figure(fig,outdir,"figureS06_detailed_scaling",manifest,["scaling.json :: sections.harmonic_transform","scaling.json :: sections.interaction"],"Individual size-axis scaling sweeps supporting the work-collapse main figure.")


def figS7_memory(data,outdir,manifest):
    hm=data["harmonic_memory"];im=data["interaction_memory"]
    fig,axs=plt.subplots(2,2,figsize=(MAIN_WIDTH,4.9),gridspec_kw={"wspace":.34,"hspace":.4})
    configs=[(axs[0,0],hm,None,"N_q",r"HarmonicTransform: $N_q$","(a)"),(axs[0,1],hm,None,"N_r",r"HarmonicTransform: $N_r$","(b)"),(axs[1,0],im,"gl4","N_q",r"Interaction finite (GL4): $N_q$","(c)"),(axs[1,1],im,"fftlog","N_F",r"Interaction FFTLog: $N_F$","(d)")]
    for ax,d,branch,axis,title,label in configs:
        rr=[r for r in d["rows"] if r["axis"]==axis and (branch is None or r.get("branch")==branch)]
        x=np.array([r["value"] for r in rr],float);y=np.array([r["incremental_peak_rss_median_bytes"]/(1024**2) for r in rr],float);q1=np.array([r["incremental_peak_rss_q25_bytes"]/(1024**2) for r in rr]);q3=np.array([r["incremental_peak_rss_q75_bytes"]/(1024**2) for r in rr])
        ax.errorbar(x,y,yerr=np.vstack((y-q1,q3-y)),marker="o",capsize=2);ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(title.split(": ")[-1]);ax.set_ylabel("incremental peak RSS (MiB)");ax.set_title(title.split(":")[0],pad=3);finish_axis(ax,grid=True);panel_label(ax,label)
    save_figure(fig,outdir,"figureS07_memory_scaling",manifest,["memory.json :: sections.harmonic_transform","memory.json :: sections.interaction"],"Fresh-process peak-RSS scaling for the two numerical stages.")


def figS8_reference_stability(data,outdir,manifest):
    d=data["interaction_convergence"]
    fig,axs=plt.subplots(1,2,figsize=(MAIN_WIDTH,2.6),gridspec_kw={"wspace":.34})
    ax=axs[0]
    for case in ("isotropic_coulomb","anisotropic_rk_strong","complex_gate","nodal_rpa"):
        meta=d["reference_stability"][f"large::{case}"];levels=meta["levels"]
        x=[];y=[]
        for lev in levels:
            if lev.get("relative_l2_change") is not None:
                x.append(int(lev["level"]));y.append(float(lev["relative_l2_change"]))
        if x: ax.plot(x,y,marker="o",label=CASE_LABEL[case])
    ax.axhline(1e-5,color="0.35",linestyle=":",linewidth=.8);ax.set_yscale("log");ax.set_xlabel("reference refinement level");ax.set_ylabel("relative $L_2$ change");ax.legend(fontsize=6.4);finish_axis(ax,grid=True);panel_label(ax,"(a)")
    ax=axs[1]
    # Show the four large-delta GL4 q-boundary checks directly.
    vals=[];labs=[];status=[]
    for case in ("isotropic_coulomb","anisotropic_rk_strong","complex_gate","nodal_rpa"):
        row=next(r for r in data["cross_stage_large"]["rows"] if r["method"]=="gl4" and r["case"]==CROSS_CASE_ID[case])
        qb=((row.get("convergence") or {}).get("search") or {}).get("metadata",{}).get("q_boundary_robustness",{})
        if qb.get("relative_l2_change") is not None:
            vals.append(float(qb["relative_l2_change"]));labs.append(CASE_LABEL[case]);status.append(qb.get("status"))
    bars=ax.bar(np.arange(len(vals)),vals,width=.65)
    for b,s in zip(bars,status):
        if s and s!="passed": b.set_hatch("///")
    ax.axhline(5e-5,color="0.35",linestyle=":",linewidth=.8,label="boundary budget");ax.set_yscale("log");ax.set_xticks(np.arange(len(vals)),labs,rotation=35,ha="right",fontsize=6.6);ax.set_ylabel("q-boundary relative $L_2$ change");ax.legend(fontsize=6.4);finish_axis(ax,grid=True);panel_label(ax,"(b)")
    fig.subplots_adjust(bottom=.26)
    save_figure(fig,outdir,"figureS08_reference_stability",manifest,["interaction_convergence.json :: sections.primary_tolerance","pipeline.json :: sections.cross_stage_large_delta"],"Independent large-delta reference refinement and q-boundary robustness, including the intentional safe refusal.")


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)


def tex_escape(s) -> str:
    s=str(s)
    return s.replace("_",r"\_").replace("%",r"\%")


def write_tex_table(path: Path, headers: list[str], rows: list[list], caption: str, label: str, align: str | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align=align or ("l"+"r"*(len(headers)-1))
    lines=[r"\begin{table*}[t]",r"\centering",r"\small",rf"\caption{{{caption}}}",rf"\label{{{label}}}",rf"\begin{{tabular}}{{{align}}}",r"\toprule", " & ".join(headers)+r" \\",r"\midrule"]
    for row in rows: lines.append(" & ".join(str(x) for x in row)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def generate_tables(data,outdir:Path,manifest:dict):
    # Main Table 1: true end-to-end accuracy.
    headers=["domain","case","PETAL2D L2","Harmonic L2","Interaction L2","Interaction peak"]
    rows=[]
    for key,dom in [("end_to_end_standard","standard"),("end_to_end_large","large")]:
        for c in data[key]["cases"]:
            inter=c["interactions"][0]
            rows.append([dom,CASE_LABEL.get(c["name"],c["name"].replace("_"," ")),f"{c['petal2d']['relative_l2']:.3e}",f"{c['harmonic_transform']['analytic_error']['relative_l2']:.3e}",f"{inter['relative_l2']:.3e}",f"{inter['relative_peak']:.3e}"])
    write_csv(outdir/"table01_end_to_end_accuracy.csv",headers,rows)
    write_tex_table(outdir/"table01_end_to_end_accuracy.tex",headers,[[tex_escape(x) for x in r] for r in rows],"True end-to-end validation through PETAL2D, HarmonicTransform, and Interaction.","tab:end_to_end")
    manifest["table01_end_to_end_accuracy"]={"sources":["pipeline.json :: sections.end_to_end_standard_delta","pipeline.json :: sections.end_to_end_large_delta"]}

    # Main Table 2: automatic convergence at the primary target.
    d=data["interaction_convergence"];headers=["domain","method","accepted + ref pass / cases","accepted + ref fail","capable selector misses"]
    rows=[]
    for s in d["summary"]["by_method_and_domain"]:
        rows.append([s["delta_domain"],METHOD_LABEL.get(s["method"],s["method"]),f"{s['n_certified_reference_pass']}/{s['n_rows']}",s["n_false_positive"],s["n_selector_miss_backend_capable"]])
    write_csv(outdir/"table02_automatic_convergence_outcomes.csv",headers,rows)
    write_tex_table(outdir/"table02_automatic_convergence_outcomes.tex",headers,[[tex_escape(x) for x in r] for r in rows],"Automatic Interaction convergence evaluated against independent references at the primary $10^{-4}$ target.","tab:auto_convergence")
    manifest["table02_automatic_convergence_outcomes"]={"sources":["interaction_convergence.json :: sections.primary_tolerance"]}

    # Supplement Table S1: broad Interaction accuracy/timing summary.
    d=data["interaction_accuracy"];headers=["method","target","reference pass","capability fraction","worst L2","worst peak","median prod (ms)","median cal (ms)"]
    rows=[]
    for s in d["summary"]:
        rows.append([METHOD_LABEL.get(s["method"],s["method"]),f"{s['requested_tolerance']:.0e}",f"{s['n_reference_pass']}/{s['n_cases']}",f"{s['tested_capability_fraction']:.3f}",fmt_e(s.get("worst_relative_l2")),fmt_e(s.get("worst_relative_peak")),fmt_ms(s.get("median_production_seconds")),fmt_ms(s.get("median_calibration_seconds"))])
    write_csv(outdir/"tableS01_interaction_accuracy_summary.csv",headers,rows)
    write_tex_table(outdir/"tableS01_interaction_accuracy_summary.tex",headers,[[tex_escape(x) for x in r] for r in rows],"Broad 74-case Interaction accuracy and timing summary.","tab:s_interaction_accuracy")
    manifest["tableS01_interaction_accuracy_summary"]={"sources":["interaction_accuracy.json :: sections.broad_accuracy_and_timing"]}

    # Supplement Table S2: primary method matrix including finite alternatives.
    d=data["interaction_method_matrix_1e-4"];headers=["method","reference pass","capability fraction","worst L2","worst peak","median prod (ms)","median cal (ms)"]
    rows=[]
    for s in d["summary"]:
        rows.append([METHOD_LABEL.get(s["method"],s["method"]),f"{s['n_reference_pass']}/{s['n_cases']}",f"{s['tested_capability_fraction']:.3f}",fmt_e(s.get("worst_relative_l2")),fmt_e(s.get("worst_relative_peak")),fmt_ms(s.get("median_production_seconds")),fmt_ms(s.get("median_calibration_seconds"))])
    write_csv(outdir/"tableS02_interaction_method_matrix.csv",headers,rows)
    write_tex_table(outdir/"tableS02_interaction_method_matrix.tex",headers,[[tex_escape(x) for x in r] for r in rows],"Public Interaction method matrix at $10^{-4}$ for the canonical 74-case standard-domain suite.","tab:s_method_matrix")
    manifest["tableS02_interaction_method_matrix"]={"sources":["interaction_accuracy.json :: sections.primary_tolerance_method_matrix"]}

    # Supplement Table S3: large-delta practical fixed production, case-resolved.
    d=data["fixed_performance_large_1e-3"];headers=["case","method","qualification","error","prod (ms)"]
    rows=[]
    for r in d["rows"]:
        qual_label={"automatic_certificate":"automatic selection","independent_oracle_qualification":"independent reference","independent_terminal_qualification":"terminal reference"}.get(r["qualification_kind"],r["qualification_kind"].replace("_"," "))
        rows.append([CASE_LABEL.get(r["case"],r["case"].replace("_"," ")),METHOD_LABEL.get(r["method"],r["method"]),qual_label,f"{eps_ref(r['qualification_reference_error']):.3e}",f"{1e3*r['production_timing']['median_seconds']:.3f}"])
    write_csv(outdir/"tableS03_large_delta_fixed_performance.csv",headers,rows)
    write_tex_table(outdir/"tableS03_large_delta_fixed_performance.tex",headers,[[tex_escape(x) for x in r] for r in rows],r"Reference-qualified fixed-parameter production timing for $10^{-3}$ and $10^2\leq\delta\leq10^4$.","tab:s_large_fixed")
    manifest["tableS03_large_delta_fixed_performance"]={"sources":["performance.json :: sections.qualified_fixed_configuration_large_delta_practical_tolerance"]}

    # Supplement Table S4: cross-stage summary.
    headers=["domain","method","converged","reference pass","q-boundary refusals","numerical-method refusals","worst L2","worst peak"]
    rows=[]
    for key,dom in [("cross_stage_standard","standard"),("cross_stage_large","large")]:
        for s in data[key]["summary"]:
            rows.append([dom,METHOD_LABEL.get(s["method"],s["method"]),s["n_converged"],s["n_reference_pass"],s["n_upstream_boundary_refusal"],s["n_backend_refusal"],f"{s['worst_relative_l2']:.3e}" if s["worst_relative_l2"] is not None else "--",f"{s['worst_relative_peak']:.3e}" if s["worst_relative_peak"] is not None else "--"])
    write_csv(outdir/"tableS04_cross_stage_summary.csv",headers,rows)
    write_tex_table(outdir/"tableS04_cross_stage_summary.tex",headers,[[tex_escape(x) for x in r] for r in rows],"Adaptive HarmonicTransform to Interaction cross-stage validation.","tab:s_cross_stage")
    manifest["tableS04_cross_stage_summary"]={"sources":["pipeline.json :: sections.cross_stage_standard_delta","pipeline.json :: sections.cross_stage_large_delta"]}

    # Supplement Table S5: scaling fits.
    headers=["stage/branch","axis","expected","measured tail slope","R2"]
    rows=[]
    for axis,f in data["harmonic_scaling"]["fits"].items():
        t=f["asymptotic_tail"];rows.append(["HarmonicTransform",axis,f"{f['expected_alpha']:.1f}",f"{t['alpha']:.3f}",f"{t['r2_log']:.4f}"])
    for name,f in data["interaction_scaling"]["fits"].items():
        if name == "fftlog_N_F_work":
            t=f["power_vs_work"]["asymptotic_tail"]
            expected="1.0"
        else:
            t=f.get("asymptotic_tail",f.get("all_points",f))
            expected=f"{float(f.get('expected_alpha',1.0)):.1f}" if f.get("expected_alpha") is not None else "--"
        alpha=t.get("alpha");r2=t.get("r2_log")
        rows.append(["Interaction",name,expected,f"{alpha:.3f}" if alpha is not None else "--",f"{r2:.4f}" if r2 is not None else "--"])
    write_csv(outdir/"tableS05_scaling_fits.csv",headers,rows)
    write_tex_table(outdir/"tableS05_scaling_fits.tex",headers,[[tex_escape(x) for x in r] for r in rows],"Finite-range runtime scaling fits. Source-derived asymptotic expectations are distinguished from measured slopes.","tab:s_scaling_fits")
    manifest["tableS05_scaling_fits"]={"sources":["scaling.json :: sections.harmonic_transform","scaling.json :: sections.interaction"]}

    # Supplement Table S6: memory fits.
    headers=["stage","fit","slope","R2"]
    rows=[]
    for name,f in data["harmonic_memory"]["fits"].items():
        if isinstance(f,dict): rows.append(["HarmonicTransform",name,f"{f.get('slope_bytes_per_x',float('nan')):.4g}",f"{f.get('r2',float('nan')):.4f}"])
    for name,f in data["interaction_memory"]["fits"].items():
        if isinstance(f,dict):
            slope=f.get("slope_bytes_per_x",f.get("slope_bytes_per_unit",float('nan')));r2=f.get("r2",float('nan'))
            rows.append(["Interaction",name,f"{slope:.4g}",f"{r2:.4f}"])
    write_csv(outdir/"tableS06_memory_fits.csv",headers,rows)
    write_tex_table(outdir/"tableS06_memory_fits.tex",headers,[[tex_escape(x) for x in r] for r in rows],"Fresh-process peak-memory scaling fits.","tab:s_memory_fits")
    manifest["tableS06_memory_fits"]={"sources":["memory.json :: sections.harmonic_transform","memory.json :: sections.interaction"]}


# -----------------------------------------------------------------------------
# Captions / contract
# -----------------------------------------------------------------------------


def write_caption_file(path:Path):
    text=r"""# QUARTIC2D manuscript figure captions

## Figure 1 - Method and representative calculation
(a) Representative anisotropic transition field. (b) Retained radial angular harmonics. (c) Corresponding Fourier-Bessel form factors, comparing the benchmark-selected numerical transform with the analytic form factors. (d) Screened interaction along the x direction, comparing the selected QUARTIC2D calculation with an independently integrated direct-q reference. The calculation illustrates the sequence rho(x,y) -> rho_m(r) -> F_m(q) -> V(delta); quantitative validation is reported in Figs. 2-5.

## Figure 2 - Direct numerical accuracy
Numerical/reference comparisons for (a) an isotropic Gaussian Hankel transform, (b) the m=2 component of an anisotropic Gaussian, (c) an anisotropic strongly screened Rytova-Keldysh interaction, and (d) the nodal 2DEG-RPA benchmark. Lower panels show pointwise absolute numerical-reference differences normalized by the peak reference magnitude. Their maxima give the peak-normalized error; they are not pointwise relative-error or `rtol` bounds.

## Figure 3 - Automatic error control
(a) Independent aggregate reference error versus requested Interaction `rtol` over the broad fixed-field benchmark. Large symbols denote per-method medians. The diagonal compares two distinct global quantities and is not a pointwise-error guarantee. (b) Finite-rule Richardson estimate versus independent relative L2 error for automatically accepted Simpson and GL4 calculations. (c,d) Case-level automatic-convergence outcomes for the canonical standard- and large-displacement matrices. Symbols distinguish automatic acceptance followed by independent-reference passing, reference-qualified points found after automatic refusal, unresolved automatic refusals, and cases for which no capable point was demonstrated in the declared tested box.

## Figure 4 - Accuracy versus computational cost
Production execution time versus achieved independent-reference error for representative standard- and large-separation workloads. The standard-domain panels include the three requested-accuracy levels where available; the large-separation panels use independently qualified fixed configurations at the practical 10^-3 target. A method is shown only when an independently reference-qualified production timing exists in the relevant benchmark. Public Ogata was not part of the older canonical standard-domain timing sweep. Trapezoid and GL8 were not part of the practical large-separation fixed-configuration timing run. FFTLog is additionally absent from the large-separation complex dual-gate panel because no tested configuration met the target. Comparisons are workload-specific and do not define a universal method ranking.

## Figure 5 - Scaling and reusable calibration
(a) HarmonicTransform runtime versus the dominant work measure N_m N_q N_s. (b) Finite-grid Interaction runtime versus N_p N_D N_q,s. (c) FFTLog Interaction runtime versus N_p[N_F log2 N_F + N_D]; the dashed proportional line is a source-derived work guide, not a fitted asymptotic law. (d) Effective per-evaluation time t_eff=t_prod+t_cal/M for two prequalified repeated-use workloads, separating one-time calibration from fixed-parameter production.

## Supplementary figures
S1: Per-workload HarmonicTransform independent-reference validation at requested `rtol` values 10^-3, 10^-4, and 10^-5, shown as independent categorical points rather than connected progressions.
S2: Cumulative independent-reference error and coverage over the 74-case broad validation suite. The vertical requested-level guide is compared with the global reference-error scalar; plateaus below one expose incomplete qualification in the declared search.
S3: FFTLog bounded (N,q_bias) capability maps and Ogata coupled h-N refinement paths for specialist numerical methods.
S4: Standard and large-delta cross-stage accuracy together with the q-boundary safety test.
S5: Stage-resolved true end-to-end errors through PETAL2D, HarmonicTransform, and Interaction.
S6: Individual runtime scaling sweeps underlying the work-collapse main figure.
S7: Fresh-process peak-memory scaling.
S8: Independent large-delta reference refinement and q-boundary robustness, including the intentional upstream_q_boundary_not_robust refusal.
"""
    path.write_text(text,encoding="utf-8")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=Path("benchmarks/results"))
    parser.add_argument("--output",type=Path,default=Path("benchmarks/results/manuscript_artifacts"))
    args=parser.parse_args()
    apply_style()
    r=args.results
    data, provenance = load_publication_results(r)
    out=args.output;mainout=out/"main";supp=out/"supplement";tables=out/"tables"
    manifest={}
    fig1_workflow(data,mainout,manifest)
    fig2_direct_accuracy(data,mainout,manifest)
    fig3_error_control(data,mainout,manifest)
    fig4_accuracy_cost(data,mainout,manifest)
    fig5_scaling_reuse(data,mainout,manifest)
    figS1_harmonic_validation(data,supp,manifest)
    figS2_interaction_distributions(data,supp,manifest)
    figS3_numerical_method_parameters(data,supp,manifest)
    figS4_cross_stage_and_refusal(data,supp,manifest)
    figS5_end_to_end(data,supp,manifest)
    figS6_detailed_scaling(data,supp,manifest)
    figS7_memory(data,supp,manifest)
    figS8_reference_stability(data,supp,manifest)
    generate_tables(data,tables,manifest)
    write_caption_file(out/"CAPTIONS.md")
    (out/"manifest.json").write_text(
        json.dumps({"inputs": provenance, "artifacts": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manuscript artifacts to {out}")


if __name__=="__main__":
    main()
