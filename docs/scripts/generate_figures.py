"""Regenerate QUARTIC2D documentation figures.

``examples`` is self-contained and uses the public package API. ``validation``
reads one explicit consolidated publication JSON bundle. The two paths are kept
separate so a fresh source checkout can regenerate tutorial figures without
requiring ignored publication results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from petal2d import PolarDecomposition
from scipy.integrate import simpson
from scipy.special import i0e, jv

from generate_evidence import publication_fingerprint
from quartic2d import HarmonicTransform, Interaction
from quartic2d._plot_style import (
    CRITERION_LINE_STYLE,
    DOCUMENTATION_RC_PARAMS,
    GUIDE_LINE_STYLE,
    REFERENCE_LINE_STYLE,
    ZERO_LINE_STYLE,
    domain_style,
    method_label,
    method_style,
    status_style,
)

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "source" / "_static"
EXAMPLE_DIR = STATIC / "examples"
VALIDATION_DIR = STATIC / "validation"
ACTIVE_PUBLICATION_FINGERPRINT: str | None = None


plt.rcParams.update({**DOCUMENTATION_RC_PARAMS, "figure.constrained_layout.use": True})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    if path.suffix.lower() == ".svg":
        # Matplotlib SVG path data contains harmless trailing spaces. Strip them
        # so generated documentation figures remain clean under git diff --check.
        text = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        if ACTIVE_PUBLICATION_FINGERPRINT is not None:
            marker = f"<!-- publication-bundle-sha256: {ACTIVE_PUBLICATION_FINGERPRINT} -->\n"
            if marker not in text:
                lines = text.splitlines(keepends=True)
                insert_at = 1 if lines and lines[0].startswith("<?xml") else 0
                lines.insert(insert_at, marker)
                text = "".join(lines)
        path.write_text(text)
    print(path.relative_to(ROOT))


def _set_reference_error_limits(ax, *series, ceiling=1.0e-2, floor=1.0e-10):
    """Choose a useful log range without letting roundoff dominate the panel."""
    positive = []
    for values in series:
        arr = np.asarray(values, dtype=float)
        positive.extend(arr[np.isfinite(arr) & (arr > floor)].tolist())
    if positive:
        low = max(floor, 10.0 ** np.floor(np.log10(min(positive))))
        high = max(low * 1.0e2, min(float(ceiling), 10.0 ** np.ceil(np.log10(max(positive)))))
    else:
        low, high = floor, ceiling
    if high <= low:
        high = 10.0 * low
    ax.set_ylim(low, high)


def _gaussian_end_to_end_problem():
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)

    def density(x, y):
        return np.exp(-(x**2 + y**2)) / np.pi

    dec = PolarDecomposition(
        density,
        x,
        y,
        Nr=181,
        Ntheta=256,
        rmax=5.5,
        origin=(0.0, 0.0),
        recon_err_tol=1.0e-10,
    )
    return x, y, density, dec



def gaussian_petal2d() -> None:
    x, y, density, dec = _gaussian_end_to_end_problem()
    X, Y = np.meshgrid(x, y, indexing="xy")
    original = density(X, Y)
    rho0 = np.asarray(dec.rho[0]).real

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(7.4, 3.25), gridspec_kw={"width_ratios": (1.0, 1.18)}
    )
    image = ax0.imshow(
        original,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        vmin=0.0,
        vmax=float(np.max(original)),
        aspect="equal",
        cmap="magma",
    )
    ax0.set_xlabel(r"$x/\sigma$")
    ax0.set_ylabel(r"$y/\sigma$")
    ax0.set_title(r"$\rho(x,y)$")
    fig.colorbar(image, ax=ax0, shrink=0.84, pad=0.035, label=r"$\rho$")

    ax1.plot(dec.r, np.exp(-dec.r**2) / np.pi, **REFERENCE_LINE_STYLE, label="analytic")
    ax1.plot(
        dec.r,
        rho0,
        "o",
        ms=3.0,
        markevery=max(1, len(dec.r) // 18),
        fillstyle="none",
        label="PETAL2D",
    )
    ax1.set_xlabel(r"$r/\sigma$")
    ax1.set_ylabel(r"$\rho_0(r)$")
    ax1.set_title(r"retained harmonic $m=0$")
    ax1.legend(frameon=True, framealpha=1.0, loc="upper right")
    _save(fig, EXAMPLE_DIR / "gaussian_petal2d.svg")




def gaussian_hankel_end_to_end() -> None:
    _, _, _, dec = _gaussian_end_to_end_problem()
    field = HarmonicTransform(dec)
    q = field.q
    exact = np.exp(-q**2 / 4.0) / (2.0 * np.pi)
    pointwise_peak_error = np.abs(field.F_q[0].real - exact) / np.max(np.abs(exact))

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.5, 4.9), sharex=True,
        gridspec_kw={"height_ratios": (2.5, 1.0)}
    )
    ax0.plot(q, exact, **REFERENCE_LINE_STYLE, label="analytic")
    ax0.plot(
        q, field.F_q[0].real, "o", ms=3.4,
        markevery=max(1, q.size // 14), fillstyle="none", label="QUARTIC2D"
    )
    ax0.set_ylabel(r"$F_0(q)$")
    ax0.legend(frameon=True, framealpha=1.0, loc="upper right")

    plotted = np.maximum(pointwise_peak_error, 1.0e-10)
    ax1.semilogy(q, plotted, lw=1.25)
    _set_reference_error_limits(ax1, pointwise_peak_error, ceiling=1.0e-2, floor=1.0e-10)
    ax1.set_xlabel(r"$q\sigma$")
    ax1.set_ylabel(r"$|\Delta F|/\max|F_{\rm ref}|$")
    _save(fig, EXAMPLE_DIR / "gaussian_hankel_end_to_end.svg")




def _gaussian_interaction_reference(delta, kernel):
    q = np.linspace(1.0e-7, 16.0, 12001)
    form = np.exp(-q**2 / 4.0) / (2.0 * np.pi)
    values = []
    for d in np.asarray(delta, dtype=float):
        integrand = 2.0 * np.pi * q * kernel(q) * np.abs(form) ** 2 * jv(0, q * d)
        values.append(simpson(integrand, x=q).real)
    return np.asarray(values)



def gaussian_interaction_end_to_end() -> None:
    _, _, _, dec = _gaussian_end_to_end_problem()

    def coulomb(q):
        return 2.0 * np.pi / q

    def rk(q):
        return 2.0 * np.pi / (q * (1.0 + q))

    default_field = HarmonicTransform(dec)
    hcal = HarmonicTransform.converge_parameters(
        dec, rtol=1.0e-4, atol=1.0e-12, q_tail_rtol=1.0e-3,
        method="simpson", verbose=False,
    )
    converged_field = hcal.transform(dec)

    delta = np.linspace(0.0, 4.0, 81)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    references = {
        "Coulomb": _gaussian_interaction_reference(delta, coulomb),
        "Rytova–Keldysh": _gaussian_interaction_reference(delta, rk),
    }
    defaults = {
        "Coulomb": Interaction(vectors, default_field, default_field, coulomb).V.real,
        "Rytova–Keldysh": Interaction(vectors, default_field, default_field, rk).V.real,
    }

    cal_delta = np.geomspace(1.0e-2, 1.0e2, 32)
    cal_vectors = np.column_stack((cal_delta, np.zeros_like(cal_delta)))
    ccal = Interaction.converge_parameters(
        cal_vectors, converged_field, converged_field, coulomb,
        rtol=1.0e-4, atol=1.0e-12, method="gl4",
    )
    rcal = Interaction.converge_parameters(
        cal_vectors, converged_field, converged_field, rk,
        rtol=1.0e-4, atol=1.0e-12, method="gl4",
    )
    converged = {
        "Coulomb": ccal.interaction(vectors, converged_field, converged_field, coulomb).V.real,
        "Rytova–Keldysh": rcal.interaction(vectors, converged_field, converged_field, rk).V.real,
    }

    colors = {"Coulomb": "#0072B2", "Rytova–Keldysh": "#D55E00"}
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sample = np.arange(0, delta.size, 8)
    for name in ("Coulomb", "Rytova–Keldysh"):
        color = colors[name]
        ax.plot(delta, references[name], color=color, lw=1.55, label=name)
        ax.plot(
            delta[sample], defaults[name][sample], "o", color=color, ms=3.6,
            fillstyle="none", label="_nolegend_",
        )
        ax.plot(
            delta[sample], converged[name][sample], "s", color=color, ms=3.4,
            fillstyle="none", label="_nolegend_",
        )
    ax.set_xlabel(r"$\delta/\sigma$")
    ax.set_ylabel(r"$V(\delta)$")
    legend_handles = [
        Line2D([0], [0], color="0.2", lw=1.55, label="reference"),
        Line2D(
            [0], [0], color="0.2", marker="o", linestyle="none",
            fillstyle="none", label="QUARTIC2D",
        ),
    ]
    ax.legend(
        handles=legend_handles, frameon=True, framealpha=1.0,
        loc="upper right", fontsize=8.5,
    )
    _save(fig, EXAMPLE_DIR / "gaussian_interaction_end_to_end.svg")



def _analytic_orbital_form_factors(q, phi):
    qq = q[:, None]
    c = np.cos(phi)[None, :]
    envelope = np.exp(-qq**2 / 4.0)
    rho_ss = envelope / (2.0 * np.pi)
    rho_pp = (2.0 / np.pi) * envelope * (0.25 - qq**2 * c**2 / 8.0)
    rho_sp = -1j * np.sqrt(2.0) * qq * c * envelope / (4.0 * np.pi)
    return rho_ss, rho_pp, rho_sp


def _direct_phi_reference(deltas, kappa=0.35):
    q = np.linspace(0.0, 14.0, 2601)
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    qq = q[:, None]
    c = np.cos(phi)[None, :]
    rho_ss, rho_pp, rho_sp = _analytic_orbital_form_factors(q, phi)
    kernel = 2.0 * np.pi / np.sqrt(qq**2 + kappa**2)
    direct = []
    exchange = []
    for delta in deltas:
        phase = np.exp(-1j * qq * float(delta) * c)
        direct_phi = (
            qq * rho_ss * kernel * np.conj(rho_pp) * phase
        ).mean(axis=1) * (2.0 * np.pi)
        exchange_phi = (
            qq * rho_sp * kernel * np.conj(rho_sp) * phase
        ).mean(axis=1) * (2.0 * np.pi)
        direct.append(simpson(direct_phi, x=q).real)
        exchange.append(simpson(exchange_phi, x=q).real)
    return np.asarray(direct), np.asarray(exchange)


def _direct_exchange_problem():
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)

    def orbital_s(x, y):
        return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)

    def orbital_px(x, y):
        return np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * (x**2 + y**2))

    def rho_ss(x, y):
        psi = orbital_s(x, y)
        return np.conj(psi) * psi

    def rho_pp(x, y):
        psi = orbital_px(x, y)
        return np.conj(psi) * psi

    def rho_sp(x, y):
        return np.conj(orbital_s(x, y)) * orbital_px(x, y)

    def decompose(field):
        return PolarDecomposition(
            field,
            x,
            y,
            Nr=181,
            Ntheta=256,
            rmax=5.5,
            origin=(0.0, 0.0),
            recon_err_tol=1.0e-4,
        )

    return (
        x,
        y,
        (rho_ss, rho_pp, rho_sp),
        (decompose(rho_ss), decompose(rho_pp), decompose(rho_sp)),
    )




def _harmonic_map(field):
    return {
        int(m): np.asarray(field.F_q[int(m)])
        for m in field.m_values
    }





def direct_exchange() -> None:
    _, _, _, decompositions = _direct_exchange_problem()
    calibrations = [
        HarmonicTransform.converge_parameters(
            dec, rtol=1.0e-4, atol=1.0e-12, q_tail_rtol=1.0e-3,
            method="simpson", verbose=False,
        )
        for dec in decompositions
    ]
    field_ss, field_pp, field_sp = [
        calibration.transform(dec) for calibration, dec in zip(calibrations, decompositions)
    ]

    def kernel(q):
        return 2.0 * np.pi / np.sqrt(q**2 + 0.35**2)

    sample_delta = np.linspace(0.0, 4.0, 81)
    vectors = np.column_stack((sample_delta, np.zeros_like(sample_delta)))
    calibration_delta = np.linspace(0.0, 4.0, 25)
    calibration_vectors = np.column_stack((calibration_delta, np.zeros_like(calibration_delta)))
    direct_cal = Interaction.converge_parameters(
        calibration_vectors, field_ss, field_pp, kernel,
        rtol=1.0e-4, atol=1.0e-12, method="gl4",
    )
    exchange_cal = Interaction.converge_parameters(
        calibration_vectors, field_sp, field_sp, kernel,
        rtol=1.0e-4, atol=1.0e-12, method="gl4",
    )
    direct = direct_cal.interaction(vectors, field_ss, field_pp, kernel).V.real
    exchange = exchange_cal.interaction(vectors, field_sp, field_sp, kernel).V.real

    reference_delta = np.linspace(0.0, 4.0, 241)
    direct_ref, exchange_ref = _direct_phi_reference(reference_delta)
    channel_styles = {
        "direct": {"color": "#0072B2", "marker": "o", "label": r"$U_{spsp}$"},
        "exchange": {"color": "#D55E00", "marker": "s", "label": r"$U_{spps}$"},
    }

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.5, 5.4), sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)}
    )
    for key, ref, values in (
        ("direct", direct_ref, direct),
        ("exchange", exchange_ref, exchange),
    ):
        style = channel_styles[key]
        ax0.plot(reference_delta, ref, color=style["color"], lw=1.55, label=style["label"])
        ax0.plot(
            sample_delta, values, linestyle="none", marker=style["marker"],
            color=style["color"], ms=3.4, markevery=5, fillstyle="none",
            label="_nolegend_",
        )
    ax0.axhline(0.0, **ZERO_LINE_STYLE)
    ax0.set_ylabel(r"$U_{1234}(\delta)$")
    legend_handles = [
        Line2D([0], [0], color=channel_styles["direct"]["color"], lw=1.5, label=r"$U_{spsp}$"),
        Line2D([0], [0], color=channel_styles["exchange"]["color"], lw=1.5, label=r"$U_{spps}$"),
        Line2D([0], [0], color="0.2", lw=1.5, label="reference"),
        Line2D([0], [0], color="0.2", marker="o", linestyle="none", fillstyle="none", label="QUARTIC2D"),
    ]
    ax0.legend(handles=legend_handles, frameon=True, framealpha=1.0, loc="best", ncol=2)

    ratio_ref = exchange_ref / direct_ref
    ratio = exchange / direct
    ax1.plot(reference_delta, ratio_ref, **REFERENCE_LINE_STYLE)
    ax1.plot(sample_delta, ratio, "o", ms=3.2, markevery=5, fillstyle="none")
    ax1.axhline(0.0, **ZERO_LINE_STYLE)
    ax1.set_xlabel(r"displacement $\delta$")
    ax1.set_ylabel(r"$U_{spps}/U_{spsp}$")
    _save(fig, EXAMPLE_DIR / "direct_exchange.svg")


def _anisotropic_problem():
    anisotropy = 0.35
    kappa = 0.4
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)

    def transition_field(x, y):
        r2 = x**2 + y**2
        return (
            np.exp(-r2)
            * (1.0 + anisotropy * (x**2 - y**2))
            / np.pi
        )

    def kernel(q):
        return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)

    dec = PolarDecomposition(
        transition_field,
        x,
        y,
        Nr=181,
        Ntheta=256,
        rmax=5.5,
        origin=(0.0, 0.0),
        recon_err_tol=1.0e-4,
    )
    return anisotropy, kappa, x, y, transition_field, kernel, dec


def _polar_reconstruction(dec, x, y):
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    reconstruction = np.zeros_like(X, dtype=np.complex128)
    for m in dec.m_sorted:
        m_int = int(m)
        radial = np.asarray(dec.rho[m_int])
        values = np.interp(radius.ravel(), dec.r, radial).reshape(radius.shape)
        reconstruction += values * np.exp(1j * m_int * theta)
    return reconstruction



def sampled_data() -> None:
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)
    Xij, Yij = np.meshgrid(x, y, indexing="ij")
    r2 = Xij**2 + Yij**2

    psi_s = np.exp(-0.5 * r2) / np.sqrt(np.pi)
    psi_p_plus = (Xij + 1j * Yij) * np.exp(-0.5 * r2) / np.sqrt(np.pi)
    sampled = np.conj(psi_s) * psi_p_plus

    dec = PolarDecomposition(
        sampled, x, y, Nr=181, Ntheta=256, rmax=5.5,
        origin=(0.0, 0.0), interp_method="cubic", recon_err_tol=1.0e-2,
    )
    exact_cartesian = sampled.T

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    amplitude = float(np.max(np.abs(exact_cartesian)))
    image = None
    for ax, data, title in (
        (axes[0], exact_cartesian.real, r"$\mathrm{Re}\,\rho_{s+}(x,y)$"),
        (axes[1], exact_cartesian.imag, r"$\mathrm{Im}\,\rho_{s+}(x,y)$"),
    ):
        image = ax.imshow(
            data, origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
            aspect="equal", cmap="RdBu_r", vmin=-amplitude, vmax=amplitude,
        )
        ax.set_xlabel(r"$x$")
        ax.set_title(title)
    axes[0].set_ylabel(r"$y$")
    fig.colorbar(image, ax=axes, shrink=0.88, pad=0.025, label=r"$\rho_{s+}$ component")
    _save(fig, EXAMPLE_DIR / "sampled_data_field.svg")

    hcal = HarmonicTransform.converge_parameters(
        dec, rtol=1.0e-4, atol=1.0e-12, q_tail_rtol=1.0e-3,
        method="simpson", verbose=False,
    )
    field = hcal.transform(dec)
    q = np.asarray(field.q, dtype=float)
    numerical = np.asarray(field.F_q[1])
    exact_transform = q * np.exp(-q**2 / 4.0) / (4.0 * np.pi)
    pointwise_peak_error = np.abs(numerical - exact_transform) / max(float(np.max(np.abs(exact_transform))), 1.0e-15)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.4, 4.8), sharex=True,
        gridspec_kw={"height_ratios": (2.4, 1.0)}
    )
    ax0.plot(q, exact_transform, **REFERENCE_LINE_STYLE, label="analytic")
    ax0.plot(
        q, numerical.real, "o", ms=3.0, markevery=max(1, len(q) // 16),
        fillstyle="none", label="QUARTIC2D",
    )
    ax0.set_ylabel(r"$F_1(q)$")
    ax0.legend(frameon=True, framealpha=1.0, loc="upper right")

    ax1.semilogy(q, np.maximum(pointwise_peak_error, 1.0e-10), lw=1.2)
    _set_reference_error_limits(ax1, pointwise_peak_error, ceiling=1.0e-2, floor=1.0e-10)
    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$|\Delta F|/\max|F_{\rm ref}|$")
    _save(fig, EXAMPLE_DIR / "sampled_data_transform.svg")





def anisotropic_hankel() -> None:
    anisotropy, _, _, _, _, _, dec = _anisotropic_problem()
    hcal = HarmonicTransform.converge_parameters(
        dec, rtol=1.0e-4, atol=1.0e-12, q_tail_rtol=1.0e-3,
        method="simpson", verbose=False,
    )
    field = hcal.transform(dec)
    numerical = _harmonic_map(field)
    q = field.q
    exact0 = np.exp(-q**2 / 4.0) / (2.0 * np.pi)
    exact2 = anisotropy * q**2 * np.exp(-q**2 / 4.0) / (16.0 * np.pi)
    err0 = np.abs(numerical[0].real - exact0) / max(float(np.max(np.abs(exact0))), 1.0e-15)
    errp = np.abs(numerical[2].real - exact2) / max(float(np.max(np.abs(exact2))), 1.0e-15)
    errm = np.abs(numerical[-2].real - exact2) / max(float(np.max(np.abs(exact2))), 1.0e-15)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.8, 5.3), sharex="col",
        gridspec_kw={"height_ratios": (2.15, 1.0)}
    )
    axes[0, 0].plot(q, exact0, **REFERENCE_LINE_STYLE, label="analytic")
    axes[0, 0].plot(q, numerical[0].real, "o", ms=3.0, markevery=max(1, len(q)//16), fillstyle="none", label="QUARTIC2D")
    axes[0, 0].set_ylabel(r"$F_0(q)$")
    axes[0, 0].set_title(r"$m=0$")
    axes[0, 0].legend(frameon=True, framealpha=1.0, loc="upper right")
    axes[1, 0].semilogy(q, np.maximum(err0, 1.0e-10), lw=1.2)
    axes[1, 0].set_xlabel(r"$q$")
    axes[1, 0].set_ylabel(r"$|\Delta F|/\max|F_{\rm ref}|$")
    _set_reference_error_limits(axes[1, 0], err0, ceiling=1.0e-2, floor=1.0e-10)

    axes[0, 1].plot(q, exact2, **REFERENCE_LINE_STYLE, label="analytic")
    axes[0, 1].plot(q, numerical[2].real, "o", ms=3.0, markevery=max(1, len(q)//16), fillstyle="none", label=r"$m=+2$")
    axes[0, 1].plot(q, numerical[-2].real, "s", ms=2.8, markevery=max(1, len(q)//16), fillstyle="none", label=r"$m=-2$")
    axes[0, 1].set_ylabel(r"$F_{\pm2}(q)$")
    axes[0, 1].set_title(r"$m=\pm2$")
    axes[0, 1].legend(frameon=True, framealpha=1.0, loc="upper right")
    axes[1, 1].semilogy(q, np.maximum(errp, 1.0e-10), lw=1.2, label=r"$m=+2$")
    axes[1, 1].semilogy(q, np.maximum(errm, 1.0e-10), lw=1.2, ls="--", label=r"$m=-2$")
    axes[1, 1].set_xlabel(r"$q$")
    axes[1, 1].set_ylabel(r"$|\Delta F|/\max|F_{\rm ref}|$")
    _set_reference_error_limits(axes[1, 1], errp, errm, ceiling=1.0e-2, floor=1.0e-10)
    axes[1, 1].legend(frameon=True, framealpha=1.0, loc="upper right")
    _save(fig, EXAMPLE_DIR / "anisotropic_hankel.svg")


def _anisotropic_interaction_calculation():
    _, _, _, _, _, kernel, dec = _anisotropic_problem()
    hcal = HarmonicTransform.converge_parameters(
        dec,
        rtol=1.0e-4,
        atol=1.0e-12,
        q_tail_rtol=1.0e-3,
        method="simpson",
        verbose=False,
    )
    field = hcal.transform(dec)
    phi = np.linspace(0.0, 2.0 * np.pi, 181)
    radii = np.asarray([0.75, 1.50, 2.50])
    vectors = np.vstack(
        [
            np.column_stack((radius * np.cos(phi), radius * np.sin(phi)))
            for radius in radii
        ]
    )
    ical = Interaction.converge_parameters(
        vectors,
        field,
        field,
        kernel,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
    )
    interaction = ical.interaction(vectors, field, field, kernel)
    return field, phi, radii, interaction



def anisotropic_orientation() -> None:
    _, phi, radii, interaction = _anisotropic_interaction_calculation()
    values = interaction.V.real.reshape(len(radii), len(phi))
    styles = (
        {"color": "#0072B2", "linestyle": "-"},
        {"color": "#D55E00", "linestyle": "--"},
        {"color": "#009E73", "linestyle": "-."},
    )

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.5, 5.3), sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)}
    )
    for radius, curve, style in zip(radii, values, styles):
        ax0.plot(phi / np.pi, curve, lw=1.5, label=rf"$\delta={radius:.2f}$", **style)
    ax0.set_ylabel(r"$V(\delta,\varphi_\delta)$")
    ax0.legend(frameon=True, framealpha=1.0, ncol=1, loc="best")

    for curve, style in zip(values, styles):
        mean = float(np.mean(curve))
        ax1.plot(phi / np.pi, (curve - mean) / mean, lw=1.4, **style)
    ax1.axhline(0.0, **ZERO_LINE_STYLE)
    ax1.set_xlabel(r"$\varphi_\delta/\pi$")
    ax1.set_ylabel(r"$(V-\langle V\rangle_\varphi)/\langle V\rangle_\varphi$")
    _save(fig, EXAMPLE_DIR / "anisotropic_orientation.svg")




def anisotropic_pair_contributions() -> None:
    field, phi, _radii, interaction = _anisotropic_interaction_calculation()
    nphi = len(phi)
    middle = slice(nphi, 2 * nphi)
    grouped = {
        0: np.zeros(nphi, dtype=np.complex128),
        2: np.zeros(nphi, dtype=np.complex128),
        4: np.zeros(nphi, dtype=np.complex128),
    }
    for i, m in enumerate(field.m_values):
        for j, mp in enumerate(field.m_values):
            grouped[abs(int(m - mp))] += interaction.V_mm[i, j, middle]
    total = sum(grouped.values()).real

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    styles = {
        0: {"color": "#0072B2", "linestyle": "-"},
        2: {"color": "#D55E00", "linestyle": "--"},
        4: {"color": "#009E73", "linestyle": ":"},
    }
    for difference, label in ((0, r"$|m-m'|=0$"), (2, r"$|m-m'|=2$"), (4, r"$|m-m'|=4$")):
        ax.plot(phi / np.pi, grouped[difference].real, lw=1.4, label=label, **styles[difference])
    ax.plot(phi / np.pi, total, color="0.15", lw=2.1, label="total")
    ax.set_xlabel(r"$\varphi_\delta/\pi$")
    ax.set_ylabel(r"contribution to $V(\delta,\varphi_\delta)$")
    ax.legend(ncol=2, frameon=True, framealpha=1.0, loc="best")
    _save(fig, EXAMPLE_DIR / "anisotropic_pair_contributions.svg")



def _complex_orbital_problem():
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)

    def orbital_s(x, y):
        r2 = x**2 + y**2
        return np.exp(-0.5 * r2) / np.sqrt(np.pi)

    def orbital_h(x, y):
        r2 = x**2 + y**2
        z = x + 1j * y
        zbar = x - 1j * y
        return (
            np.exp(-0.5 * r2)
            / (2.0 * np.sqrt(np.pi))
            * (
                np.sqrt(2.0) * (1.0 - r2)
                + 1j * z
                + zbar**2 / np.sqrt(2.0)
            )
        )

    def rho_ss(x, y):
        psi = orbital_s(x, y)
        return np.conj(psi) * psi

    def rho_sh(x, y):
        return np.conj(orbital_s(x, y)) * orbital_h(x, y)

    def rho_hs(x, y):
        return np.conj(orbital_h(x, y)) * orbital_s(x, y)

    def decompose(field):
        return PolarDecomposition(
            field,
            x,
            y,
            Nr=181,
            Ntheta=256,
            rmax=5.5,
            origin=(0.0, 0.0),
            recon_err_tol=1.0e-4,
        )

    return {
        "x": x,
        "y": y,
        "rho_ss": rho_ss,
        "rho_sh": rho_sh,
        "rho_hs": rho_hs,
        "dec_ss": decompose(rho_ss),
        "dec_sh": decompose(rho_sh),
        "dec_hs": decompose(rho_hs),
    }




def _complex_sh_exact(m, q):
    envelope = np.exp(-q**2 / 4.0)
    if m == 0:
        return q**2 * envelope / (8.0 * np.pi * np.sqrt(2.0))
    if m == 1:
        return 1j * q * envelope / (8.0 * np.pi)
    if m == -2:
        return q**2 * envelope / (16.0 * np.pi * np.sqrt(2.0))
    raise KeyError(m)



def complex_orbital_hankel() -> None:
    problem = _complex_orbital_problem()
    hcal = HarmonicTransform.converge_parameters(
        problem["dec_sh"], rtol=1.0e-4, atol=1.0e-12,
        q_tail_rtol=1.0e-3, method="simpson", verbose=False,
    )
    field = hcal.transform(problem["dec_sh"])
    numerical = _harmonic_map(field)
    q = field.q
    mode_specs = (
        (0, "real", "#0072B2", "o", r"$m=0$ (Re)"),
        (1, "imag", "#D55E00", "s", r"$m=+1$ (Im)"),
        (-2, "real", "#009E73", "^", r"$m=-2$ (Re)"),
    )

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.7, 5.1), sharex=True,
        gridspec_kw={"height_ratios": (2.25, 1.0)}
    )
    errors = []
    for m, component, color, marker, label in mode_specs:
        exact = _complex_sh_exact(m, q)
        exact_component = exact.real if component == "real" else exact.imag
        numerical_component = numerical[m].real if component == "real" else numerical[m].imag
        ax0.plot(q, exact_component, color=color, lw=1.5, label=label)
        ax0.plot(
            q, numerical_component, linestyle="none", marker=marker, color=color,
            ms=3.0, markevery=max(1, len(q)//14), fillstyle="none", label="_nolegend_",
        )
        scale = max(float(np.max(np.abs(exact))), 1.0e-15)
        error = np.abs(numerical[m] - exact) / scale
        errors.append(error)
        ax1.semilogy(q, np.maximum(error, 1.0e-10), color=color, lw=1.25, label=label)
    ax0.set_ylabel(r"$F_m(q)$ component")
    ax0.legend(frameon=True, framealpha=1.0, ncol=3, loc="upper right")
    ax1.set_xlabel(r"$q$")
    ax1.set_ylabel(r"$|\Delta F|/\max|F_{\rm ref}|$")
    _set_reference_error_limits(ax1, *errors, ceiling=1.0e-2, floor=1.0e-10)
    _save(fig, EXAMPLE_DIR / "complex_orbital_hankel.svg")


def _complex_reference_modes():
    return {
        "ss": (0,),
        "sh": (0, 1, -2),
        "hs": (0, -1, 2),
    }


def _complex_exact_transform(kind, m, q):
    if kind == "ss":
        if m != 0:
            raise KeyError(m)
        return np.exp(-q**2 / 4.0) / (2.0 * np.pi)
    if kind == "sh":
        return _complex_sh_exact(m, q)
    if kind == "hs":
        envelope = np.exp(-q**2 / 4.0)
        if m == 0:
            return q**2 * envelope / (8.0 * np.pi * np.sqrt(2.0))
        if m == -1:
            return 1j * q * envelope / (8.0 * np.pi)
        if m == 2:
            return q**2 * envelope / (16.0 * np.pi * np.sqrt(2.0))
    raise KeyError((kind, m))


def _bessel_parity(m, mp):
    return -1.0 if m > mp and (m - mp) % 2 != 0 else 1.0


def _complex_channel_reference(delta, first, second):
    q = np.linspace(1.0e-7, 14.0, 16001)
    kernel = 2.0 * np.pi * np.tanh(q) / q
    modes = _complex_reference_modes()
    delta = np.asarray(delta, dtype=float)
    bessel = {
        order: jv(order, np.outer(delta, q))
        for order in range(5)
    }
    total = np.zeros(delta.size, dtype=np.complex128)
    for m in modes[first]:
        f_m = _complex_exact_transform(first, m, q)
        for mp in modes[second]:
            f_mp = _complex_exact_transform(second, mp, q)
            base = q * kernel * f_m * np.conj(f_mp)
            radial = simpson(bessel[abs(m - mp)] * base[None, :], x=q, axis=1)
            total += 2.0 * np.pi * _bessel_parity(m, mp) * radial
    return total



def complex_orbital_channels() -> None:
    delta = np.linspace(0.0, 4.0, 81)
    exchange = _complex_channel_reference(delta, "sh", "sh")
    pair = _complex_channel_reference(delta, "sh", "hs")
    correlated = _complex_channel_reference(delta, "ss", "hs")

    q_kernel = np.geomspace(1.0e-3, 30.0, 500)
    kernel = 2.0 * np.pi * np.tanh(q_kernel) / q_kernel
    coulomb = 2.0 * np.pi / q_kernel
    channel_specs = (
        (exchange, "#0072B2", "-", "exchange"),
        (pair, "#D55E00", "--", "pair hopping"),
        (correlated, "#009E73", "-.", "correlated hopping"),
    )

    fig = plt.figure(figsize=(6.8, 6.8))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.65, 1.1))
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax1)

    ax0.loglog(q_kernel, coulomb, color="0.25", lw=1.3, label="Coulomb")
    ax0.loglog(q_kernel, kernel, color="#0072B2", lw=1.5, label="dual gate")
    ax0.axvline(1.0, **GUIDE_LINE_STYLE)
    ax0.set_xlabel(r"$q$")
    ax0.set_ylabel(r"$U(q)$")
    ax0.legend(frameon=True, framealpha=1.0, loc="upper right")

    for values, color, linestyle, label in channel_specs:
        ax1.plot(delta, np.abs(values), color=color, linestyle=linestyle, lw=1.5, label=label)
    ax1.set_ylabel(r"$|U_{1234}(\delta)|$")
    ax1.legend(frameon=True, framealpha=1.0, ncol=1, loc="upper right")

    for values, color, linestyle, _label in channel_specs:
        ax2.plot(delta, np.unwrap(np.angle(values)), color=color, linestyle=linestyle, lw=1.4)
    ax2.set_xlabel(r"$\delta$")
    ax2.set_ylabel("phase [rad]")
    _save(fig, EXAMPLE_DIR / "complex_orbital_channels.svg")



def _load(path: Path):
    return json.loads(path.read_text())


def _selector_status(record: dict) -> str:
    """Map a canonical convergence record to a presentation-level status."""
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


def end_to_end_accuracy(results_dir: Path) -> None:
    data = _load(results_dir / "pipeline.json")
    sections = {
        "standard": data["sections"]["end_to_end_standard_delta"],
        "large": data["sections"]["end_to_end_large_delta"],
    }
    case_labels = {
        "gaussian_coulomb": "Gaussian\nCoulomb",
        "anisotropic_rk_strong": "anisotropic\nRK",
        "nodal_rpa": "nodal\nRPA",
    }
    stage_specs = (
        ("PETAL2D", "#0072B2", "o"),
        ("HarmonicTransform", "#D55E00", "s"),
        ("Interaction", "#009E73", "^"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1), sharey=True)
    for ax, domain in zip(axes, ("standard", "large")):
        cases = sections[domain]["cases"]
        x = np.arange(len(cases), dtype=float)
        petal = [case["petal2d"]["relative_l2"] for case in cases]
        harmonic = [case["harmonic_transform"]["analytic_error"]["relative_l2"] for case in cases]
        interaction = [case["interactions"][0]["relative_l2"] for case in cases]
        for (label, color, marker), values in zip(stage_specs, (petal, harmonic, interaction)):
            ax.scatter(x, values, color=color, marker=marker, s=58, label=label, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x, [case_labels.get(case["name"], case["name"]) for case in cases])
        ax.set_title("standard displacement" if domain == "standard" else "large displacement")
        ax.grid(True, which="major", axis="y", alpha=0.20, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"relative $L^2$ reference error")
    axes[0].legend(frameon=True, framealpha=1.0, loc="upper left")
    _save(fig, VALIDATION_DIR / "end_to_end_accuracy.svg")


def large_delta_timing(results_dir: Path) -> None:
    data = _load(results_dir / "performance.json")
    rows = data["sections"]["qualified_fixed_configuration_large_delta_practical_tolerance"]["rows"]
    case_order = ["isotropic_coulomb", "anisotropic_rk_strong", "complex_gate", "nodal_rpa"]
    method_order = ["simpson", "gl4", "fftlog", "ogata"]
    labels = ["isotropic\nCoulomb", "anisotropic\nRK", "complex\ndual gate", "nodal\nRPA"]
    x = np.arange(len(case_order))
    offsets = np.linspace(-0.27, 0.27, len(method_order))

    fig = plt.figure(figsize=(7.4, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=(5.0, 0.72), hspace=0.04)
    ax = fig.add_subplot(gs[0])
    status_ax = fig.add_subplot(gs[1], sharex=ax)
    for offset, method in zip(offsets, method_order):
        style = method_style(method)
        for i, case in enumerate(case_order):
            match = [r for r in rows if r["case"] == case and r["method"] == method]
            xpos = x[i] + offset
            if match:
                ax.scatter(
                    xpos, match[0]["production_timing"]["median_seconds"],
                    s=52, color=style["color"], marker=style["marker"], zorder=3,
                )
            else:
                status_ax.scatter(
                    xpos, 0.0, marker=style["marker"], facecolors="none",
                    edgecolors=style["color"], s=48, linewidths=1.2,
                )
    ax.set_yscale("log")
    ax.set_ylabel("production time [s]")
    ax.grid(True, which="major", axis="y", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelbottom=False)

    status_ax.set_ylim(-0.5, 0.5)
    status_ax.set_yticks([0], ["not qualified"])
    status_ax.set_xticks(x, labels)
    status_ax.spines["top"].set_visible(False)
    status_ax.spines["right"].set_visible(False)
    status_ax.spines["left"].set_visible(False)
    status_ax.tick_params(axis="y", length=0, labelsize=8.0)
    status_ax.grid(False)

    handles = []
    for method in method_order:
        style = method_style(method, label=True)
        handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"],
                              linestyle="none", label=style["label"]))
    ax.legend(handles=handles, ncol=2, frameon=True, framealpha=1.0, loc="upper right")
    _save(fig, VALIDATION_DIR / "large_delta_timing.svg")


def interaction_scaling(results_dir: Path) -> None:
    data = _load(results_dir / "scaling.json")
    rows = data["sections"]["interaction"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for branch, method in (("finite", "gl4"), ("fftlog", "fftlog")):
        selected = [r for r in rows if r["branch"] == branch and r["axis"] == "N_D"]
        selected.sort(key=lambda r: r["N_D"])
        x = np.asarray([r["N_D"] for r in selected], dtype=float)
        y = np.asarray([r["median_seconds"] for r in selected], dtype=float)
        q25 = np.asarray([r["q25_seconds"] for r in selected], dtype=float)
        q75 = np.asarray([r["q75_seconds"] for r in selected], dtype=float)
        style = method_style(method, label=True)
        ax.errorbar(
            x, y, yerr=np.vstack((y - q25, q75 - y)),
            color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], capsize=2.2, label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of displacements $N_D$")
    ax.set_ylabel("median evaluation time [s]")
    ax.legend(frameon=True, framealpha=1.0, loc="upper left")
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "interaction_scaling.svg")

def benchmark_workloads() -> None:
    r = np.linspace(0.0, 8.0, 1200)
    groups = (
        (
            "smooth localized",
            {
                "Gaussian": np.exp(-r * r),
                "nodal Gaussian": (1.0 - r * r) * np.exp(-r * r),
            },
        ),
        (
            "difficult radial structure",
            {
                "cusp": np.exp(-r),
                "oscillatory exponential": np.exp(-0.35 * r) * np.cos(5.0 * r),
                "algebraic tail": (1.0 + r * r) ** -2.0,
            },
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.75), sharex=True, sharey=True)
    styles = ["-", "--", "-.", ":", (0, (5, 2))]
    colors = plt.get_cmap("tab10").colors
    style_index = 0
    for ax, (title, profiles) in zip(axes, groups):
        for label, values in profiles.items():
            ax.plot(r, values, label=label, color=colors[style_index],
                    linestyle=styles[style_index])
            style_index += 1
        ax.axhline(0.0, **ZERO_LINE_STYLE)
        ax.set_xlim(0.0, 8.0)
        ax.set_xlabel(r"$r$")
        ax.set_title(title)
        ax.legend(frameon=True, framealpha=1.0, loc="upper right")
        ax.grid(True, which="major", alpha=0.16, linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("dimensionless radial profile")
    _save(fig, VALIDATION_DIR / "benchmark_workloads.svg")

def benchmark_kernels() -> None:
    q = np.linspace(0.0, 6.0, 1000)
    stern = np.ones_like(q)
    above = q > 2.0
    stern[above] = 1.0 - np.sqrt(np.maximum(0.0, 1.0 - (2.0 / q[above]) ** 2))
    groups = (
        (
            "long-range and geometric screening",
            {
                "Coulomb": np.ones_like(q),
                r"RK $r_0=10$": 1.0 / (1.0 + 10.0 * q),
                r"dual gate $d=1$": np.tanh(q),
                r"interlayer $d=1$": np.exp(-q),
            },
        ),
        (
            "screened and nonanalytic kernels",
            {
                "Thomas-Fermi": q / (q + 1.0),
                r"Yukawa $\kappa=1$": q / np.sqrt(q * q + 1.0),
                "2DEG RPA": q / (q + stern),
            },
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.75), sharex=True, sharey=True)
    styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    colors = plt.get_cmap("tab10").colors
    style_index = 0
    for ax, (title, curves) in zip(axes, groups):
        for label, values in curves.items():
            ax.plot(q, values, label=label, color=colors[style_index],
                    linestyle=styles[style_index])
            style_index += 1
        ax.set_xlabel(r"$q$")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title)
        ax.legend(frameon=True, framealpha=1.0, loc="best")
        ax.grid(True, which="major", alpha=0.16, linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$qU(q)/(2\pi)$")
    _save(fig, VALIDATION_DIR / "benchmark_kernels.svg")

def harmonic_method_accuracy_cost(results_dir: Path) -> None:
    data = _load(results_dir / "harmonic.json")
    section = data["sections"]["finite_quadrature_method_matrix"]
    methods = ("trapezoid", "simpson", "gl4", "gl8")
    fig, ax = plt.subplots(figsize=(6.5, 4.35))
    for method in methods:
        x = []
        y = []
        for row in section["rows"]:
            record = row["methods"][method]
            x.append(record["timing"]["median_seconds"])
            y.append(record["worst_in_domain_relative_l2_total_norm"])
        style = method_style(method, label=True)
        ax.scatter(x, y, s=38, color=style["color"], marker=style["marker"],
                   alpha=0.85, label=style["label"], zorder=3)
    ax.axhline(1.0e-4, **CRITERION_LINE_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("transform time [s]")
    ax.set_ylabel(r"relative $L^2$ reference error")
    ax.legend(frameon=True, framealpha=1.0, loc="best", ncol=2)
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "harmonic_method_accuracy_cost.svg")



def interaction_method_accuracy_cost(results_dir: Path) -> None:
    data = _load(results_dir / "interaction_accuracy.json")
    rows = data["sections"]["primary_tolerance_method_matrix"]["rows"]
    methods = ("trapezoid", "simpson", "gl4", "gl8", "fftlog")
    fig, ax = plt.subplots(figsize=(6.7, 4.45))
    for method in methods:
        selected = [
            row for row in rows
            if row["method"] == method
            and row["requested_tolerance"] == 1.0e-4
            and row.get("reference_pass") is True
            and row.get("selected_reference_error") is not None
            and row.get("production_timing") is not None
        ]
        x = [row["production_timing"]["median_seconds"] for row in selected]
        y = [
            max(row["selected_reference_error"]["relative_l2"],
                row["selected_reference_error"]["relative_peak"])
            for row in selected
        ]
        style = method_style(method, label=True)
        ax.scatter(x, y, s=26, color=style["color"], marker=style["marker"],
                   alpha=0.72, label=style["label"], zorder=3)
    ax.axhline(1.0e-4, **CRITERION_LINE_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("qualified production time [s]")
    ax.set_ylabel("independent reference error")
    ax.legend(frameon=True, framealpha=1.0, loc="best", ncol=2)
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "interaction_method_accuracy_cost.svg")


def fftlog_workload_coverage(results_dir: Path) -> None:
    data = _load(results_dir / "interaction_accuracy.json")
    rows = data["sections"]["primary_tolerance_method_matrix"]["rows"]
    workloads = (
        "gaussian_isotropic", "gaussian_odd_pair", "gaussian_anisotropic_m2",
        "gaussian_high_order_m4", "nodal_mixed", "complex_mixed",
        "exponential_cusp", "oscillatory_exponential", "algebraic_mixed",
    )
    labels = (
        "Gaussian m=0", "odd pair", "anisotropic m=2", "high-order m=4",
        "nodal", "complex", "cusp", "oscillatory", "algebraic",
    )
    accepted = []
    tested = []
    for workload in workloads:
        selected = [row for row in rows if row["method"] == "fftlog" and row["workload"] == workload]
        tested.append(len(selected))
        accepted.append(sum(bool(row["automatic_converged"]) for row in selected))
    fractions = np.divide(accepted, tested, dtype=float)
    y = np.arange(len(labels))
    style = method_style("fftlog")
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.hlines(y, 0.0, fractions, color="0.78", linewidth=1.4, zorder=1)
    ax.scatter(fractions, y, color=style["color"], marker=style["marker"], s=48, zorder=3)
    for yi, frac, nacc, ntot in zip(y, fractions, accepted, tested):
        ax.text(min(1.02, frac + 0.035), yi, f"{nacc}/{ntot}", va="center", fontsize=8.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.03, 1.13)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("automatic acceptance fraction")
    ax.invert_yaxis()
    ax.grid(True, which="major", axis="x", alpha=0.18, linewidth=0.55)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "fftlog_workload_coverage.svg")

def fftlog_kernel_coverage(results_dir: Path) -> None:
    data = _load(results_dir / "interaction_accuracy.json")
    rows = data["sections"]["primary_tolerance_method_matrix"]["rows"]
    family_order = (
        "coulomb", "rytova_keldysh", "interlayer_coulomb", "gated_interlayer",
        "single_gate", "dual_gate", "thomas_fermi", "yukawa",
        "helmholtz_yukawa_2d", "static_2deg_rpa", "numerical_control",
    )
    labels = {
        "coulomb": "Coulomb", "dual_gate": "dual gate",
        "gated_interlayer": "gated interlayer", "helmholtz_yukawa_2d": "2D Helmholtz",
        "interlayer_coulomb": "interlayer", "numerical_control": "smooth control",
        "rytova_keldysh": "Rytova-Keldysh", "single_gate": "single gate",
        "static_2deg_rpa": "2DEG RPA", "thomas_fermi": "Thomas-Fermi",
        "yukawa": "Yukawa",
    }
    names = []
    accepted = []
    tested = []
    for family in family_order:
        selected = [row for row in rows if row["method"] == "fftlog" and row["kernel_family"] == family]
        if not selected:
            continue
        names.append(labels.get(family, family))
        tested.append(len(selected))
        accepted.append(sum(bool(row["automatic_converged"]) for row in selected))
    fractions = np.divide(accepted, tested, dtype=float)
    y = np.arange(len(names))
    style = method_style("fftlog")
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.hlines(y, 0.0, fractions, color="0.78", linewidth=1.4, zorder=1)
    ax.scatter(fractions, y, color=style["color"], marker=style["marker"], s=48, zorder=3)
    for yi, frac, nacc, ntot in zip(y, fractions, accepted, tested):
        ax.text(min(1.02, frac + 0.035), yi, f"{nacc}/{ntot}", va="center", fontsize=8.0)
    ax.set_yticks(y, names)
    ax.set_xlim(-0.03, 1.13)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("automatic acceptance fraction")
    ax.invert_yaxis()
    ax.grid(True, which="major", axis="x", alpha=0.18, linewidth=0.55)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "fftlog_kernel_coverage.svg")

def autoconvergence_standard_cost(results_dir: Path) -> None:
    data = _load(results_dir / "performance.json")
    rows = data["sections"]["automatic_convergence_cost_primary_tolerance"]["rows"]
    rows = [row for row in rows if row["stage"] == "Interaction" and row["delta_domain"] == "standard"]
    all_values = []
    fig, ax = plt.subplots(figsize=(6.5, 4.35))
    for method in ("simpson", "gl4", "fftlog"):
        selected = [
            row for row in rows
            if row["method"] == method and row["selected_production_timing"] is not None
        ]
        x = np.asarray([row["calibration_timing"]["median_seconds"] for row in selected], dtype=float)
        y = np.asarray([row["selected_production_timing"]["median_seconds"] for row in selected], dtype=float)
        all_values.extend(x.tolist())
        all_values.extend(y.tolist())
        style = method_style(method, label=True)
        ax.scatter(x, y, color=style["color"], marker=style["marker"], s=42,
                   alpha=0.82, label=style["label"], zorder=3)
    finite = np.asarray([value for value in all_values if np.isfinite(value)], dtype=float)
    lo = float(np.min(finite)) / 1.8
    hi = float(np.max(finite)) * 1.8
    ax.plot([lo, hi], [lo, hi], label="equal time", **GUIDE_LINE_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("automatic-selection time [s]")
    ax.set_ylabel("fixed-production time [s]")
    ax.legend(frameon=True, framealpha=1.0, loc="best", ncol=2)
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "autoconvergence_standard_cost.svg")


def autoconvergence_domain_cost(results_dir: Path) -> None:
    data = _load(results_dir / "performance.json")
    summary = data["sections"]["automatic_convergence_cost_primary_tolerance"]["summary"]
    methods = ("simpson", "gl4", "fftlog")
    values = {}
    for domain in ("standard", "large"):
        values[domain] = [
            next(row["median_calibration_seconds"] for row in summary
                 if row["stage"] == "Interaction" and row["method"] == method
                 and row["delta_domain"] == domain)
            for method in methods
        ]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.5, 4.35))
    for i in range(len(methods)):
        ax.plot([i, i], [values["standard"][i], values["large"][i]],
                color="0.75", linewidth=1.1, zorder=1)
    for domain in ("standard", "large"):
        style = domain_style(domain, label=True)
        ax.scatter(x, values[domain], color=style["color"], marker=style["marker"],
                   s=58, label=style["label"], zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, [method_label(method) for method in methods])
    ax.set_ylabel("median automatic-selection time [s]")
    ax.legend(frameon=True, framealpha=1.0, loc="upper left", ncol=1)
    ax.grid(True, which="major", axis="y", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "autoconvergence_domain_cost.svg")



def canonical_method_status(results_dir: Path) -> None:
    data = _load(results_dir / "interaction_convergence.json")
    rows = data["sections"]["primary_tolerance"]["rows"]
    cases = ("isotropic_coulomb", "anisotropic_rk_strong", "complex_gate", "nodal_rpa")
    case_labels = ("isotropic Coulomb", "anisotropic RK", "complex dual gate", "nodal RPA")
    methods = ("simpson", "gl4", "fftlog", "ogata")
    fig = plt.figure(figsize=(10.0, 4.1))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.9), wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    legend_ax = fig.add_subplot(grid[0, 2])
    for ax, domain in zip(axes, ("standard", "large")):
        for i, case in enumerate(cases):
            for j, method in enumerate(methods):
                record = next(row for row in rows
                              if row["delta_domain"] == domain
                              and row["case"] == case and row["method"] == method)
                status = _selector_status(record)
                style = status_style(status)
                kwargs = {"marker": style["marker"], "s": 76, "linewidths": 1.25, "zorder": 3}
                if style.get("fillstyle") == "none" and style["marker"] != "x":
                    kwargs.update(facecolors="none", edgecolors=style["color"])
                else:
                    kwargs.update(color=style["color"])
                ax.scatter(j, i, **kwargs)
        ax.set_xticks(range(len(methods)), [method_label(method) for method in methods])
        ax.set_xlim(-0.5, len(methods) - 0.5)
        ax.set_ylim(len(cases) - 0.5, -0.5)
        ax.set_title("standard displacement" if domain == "standard" else "large displacement")
        ax.grid(True, which="major", color="0.90", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(cases)), case_labels)
    axes[1].set_yticks(range(len(cases)), [])
    display_labels = {
        "accepted_reference_pass": "accepted + reference pass",
        "qualified_after_refusal": "qualified after refusal",
        "conservative_refusal": "unresolved refusal",
        "tested_box_not_capable": "no capable point in tested box",
    }
    handles = []
    for status, label in display_labels.items():
        style = status_style(status)
        kwargs = {"marker": style["marker"], "linestyle": "none", "label": label}
        if style.get("fillstyle") == "none" and style["marker"] != "x":
            kwargs.update(markerfacecolor="none", markeredgecolor=style["color"], color=style["color"])
        else:
            kwargs.update(color=style["color"])
        handles.append(Line2D([0], [0], **kwargs))
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, frameon=True, framealpha=1.0, loc="center left")
    _save(fig, VALIDATION_DIR / "canonical_method_status.svg")









def _rk_kernel(r0):
    r0 = float(r0)

    def kernel(q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                q > 0.0,
                2.0 * np.pi / (q * (1.0 + r0 * q)),
                np.inf,
            )

    return kernel


def _rk_multichannel_calculation():
    problem = _complex_orbital_problem()
    decompositions = {
        "ss": problem["dec_ss"],
        "sh": problem["dec_sh"],
        "hs": problem["dec_hs"],
    }
    harmonic_calibrations = {
        name: HarmonicTransform.converge_parameters(
            dec,
            rtol=1.0e-4,
            atol=1.0e-12,
            q_tail_rtol=1.0e-3,
            method="simpson",
            verbose=False,
        )
        for name, dec in decompositions.items()
    }
    fields = {
        name: harmonic_calibrations[name].transform(decompositions[name])
        for name in decompositions
    }
    channels = {
        "exchange": (fields["sh"], fields["sh"]),
        "pair hopping": (fields["sh"], fields["hs"]),
        "correlated hopping": (fields["ss"], fields["hs"]),
    }
    representatives = (0.1, 1.0, 10.0)
    calibration_delta = np.linspace(0.0, 4.0, 25)
    calibration_vectors = np.column_stack(
        (calibration_delta, np.zeros_like(calibration_delta))
    )
    calibrations = {}
    for name, (first, second) in channels.items():
        for r0 in representatives:
            calibrations[(name, r0)] = Interaction.converge_parameters(
                calibration_vectors,
                first,
                second,
                _rk_kernel(r0),
                rtol=1.0e-4,
                atol=1.0e-12,
                method="gl4",
                verbose=False,
            )

    subdivisions = max(
        result.parameters["subdivisions"] for result in calibrations.values()
    )
    production_parameters = {
        "method": "gl4",
        "interpolator": "cubic",
        "subdivisions": subdivisions,
    }
    r0_values = np.geomspace(0.1, 10.0, 31)
    delta = np.linspace(0.0, 4.0, 41)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    production = {
        name: np.empty((r0_values.size, delta.size), dtype=np.complex128)
        for name in channels
    }
    for i, r0 in enumerate(r0_values):
        kernel = _rk_kernel(r0)
        for name, (first, second) in channels.items():
            production[name][i] = Interaction(
                vectors,
                first,
                second,
                kernel,
                **production_parameters,
            ).V
    return representatives, calibrations, r0_values, delta, production



def rk_multiorbital_screening() -> None:
    _, _, r0_values, delta, production = _rk_multichannel_calculation()

    channel_order = ("exchange", "pair hopping", "correlated hopping")
    channel_labels = ("exchange", "pair hopping", "correlated hopping")
    magnitudes = [np.abs(production[name]).T for name in channel_order]
    positive = np.concatenate([values[values > 0.0] for values in magnitudes])
    global_max = max(float(np.max(values)) for values in magnitudes)
    global_min = max(float(np.min(positive)), global_max * 1.0e-5)
    norm = LogNorm(vmin=global_min, vmax=global_max)

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.6), sharex=True, sharey=True)
    image = None
    for ax, magnitude, label in zip(axes, magnitudes, channel_labels):
        image = ax.pcolormesh(
            r0_values, delta, magnitude, shading="auto", norm=norm, rasterized=True,
        )
        ax.set_xscale("log")
        ax.set_ylabel(r"$\delta$")
        ax.set_title(label, loc="left")
    axes[-1].set_xlabel(r"screening length $r_0$")
    fig.colorbar(image, ax=axes, shrink=0.88, pad=0.025, label=r"$|U_{1234}|$")
    _save(fig, EXAMPLE_DIR / "rk_multichannel_map.svg")

    cut_values = (0.0, 1.0, 2.0, 4.0)
    indices = [int(np.argmin(np.abs(delta - value))) for value in cut_values]
    cut_styles = (
        {"color": "#0072B2", "linestyle": "-"},
        {"color": "#D55E00", "linestyle": "--"},
        {"color": "#009E73", "linestyle": "-."},
        {"color": "#CC79A7", "linestyle": ":"},
    )
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.0), sharex=True)
    for ax, name, label in zip(axes, channel_order, channel_labels):
        for value, index, style in zip(cut_values, indices, cut_styles):
            ax.loglog(r0_values, np.abs(production[name][:, index]), lw=1.4, **style)
        ax.set_ylabel(r"$|U_{1234}|$")
        ax.set_title(label, loc="left")
    axes[-1].set_xlabel(r"screening length $r_0$")
    handles = [Line2D([0], [0], lw=1.4, label=rf"$\delta={value:g}$", **style) for value, style in zip(cut_values, cut_styles)]
    axes[0].legend(handles=handles, frameon=True, framealpha=1.0, ncol=2, loc="best")
    _save(fig, EXAMPLE_DIR / "rk_multichannel_cuts.svg")





def large_delta_methods() -> None:
    _, _, _, dec = _gaussian_end_to_end_problem()
    hcal = HarmonicTransform.converge_parameters(
        dec, rtol=1.0e-4, atol=1.0e-12, q_tail_rtol=1.0e-3,
        method="simpson", verbose=False,
    )
    field = hcal.transform(dec)

    def coulomb(q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(q > 0.0, 2.0 * np.pi / q, np.inf)

    delta = np.geomspace(1.0e2, 1.0e4, 81)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    exact = np.sqrt(np.pi / 2.0) * i0e(delta**2 / 4.0)
    results = {
        "gl4": Interaction(vectors, field, field, coulomb, method="gl4", subdivisions=256).V.real,
        "fftlog": Interaction(vectors, field, field, coulomb, method="fftlog", n=256, bias=0.0).V.real,
        "ogata": Interaction(vectors, field, field, coulomb, method="ogata", N=256, h=0.025).V.real,
    }

    fig, axes = plt.subplots(
        3, 1, figsize=(6.7, 6.9), sharex=True,
        gridspec_kw={"height_ratios": (2.25, 1.15, 1.15)}
    )
    ax0, ax1, ax2 = axes
    ax0.loglog(delta, exact, **REFERENCE_LINE_STYLE, label="analytic reference")
    ax0.loglog(delta, 1.0 / delta, label=r"$1/\delta$ asymptote", **GUIDE_LINE_STYLE)
    for method, values in results.items():
        style = method_style(method, label=True)
        ax0.loglog(delta, values, color=style["color"], marker=style["marker"],
                   linestyle="none", ms=3.2, markevery=8, fillstyle="none")
    ax0.set_ylabel(r"$V_{\rm C}(\delta)$")
    ax0.legend(frameon=True, framealpha=1.0, ncol=1, loc="lower left")

    reference_scale = max(float(np.max(np.abs(exact))), 1.0e-300)
    for method, values in results.items():
        style = method_style(method, label=True)
        error = np.abs(values - exact) / reference_scale
        ax1.loglog(delta, np.maximum(error, 1.0e-16), color=style["color"],
                   marker=style["marker"], linestyle=style["linestyle"],
                   ms=3.0, markevery=8, label=style["label"])
    ax1.set_ylabel(r"$|\Delta V|/\max|V_{\rm ref}|$")
    ax1.legend(frameon=True, framealpha=1.0, ncol=3, loc="best")

    finite_size = delta * exact - 1.0
    asymptotic_correction = 1.0 / (2.0 * delta**2)
    ax2.loglog(delta, finite_size, **REFERENCE_LINE_STYLE, label=r"$\delta V_{\rm ref}-1$")
    ax2.loglog(delta, asymptotic_correction, label=r"$1/(2\delta^2)$", **GUIDE_LINE_STYLE)
    ax2.set_xlabel(r"$\delta/\sigma$")
    ax2.set_ylabel(r"$\delta V_{\rm C}-1$")
    ax2.legend(frameon=True, framealpha=1.0, ncol=2, loc="upper right")
    _save(fig, EXAMPLE_DIR / "large_delta_methods.svg")




def interaction_memory_scaling(results_dir: Path) -> None:
    data = _load(results_dir / "memory.json")
    rows = data["sections"]["interaction"]["rows"]
    fig, ax = plt.subplots(figsize=(6.5, 4.35))
    for branch, method in (("gl4", "gl4"), ("fftlog", "fftlog")):
        selected = sorted(
            [row for row in rows if row["branch"] == branch and row["axis"] == "N_D"],
            key=lambda row: row["N_D"],
        )
        x = np.asarray([row["N_D"] for row in selected], dtype=float)
        y = np.asarray([row["incremental_peak_rss_median_bytes"] / 1024.0**2 for row in selected])
        q25 = np.asarray([row["incremental_peak_rss_q25_bytes"] / 1024.0**2 for row in selected])
        q75 = np.asarray([row["incremental_peak_rss_q75_bytes"] / 1024.0**2 for row in selected])
        style = method_style(method, label=True)
        ax.errorbar(
            x, y, yerr=np.vstack((y - q25, q75 - y)),
            color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], capsize=2.2, label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of displacements $N_D$")
    ax.set_ylabel("incremental peak RSS [MiB]")
    ax.legend(frameon=True, framealpha=1.0, loc="upper left")
    ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
    ax.set_axisbelow(True)
    _save(fig, VALIDATION_DIR / "interaction_memory_scaling.svg")

def generate_examples() -> None:
    """Regenerate the tracked tutorial/README figures without benchmark JSON."""
    gaussian_petal2d()
    gaussian_hankel_end_to_end()
    gaussian_interaction_end_to_end()
    direct_exchange()
    anisotropic_hankel()
    anisotropic_orientation()
    anisotropic_pair_contributions()
    sampled_data()
    complex_orbital_hankel()
    complex_orbital_channels()
    rk_multiorbital_screening()
    large_delta_methods()


def generate_validation(results_dir: Path) -> None:
    """Regenerate validation figures from one canonical publication bundle."""
    global ACTIVE_PUBLICATION_FINGERPRINT
    ACTIVE_PUBLICATION_FINGERPRINT, _ = publication_fingerprint(results_dir)

    benchmark_workloads()
    benchmark_kernels()
    harmonic_method_accuracy_cost(results_dir)
    interaction_method_accuracy_cost(results_dir)
    fftlog_workload_coverage(results_dir)
    fftlog_kernel_coverage(results_dir)
    autoconvergence_standard_cost(results_dir)
    autoconvergence_domain_cost(results_dir)
    canonical_method_status(results_dir)
    end_to_end_accuracy(results_dir)
    large_delta_timing(results_dir)
    interaction_scaling(results_dir)
    interaction_memory_scaling(results_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "group",
        nargs="?",
        choices=("examples", "validation", "all"),
        default="examples",
        help="artifact group to regenerate; default: examples",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "benchmarks" / "results",
        help="canonical consolidated publication result directory",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    group = "examples" if args.examples_only else args.group

    if group in {"examples", "all"}:
        generate_examples()
    if group in {"validation", "all"}:
        generate_validation(args.results_dir)


if __name__ == "__main__":
    main()
