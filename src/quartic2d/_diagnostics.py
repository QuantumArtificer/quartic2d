"""Fast numerical sanity checks for momentum-space harmonic transforms.

The checks in this module are deliberately cheap.  They inspect quantities that
already exist after a transform has been computed and never launch a second
Hankel transform.  They are fault detectors, not convergence proofs: a warning
means that a numerical choice deserves attention, while the explicit
convergence helpers perform the expensive verification when requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.integrate import trapezoid


# These defaults were calibrated against the analytic harmonic-transform benchmark suite.
# They are intentionally conservative for smooth localized profiles.  The fast
# diagnostics below catch the classes for which a one-shot estimate is not
# reliable (for example cusps and discontinuities with slowly decaying spectra).
_DEFAULT_Q_RMS_MULTIPLIER = 4.0
_DEFAULT_Q_OVERSAMPLING = 5.0
_DEFAULT_MIN_NQ = 32
_DEFAULT_MAX_NQ = 512
_DEFAULT_Q_CEILING_FRACTION = 0.8

_BOUNDARY_AMPLITUDE_WARNING = 1.0e-2
_BOUNDARY_POWER_WARNING = 1.0e-4
_BOUNDARY_AMPLITUDE_WINDOW = 0.05
_BOUNDARY_POWER_WINDOW = 0.10
_DEFAULT_SAMPLING_WARNING = 5.0


class Quartic2DNumericalWarning(UserWarning):
    """Warning category for cheap numerical fault detectors."""


@dataclass(frozen=True)
class DiagnosticIssue:
    """One numerical condition that may make a transform unreliable."""

    code: str
    message: str
    harmonics: tuple[int, ...] = ()


@dataclass
class HarmonicTransformDiagnostics:
    """Result of the inexpensive post-transform sanity checks.

    Notes
    -----
    These diagnostics are designed to answer a practical question: "does
    anything about this result look obviously under-resolved?"  They do not
    certify a requested error tolerance.  Use
    :meth:`quartic2d.HarmonicTransform.converge_parameters` when a quantitative
    convergence certificate is needed.
    """

    q_ceiling: float
    q_max: float
    n_q: int
    dq: float
    q_sampling_factor: float
    per_harmonic: dict[int, dict[str, float]] = field(default_factory=dict)
    issues: list[DiagnosticIssue] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """Whether none of the fast fault detectors raised a concern."""
        return not self.issues

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "q_ceiling": float(self.q_ceiling),
            "q_max": float(self.q_max),
            "n_q": int(self.n_q),
            "dq": float(self.dq),
            "q_sampling_factor": float(self.q_sampling_factor),
            "per_harmonic": {
                str(m): {key: float(value) for key, value in values.items()}
                for m, values in self.per_harmonic.items()
            },
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "harmonics": list(issue.harmonics),
                }
                for issue in self.issues
            ],
        }

    def warning_message(self) -> str:
        """Return one compact user-facing warning for all detected issues."""
        if self.healthy:
            return ""
        lines = ["QUARTIC2D detected possible harmonic-transform numerical under-resolution:"]
        lines.extend(f"- {issue.message}" for issue in self.issues)
        lines.append(
            "The transform was returned, but these checks are indicators rather "
            "than error estimates. Run HarmonicTransform.converge_parameters(...) "
            "to obtain parameters verified against a requested tolerance."
        )
        return "\n".join(lines)


def radial_q_ceiling(r: np.ndarray) -> float:
    """Largest momentum scale supported by the supplied radial sample spacing.

    For a radial grid with largest spacing ``dr``, ``pi / dr`` is the Nyquist
    momentum associated with that sampling.  It is a ceiling, not a recommended
    ``q_max``: useful form factors normally decay well before this value.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size < 2 or np.any(~np.isfinite(r)) or np.any(np.diff(r) <= 0.0):
        raise ValueError("The PETAL2D radial grid must be finite and strictly increasing.")
    return float(np.pi / np.max(np.diff(r)))


def _mode_arrays(decomposition, m: int) -> tuple[np.ndarray, np.ndarray, float]:
    r = np.asarray(decomposition.r, dtype=float)
    cutoff = float(decomposition.cutoff_radius[int(m)])
    mask = r <= cutoff
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"cutoff_radius[{m}] retains fewer than two radial samples.")
    return r[mask], np.asarray(decomposition[int(m)])[mask], float(r[mask][-1])


def _rms_momentum_scale(r: np.ndarray, rho: np.ndarray, m: int) -> float:
    r"""Estimate the natural momentum scale of one radial harmonic.

    Hankel Parseval identities give

    ``<q^2> = integral r (|d rho/dr|^2 + m^2 |rho|^2/r^2) dr / integral r |rho|^2 dr``.

    The square root therefore measures the typical momentum carried by the
    sampled radial profile without performing a Hankel transform.  It is used
    only to choose a conservative default q range; convergence remains an
    explicit opt-in operation.
    """
    rho = np.asarray(rho)
    power = float(np.real(trapezoid(r * np.abs(rho) ** 2, x=r)))
    if not np.isfinite(power) or power <= 0.0:
        return 0.0

    derivative = np.gradient(rho, r, edge_order=2 if r.size >= 3 else 1)
    density = r * np.abs(derivative) ** 2
    if int(m) != 0:
        centrifugal = np.zeros_like(r, dtype=float)
        centrifugal[1:] = (int(m) ** 2) * np.abs(rho[1:]) ** 2 / r[1:]
        density = density + centrifugal
    gradient_power = float(np.real(trapezoid(density, x=r)))
    if not np.isfinite(gradient_power) or gradient_power <= 0.0:
        return 0.0
    return float(math.sqrt(gradient_power / power))


def default_q_grid(decomposition) -> tuple[float, int, dict[str, float]]:
    """Choose the inexpensive default q range and q-grid size.

    The q range is estimated from the largest RMS momentum scale among the
    retained PETAL2D harmonics.  Four RMS widths are used by default, then the
    estimate is capped below the radial Nyquist ceiling.  The factor four is
    calibrated against the analytic validation suite for a default omitted-q
    L2 target of 1e-3 when the fast support checks pass.  This rule is
    scale-aware: rescaling the real-space orbital rescales the suggested
    momentum range automatically.

    The number of q samples aims for five samples per Nyquist interval of the
    shortest oscillation implied by the retained radial support.  This density
    keeps both relative L2 and relative maximum interpolation errors below
    1e-4 on the non-discontinuous validation suite.  The grid is kept between
    32 and 512 points for predictable memory and runtime.  The subsequent fast
    fault detectors warn if these defaults look inadequate for a particular
    profile.  Exact tolerance certification is intentionally left to the
    explicit convergence helper.
    """
    m_values = [int(m) for m in decomposition.m_sorted]
    if not m_values:
        raise ValueError("decomposition contains no retained harmonics.")
    r = np.asarray(decomposition.r, dtype=float)
    q_ceiling = radial_q_ceiling(r)

    q_rms_values = []
    supports = []
    for m in m_values:
        r_m, rho_m, support = _mode_arrays(decomposition, m)
        supports.append(support)
        q_rms_values.append(_rms_momentum_scale(r_m, rho_m, m))

    q_rms = max(q_rms_values)
    if not np.isfinite(q_rms) or q_rms <= 0.0:
        # A constant m=0 profile has zero gradient energy.  Use the reciprocal
        # support as a finite, scale-aware fallback rather than a dimensionful
        # magic number.
        support = max(supports)
        q_rms = 1.0 / support

    q_max_unclipped = _DEFAULT_Q_RMS_MULTIPLIER * q_rms
    q_max = min(q_max_unclipped, _DEFAULT_Q_CEILING_FRACTION * q_ceiling)
    support_radius = max(supports)
    predicted = 1 + int(
        math.ceil(_DEFAULT_Q_OVERSAMPLING * q_max * support_radius / np.pi)
    )
    n_q = min(_DEFAULT_MAX_NQ, max(_DEFAULT_MIN_NQ, predicted))
    return float(q_max), int(n_q), {
        "q_ceiling": float(q_ceiling),
        "q_rms": float(q_rms),
        "q_max_unclipped": float(q_max_unclipped),
        "support_radius": float(support_radius),
        "predicted_n_q": int(predicted),
    }


def inspect_transform(decomposition, q: np.ndarray, F_q: dict[int, np.ndarray]) -> HarmonicTransformDiagnostics:
    """Run cheap fault detectors using only the already-computed transform.

    No extra Hankel transform, interpolation study, or quadrature sweep is
    performed.  The work is linear in the number of values already stored in
    ``F_q`` and is normally negligible compared with evaluating the Bessel
    integrals themselves.
    """
    q = np.asarray(q, dtype=float)
    q_max = float(q[-1])
    dq = float(np.max(np.diff(q)))
    q_ceiling = radial_q_ceiling(np.asarray(decomposition.r, dtype=float))
    support_radius = max(float(decomposition.cutoff_radius[int(m)]) for m in decomposition.m_sorted)
    sampling_factor = float(np.pi / (dq * support_radius)) if support_radius > 0 else math.inf

    issues: list[DiagnosticIssue] = []
    per_harmonic: dict[int, dict[str, float]] = {}

    if q_max >= 0.9 * q_ceiling:
        issues.append(
            DiagnosticIssue(
                "q_near_radial_ceiling",
                f"q_max={q_max:.4g} is close to the radial-sampling ceiling "
                f"pi/dr={q_ceiling:.4g}. Results near the upper q boundary may "
                "depend on the PETAL2D radial resolution.",
            )
        )

    if sampling_factor < 1.0:
        issues.append(
            DiagnosticIssue(
                "q_grid_below_nyquist",
                f"The q grid is too coarse for the retained radial support "
                f"(sampling factor={sampling_factor:.2f}, below the Nyquist lower "
                "bound of 1). Increase n_q or run convergence.",
            )
        )
    elif sampling_factor < _DEFAULT_SAMPLING_WARNING:
        issues.append(
            DiagnosticIssue(
                "q_grid_below_default_accuracy_density",
                f"The q grid has sampling factor={sampling_factor:.2f}. The "
                f"benchmarked default density is {_DEFAULT_SAMPLING_WARNING:.0f}, "
                "which kept q-space interpolation errors below 1e-4 on the "
                "non-discontinuous validation suite. Increase n_q or run "
                "convergence if that accuracy is required.",
            )
        )

    r = np.asarray(decomposition.r, dtype=float)
    dr = float(np.max(np.diff(r)))
    boundary_modes = []
    for m_raw in decomposition.m_sorted:
        m = int(m_raw)
        if float(decomposition.cutoff_radius[m]) >= float(r[-1] - 1.5 * dr):
            boundary_modes.append(m)
    if boundary_modes:
        issues.append(
            DiagnosticIssue(
                "radial_support_at_boundary",
                "PETAL2D reports retained radial support reaching the edge of the "
                f"available real-space interval for harmonic(s) {boundary_modes}. "
                "The input radial domain may be too short.",
                tuple(boundary_modes),
            )
        )

    support_modes = []
    for m_raw in decomposition.m_sorted:
        m = int(m_raw)
        values = np.asarray(F_q[m])
        peak = float(np.max(np.abs(values)))
        amp_start = max(0, int(math.floor((1.0 - _BOUNDARY_AMPLITUDE_WINDOW) * q.size)))
        power_start = max(0, int(math.floor((1.0 - _BOUNDARY_POWER_WINDOW) * q.size)))
        boundary_amplitude_ratio = (
            0.0 if peak == 0.0 else float(np.max(np.abs(values[amp_start:])) / peak)
        )
        density = q * np.abs(values) ** 2
        total_power = float(np.real(trapezoid(density, x=q)))
        boundary_power_fraction = (
            0.0
            if total_power <= 0.0
            else float(
                np.real(trapezoid(density[power_start:], x=q[power_start:]))
                / total_power
            )
        )
        per_harmonic[m] = {
            "boundary_amplitude_ratio": boundary_amplitude_ratio,
            "boundary_power_fraction": boundary_power_fraction,
        }
        if (
            boundary_amplitude_ratio > _BOUNDARY_AMPLITUDE_WARNING
            or boundary_power_fraction > _BOUNDARY_POWER_WARNING
        ):
            support_modes.append(m)

    if support_modes:
        details = ", ".join(
            f"m={m}: boundary amplitude={per_harmonic[m]['boundary_amplitude_ratio']:.2e}, "
            f"boundary power={per_harmonic[m]['boundary_power_fraction']:.2e}"
            for m in support_modes
        )
        issues.append(
            DiagnosticIssue(
                "q_support_not_decayed",
                "The transformed profile still carries noticeable weight near "
                f"q_max ({details}). 'Boundary amplitude' is the largest |F_m| "
                "in the final 5% of the sampled q range divided by that harmonic's "
                "peak. 'Boundary power' is the fraction of the represented "
                "q-weighted L2 norm lying in the final 10% of the sampled range. "
                "Neither is an omitted-tail error estimate; they are fast signs "
                "that q_max may be too small.",
                tuple(support_modes),
            )
        )

    return HarmonicTransformDiagnostics(
        q_ceiling=q_ceiling,
        q_max=q_max,
        n_q=int(q.size),
        dq=dq,
        q_sampling_factor=sampling_factor,
        per_harmonic=per_harmonic,
        issues=issues,
    )
