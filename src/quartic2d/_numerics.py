"""Numerical integration and interpolation backends used by QUARTIC2D."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy import fft as scipy_fft
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import jv, roots_legendre

_RADIAL_METHODS = ("trapezoid", "simpson", "gl4", "gl8")
_INTERACTION_METHODS = ("fftlog", "trapezoid", "simpson", "gl4", "gl8")
_EXPERIMENTAL_RADIAL_METHODS = ("ogata",)
_EXPERIMENTAL_INTERACTION_METHODS = ("ogata",)
_INTERPOLATORS = ("linear", "cubic", "pchip")


def _validate_radial_method(method: str, *, allow_experimental: bool = False) -> str:
    method = str(method).lower()
    allowed_methods = _RADIAL_METHODS
    if allow_experimental:
        allowed_methods += _EXPERIMENTAL_RADIAL_METHODS
    if method not in allowed_methods:
        allowed = ", ".join(allowed_methods)
        raise ValueError(f"method must be one of: {allowed}.")
    return method


def validate_radial_method(method: str) -> str:
    """Return a normalized supported radial quadrature name."""
    return _validate_radial_method(method, allow_experimental=False)


def _validate_interaction_method(method: str, *, allow_experimental: bool = False) -> str:
    method = str(method).lower()
    allowed_methods = _INTERACTION_METHODS
    if allow_experimental:
        allowed_methods += _EXPERIMENTAL_INTERACTION_METHODS
    if method not in allowed_methods:
        allowed = ", ".join(allowed_methods)
        raise ValueError(f"method must be one of: {allowed}.")
    return method


def validate_interaction_method(method: str) -> str:
    """Return a normalized supported interaction-transform name."""
    return _validate_interaction_method(method, allow_experimental=False)


def _require_hankel():
    try:
        import hankel
    except ImportError as exc:
        raise ImportError(
            "The experimental Ogata backend requires the optional 'hankel' dependency. "
            "Install quartic2d[experimental]."
        ) from exc
    return hankel


def validate_interpolator(method: str) -> str:
    """Return a normalized interpolator name."""
    method = str(method).lower()
    if method not in _INTERPOLATORS:
        allowed = ", ".join(_INTERPOLATORS)
        raise ValueError(f"interpolator must be one of: {allowed}.")
    return method


class Interpolator1D:
    """One-dimensional real/complex interpolator with zero exterior values.

    ``cubic_bc_type`` is forwarded to :class:`scipy.interpolate.CubicSpline`.
    The default remains ``"natural"`` for backward-compatible radial-profile
    interpolation. Momentum-space form factors use ``"not-a-knot"`` at their
    call sites because a natural boundary condition generally imposes an
    incorrect second derivative at ``q=0``.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "cubic",
        *,
        cubic_bc_type: str = "natural",
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y)
        self.method = validate_interpolator(method)
        self.cubic_bc_type = cubic_bc_type

        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be one-dimensional.")
        if self.x.size != self.y.size:
            raise ValueError("x and y must have the same length.")
        if self.x.size < 2 or np.any(np.diff(self.x) <= 0.0):
            raise ValueError("x must contain at least two strictly increasing samples.")

        self._complex = np.iscomplexobj(self.y)
        if self.method == "cubic":
            self._real = CubicSpline(
                self.x, self.y.real, bc_type=self.cubic_bc_type, extrapolate=False
            )
            self._imag = (
                CubicSpline(
                    self.x, self.y.imag, bc_type=self.cubic_bc_type, extrapolate=False
                )
                if self._complex
                else None
            )
        elif self.method == "pchip":
            self._real = PchipInterpolator(self.x, self.y.real, extrapolate=False)
            self._imag = (
                PchipInterpolator(self.x, self.y.imag, extrapolate=False)
                if self._complex
                else None
            )
        else:
            self._real = None
            self._imag = None

    def __call__(self, z):
        z_arr = np.asarray(z, dtype=float)

        if self.method == "linear":
            real = np.interp(z_arr, self.x, self.y.real, left=0.0, right=0.0)
            if self._complex:
                imag = np.interp(
                    z_arr, self.x, self.y.imag, left=0.0, right=0.0
                )
                out = real + 1j * imag
            else:
                out = real
        else:
            real = self._real(z_arr)
            if self._complex:
                out = real + 1j * self._imag(z_arr)
            else:
                out = real
            out = np.where(np.isfinite(out), out, 0.0)

        return out.item() if z_arr.ndim == 0 else out


@lru_cache(maxsize=2)
def _gauss_legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order not in (4, 8):
        raise ValueError("Gauss-Legendre order must be 4 or 8.")
    nodes, weights = roots_legendre(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def composite_gauss_nodes_weights(x: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened Gauss-Legendre nodes and weights on every x interval."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("x must contain at least two strictly increasing samples.")
    nodes, weights = _gauss_legendre_rule(int(order))
    a = x[:-1]
    b = x[1:]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    z = mid[:, None] + half[:, None] * nodes[None, :]
    w = half[:, None] * weights[None, :]
    return z.reshape(-1), w.reshape(-1)


def subdivide_grid(x: np.ndarray, factor: int) -> np.ndarray:
    """Subdivide every interval of a strictly increasing grid by ``factor``."""
    x = np.asarray(x, dtype=float)
    factor = int(factor)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("x must contain at least two strictly increasing samples.")
    if factor < 1:
        raise ValueError("factor must be a positive integer.")
    if factor == 1:
        return x.copy()

    offsets = np.arange(factor, dtype=float) / factor
    intervals = x[:-1, None] + np.diff(x)[:, None] * offsets[None, :]
    return np.concatenate((intervals.reshape(-1), x[-1:]))


def integrate_function_on_grid(
    x: np.ndarray,
    func: Callable,
    method: str,
) -> complex:
    """Integrate a callable over a finite grid using the selected backend."""
    x = np.asarray(x, dtype=float)
    method = validate_radial_method(method)

    if method == "trapezoid":
        return trapezoid(func(x), x=x)
    if method == "simpson":
        return simpson(func(x), x=x)
    if method in ("gl4", "gl8"):
        order = 4 if method == "gl4" else 8
        z, w = composite_gauss_nodes_weights(x, order)
        return np.sum(w * func(z))
    raise ValueError("Ogata integration requires a Bessel-transform context.")


def _finite_hankel_many(
    r: np.ndarray,
    values: np.ndarray,
    q: np.ndarray,
    order: int,
    method: str,
    interpolator: str,
    batch_size: int = 256,
    subdivisions: int = 1,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    values = np.asarray(values)
    q = np.asarray(q, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != r.size:
        raise ValueError("values must have shape (n_profiles, r.size).")
    order = int(order)
    subdivisions = int(subdivisions)
    if subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer.")

    interps = [Interpolator1D(r, profile, interpolator) for profile in values]
    panel_grid = subdivide_grid(r, subdivisions)
    dtype = np.result_type(values, np.complex128)
    out = np.empty((values.shape[0], q.size), dtype=dtype)

    if method in ("trapezoid", "simpson"):
        sample_x = panel_grid
        weighted_values = np.stack(
            [sample_x * interp(sample_x) for interp in interps], axis=0
        )
        integrate = trapezoid if method == "trapezoid" else simpson
        for start in range(0, q.size, batch_size):
            qb = q[start : start + batch_size]
            bessel = jv(order, np.outer(qb, sample_x))
            for profile_index, weighted in enumerate(weighted_values):
                out[profile_index, start : start + batch_size] = integrate(
                    bessel * weighted[None, :], x=sample_x, axis=1
                )
        return out

    if method in ("gl4", "gl8"):
        gl_order = 4 if method == "gl4" else 8
        z, w = composite_gauss_nodes_weights(panel_grid, gl_order)
        weighted_values = np.stack(
            [w * z * interp(z) for interp in interps], axis=0
        )
        for start in range(0, q.size, batch_size):
            qb = q[start : start + batch_size]
            bessel = jv(order, np.outer(qb, z))
            out[:, start : start + batch_size] = (bessel @ weighted_values.T).T
        return out

    raise ValueError(method)


def _finite_hankel(
    r: np.ndarray,
    values: np.ndarray,
    q: np.ndarray,
    order: int,
    method: str,
    interpolator: str,
    batch_size: int = 256,
    subdivisions: int = 1,
) -> np.ndarray:
    result = _finite_hankel_many(
        r,
        np.asarray(values)[None, :],
        q,
        order,
        method,
        interpolator,
        batch_size=batch_size,
        subdivisions=subdivisions,
    )
    return result[0]

def hankel_transform_sampled(
    r: np.ndarray,
    values: np.ndarray,
    q: np.ndarray,
    order: int,
    *,
    method: str,
    interpolator: str,
    ogata_N: int = 2048,
    ogata_h: float | None = None,
    ogata_object: object | None = None,
    subdivisions: int = 1,
    _allow_experimental: bool = False,
) -> np.ndarray:
    """Evaluate a Hankel transform of sampled finite-support data."""
    r = np.asarray(r, dtype=float)
    values = np.asarray(values)
    q = np.asarray(q, dtype=float)
    method = _validate_radial_method(method, allow_experimental=_allow_experimental)
    interpolator = validate_interpolator(interpolator)
    order = int(order)

    if method != "ogata":
        return _finite_hankel(
            r, values, q, order, method, interpolator,
            subdivisions=subdivisions,
        )

    hankel = _require_hankel()
    interp = Interpolator1D(r, values, interpolator)
    h = np.pi / int(ogata_N) if ogata_h is None else float(ogata_h)
    ht = ogata_object or hankel.HankelTransform(nu=order, N=int(ogata_N), h=h)

    out = np.zeros(q.size, dtype=np.result_type(values, np.complex128))
    zero = np.isclose(q, 0.0)
    nonzero = ~zero
    if np.any(nonzero):
        out[nonzero] = ht.transform(interp, q[nonzero], ret_err=False)
    if np.any(zero) and order == 0:
        # The sampled-data problem is finite by construction.  GL8 avoids an
        # unnecessary infinite-interval adaptive special case at q=0.
        out[zero] = integrate_function_on_grid(
            r, lambda x: x * interp(x), method="gl8"
        )
    return out



def _fftlog_hankel_transform_many(
    func: Callable,
    n_profiles: int,
    deltas: np.ndarray,
    order: int,
    *,
    native_q_max: float,
    native_q_step: float,
    n: int = 512,
    bias: float = -0.5,
    interpolator: str = "cubic",
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate multiple FFTLog transforms on one logarithmic grid."""
    deltas = np.asarray(deltas, dtype=float)
    order = int(order)
    n = int(n)
    n_profiles = int(n_profiles)
    bias = float(bias)
    interpolator = validate_interpolator(interpolator)

    if deltas.ndim != 1 or np.any(deltas <= 0.0) or np.any(~np.isfinite(deltas)):
        raise ValueError("FFTLog requires finite strictly positive displacements.")
    if n < 16:
        raise ValueError("FFTLog n must be at least 16.")
    if n_profiles < 1:
        raise ValueError("n_profiles must be a positive integer.")
    if not np.isfinite(native_q_max) or native_q_max <= 0.0:
        raise ValueError("native_q_max must be finite and positive.")
    if not np.isfinite(native_q_step) or native_q_step <= 0.0:
        raise ValueError("native_q_step must be finite and positive.")

    dmin = float(np.min(deltas))
    dmax = float(np.max(deltas))
    qmin = min(native_q_step * 1.0e-3, 1.0e-4 / dmax)
    qmin = max(qmin, np.finfo(float).tiny ** 0.25)
    qmax = max(native_q_max, 10.0 / dmin)

    q = np.geomspace(qmin, qmax, n)
    dln = float(np.log(q[1] / q[0]))
    qcenter = float(np.sqrt(qmin * qmax))
    dcenter = float(np.sqrt(dmin * dmax))
    initial = float(np.log(qcenter * dcenter))
    offset = float(
        scipy_fft.fhtoffset(dln, mu=order, initial=initial, bias=bias)
    )
    dgrid = np.exp(offset) / q[::-1]

    a = np.zeros((n_profiles, n), dtype=np.complex128)
    inside = q <= native_q_max
    qi = q[inside]
    sampled = np.asarray(func(qi), dtype=np.complex128)
    if sampled.shape != (n_profiles, qi.size):
        raise ValueError("func(q) must return shape (n_profiles, q.size).")
    a[:, inside] = qi[None, :] * sampled

    A = scipy_fft.fht(a.real, dln, mu=order, offset=offset, bias=bias)
    A = A + 1j * scipy_fft.fht(a.imag, dln, mu=order, offset=offset, bias=bias)
    H = A / dgrid[None, :]

    logd = np.log(dgrid)
    target = np.log(deltas)
    if np.min(target) < logd[0] or np.max(target) > logd[-1]:
        raise ValueError("Requested displacement lies outside the FFTLog output grid.")

    if interpolator == "linear":
        out = np.stack(
            [
                np.interp(target, logd, row.real)
                + 1j * np.interp(target, logd, row.imag)
                for row in H
            ],
            axis=0,
        )
    elif interpolator == "pchip":
        real = PchipInterpolator(logd, H.real.T, axis=0, extrapolate=False)(target).T
        imag = PchipInterpolator(logd, H.imag.T, axis=0, extrapolate=False)(target).T
        out = real + 1j * imag
    else:
        real = CubicSpline(logd, H.real.T, axis=0, extrapolate=False)(target).T
        imag = CubicSpline(logd, H.imag.T, axis=0, extrapolate=False)(target).T
        out = real + 1j * imag

    return np.asarray(out, dtype=np.complex128), {
        "qmin": float(qmin),
        "qmax": float(qmax),
        "offset": float(offset),
        "delta_min": float(dgrid[0]),
        "delta_max": float(dgrid[-1]),
    }


def fftlog_hankel_transform(
    func: Callable,
    deltas: np.ndarray,
    order: int,
    *,
    native_q_max: float,
    native_q_step: float,
    n: int = 512,
    bias: float = -0.5,
    interpolator: str = "cubic",
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate ``integral q*func(q)*J_order(q*delta) dq`` with SciPy FFTLog."""

    def sampled(q: np.ndarray) -> np.ndarray:
        return np.asarray(func(q), dtype=np.complex128)[None, :]

    values, metadata = _fftlog_hankel_transform_many(
        sampled,
        1,
        deltas,
        order,
        native_q_max=native_q_max,
        native_q_step=native_q_step,
        n=n,
        bias=bias,
        interpolator=interpolator,
    )
    return values[0], metadata


def common_grid(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Return the sorted union of two grids on their common finite interval."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    upper = min(float(x1[-1]), float(x2[-1]))
    values = np.concatenate((x1[x1 <= upper], x2[x2 <= upper]))
    grid = np.unique(values)
    if grid.size < 2:
        raise ValueError("The two grids do not share a usable integration interval.")
    return grid


def extrapolate_q_times_kernel_zero(U_q: Callable, first_positive_q: float) -> complex:
    """Estimate lim(q->0+) q*U(q) without evaluating U at q=0."""
    scale = float(first_positive_q)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("first_positive_q must be finite and positive.")

    q = scale * np.array([0.125, 0.25, 0.5, 0.75], dtype=float)
    y = q * np.asarray(U_q(q))
    if np.any(~np.isfinite(y)):
        raise ValueError("q*U_q(q) is not finite near q=0.")

    # A low-order polynomial is used only to supply endpoint values to
    # Newton-Cotes rules.  Gauss-Legendre methods never call this function.
    deg = min(2, q.size - 1)
    if np.iscomplexobj(y):
        cr = np.polyfit(q, y.real, deg)
        ci = np.polyfit(q, y.imag, deg)
        return complex(np.polyval(cr, 0.0), np.polyval(ci, 0.0))
    coeff = np.polyfit(q, y, deg)
    return np.polyval(coeff, 0.0)
