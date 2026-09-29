"""Momentum-support and sampling convergence for PETAL2D harmonic transforms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_simpson, simpson

from ._convergence import absolute_linf_error, relative_l2_error, relative_linf_error
from ._numerics import Interpolator1D, _finite_hankel_many, subdivide_grid


@dataclass
class QTailModeResult:
    """Momentum-tail diagnostic for one retained angular harmonic."""

    m: int
    q_required: float | None
    tail_error_at_q_max: float
    parseval_relative_error: float
    resolved: bool

    def to_dict(self) -> dict:
        return {
            "m": int(self.m),
            "q_required": None if self.q_required is None else float(self.q_required),
            "tail_error_at_q_max": float(self.tail_error_at_q_max),
            "parseval_relative_error": float(self.parseval_relative_error),
            "resolved": bool(self.resolved),
        }


@dataclass
class QInterpolationStep:
    """One q-grid interpolation refinement level."""

    oversampling: float
    n_q: int
    harmonic_errors: dict[int, dict[str, float]]
    converged: bool
    grid_kind: str = "uniform"
    refined_intervals: int = 0
    min_dq: float | None = None
    max_dq: float | None = None

    def to_dict(self) -> dict:
        return {
            "oversampling": float(self.oversampling),
            "n_q": int(self.n_q),
            "harmonic_errors": {
                str(m): {key: float(value) for key, value in errors.items()}
                for m, errors in self.harmonic_errors.items()
            },
            "converged": bool(self.converged),
            "grid_kind": str(self.grid_kind),
            "refined_intervals": int(self.refined_intervals),
            "min_dq": None if self.min_dq is None else float(self.min_dq),
            "max_dq": None if self.max_dq is None else float(self.max_dq),
        }


@dataclass
class QSamplingResult:
    """Automatic q-support and q-grid convergence result."""

    q_ceiling: float
    q_max: float
    n_q: int
    support_radius: float
    q_tail_rtol: float
    interpolation_rtol: float
    interpolation_atol: float
    tail_pilot_oversampling: float
    interpolation_oversampling: float
    tail_modes: dict[int, QTailModeResult]
    q_grid: np.ndarray | None = None
    grid_kind: str = "uniform"
    interpolation_steps: list[QInterpolationStep] = field(default_factory=list)
    tail_converged: bool = True
    interpolation_converged: bool = True

    @property
    def converged(self) -> bool:
        return bool(self.tail_converged and self.interpolation_converged)

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "q_ceiling": float(self.q_ceiling),
            "q_max": float(self.q_max),
            "n_q": int(self.n_q),
            "support_radius": float(self.support_radius),
            "q_tail_rtol": float(self.q_tail_rtol),
            "interpolation_rtol": float(self.interpolation_rtol),
            "interpolation_atol": float(self.interpolation_atol),
            "tail_pilot_oversampling": float(self.tail_pilot_oversampling),
            "interpolation_oversampling": float(self.interpolation_oversampling),
            "tail_converged": bool(self.tail_converged),
            "interpolation_converged": bool(self.interpolation_converged),
            "tail_modes": {str(m): result.to_dict() for m, result in self.tail_modes.items()},
            "grid_kind": str(self.grid_kind),
            "q_grid": None if self.q_grid is None else [float(x) for x in np.asarray(self.q_grid)],
            "interpolation_steps": [step.to_dict() for step in self.interpolation_steps],
        }


def _mode_support(decomposition, m: int) -> tuple[np.ndarray, np.ndarray, float]:
    r = np.asarray(decomposition.r, dtype=float)
    cutoff = float(decomposition.cutoff_radius[int(m)])
    mask = r <= cutoff
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"cutoff_radius[{m}] retains fewer than two radial samples.")
    return r[mask], np.asarray(decomposition[int(m)])[mask], float(r[mask][-1])


def radial_q_ceiling(decomposition) -> float:
    """Return the conservative radial-sampling ceiling ``pi / max(diff(r))``."""
    r = np.asarray(decomposition.r, dtype=float)
    if r.ndim != 1 or r.size < 2 or np.any(~np.isfinite(r)) or np.any(np.diff(r) <= 0):
        raise ValueError("decomposition.r must be finite and strictly increasing.")
    return float(np.pi / np.max(np.diff(r)))


def _represented_radial_power(
    r: np.ndarray,
    values: np.ndarray,
    *,
    interpolator: str,
    subdivisions: int,
) -> float:
    interp = Interpolator1D(r, values, interpolator)
    grid = subdivide_grid(r, max(8, 2 * int(subdivisions)))
    return float(simpson(grid * np.abs(interp(grid)) ** 2, x=grid))


def _grouped_hankel_transforms(
    supports: dict[int, tuple[np.ndarray, np.ndarray, float]],
    q: np.ndarray,
    *,
    method: str,
    interpolator: str,
    subdivisions: int,
) -> dict[int, np.ndarray]:
    groups: dict[tuple[int, bytes], list[int]] = {}
    for m, (r, _, _) in supports.items():
        key = (abs(int(m)), np.asarray(r, dtype=float).tobytes())
        groups.setdefault(key, []).append(int(m))

    transforms: dict[int, np.ndarray] = {}
    for (order, _), modes in groups.items():
        r = supports[modes[0]][0]
        profiles = np.stack([supports[m][1] for m in modes], axis=0)
        values = _finite_hankel_many(
            r,
            profiles,
            q,
            order,
            method,
            interpolator,
            subdivisions=int(subdivisions),
        )
        for index, m in enumerate(modes):
            transforms[m] = values[index]
    return transforms




class _DirectHankelCache:
    """Cache direct Hankel evaluations at arbitrary q values.

    Convergence studies repeatedly revisit the same dyadic support and
    interpolation points.  The cache is purely an execution optimization: it
    does not interpolate or approximate missing values.  Every uncached q point
    is evaluated by the requested Hankel backend exactly as before.
    """

    def __init__(
        self,
        supports: dict[int, tuple[np.ndarray, np.ndarray, float]],
        *,
        method: str,
        interpolator: str,
        subdivisions: int,
    ):
        self.supports = supports
        self.method = method
        self.interpolator = interpolator
        self.subdivisions = int(subdivisions)
        self._values: dict[int, dict[str, complex]] = {
            int(m): {} for m in supports
        }

    @staticmethod
    def _key(value: float) -> str:
        # Fifteen significant decimal digits collapse only round-off-level
        # differences between mathematically identical nested-grid points.
        # This is many orders of magnitude below every supported convergence
        # tolerance and avoids missing reuse solely because np.linspace took a
        # slightly different floating-point path.
        value = 0.0 if float(value) == 0.0 else float(value)
        return format(value, ".15g")

    def evaluate(self, q: np.ndarray) -> dict[int, np.ndarray]:
        q = np.asarray(q, dtype=float)
        flat = q.ravel()
        keys = [self._key(x) for x in flat]
        missing_by_key: dict[str, float] = {}
        reference_mode = next(iter(self._values))
        reference_cache = self._values[reference_mode]
        for key, value in zip(keys, flat):
            if key not in reference_cache:
                missing_by_key.setdefault(key, float(value))

        if missing_by_key:
            missing_keys = list(missing_by_key)
            missing_q = np.asarray([missing_by_key[key] for key in missing_keys], dtype=float)
            order = np.argsort(missing_q)
            sorted_q = missing_q[order]
            transforms = _grouped_hankel_transforms(
                self.supports,
                sorted_q,
                method=self.method,
                interpolator=self.interpolator,
                subdivisions=self.subdivisions,
            )
            sorted_keys = [missing_keys[index] for index in order]
            for m, values in transforms.items():
                cache = self._values[int(m)]
                for key, value in zip(sorted_keys, values):
                    cache[key] = complex(value)

        return {
            int(m): np.asarray([cache[key] for key in keys], dtype=np.complex128).reshape(q.shape)
            for m, cache in self._values.items()
        }


def _dyadic_uniform_grid(q_max: float, target_intervals: int) -> np.ndarray:
    """Uniform grid whose interval count is a power of two.

    Dyadic interval counts make successive density refinements and doubled
    support windows exactly nested, enabling direct-transform reuse.
    """
    target_intervals = max(4, int(target_intervals))
    intervals = 1 << math.ceil(math.log2(target_intervals))
    return np.linspace(0.0, float(q_max), intervals + 1)


def _adaptive_interpolation_grid(
    decomposition,
    q_max: float,
    *,
    method: str,
    interpolator: str,
    subdivisions: int,
    rtol: float,
    atol: float,
    initial_intervals: int = 16,
    maximum_points: int = 4097,
    maximum_iterations: int = 14,
) -> tuple[np.ndarray, float, list[QInterpolationStep]]:
    """Converge a nonuniform q grid by direct interval-wise transform checks.

    The refinement strategy makes no smoothness or functional-form assumption.
    At every iteration the interpolant is checked against direct Hankel values
    at the quarter, midpoint, and three-quarter points of *every* current
    interval.  Refinement is local, but acceptance uses the same global relative
    L2 and peak-normalized absolute criteria as the legacy uniform-grid study.

    If the adaptive grid cannot satisfy the checks within ``maximum_points`` or
    ``maximum_iterations``, the caller may fall back to the legacy uniform-grid
    search without weakening the convergence requirement.
    """
    q_max = float(q_max)
    supports = {int(m): _mode_support(decomposition, int(m)) for m in decomposition.m_sorted}
    modes = list(supports)
    R_max = max(item[2] for item in supports.values())
    cache = _DirectHankelCache(
        supports,
        method=method,
        interpolator=interpolator,
        subdivisions=subdivisions,
    )
    q_nodes = np.linspace(0.0, q_max, int(initial_intervals) + 1)
    steps: list[QInterpolationStep] = []

    for _iteration in range(int(maximum_iterations)):
        grid_values = cache.evaluate(q_nodes)
        widths = np.diff(q_nodes)
        q_check_matrix = np.stack(
            [q_nodes[:-1] + fraction * widths for fraction in (0.25, 0.5, 0.75)],
            axis=1,
        )
        q_check = q_check_matrix.ravel()
        direct_values = cache.evaluate(q_check)

        harmonic_errors: dict[int, dict[str, float]] = {}
        all_ok = True
        interval_scores = np.zeros(widths.size, dtype=float)
        l2_contributions = np.zeros(widths.size, dtype=float)

        for m in modes:
            interp = Interpolator1D(
                q_nodes,
                grid_values[m],
                interpolator,
                cubic_bc_type="not-a-knot",
            )
            direct = direct_values[m]
            approx = interp(q_check)
            error = approx - direct
            rel_l2 = relative_l2_error(approx, direct)
            rel_linf = relative_linf_error(approx, direct)
            abs_linf = absolute_linf_error(approx, direct)
            peak = float(np.max(np.abs(direct))) if direct.size else 0.0
            absolute_budget = float(atol + rtol * peak)
            ok = bool(rel_l2 <= rtol and abs_linf <= absolute_budget)
            all_ok = all_ok and ok
            harmonic_errors[m] = {
                "relative_l2": rel_l2,
                "relative_linf": rel_linf,
                "absolute_linf": abs_linf,
            }

            local_abs = np.max(np.abs(error).reshape(-1, 3), axis=1)
            denom = max(absolute_budget, np.finfo(float).tiny)
            interval_scores = np.maximum(interval_scores, local_abs / denom)

            # Approximate the global L2 error contribution of each interval.
            # This is used only to choose what to refine; final acceptance still
            # comes from direct checks at all final interval points.
            direct_scale = max(float(np.sum(np.abs(direct) ** 2)), np.finfo(float).tiny)
            local_sq = np.sum(np.abs(error).reshape(-1, 3) ** 2, axis=1)
            l2_contributions += local_sq / direct_scale

        effective_oversampling = float((q_nodes.size - 1) * np.pi / max(q_max * R_max, np.finfo(float).tiny))
        step = QInterpolationStep(
            oversampling=effective_oversampling,
            n_q=int(q_nodes.size),
            harmonic_errors=harmonic_errors,
            converged=all_ok,
            grid_kind="adaptive",
            refined_intervals=0,
            min_dq=float(np.min(widths)),
            max_dq=float(np.max(widths)),
        )
        steps.append(step)
        if all_ok:
            return q_nodes, effective_oversampling, steps

        # Any interval violating the global peak-normalized absolute criterion
        # is necessarily refined.  If the remaining failure is global-L2 only,
        # refine the intervals responsible for at least 80% of the observed
        # squared interpolation discrepancy.  Always refine at least one.
        refine = interval_scores > 1.0
        if not np.any(refine):
            order = np.argsort(l2_contributions)[::-1]
            total = float(np.sum(l2_contributions))
            if total > 0.0:
                cumulative = 0.0
                for idx in order:
                    refine[idx] = True
                    cumulative += float(l2_contributions[idx])
                    if cumulative >= 0.8 * total:
                        break
            else:
                refine[int(np.argmax(interval_scores))] = True
        else:
            # Refine near-threshold neighbours too.  This reduces spline
            # boundary effects without globally doubling the q grid.
            hot = np.flatnonzero(interval_scores > 0.5)
            refine[hot] = True

        indices = np.flatnonzero(refine)
        steps[-1].refined_intervals = int(indices.size)
        if q_nodes.size + indices.size > int(maximum_points):
            break
        midpoints = 0.5 * (q_nodes[indices] + q_nodes[indices + 1])
        q_nodes = np.unique(np.concatenate([q_nodes, midpoints]))

    raise RuntimeError(
        "Adaptive q-grid interpolation did not converge within the configured "
        "point/iteration limit."
    )


def _adaptive_parseval_support(
    decomposition,
    supports: dict[int, tuple[np.ndarray, np.ndarray, float]],
    radial_power: dict[int, float],
    *,
    method: str,
    interpolator: str,
    subdivisions: int,
    q_tail_rtol: float,
    q_ceiling: float,
    maximum_points: int = 8193,
    maximum_iterations: int = 20,
) -> tuple[float, float, float, dict[int, QTailModeResult]]:
    """Select q support with adaptive integration of the Parseval density.

    For each retained harmonic the non-negative spectral density
    ``q |F_m(q)|^2`` integrates to the represented radial norm by Plancherel.
    The known radial norm therefore supplies the target cumulative power; q
    intervals are refined only where the power integral itself is unresolved.

    Composite Simpson estimates on each interval are compared with the same
    interval split in half.  The difference provides the local refinement
    indicator.  The support threshold is selected from a conservative
    cumulative lower estimate ``I - error``.  This is an accelerator, not the
    sole safety mechanism: the public support routine falls back to the dense
    nested pilot if this adaptive study cannot satisfy the requested budget.
    """
    m_values = list(supports)
    nonzero_modes = [m for m in m_values if radial_power[m] > 0.0]
    if not nonzero_modes:
        return (
            float(q_ceiling),
            0.0,
            0.0,
            {
                m: QTailModeResult(m, 0.0, 0.0, 0.0, True)
                for m in m_values
            },
        )

    R_max = max(item[2] for item in supports.values())
    tail_power_budget = max(q_tail_rtol**2, 100.0 * np.finfo(float).eps)
    # Reserve a small part of the tail-power budget for numerical integration
    # of q |F|^2.  The remainder is available for actual omitted q support.
    integration_target = 0.05 * tail_power_budget

    from ._diagnostics import default_q_grid

    seeded_q_max = float(default_q_grid(decomposition)[0])
    minimum_window = min(q_ceiling, 4.0 * np.pi / R_max)
    q_window = min(q_ceiling, max(seeded_q_max, minimum_window))
    cache = _DirectHankelCache(
        supports,
        method=method,
        interpolator=interpolator,
        subdivisions=subdivisions,
    )

    for _expansion in range(32):
        edges = np.linspace(0.0, q_window, 17)
        integrals: dict[int, np.ndarray] = {}
        errors: dict[int, np.ndarray] = {}

        for _iteration in range(int(maximum_iterations)):
            widths = np.diff(edges)
            points = np.stack(
                [
                    edges[:-1],
                    edges[:-1] + 0.25 * widths,
                    edges[:-1] + 0.50 * widths,
                    edges[:-1] + 0.75 * widths,
                    edges[1:],
                ],
                axis=1,
            )
            q_eval = np.unique(points.ravel())
            transforms = cache.evaluate(q_eval)
            indices = np.searchsorted(q_eval, points)
            interval_priority = np.zeros(widths.size, dtype=float)
            worst_integral_error = 0.0

            for m in m_values:
                power = radial_power[m]
                if power <= 0.0:
                    integrals[m] = np.zeros(widths.size, dtype=float)
                    errors[m] = np.zeros(widths.size, dtype=float)
                    continue
                density = q_eval * np.abs(transforms[m]) ** 2
                values = density[indices]
                coarse = widths / 6.0 * (
                    values[:, 0] + 4.0 * values[:, 2] + values[:, 4]
                )
                fine = widths / 12.0 * (
                    values[:, 0]
                    + 4.0 * values[:, 1]
                    + 2.0 * values[:, 2]
                    + 4.0 * values[:, 3]
                    + values[:, 4]
                )
                # Standard composite-Simpson local error estimate.  It is used
                # conservatively through the cumulative lower estimate below
                # and checked by repeated local refinement.
                error = np.abs(fine - coarse) / 15.0
                integrals[m] = np.asarray(fine, dtype=float)
                errors[m] = np.asarray(error, dtype=float)
                relative_error = float(np.sum(error) / power)
                worst_integral_error = max(worst_integral_error, relative_error)
                interval_priority = np.maximum(interval_priority, error / power)

            if worst_integral_error <= integration_target:
                break

            if edges.size >= int(maximum_points):
                raise RuntimeError("adaptive Parseval integration reached its point limit")

            # Refine intervals responsible for most of the unresolved power
            # integral.  This is only a work-selection heuristic; every final
            # interval remains represented in the cumulative support accounting.
            order = np.argsort(interval_priority)[::-1]
            refine = np.zeros(widths.size, dtype=bool)
            total_priority = float(np.sum(interval_priority))
            accumulated = 0.0
            for index in order:
                if interval_priority[index] <= 0.0:
                    break
                refine[index] = True
                accumulated += float(interval_priority[index])
                if accumulated >= 0.85 * total_priority:
                    break
            if not np.any(refine):
                raise RuntimeError("adaptive Parseval refinement stalled")
            if edges.size + int(np.count_nonzero(refine)) > int(maximum_points):
                raise RuntimeError("adaptive Parseval integration reached its point limit")
            midpoints = 0.5 * (edges[:-1] + edges[1:])
            edges = np.sort(np.concatenate([edges, midpoints[refine]]))
        else:
            raise RuntimeError("adaptive Parseval integration did not stabilize")

        mode_results: dict[int, QTailModeResult] = {}
        q_required_values: list[float] = []
        window_sufficient = True
        for m in m_values:
            power = radial_power[m]
            if power <= 0.0:
                mode_results[m] = QTailModeResult(m, 0.0, 0.0, 0.0, True)
                q_required_values.append(0.0)
                continue

            interval_integral = integrals[m]
            interval_error = errors[m]
            lower_cumulative = np.cumsum(
                np.maximum(0.0, interval_integral - interval_error)
            )
            required_power = (1.0 - tail_power_budget) * power
            crossing = int(np.searchsorted(lower_cumulative, required_power))
            total_integral = float(np.sum(interval_integral))
            endpoint_mismatch = float(abs(power - total_integral) / power)

            if crossing >= interval_integral.size:
                window_sufficient = False
                q_required = None
                tail_error = math.sqrt(
                    max(power - total_integral, 0.0) / power
                )
                resolved = False
            else:
                # The upper edge is deliberately conservative: the cumulative
                # lower estimate has already reached the required represented
                # power by this point.
                q_required = float(edges[crossing + 1])
                cumulative_estimate = float(np.sum(interval_integral[: crossing + 1]))
                tail_error = math.sqrt(
                    max(power - cumulative_estimate, 0.0) / power
                )
                resolved = True
                q_required_values.append(q_required)

            mode_results[m] = QTailModeResult(
                m=m,
                q_required=q_required,
                tail_error_at_q_max=float(tail_error),
                parseval_relative_error=endpoint_mismatch,
                resolved=resolved,
            )

        if window_sufficient:
            q_max = max(q_required_values)
            minimum_width = float(np.min(np.diff(edges)))
            effective_oversampling = float(
                np.pi / max(minimum_width * R_max, np.finfo(float).tiny)
            )
            return q_ceiling, q_max, effective_oversampling, mode_results

        if q_window >= q_ceiling * (1.0 - 10.0 * np.finfo(float).eps):
            raise RuntimeError("adaptive Parseval support reached the radial q ceiling")
        q_window = min(q_ceiling, 2.0 * q_window)

    raise RuntimeError("adaptive Parseval support exceeded the expansion limit")


def estimate_q_support(
    decomposition,
    *,
    method: str = "simpson",
    interpolator: str = "cubic",
    subdivisions: int = 2,
    q_tail_rtol: float = 1.0e-3,
    pilot_oversampling: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0),
) -> tuple[float, float, float, dict[int, QTailModeResult]]:
    """Estimate the smallest q support satisfying a per-harmonic Parseval tail.

    The radial Nyquist scale ``q_ceiling = pi / max(diff(r))`` is a hard upper
    bound, not an efficient pilot interval.  The support search therefore starts
    from the inexpensive scale-aware q-range estimate used by the normal
    constructor and expands that interval only when the requested tail
    criterion cannot be met.  Within each interval, the q integration is refined
    until both
    the Parseval remainder and the inferred threshold crossing stabilize.

    This avoids evaluating a dense Hankel transform over momentum ranges that
    contain essentially no represented norm.  Failure to satisfy the tail
    criterion for a retained harmonic before the radial-sampling ceiling still raises
    ``RuntimeError``.
    """
    q_tail_rtol = float(q_tail_rtol)
    if not np.isfinite(q_tail_rtol) or not (0.0 < q_tail_rtol < 1.0):
        raise ValueError("q_tail_rtol must satisfy 0 < q_tail_rtol < 1.")

    m_values = [int(m) for m in decomposition.m_sorted]
    if not m_values:
        raise ValueError("decomposition contains no retained harmonics.")
    q_ceiling = radial_q_ceiling(decomposition)
    supports = {m: _mode_support(decomposition, m) for m in m_values}
    radial_power = {
        m: _represented_radial_power(
            supports[m][0],
            supports[m][1],
            interpolator=interpolator,
            subdivisions=subdivisions,
        )
        for m in m_values
    }
    R_max = max(item[2] for item in supports.values())
    if R_max <= 0.0:
        raise ValueError("retained harmonic support radius must be positive.")

    try:
        return _adaptive_parseval_support(
            decomposition,
            supports,
            radial_power,
            method=method,
            interpolator=interpolator,
            subdivisions=subdivisions,
            q_tail_rtol=q_tail_rtol,
            q_ceiling=q_ceiling,
        )
    except RuntimeError:
        # The adaptive Parseval integration is only an accelerator.  Difficult
        # or pathological cases fall back to the dense nested pilot below,
        # which preserves the previous acceptance semantics.
        pass

    # A relative-L2 tail target q_tail_rtol corresponds to a tail-power
    # fraction q_tail_rtol**2.  The exact represented radial power supplies the
    # Parseval normalization, so the pilot only needs to integrate far enough
    # in q to exhaust that budget; it does not need to march to q_ceiling.
    tail_power_budget = max(q_tail_rtol**2, 100.0 * np.finfo(float).eps)

    # Reuse the cheap scale-aware production estimate as the first support
    # window.  It is intentionally only a seed: difficult/cusped profiles can
    # expand geometrically until the hard radial-sampling ceiling is reached.
    from ._diagnostics import default_q_grid

    seeded_q_max = float(default_q_grid(decomposition)[0])
    minimum_window = min(q_ceiling, 4.0 * np.pi / R_max)
    q_window = min(q_ceiling, max(seeded_q_max, minimum_window))
    growth = 2.0

    last_results: dict[int, QTailModeResult] = {}
    chosen_oversampling = float(pilot_oversampling[-1])
    transform_cache = _DirectHankelCache(
        supports,
        method=method,
        interpolator=interpolator,
        subdivisions=subdivisions,
    )

    while True:
        previous_q_required: dict[int, float | None] = {}
        previous_dq: float | None = None
        need_expand = False

        for oversampling in pilot_oversampling:
            oversampling = float(oversampling)
            if oversampling <= 0.0:
                raise ValueError("pilot_oversampling values must be positive.")
            target_intervals = max(
                32,
                math.ceil(oversampling * q_window * R_max / np.pi),
            )
            q = _dyadic_uniform_grid(q_window, target_intervals)
            transforms = transform_cache.evaluate(q)
            mode_results: dict[int, QTailModeResult] = {}
            all_resolved = True
            window_insufficient = False

            for m in m_values:
                power_r = radial_power[m]
                if power_r <= 0.0:
                    mode_results[m] = QTailModeResult(m, 0.0, 0.0, 0.0, True)
                    continue

                F = transforms[m]
                q_power = q * np.abs(F) ** 2
                cumulative = cumulative_simpson(q_power, x=q, initial=0.0)
                remainder = (power_r - cumulative) / power_r
                endpoint_mismatch = float(abs(remainder[-1]))
                parseval_ok = endpoint_mismatch <= tail_power_budget

                remaining_fraction = np.maximum(0.0, remainder)
                tail = np.sqrt(remaining_fraction)
                found = np.flatnonzero(tail <= q_tail_rtol)
                q_required = float(q[found[0]]) if found.size else None
                if q_required is None and float(remainder[-1]) > tail_power_budget:
                    # The current q window still contains more physical norm
                    # than the requested tail budget.  Refining the same window
                    # cannot fix missing support, so expand it immediately.
                    window_insufficient = True

                # q_required is read from a discrete pilot grid.  Acceptance
                # therefore requires agreement under one q-grid refinement to
                # within one cell of the coarser pilot.
                previous = previous_q_required.get(m)
                stable = False
                if q_required is not None and previous is not None and previous_dq is not None:
                    stable = abs(q_required - previous) <= previous_dq

                resolved = bool(parseval_ok and q_required is not None and stable)
                all_resolved = all_resolved and resolved
                mode_results[m] = QTailModeResult(
                    m=m,
                    q_required=q_required,
                    tail_error_at_q_max=float(tail[found[0]]) if found.size else float(tail[-1]),
                    parseval_relative_error=endpoint_mismatch,
                    resolved=resolved,
                )

            last_results = mode_results
            if all_resolved:
                chosen_oversampling = oversampling
                q_max = max(float(item.q_required) for item in mode_results.values())
                # One pilot interval of headroom avoids selecting a support
                # boundary exactly at the first threshold crossing.
                dq = float(q[1] - q[0])
                q_max = min(q_ceiling, q_max + dq)
                return q_ceiling, q_max, chosen_oversampling, mode_results

            if window_insufficient:
                need_expand = True
                break

            previous_q_required = {m: result.q_required for m, result in mode_results.items()}
            previous_dq = float(q[1] - q[0])

        if not need_expand:
            details = ", ".join(
                f"m={m}: parseval={last_results[m].parseval_relative_error:.3e}, "
                f"q_required={last_results[m].q_required}"
                for m in m_values
            )
            raise RuntimeError(
                "q-support pilot density did not stabilize on the current physical "
                "support window. Increase pilot_oversampling. " + details
            )

        if q_window >= q_ceiling * (1.0 - 10.0 * np.finfo(float).eps):
            break
        q_window = min(q_ceiling, growth * q_window)

    unresolved = [m for m, result in last_results.items() if not result.resolved]
    details = ", ".join(
        f"m={m}: parseval={last_results[m].parseval_relative_error:.3e}, "
        f"q_required={last_results[m].q_required}"
        for m in unresolved
    )
    raise RuntimeError(
        "Could not satisfy q-space support before the radial-sampling ceiling. "
        "The PETAL2D radial sampling/support may be insufficient for the requested "
        f"q_tail_rtol={q_tail_rtol:.3e}. {details}"
    )


def converge_q_interpolation(
    decomposition,
    q_max: float,
    *,
    method: str = "simpson",
    interpolator: str = "cubic",
    subdivisions: int = 4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-12,
    oversampling_values: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0),
    adaptive: bool = True,
    adaptive_initial_intervals: int = 16,
    adaptive_maximum_points: int = 4097,
) -> tuple[np.ndarray, float, list[QInterpolationStep]]:
    """Converge q-grid interpolation by direct off-grid transform checks.

    The default path builds a nonuniform grid by local refinement.  It makes no
    assumption about the analytic form or smoothness class of the transformed
    profile: every interval of the final grid is checked against direct Hankel
    evaluations at its quarter, midpoint, and three-quarter points.  Acceptance
    uses the same global relative-L2 and peak-normalized absolute criteria as
    the legacy uniform-grid study.

    The adaptive search is an accelerator and does not weaken the acceptance
    criteria.  If it cannot satisfy them within the configured point limit, this
    function falls back to the legacy uniform-grid search over
    ``oversampling_values``.
    """
    q_max = float(q_max)
    rtol = float(rtol)
    atol = float(atol)
    if q_max <= 0.0 or not np.isfinite(q_max):
        raise ValueError("q_max must be finite and positive.")
    if rtol < 0.0 or atol < 0.0 or not np.isfinite(rtol) or not np.isfinite(atol):
        raise ValueError("rtol and atol must be finite and non-negative.")

    if adaptive:
        try:
            return _adaptive_interpolation_grid(
                decomposition,
                q_max,
                method=method,
                interpolator=interpolator,
                subdivisions=subdivisions,
                rtol=rtol,
                atol=atol,
                initial_intervals=adaptive_initial_intervals,
                maximum_points=adaptive_maximum_points,
            )
        except RuntimeError:
            # Conservative fallback: use the previous all-interval uniform-grid
            # acceptance check rather than relaxing the acceptance condition.
            pass

    m_values = [int(m) for m in decomposition.m_sorted]
    supports = {m: _mode_support(decomposition, m) for m in m_values}
    R_max = max(item[2] for item in supports.values())
    steps: list[QInterpolationStep] = []
    cache = _DirectHankelCache(
        supports,
        method=method,
        interpolator=interpolator,
        subdivisions=subdivisions,
    )

    for oversampling in oversampling_values:
        oversampling = float(oversampling)
        n_q = max(4, 1 + math.ceil(oversampling * q_max * R_max / np.pi))
        q = np.linspace(0.0, q_max, n_q)
        dq = np.diff(q)
        q_check = np.sort(
            np.concatenate(
                [q[:-1] + fraction * dq for fraction in (0.25, 0.5, 0.75)]
            )
        )
        harmonic_errors: dict[int, dict[str, float]] = {}
        all_ok = True

        grid_transforms = cache.evaluate(q)
        direct_transforms = cache.evaluate(q_check)

        for m in m_values:
            F_grid = grid_transforms[m]
            q_interp = Interpolator1D(
                q,
                F_grid,
                interpolator,
                cubic_bc_type="not-a-knot",
            )
            direct = direct_transforms[m]
            approx = q_interp(q_check)
            rel_l2 = relative_l2_error(approx, direct)
            rel_linf = relative_linf_error(approx, direct)
            abs_linf = absolute_linf_error(approx, direct)
            peak = float(np.max(np.abs(direct))) if direct.size else 0.0
            ok = bool(rel_l2 <= rtol and abs_linf <= atol + rtol * peak)
            all_ok = all_ok and ok
            harmonic_errors[m] = {
                "relative_l2": rel_l2,
                "relative_linf": rel_linf,
                "absolute_linf": abs_linf,
            }

        step = QInterpolationStep(
            oversampling=oversampling,
            n_q=n_q,
            harmonic_errors=harmonic_errors,
            converged=all_ok,
            grid_kind="uniform_fallback",
            min_dq=float(np.min(dq)),
            max_dq=float(np.max(dq)),
        )
        steps.append(step)
        if all_ok:
            return q, oversampling, steps

    raise RuntimeError(
        "q-grid interpolation did not converge for all retained harmonics. "
        "Increase oversampling_values or improve the PETAL2D input sampling."
    )

def converge_q_sampling(
    decomposition,
    *,
    method: str = "simpson",
    interpolator: str = "cubic",
    subdivisions: int = 4,
    q_tail_rtol: float = 1.0e-3,
    interpolation_rtol: float = 1.0e-4,
    interpolation_atol: float = 1.0e-12,
    tail_pilot_oversampling: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0),
    interpolation_oversampling: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0),
) -> QSamplingResult:
    """Converge the q support and q sampling for all retained harmonics."""
    q_ceiling, q_max, tail_s, tail_modes = estimate_q_support(
        decomposition,
        method=method,
        interpolator=interpolator,
        subdivisions=subdivisions,
        q_tail_rtol=q_tail_rtol,
        pilot_oversampling=tail_pilot_oversampling,
    )
    q_grid, interp_s, steps = converge_q_interpolation(
        decomposition,
        q_max,
        method=method,
        interpolator=interpolator,
        subdivisions=subdivisions,
        rtol=interpolation_rtol,
        atol=interpolation_atol,
        oversampling_values=interpolation_oversampling,
    )
    support_radius = max(
        _mode_support(decomposition, int(m))[2] for m in decomposition.m_sorted
    )
    return QSamplingResult(
        q_ceiling=q_ceiling,
        q_max=q_max,
        n_q=int(np.asarray(q_grid).size),
        support_radius=support_radius,
        q_tail_rtol=q_tail_rtol,
        interpolation_rtol=interpolation_rtol,
        interpolation_atol=interpolation_atol,
        tail_pilot_oversampling=tail_s,
        interpolation_oversampling=interp_s,
        tail_modes=tail_modes,
        q_grid=np.asarray(q_grid, dtype=float),
        grid_kind=str(steps[-1].grid_kind if steps else "uniform"),
        interpolation_steps=steps,
    )
