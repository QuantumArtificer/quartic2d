"""Numerical convergence utilities for QUARTIC2D transform backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from time import perf_counter
from typing import Any

import numpy as np

ArrayEvaluator = Callable[[Mapping[str, Any]], np.ndarray]


def relative_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``||a-b||_2 / ||b||_2`` with a finite zero-reference rule."""
    a = np.asarray(a)
    b = np.asarray(b)
    denom = float(np.linalg.norm(b))
    diff = float(np.linalg.norm(a - b))
    if denom == 0.0:
        return 0.0 if diff == 0.0 else np.inf
    return diff / denom


def relative_linf_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``||a-b||_inf / ||b||_inf`` with a finite zero-reference rule."""
    a = np.asarray(a)
    b = np.asarray(b)
    denom = float(np.max(np.abs(b))) if b.size else 0.0
    diff = float(np.max(np.abs(a - b))) if b.size else 0.0
    if denom == 0.0:
        return 0.0 if diff == 0.0 else np.inf
    return diff / denom


def absolute_linf_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``||a-b||_inf``."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b))) if b.size else 0.0


@dataclass
class ConvergenceStep:
    """One evaluated point in a numerical convergence study."""

    parameters: dict[str, Any]
    runtime_seconds: float
    relative_l2_change: float | None = None
    relative_linf_change: float | None = None
    absolute_linf_change: float | None = None
    converged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "parameters": dict(self.parameters),
            "runtime_seconds": float(self.runtime_seconds),
            "relative_l2_change": self.relative_l2_change,
            "relative_linf_change": self.relative_linf_change,
            "absolute_linf_change": self.absolute_linf_change,
            "converged": bool(self.converged),
            "metadata": dict(self.metadata),
        }


@dataclass
class ConvergenceResult:
    """Result of a numerical convergence study."""

    method: str
    rtol: float
    atol: float
    selected_parameters: dict[str, Any]
    selected_values: np.ndarray
    steps: list[ConvergenceStep]
    resolution_converged: bool
    hyperparameter_robust: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """Whether resolution and any tested conditioning choices are robust."""
        return bool(self.resolution_converged and self.hyperparameter_robust)

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        out = {
            "method": str(self.method),
            "converged": self.converged,
            "resolution_converged": bool(self.resolution_converged),
            "hyperparameter_robust": bool(self.hyperparameter_robust),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "selected_parameters": dict(self.selected_parameters),
            "steps": [step.to_dict() for step in self.steps],
            "metadata": dict(self.metadata),
        }
        if include_values:
            values = np.asarray(self.selected_values)
            out["selected_values_real"] = values.real.tolist()
            if np.iscomplexobj(values):
                out["selected_values_imag"] = values.imag.tolist()
        return out


def _validate_tolerances(rtol: float, atol: float) -> tuple[float, float]:
    rtol = float(rtol)
    atol = float(atol)
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and non-negative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative.")
    return rtol, atol


def _evaluate(
    evaluator: ArrayEvaluator, parameters: Mapping[str, Any]
) -> tuple[np.ndarray, float]:
    start = perf_counter()
    values = np.asarray(evaluator(parameters), dtype=np.complex128)
    elapsed = perf_counter() - start
    if values.ndim == 0:
        values = values.reshape(1)
    if np.any(~np.isfinite(values.real)) or np.any(~np.isfinite(values.imag)):
        raise FloatingPointError(f"non-finite values for parameters={dict(parameters)!r}")
    return values, elapsed


def _meets_tolerance(
    current: np.ndarray,
    previous: np.ndarray,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float, float]:
    rel_l2 = relative_l2_error(current, previous)
    rel_linf = relative_linf_error(current, previous)
    abs_linf = absolute_linf_error(current, previous)
    peak = float(np.max(np.abs(previous))) if previous.size else 0.0
    max_ok = abs_linf <= atol + rtol * peak
    l2_ok = rel_l2 <= rtol if np.isfinite(rel_l2) else False
    return bool(l2_ok and max_ok), rel_l2, rel_linf, abs_linf


def converge_sequence(
    method: str,
    parameter_sets: Iterable[Mapping[str, Any]],
    evaluator: ArrayEvaluator,
    *,
    rtol: float = 1.0e-3,
    atol: float = 0.0,
    require_consecutive: int = 1,
) -> ConvergenceResult:
    """Converge an ordered coarse-to-fine numerical-resolution sequence.

    The stopping criterion uses the standard ``rtol``/``atol`` vocabulary. A
    refinement is accepted when the relative L2 change is at most ``rtol`` and
    the maximum absolute change is at most ``atol + rtol * max(abs(previous))``.

    This is a self-convergence test. It measures stability of the numerical
    method on the represented input and does not establish convergence of the
    input sampling or finite support itself.
    """
    rtol, atol = _validate_tolerances(rtol, atol)
    require_consecutive = int(require_consecutive)
    if require_consecutive < 1:
        raise ValueError("require_consecutive must be at least 1.")

    parameter_sets = [dict(p) for p in parameter_sets]
    if len(parameter_sets) < 2:
        raise ValueError("parameter_sets must contain at least two refinement levels.")

    steps: list[ConvergenceStep] = []
    previous: np.ndarray | None = None
    selected_values: np.ndarray | None = None
    selected_parameters = parameter_sets[-1]
    stable_count = 0
    resolution_converged = False

    for parameters in parameter_sets:
        values, runtime = _evaluate(evaluator, parameters)
        step_converged = False
        rel_l2 = rel_linf = abs_linf = None
        if previous is not None:
            ok, rel_l2, rel_linf, abs_linf = _meets_tolerance(values, previous, rtol, atol)
            stable_count = stable_count + 1 if ok else 0
            step_converged = stable_count >= require_consecutive

        steps.append(
            ConvergenceStep(
                parameters=parameters,
                runtime_seconds=runtime,
                relative_l2_change=rel_l2,
                relative_linf_change=rel_linf,
                absolute_linf_change=abs_linf,
                converged=step_converged,
            )
        )
        previous = values
        selected_values = values
        if step_converged:
            selected_parameters = parameters
            resolution_converged = True
            break

    assert selected_values is not None
    return ConvergenceResult(
        method=str(method),
        rtol=rtol,
        atol=atol,
        selected_parameters=selected_parameters,
        selected_values=selected_values,
        steps=steps,
        resolution_converged=resolution_converged,
        metadata={"kind": "ordered_sequence", "require_consecutive": require_consecutive},
    )


def converge_ogata(
    f: Callable,
    nu: int,
    K=None,
    *,
    rtol: float = 1.0e-3,
    atol: float = 1.0e-3,
    hstart: float = 0.05,
    hdecrement: float = 2.0,
    maxiter: int = 15,
) -> ConvergenceResult:
    """Determine Ogata ``h`` and ``N`` using :func:`hankel.get_h`.

    Parameters use the nomenclature of the upstream ``hankel`` package. ``K``
    may be a down-sampled array of transform scales, as recommended by
    ``hankel.get_h`` for efficiency.
    """
    rtol, atol = _validate_tolerances(rtol, atol)
    nu = int(nu)
    hstart = float(hstart)
    hdecrement = float(hdecrement)
    maxiter = int(maxiter)
    if hstart <= 0.0 or not np.isfinite(hstart):
        raise ValueError("hstart must be finite and positive.")
    if hdecrement <= 1.0 or not np.isfinite(hdecrement):
        raise ValueError("hdecrement must be finite and greater than 1.")
    if maxiter < 1:
        raise ValueError("maxiter must be positive.")

    from ._numerics import _require_hankel

    hankel = _require_hankel()
    start = perf_counter()
    try:
        h, values, N = hankel.get_h(
            f,
            nu,
            K=K,
            hstart=hstart,
            hdecrement=hdecrement,
            atol=atol,
            rtol=rtol,
            maxiter=maxiter,
        )
    except RuntimeError as exc:
        # hankel.get_h raises RuntimeError when its bounded search does
        # not satisfy the requested tolerance. Treat that documented
        # non-convergence as a convergence result, not as a package crash.
        if "Maxiter reached while checking convergence" not in str(exc):
            raise
        runtime = perf_counter() - start
        step = ConvergenceStep(
            parameters={},
            runtime_seconds=runtime,
            converged=False,
            metadata={
                "source": "hankel.get_h",
                "status": "maxiter_reached",
                "exception": str(exc),
            },
        )
        return ConvergenceResult(
            method="ogata",
            rtol=rtol,
            atol=atol,
            selected_parameters={},
            selected_values=np.empty(0, dtype=np.complex128),
            steps=[step],
            resolution_converged=False,
            metadata={
                "kind": "hankel.get_h",
                "nu": nu,
                "hstart": hstart,
                "hdecrement": hdecrement,
                "maxiter": maxiter,
                "status": "maxiter_reached",
            },
        )

    runtime = perf_counter() - start
    values = np.asarray(values, dtype=np.complex128)
    if values.ndim == 0:
        values = values.reshape(1)

    selected = {"N": int(N), "h": float(h)}
    step = ConvergenceStep(
        parameters=selected,
        runtime_seconds=runtime,
        converged=True,
        metadata={"source": "hankel.get_h", "status": "converged"},
    )
    return ConvergenceResult(
        method="ogata",
        rtol=rtol,
        atol=atol,
        selected_parameters=selected,
        selected_values=values,
        steps=[step],
        resolution_converged=True,
        metadata={
            "kind": "hankel.get_h",
            "nu": nu,
            "hstart": hstart,
            "hdecrement": hdecrement,
            "maxiter": maxiter,
            "status": "converged",
        },
    )


def _criterion_from_estimate(
    relative_l2: float,
    absolute_linf: float,
    values: np.ndarray,
    rtol: float,
    atol: float,
) -> bool:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    l2_ok = bool(np.isfinite(relative_l2) and relative_l2 <= rtol)
    max_ok = bool(np.isfinite(absolute_linf) and absolute_linf <= atol + rtol * peak)
    return bool(l2_ok and max_ok)


def _observed_order(coarse_change: float, fine_change: float, factor: float) -> float:
    if coarse_change <= 0.0 and fine_change <= 0.0:
        return np.inf
    if coarse_change <= 0.0 or fine_change <= 0.0:
        return np.nan
    return float(np.log(coarse_change / fine_change) / np.log(factor))


def converge_fixed_order_sequence(
    method: str,
    parameter_name: str,
    parameter_values: Iterable[int],
    evaluator: ArrayEvaluator,
    *,
    expected_order: float,
    rtol: float = 1.0e-3,
    atol: float = 0.0,
    refinement_factor: float = 2.0,
    minimum_order_fraction: float = 0.5,
    floor_fraction: float = 0.1,
) -> ConvergenceResult:
    """Converge a known-order nested quadrature sequence with look-ahead.

    Three successive resolutions are used to verify that refinement is in the
    expected asymptotic regime.  Richardson estimates then decide whether the
    coarsest or middle member of the accepted triplet already meets the target.
    This allows an expensive calibration run to select the cheapest converged
    production resolution rather than automatically returning the finer check.
    """
    rtol, atol = _validate_tolerances(rtol, atol)
    expected_order = float(expected_order)
    refinement_factor = float(refinement_factor)
    minimum_order_fraction = float(minimum_order_fraction)
    floor_fraction = float(floor_fraction)
    if expected_order <= 0.0 or not np.isfinite(expected_order):
        raise ValueError("expected_order must be finite and positive.")
    if refinement_factor <= 1.0 or not np.isfinite(refinement_factor):
        raise ValueError("refinement_factor must be finite and greater than 1.")
    if not (0.0 < minimum_order_fraction <= 1.0):
        raise ValueError("minimum_order_fraction must satisfy 0 < value <= 1.")
    if not (0.0 < floor_fraction <= 1.0):
        raise ValueError("floor_fraction must satisfy 0 < value <= 1.")

    raw_values = [int(v) for v in parameter_values]
    if len(raw_values) < 3 or any(v < 1 for v in raw_values):
        raise ValueError("parameter_values must contain at least three positive levels.")
    for left, right in pairwise(raw_values):
        ratio = right / left
        if not np.isclose(ratio, refinement_factor, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                "known-order convergence requires a geometric refinement sequence; "
                f"expected factor {refinement_factor:g}, got {left}->{right}."
            )

    parameter_sets = [{str(parameter_name): value} for value in raw_values]
    steps: list[ConvergenceStep] = []
    values: list[np.ndarray] = []
    denominator = 1.0 - refinement_factor ** (-expected_order)

    selected_index: int | None = None
    verification: dict[str, Any] = {}

    for i, parameters in enumerate(parameter_sets):
        current, runtime = _evaluate(evaluator, parameters)
        rel = rel_inf = abs_inf = None
        if values:
            rel = relative_l2_error(current, values[-1])
            rel_inf = relative_linf_error(current, values[-1])
            abs_inf = absolute_linf_error(current, values[-1])
        steps.append(
            ConvergenceStep(
                parameters=parameters,
                runtime_seconds=runtime,
                relative_l2_change=rel,
                relative_linf_change=rel_inf,
                absolute_linf_change=abs_inf,
                converged=False,
            )
        )
        values.append(current)
        if i < 2:
            continue

        a, b, c = values[i - 2], values[i - 1], values[i]
        dab_rel = relative_l2_error(b, a)
        dbc_rel = relative_l2_error(c, b)
        dab_abs = absolute_linf_error(b, a)
        dbc_abs = absolute_linf_error(c, b)
        p_l2 = _observed_order(dab_rel, dbc_rel, refinement_factor)
        p_abs = _observed_order(dab_abs, dbc_abs, refinement_factor)

        peak_a = float(np.max(np.abs(a))) if a.size else 0.0
        peak_b = float(np.max(np.abs(b))) if b.size else 0.0
        rel_floor = max(dab_rel, dbc_rel) <= floor_fraction * max(rtol, np.finfo(float).eps)
        abs_floor = max(dab_abs, dbc_abs) <= floor_fraction * max(
            atol + rtol * max(peak_a, peak_b), np.finfo(float).eps
        )
        l2_order_ok = rel_floor or (
            np.isfinite(p_l2) and p_l2 >= minimum_order_fraction * expected_order
        )
        abs_order_ok = abs_floor or (
            np.isfinite(p_abs) and p_abs >= minimum_order_fraction * expected_order
        )
        numerical_floor_verified = bool(rel_floor and abs_floor)
        asymptotic_ok = bool(
            numerical_floor_verified
            or (l2_order_ok and abs_order_ok and dbc_rel <= dab_rel and dbc_abs <= dab_abs)
        )

        candidate_checks = []
        for candidate_index, change_rel, change_abs in (
            (i - 2, dab_rel, dab_abs),
            (i - 1, dbc_rel, dbc_abs),
        ):
            estimate_rel = float(change_rel / denominator)
            estimate_abs = float(change_abs / denominator)
            ok = asymptotic_ok and _criterion_from_estimate(
                estimate_rel,
                estimate_abs,
                values[candidate_index],
                rtol,
                atol,
            )
            candidate_checks.append(
                {
                    "index": candidate_index,
                    "parameters": parameter_sets[candidate_index],
                    "estimated_relative_l2_error": estimate_rel,
                    "estimated_absolute_linf_error": estimate_abs,
                    "meets_tolerance": bool(ok),
                }
            )
            if ok:
                selected_index = candidate_index
                break

        verification = {
            "triplet": [parameter_sets[i - 2], parameter_sets[i - 1], parameters],
            "expected_order": expected_order,
            "observed_order_l2": p_l2,
            "observed_order_linf": p_abs,
            "asymptotic_regime_verified": asymptotic_ok,
            "numerical_floor_verified": numerical_floor_verified,
            "candidate_checks": candidate_checks,
        }
        steps[i].metadata.update(verification)
        if selected_index is not None:
            steps[selected_index].converged = True
            steps[selected_index].metadata["verified_by_triplet_ending_at"] = parameters
            break

    fallback_verification: dict[str, Any] = {}
    if selected_index is None:
        # Known-order Richardson extrapolation can become unreliable on strongly
        # oscillatory or nodal integrals even when the refinement sequence is
        # numerically stable.  Do not weaken the tolerance in that regime: fall
        # back to the generic one-level look-ahead acceptance check and require
        # two consecutive refinement changes to satisfy the same rtol/atol budget.
        stable_flags = [False] * len(values)
        for j in range(1, len(values)):
            stable, rel, rel_inf, abs_inf = _meets_tolerance(
                values[j], values[j - 1], rtol, atol
            )
            stable_flags[j] = bool(stable)
            if steps[j].relative_l2_change is None:
                steps[j].relative_l2_change = rel
                steps[j].relative_linf_change = rel_inf
                steps[j].absolute_linf_change = abs_inf
            if j >= 2 and stable_flags[j - 1] and stable_flags[j]:
                selected_index = j - 1
                steps[selected_index].converged = True
                steps[selected_index].metadata["verified_by_stable_refinement"] = (
                    parameter_sets[j]
                )
                fallback_verification = {
                    "kind": "verified_sequence_fallback",
                    "selected_index": selected_index,
                    "selected_parameters": parameter_sets[selected_index],
                    "verified_by": parameter_sets[j],
                }
                break

    if selected_index is None:
        selected_index = len(values) - 1
        resolution_converged = False
    else:
        resolution_converged = True

    return ConvergenceResult(
        method=str(method),
        rtol=rtol,
        atol=atol,
        selected_parameters=parameter_sets[selected_index],
        selected_values=values[selected_index],
        steps=steps,
        resolution_converged=resolution_converged,
        metadata={
            "kind": "fixed_order_richardson",
            "parameter_name": str(parameter_name),
            "expected_order": expected_order,
            "refinement_factor": refinement_factor,
            "minimum_order_fraction": minimum_order_fraction,
            "last_verification": verification,
            "fallback_verification": fallback_verification,
            "status": "converged" if resolution_converged else "tested_parameter_limit_reached",
        },
    )


def converge_verified_sequence(
    method: str,
    parameter_sets: Iterable[Mapping[str, Any]],
    evaluator: ArrayEvaluator,
    *,
    rtol: float = 1.0e-3,
    atol: float = 0.0,
) -> ConvergenceResult:
    """Self-converge an unknown-order sequence using one-level look-ahead.

    Two consecutive refinement changes must satisfy the tolerance.  The middle
    member of the stable triplet is selected, so the final member acts only as
    an independent look-ahead level.
    """
    rtol, atol = _validate_tolerances(rtol, atol)
    parameter_sets = [dict(p) for p in parameter_sets]
    if len(parameter_sets) < 3:
        raise ValueError("parameter_sets must contain at least three refinement levels.")

    steps: list[ConvergenceStep] = []
    values: list[np.ndarray] = []
    selected_index: int | None = None
    stable_flags: list[bool] = []

    for i, parameters in enumerate(parameter_sets):
        current, runtime = _evaluate(evaluator, parameters)
        rel = rel_inf = abs_inf = None
        ok = False
        if values:
            ok, rel, rel_inf, abs_inf = _meets_tolerance(current, values[-1], rtol, atol)
        stable_flags.append(bool(ok))
        steps.append(
            ConvergenceStep(
                parameters=parameters,
                runtime_seconds=runtime,
                relative_l2_change=rel,
                relative_linf_change=rel_inf,
                absolute_linf_change=abs_inf,
                converged=False,
            )
        )
        values.append(current)
        if i >= 2 and stable_flags[i - 1] and stable_flags[i]:
            selected_index = i - 1
            steps[selected_index].converged = True
            steps[selected_index].metadata["verified_by"] = parameters
            break

    if selected_index is None:
        selected_index = len(values) - 1
        resolution_converged = False
    else:
        resolution_converged = True

    return ConvergenceResult(
        method=str(method),
        rtol=rtol,
        atol=atol,
        selected_parameters=parameter_sets[selected_index],
        selected_values=values[selected_index],
        steps=steps,
        resolution_converged=resolution_converged,
        metadata={
            "kind": "verified_sequence",
            "lookahead_levels": 1,
            "status": "converged" if resolution_converged else "tested_parameter_limit_reached",
        },
    )



def converge_ogata_coupled(
    evaluator: ArrayEvaluator,
    *,
    n_values: Iterable[int] = (
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288,
    ),
    hstart: float = 0.05,
    hdecrement: float = 2.0,
    maxiter: int = 20,
    rtol: float = 1.0e-3,
    atol: float = 0.0,
) -> ConvergenceResult:
    """Converge Ogata quadrature directly in the coupled ``(N, h)`` space.

    ``hankel.get_h`` is deliberately not used as a gate here.  Instead, for
    each successively smaller ``h`` the assembled interaction is first
    self-converged in ``N``.  Two consecutive changes between independently
    ``N``-converged ``h`` levels must then satisfy the requested tolerance.
    The middle member of this stable three-``h`` sequence is selected, with
    the finest member serving only as a look-ahead check.

    This strategy is intended for the workflow-level Interaction object, for
    which convergence of the final assembled interaction is the relevant
    quantity.  It is especially useful when individual harmonic-pair
    integrands are oscillatory or non-smooth enough that ``hankel.get_h`` is
    not a reliable parameter selector.
    """
    rtol, atol = _validate_tolerances(rtol, atol)
    hstart = float(hstart)
    hdecrement = float(hdecrement)
    maxiter = int(maxiter)
    if not np.isfinite(hstart) or hstart <= 0.0:
        raise ValueError("hstart must be finite and positive.")
    if not np.isfinite(hdecrement) or hdecrement <= 1.0:
        raise ValueError("hdecrement must be finite and greater than 1.")
    if maxiter < 3:
        raise ValueError("maxiter must be at least 3 for the Ogata look-ahead check.")

    ns = tuple(sorted({int(n) for n in n_values}))
    if len(ns) < 3 or any(n < 8 for n in ns):
        raise ValueError("n_values must contain at least three positive Ogata node counts >= 8.")

    h_values = tuple(hstart / hdecrement**i for i in range(maxiter))
    outer_steps: list[ConvergenceStep] = []
    per_h: list[dict[str, Any]] = []
    successful: list[tuple[int, float, ConvergenceResult]] = []
    selected: tuple[int, float, ConvergenceResult] | None = None
    previous_selected_n: int | None = None
    terminal_parameters: dict[str, Any] = {"N": ns[-1], "h": h_values[0]}
    terminal_values = np.empty(0, dtype=np.complex128)
    stop_status = "h_limit_reached"

    for h_index, h in enumerate(h_values):
        # Smaller h generally requires at least as much truncation depth.  Start
        # one tested N level below the preceding solution when possible, while
        # retaining three levels for the look-ahead test.
        start_index = 0
        if previous_selected_n is not None:
            candidate_index = next(
                (i for i, n in enumerate(ns) if n >= previous_selected_n),
                len(ns) - 1,
            )
            start_index = max(0, min(candidate_index - 1, len(ns) - 3))
        n_subset = ns[start_index:]
        parameter_sets = [{"N": int(n), "h": float(h)} for n in n_subset]

        n_search = converge_verified_sequence(
            "ogata",
            parameter_sets,
            evaluator,
            rtol=rtol,
            atol=atol,
        )
        terminal_parameters = dict(n_search.selected_parameters)
        terminal_values = np.asarray(n_search.selected_values, dtype=np.complex128)
        runtime = float(sum(step.runtime_seconds for step in n_search.steps))

        rel_l2 = rel_linf = abs_linf = None
        h_stable = False
        if n_search.converged and successful:
            previous_values = successful[-1][2].selected_values
            h_stable, rel_l2, rel_linf, abs_linf = _meets_tolerance(
                n_search.selected_values,
                previous_values,
                rtol,
                atol,
            )

        outer_steps.append(
            ConvergenceStep(
                parameters=dict(n_search.selected_parameters),
                runtime_seconds=runtime,
                relative_l2_change=rel_l2,
                relative_linf_change=rel_linf,
                absolute_linf_change=abs_linf,
                converged=False,
                metadata={
                    "h_index": int(h_index),
                    "n_converged": bool(n_search.converged),
                    "h_stable_with_previous": bool(h_stable),
                    "n_search": n_search.to_dict(),
                },
            )
        )
        per_h.append(
            {
                "h": float(h),
                "n_search": n_search.to_dict(),
                "h_stable_with_previous": bool(h_stable),
                "relative_l2_change": rel_l2,
                "relative_linf_change": rel_linf,
                "absolute_linf_change": abs_linf,
            }
        )

        if not n_search.converged:
            # The hankel implementation generally needs larger N as h is
            # reduced.  Once the available N ladder cannot converge at a given
            # h, continuing to still smaller h would only exceed the tested
            # truncation budget.
            stop_status = "n_limit_reached"
            break

        previous_selected_n = int(n_search.selected_parameters["N"])
        successful.append((len(outer_steps) - 1, float(h), n_search))

        if len(successful) >= 3:
            first = successful[-3]
            middle = successful[-2]
            fine = successful[-1]
            # Require consecutive h levels, not merely any three successful
            # levels, so a failed N search cannot be skipped over silently.
            consecutive = first[0] + 1 == middle[0] and middle[0] + 1 == fine[0]
            middle_stable = bool(outer_steps[middle[0]].metadata["h_stable_with_previous"])
            fine_stable = bool(outer_steps[fine[0]].metadata["h_stable_with_previous"])
            if consecutive and middle_stable and fine_stable:
                selected = middle
                outer_steps[middle[0]].converged = True
                outer_steps[middle[0]].metadata["verified_by"] = dict(
                    fine[2].selected_parameters
                )
                stop_status = "converged"
                break

    if selected is not None:
        _, _, chosen_search = selected
        selected_parameters = dict(chosen_search.selected_parameters)
        selected_values = np.asarray(chosen_search.selected_values, dtype=np.complex128)
        resolution_converged = True
    else:
        selected_parameters = terminal_parameters
        selected_values = terminal_values
        resolution_converged = False

    # Describe the terminal h behavior without changing the convergence
    # acceptance check.  This distinguishes a bounded search that was still
    # improving when it stopped from one that had visibly stalled or become
    # non-monotonic.  It is a diagnostic, not an extrapolated error estimate.
    h_change_metrics: list[float] = []
    for item in per_h:
        rel_l2 = item.get("relative_l2_change")
        rel_linf = item.get("relative_linf_change")
        finite = [
            float(value)
            for value in (rel_l2, rel_linf)
            if value is not None and np.isfinite(value)
        ]
        if finite:
            h_change_metrics.append(max(finite))

    if resolution_converged:
        terminal_h_trend = "converged"
    elif len(h_change_metrics) >= 3:
        a, b, c = h_change_metrics[-3:]
        terminal_h_trend = (
            "monotonic_improving" if c < b < a else "stalled_or_nonmonotonic"
        )
    elif len(h_change_metrics) >= 2:
        terminal_h_trend = (
            "improving" if h_change_metrics[-1] < h_change_metrics[-2]
            else "stalled_or_nonmonotonic"
        )
    else:
        terminal_h_trend = "insufficient_data"

    terminal_h_ratio = None
    if len(h_change_metrics) >= 2 and h_change_metrics[-2] > 0.0:
        terminal_h_ratio = float(h_change_metrics[-1] / h_change_metrics[-2])

    return ConvergenceResult(
        method="ogata",
        rtol=rtol,
        atol=atol,
        selected_parameters=selected_parameters,
        selected_values=selected_values,
        steps=outer_steps,
        resolution_converged=resolution_converged,
        hyperparameter_robust=resolution_converged,
        metadata={
            "kind": "ogata_coupled_N_h",
            "status": stop_status,
            "n_values": list(ns),
            "hstart": float(hstart),
            "hdecrement": float(hdecrement),
            "maxiter": int(maxiter),
            "h_min_tested": (
                float(per_h[-1]["h"]) if per_h else None
            ),
            "n_max_available": int(ns[-1]),
            "terminal_h_trend": terminal_h_trend,
            "terminal_h_change_metric": (
                None if not h_change_metrics else float(h_change_metrics[-1])
            ),
            "terminal_h_change_ratio": terminal_h_ratio,
            "per_h": per_h,
        },
    )


def converge_fftlog_bias_resolution(
    evaluator: ArrayEvaluator,
    *,
    n_values: Iterable[int] = (128, 256, 512, 1024, 2048, 4096),
    bias_values: Iterable[float] = (-0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35),
    preferred_bias: float = -0.5,
    rtol: float = 1.0e-3,
    atol: float = 0.0,
    minimum_bias_cluster: int = 3,
) -> ConvergenceResult:
    """Converge FFTLog resolution and reject bias-sensitive false plateaus.

    The preferred bias is converged first with the complete ``n`` ladder.  Its
    selected resolution then provides a warm start for neighboring biases: a
    centered three-level ``(n/2, n, 2n)`` sequence is sufficient to apply the
    same two-consecutive-change/look-ahead acceptance rule used by
    :func:`converge_verified_sequence`.  Consecutive bias windows containing
    the preferred bias are tested from the center outward.  If no window can be
    accepted by these exact warm-start triplets, only the unresolved biases
    fall back to the complete ``n`` ladder.

    This changes search order and reuse only; the resolution acceptance rule and
    the final mutual bias-consistency test are unchanged.
    """
    rtol, atol = _validate_tolerances(rtol, atol)
    n_values = tuple(int(n) for n in n_values)
    if len(n_values) < 3 or any(n < 16 for n in n_values):
        raise ValueError("n_values must contain at least three FFT lengths >= 16.")
    if tuple(sorted(set(n_values))) != n_values:
        raise ValueError("n_values must be strictly increasing with no duplicates.")

    biases = sorted({float(b) for b in bias_values} | {float(preferred_bias)})
    if len(biases) < int(minimum_bias_cluster):
        raise ValueError("bias_values do not provide enough values for robustness testing.")
    if any(not np.isfinite(b) for b in biases):
        raise ValueError("bias_values must be finite.")
    minimum_bias_cluster = int(minimum_bias_cluster)
    if minimum_bias_cluster < 3:
        raise ValueError("minimum_bias_cluster must be at least 3.")

    preferred_index = min(range(len(biases)), key=lambda i: abs(biases[i] - preferred_bias))
    preferred_bias = float(biases[preferred_index])

    candidate_windows: list[list[float]] = []
    for start in range(len(biases) - minimum_bias_cluster + 1):
        stop = start + minimum_bias_cluster
        if start <= preferred_index < stop:
            candidate_windows.append(biases[start:stop])

    # Biases outside these windows can never contribute to an accepted result.
    admissible_biases = sorted({bias for window in candidate_windows for bias in window})
    evaluation_cache: dict[tuple[int, float], np.ndarray] = {}

    def cached_evaluator(parameters: Mapping[str, Any]) -> np.ndarray:
        key = (int(parameters["n"]), float(parameters["bias"]))
        if key not in evaluation_cache:
            evaluation_cache[key] = np.asarray(evaluator(parameters), dtype=np.complex128)
        return evaluation_cache[key]

    def full_resolution_search(bias: float) -> ConvergenceResult:
        return converge_verified_sequence(
            "fftlog",
            [{"n": n, "bias": float(bias)} for n in n_values],
            cached_evaluator,
            rtol=rtol,
            atol=atol,
        )

    def warm_resolution_search(bias: float, anchor_n: int) -> ConvergenceResult | None:
        try:
            anchor_index = n_values.index(int(anchor_n))
        except ValueError:
            return None
        if anchor_index <= 0 or anchor_index >= len(n_values) - 1:
            return None
        local_ns = n_values[anchor_index - 1 : anchor_index + 2]
        return converge_verified_sequence(
            "fftlog",
            [{"n": n, "bias": float(bias)} for n in local_ns],
            cached_evaluator,
            rtol=rtol,
            atol=atol,
        )

    def window_distance(window: list[float]) -> tuple[float, float]:
        offsets = [abs(float(b) - preferred_bias) for b in window]
        return max(offsets), sum(offsets)

    def evaluate_window(
        window_biases: list[float],
        per_bias: Mapping[float, ConvergenceResult],
    ) -> tuple[dict[str, Any], tuple[int, float, np.ndarray] | None]:
        results = [per_bias[b] for b in window_biases]
        if not all(item.resolution_converged for item in results):
            return ({"biases": window_biases, "resolution_converged": False}, None)

        common_n = max(int(item.selected_parameters["n"]) for item in results)
        verification_values = {
            b: cached_evaluator({"n": common_n, "bias": b}) for b in window_biases
        }
        pair_checks = []
        robust = True
        for i, left in enumerate(window_biases):
            for right in window_biases[i + 1 :]:
                ok, rel, rel_inf, abs_inf = _meets_tolerance(
                    verification_values[right], verification_values[left], rtol, atol
                )
                pair_checks.append(
                    {
                        "left_bias": left,
                        "right_bias": right,
                        "relative_l2_change": rel,
                        "relative_linf_change": rel_inf,
                        "absolute_linf_change": abs_inf,
                        "meets_tolerance": bool(ok),
                    }
                )
                robust = robust and bool(ok)

        entry = {
            "biases": window_biases,
            "resolution_converged": True,
            "common_n": common_n,
            "pair_checks": pair_checks,
            "robust": bool(robust),
        }
        if not robust:
            return entry, None
        selected_bias = min(window_biases, key=lambda b: abs(b - preferred_bias))
        return entry, (common_n, selected_bias, verification_values[selected_bias])

    per_bias: dict[float, ConvergenceResult] = {}
    search_mode: dict[float, str] = {}
    windows: list[dict[str, Any]] = []
    chosen: tuple[int, float, np.ndarray] | None = None

    preferred_result = full_resolution_search(preferred_bias)
    per_bias[preferred_bias] = preferred_result
    search_mode[preferred_bias] = "full"

    # Exact warm-start phase.  Search the most central admissible window first.
    # After each attempt, deprioritize windows containing a known failed warm
    # resolution so that the search naturally moves toward the promising side.
    attempted: set[tuple[float, ...]] = set()
    if preferred_result.resolution_converged:
        anchor_n = int(preferred_result.selected_parameters["n"])
        while len(attempted) < len(candidate_windows) and chosen is None:
            remaining = [w for w in candidate_windows if tuple(w) not in attempted]

            def rank(window: list[float]) -> tuple[int, float, float, int]:
                known_failed = sum(
                    1
                    for b in window
                    if b in per_bias and not per_bias[b].resolution_converged
                )
                max_distance, total_distance = window_distance(window)
                unknown = sum(1 for b in window if b not in per_bias)
                return known_failed, max_distance, total_distance, unknown

            window = min(remaining, key=rank)
            attempted.add(tuple(window))
            for bias in window:
                if bias in per_bias:
                    continue
                result = warm_resolution_search(bias, anchor_n)
                if result is None:
                    break
                per_bias[bias] = result
                search_mode[bias] = "warm_triplet"

            if not all(b in per_bias for b in window):
                windows.append({"biases": window, "resolution_converged": False})
                continue
            entry, candidate = evaluate_window(window, per_bias)
            entry["search_phase"] = "warm_start"
            windows.append(entry)
            if candidate is not None:
                chosen = candidate

    fallback_used = False
    if chosen is None:
        fallback_used = True
        # Complete only the unresolved/failed biases.  Warm-triplet successes
        # already carry the same accepted resolution result and need no redo.
        for bias in admissible_biases:
            item = per_bias.get(bias)
            if item is None or not item.resolution_converged:
                per_bias[bias] = full_resolution_search(bias)
                search_mode[bias] = "full_fallback"

        # Re-test windows from the center outward with the completed resolution
        # information.  This is the same robustness criterion as above.
        fallback_windows = sorted(candidate_windows, key=window_distance)
        for window in fallback_windows:
            entry, candidate = evaluate_window(window, per_bias)
            entry["search_phase"] = "fallback"
            windows.append(entry)
            if candidate is not None:
                chosen = candidate
                break

    if chosen is not None:
        selected_parameters = {"n": int(chosen[0]), "bias": float(chosen[1])}
        selected_values = chosen[2]
        resolution_converged = True
        hyperparameter_robust = True
    else:
        selected_parameters = dict(preferred_result.selected_parameters)
        selected_values = preferred_result.selected_values
        resolution_converged = bool(preferred_result.resolution_converged)
        hyperparameter_robust = False

    tested_biases = sorted({bias for _, bias in evaluation_cache})
    steps = []
    for bias in sorted(per_bias):
        item = per_bias[bias]
        steps.append(
            ConvergenceStep(
                parameters={"bias": bias, **item.selected_parameters},
                runtime_seconds=float(sum(step.runtime_seconds for step in item.steps)),
                converged=bool(chosen is not None and np.isclose(bias, chosen[1])),
                metadata={
                    "resolution_converged": bool(item.resolution_converged),
                    "resolution_search": item.to_dict(),
                    "search_mode": search_mode.get(bias, "unknown"),
                },
            )
        )

    return ConvergenceResult(
        method="fftlog",
        rtol=rtol,
        atol=atol,
        selected_parameters=selected_parameters,
        selected_values=np.asarray(selected_values, dtype=np.complex128),
        steps=steps,
        resolution_converged=resolution_converged,
        hyperparameter_robust=hyperparameter_robust,
        metadata={
            "kind": "fftlog_resolution_and_bias",
            "preferred_bias": float(preferred_bias),
            "minimum_bias_cluster": minimum_bias_cluster,
            "bias_values": biases,
            "admissible_bias_values": admissible_biases,
            "tested_bias_values": tested_biases,
            "bias_windows": windows,
            "warm_start_anchor_n": (
                int(preferred_result.selected_parameters["n"])
                if preferred_result.resolution_converged
                else None
            ),
            "fallback_used": bool(fallback_used),
            "status": "converged" if chosen is not None else "bias_robustness_not_established",
        },
    )
