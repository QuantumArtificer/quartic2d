"""Compact, evidence-based convergence reporting with no numerical work.

This module deliberately has no dependency on the quadrature implementation.
Reports describe the criteria that were actually tested, never a global
accuracy guarantee. All guidance uses the public QUARTIC2D API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from textwrap import wrap
from typing import Any

import numpy as np

__all__ = [
    "ConvergenceAction",
    "ConvergenceError",
    "ConvergenceFailureReport",
    "ProgressEvent",
]


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def software_versions() -> dict[str, str | None]:
    """Versions of packages relevant to numerical reproducibility."""
    from ._version import __version__

    return {
        "quartic2d": __version__,
        "petal2d": _version("petal2d"),
        "numpy": np.__version__,
        "scipy": _version("scipy"),
        "hankel": _version("hankel"),
    }


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _scalar(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _scalar(value.tolist())
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.5g}"
    if isinstance(value, (tuple, list)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _lines(title: str, passed: bool, rows: list[tuple[str, Any]], *, width: int = 69) -> str:
    title_line = f"QUARTIC2D · {title}"
    status = "PASS" if passed else "FAIL"
    width = max(52, min(92, width))
    lines = [f"{title_line:<{width-6}}{status:>6}", "─" * width]
    for key, value in rows:
        if key == "":
            lines.append("")
        elif value is None:
            text = str(key)
            indent = len(text) - len(text.lstrip())
            lines.extend(" " * indent + piece for piece in wrap(
                text.strip(), width=width - indent, break_long_words=False,
                break_on_hyphens=False,
            ))
        else:
            label = str(key)
            formatted = _fmt(value)
            initial = f"  {label:<26} "
            if len(initial) + len(formatted) <= width:
                lines.append(initial + formatted)
            else:
                parts = wrap(formatted, width=max(24, width - len(initial)),
                             break_long_words=False, break_on_hyphens=False)
                if parts:
                    lines.append(initial + parts[0])
                    lines.extend(" " * len(initial) + part for part in parts[1:])
                else:
                    lines.append(initial + "—")
    return "\n".join(lines)


@dataclass(frozen=True)
class ConvergenceAction:
    """A location-specific, evidence-dependent suggestion."""

    location: str
    parameter: str
    instruction: str

    def to_dict(self) -> dict[str, str]:
        return {
            "location": self.location,
            "parameter": self.parameter,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class ConvergenceFailureReport:
    """Structured information retained when a numerical criterion is not met."""

    code: str
    stage: str
    explanation: str
    observed: Mapping[str, Any] = field(default_factory=dict)
    requested: Mapping[str, Any] = field(default_factory=dict)
    limiting_harmonics: tuple[int, ...] = ()
    actions: tuple[ConvergenceAction, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "explanation": self.explanation,
            "observed": _scalar(self.observed),
            "requested": _scalar(self.requested),
            "limiting_harmonics": list(self.limiting_harmonics),
            "actions": [action.to_dict() for action in self.actions],
            "evidence": _scalar(self.evidence),
        }

    def format(self, *, level: str = "summary") -> str:
        if level not in {"summary", "full"}:
            raise ValueError("level must be 'summary' or 'full'.")
        rows: list[tuple[str, Any]] = [
            ("Problem", None),
            (f"  {self.explanation}", None),
        ]
        if self.limiting_harmonics:
            shown = self.limiting_harmonics if level == "full" else self.limiting_harmonics[:5]
            modes = ", ".join(str(m) for m in shown)
            if len(shown) < len(self.limiting_harmonics):
                modes += f" (+{len(self.limiting_harmonics) - len(shown)} more)"
            rows.append(("Harmonics", modes))
        for key, value in self.observed.items():
            rows.append((key.replace("_", " "), value))
        for key, value in self.requested.items():
            rows.append((f"Target: {key.replace('_', ' ')}", value))
        if self.actions:
            rows.append(("", None))
            rows.append(("Try", None))
            for action in self.actions[: (None if level == "full" else 3)]:
                rows.append((f"  {action.location}({action.parameter}=...)", None))
                rows.append((f"    {action.instruction}", None))
        if level == "full":
            rows.append(("", None))
            rows.append(("Failure code", self.code))
            rows.append(("Stage", self.stage))
            if self.evidence:
                rows.append(("Recorded evidence", None))
                for key, value in self.evidence.items():
                    if isinstance(value, (dict, list, tuple)):
                        value = f"{len(value)} entries (see report.to_dict())"
                    rows.append((key.replace("_", " "), value))
        return _lines("Convergence", False, rows)

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return self.format()


class ConvergenceError(RuntimeError):
    """Failed convergence search, carrying all available numerical evidence."""

    def __init__(self, report: ConvergenceFailureReport):
        self.report = report
        super().__init__(report.format())


@dataclass(frozen=True)
class ProgressEvent:
    """Lightweight event emitted during opt-in convergence calibration."""

    stage: str
    status: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    completed: int | None = None
    total: int | None = None


def emit_progress(callback: Callable[[ProgressEvent], None] | None, stage: str,
                  status: str, **kwargs: Any) -> None:
    if callback is not None:
        callback(ProgressEvent(stage=stage, status=status, **kwargs))


def decomposition_provenance(decomposition: Any) -> dict[str, Any]:
    """Record available upstream settings without assuming physical units."""
    d: dict[str, Any] = {"type": type(decomposition).__name__}
    for attr, label in (
        ("x", "x_grid"), ("y", "y_grid"), ("r", "r_grid"),
        ("theta", "theta_grid"),
    ):
        values = getattr(decomposition, attr, None)
        if values is not None:
            a = np.asarray(values)
            if a.ndim == 1 and a.size:
                d[label] = {"size": int(a.size), "min": float(a[0]),
                            "max": float(a[-1])}
                if a.size > 1:
                    d[label]["maximum_spacing"] = float(np.max(np.diff(a)))
    for attr in (
        "origin", "safe_rmax", "m_sorted", "cutoff_radius",
        "recon_error", "recon_error_measured", "target_reached",
        "parseval_relative_error", "_recon_err_tol", "_m_abs_max",
        "_rmax", "_normalize", "_interp_method",
        "radial_power_tail_fraction", "radial_relative_amplitude_threshold",
    ):
        if hasattr(decomposition, attr):
            d[attr.lstrip("_")] = _scalar(getattr(decomposition, attr))
    return d


def format_decomposition(decomposition: Any, *, level: str = "summary") -> str:
    """Display PETAL2D representation checks without implying input accuracy."""
    if level not in {"summary", "full"}:
        raise ValueError("level must be 'summary' or 'full'.")
    settings = decomposition_provenance(decomposition)
    target_reached = bool(getattr(decomposition, "target_reached", False))
    harmonics = list(getattr(decomposition, "m_sorted", []))
    rows: list[tuple[str, Any]] = [
        ("Harmonic selection target", "MET" if target_reached else "NOT MET"),
        ("Retained harmonics", len(harmonics)),
        ("Harmonic orders", harmonics if len(harmonics) <= 12 or level == "full"
         else f"{', '.join(map(str, harmonics[:12]))} (+{len(harmonics)-12} more)"),
        ("Reconstruction target (%)", settings.get("recon_err_tol")),
        ("Selected-power error (%)", settings.get("recon_error")),
        ("Measured reconstruction (%)", settings.get("recon_error_measured")),
        ("Parseval relative mismatch", settings.get("parseval_relative_error")),
        ("Input-field accuracy", "Not tested"),
    ]
    if level == "full":
        rows += [("", None), ("Input and decomposition settings", None)]
        for key in ("x_grid", "y_grid", "r_grid", "theta_grid"):
            grid = settings.get(key)
            if grid is not None:
                rows.append((key.replace("_", " "),
                             f"N={grid['size']}, [{_fmt(grid['min'])}, {_fmt(grid['max'])}], "
                             f"max Δ={_fmt(grid.get('maximum_spacing'))}"))
        for key in ("origin", "safe_rmax", "rmax", "m_abs_max", "normalize",
                    "interp_method", "radial_power_tail_fraction",
                    "radial_relative_amplitude_threshold"):
            if key in settings:
                rows.append((key.replace("_", " "), settings[key]))
        cutoffs = settings.get("cutoff_radius", {})
        if isinstance(cutoffs, dict):
            rows += [("", None), ("Radial cutoffs (by harmonic)", None)]
            for m, cutoff in sorted(cutoffs.items(), key=lambda entry: int(entry[0])):
                rows.append((f"m={m}", cutoff))
        rows += [("", None), ("Software versions", None)]
        rows.extend((name, value if value is not None else "Not installed")
                    for name, value in software_versions().items())
    return _lines("Polar decomposition", target_reached, rows)


def field_provenance(field: Any) -> dict[str, Any]:
    d = {"type": type(field).__name__}
    for attr in ("method", "interpolator", "m_values"):
        if hasattr(field, attr):
            d[attr] = _scalar(getattr(field, attr))
    q = getattr(field, "q", None)
    if q is not None:
        a = np.asarray(q)
        if a.ndim == 1 and a.size:
            d["q_grid"] = {"size": int(a.size), "min": float(a[0]),
                           "max": float(a[-1]), "values": a.tolist()}
    sampling = getattr(field, "sampling_convergence", None)
    if sampling is not None:
        d["sampling_convergence"] = sampling.to_dict()
    return d


def _human_params(params: Mapping[str, Any]) -> list[tuple[str, Any]]:
    rows = []
    for key, value in params.items():
        if key == "q_grid" and value is not None:
            arr = np.asarray(value)
            rows.append(("q grid", f"{arr.size} points, [{_fmt(arr[0])}, {_fmt(arr[-1])}]"))
        else:
            rows.append((key.replace("_", " "), value))
    return rows


def format_harmonic(result: Any, *, level: str = "summary") -> str:
    if level not in {"summary", "full"}:
        raise ValueError("level must be 'summary' or 'full'.")
    sampling, quadrature = result.sampling, result.quadrature
    rows: list[tuple[str, Any]] = [
        ("Method", result.method), ("Interpolator", result.interpolator),
        ("Retained harmonics", len(sampling.tail_modes)),
        ("", None), ("Selected parameters", None),
        *_human_params({k: v for k, v in result.parameters.items()
                        if k not in {"method", "interpolator"}}),
        ("", None), ("Convergence criteria", None),
        ("Momentum tail", "PASS" if sampling.tail_converged else "FAIL"),
        ("Momentum interpolation", "PASS" if sampling.interpolation_converged else "FAIL"),
        ("Radial quadrature", "PASS" if quadrature.converged else "FAIL"),
        ("Tail tolerance (relative L2)", result.q_tail_rtol),
        ("Max measured tail L2", max((v.tail_error_at_q_max
                                      for v in sampling.tail_modes.values()), default=None)),
        ("Refinement tolerance (L2)", result.rtol),
        ("Max-change rule", f"atol + rtol × peak; atol={_fmt(result.atol)}"),
    ]
    if sampling.interpolation_steps:
        final = sampling.interpolation_steps[-1]
        rows.append(("Max q interpolation L2",
                     max((e["relative_l2"] for e in final.harmonic_errors.values()),
                         default=None)))
    q_errors = [step.get("relative_l2_change")
                for h in quadrature.metadata.get("harmonics", {}).values()
                for step in h.get("steps", [])[-1:]
                if step.get("relative_l2_change") is not None]
    if q_errors:
        rows.append(("Max radial quadrature L2", max(q_errors)))
    if level == "full":
        rows += [("", None), ("Measured quantities", None),
                 ("Maximum radial-grid q", sampling.q_ceiling),
                 ("Represented radius", sampling.support_radius),
                 ("q grid kind", sampling.grid_kind)]
        for m, item in sorted(sampling.tail_modes.items()):
            rows.append((f"m={m} tail L2", item.tail_error_at_q_max))
            rows.append((f"m={m} Parseval mismatch", item.parseval_relative_error))
            rows.append((f"m={m} required q", item.q_required))
        if sampling.interpolation_steps:
            last = sampling.interpolation_steps[-1]
            rows.append(("q interpolation checks", len(sampling.interpolation_steps)))
            for m, err in sorted(last.harmonic_errors.items()):
                rows.append((f"m={m} interp relative L2", err.get("relative_l2")))
                rows.append((f"m={m} interp absolute max", err.get("absolute_linf")))
        for m, hist in sorted(quadrature.metadata.get("harmonics", {}).items()):
            steps = hist.get("steps", [])
            if steps:
                rows.append((f"m={m} quadrature L2", steps[-1].get("relative_l2_change")))
                rows.append((f"m={m} quadrature max", steps[-1].get("absolute_linf_change")))
        rows += [("", None), ("Provenance", None)]
        for key, value in result.provenance.items():
            if key == "decomposition":
                for name, detail in value.items():
                    if name == "cutoff_radius":
                        rows.append(("radial cutoffs", f"{len(detail)} mode(s); values in to_dict()"))
                    elif isinstance(detail, dict):
                        rows.append((name, ", ".join(f"{k}={_fmt(v)}" for k, v in detail.items())))
                    else:
                        label = f"{name} (%)" if name in {
                            "recon_error", "recon_error_measured", "recon_err_tol"
                        } else name
                        rows.append((label, detail))
            elif isinstance(value, dict):
                for k, v in value.items():
                    rows.append((f"{key}.{k}", v if not isinstance(v, (dict, list))
                                 else f"{len(v)} entries (see to_dict())"))
            else:
                rows.append((key, value))
        rows += [("", None), ("Scope", "Internal refinement of represented input"),
                 ("Reference accuracy", "Not tested")]
    return _lines("Harmonic convergence", result.converged, rows)


def interaction_failure(result: Any) -> ConvergenceFailureReport | None:
    search = result.search
    if search.converged:
        return None
    meta = search.metadata
    boundary = meta.get("q_boundary_robustness", {})
    if boundary.get("tested") and not boundary.get("passed"):
        return ConvergenceFailureReport(
            code="Q_BOUNDARY_NOT_ROBUST", stage="interaction_boundary",
            explanation="The interaction changes too much near the upper momentum cutoff.",
            observed={"relative L2 change": boundary.get("relative_l2_change"),
                      "absolute max change": boundary.get("absolute_linf_change")},
            requested={"relative L2 allowance": boundary.get("boundary_rtol"),
                       "absolute max rule": "boundary_atol + boundary_rtol × peak"},
            actions=(ConvergenceAction(
                "HarmonicTransform", "q_max",
                "Extend the input transforms' q range only if the radial data resolve it; recalibrate."),),
            evidence=boundary,
        )
    status = meta.get("status", "parameter_limit_reached")
    if status == "bias_robustness_not_established":
        explanation = "Changing the FFTLog bias still changes the interaction too much."
        actions = (ConvergenceAction("Interaction.converge_parameters", "bias_values",
                                     "Test a suitable bias range and inspect endpoint behavior."),)
        code = "FFTLOG_BIAS_NOT_STABLE"
    elif result.method == "ogata":
        explanation = "Refining the Ogata node count and spacing did not stabilize the interaction."
        actions = (ConvergenceAction("Interaction.converge_parameters", "ogata_n_values",
                                     "Expand the node counts tested with the paired h search."),)
        code = "OGATA_NOT_CONVERGED"
    else:
        explanation = "Increasing the interaction integration resolution still changes the result."
        actions = (ConvergenceAction("Interaction.converge_parameters",
                                     "n_values" if result.method == "fftlog" else "subdivisions",
                                     "Extend the tested refinement levels if the trend supports doing so."),)
        code = "INTERACTION_QUADRATURE_NOT_CONVERGED"
    last = next((s for s in reversed(search.steps) if s.relative_l2_change is not None), None)
    return ConvergenceFailureReport(
        code=code, stage="interaction_quadrature", explanation=explanation,
        observed={"latest relative L2 change": None if last is None else last.relative_l2_change,
                  "latest absolute max change": None if last is None else last.absolute_linf_change},
        requested={"internal relative L2 tolerance": search.rtol,
                   "maximum change rule": f"atol + rtol × peak; atol={_fmt(search.atol)}"},
        actions=actions, evidence={"backend_status": status,
                                   "tested_parameters": [s.parameters for s in search.steps]},
    )


def format_interaction(result: Any, *, level: str = "summary") -> str:
    if level not in {"summary", "full"}:
        raise ValueError("level must be 'summary' or 'full'.")
    search = result.search
    boundary = search.metadata.get("q_boundary_robustness", {})
    last = next((s for s in reversed(search.steps) if s.relative_l2_change is not None), None)
    rows: list[tuple[str, Any]] = [
        ("Method", result.method), ("Interpolator", result.interpolator),
        ("", None), ("Selected parameters", None), *_human_params(result.parameters),
        ("", None), ("Convergence criteria", None),
        ("Integration refinement", "PASS" if search.resolution_converged else "FAIL"),
        ("Backend robustness", "PASS" if search.hyperparameter_robust else "FAIL"),
        ("Requested relative L2", result.rtol),
        ("Internal relative L2", search.rtol),
        ("Absolute max rule", f"atol + rtol × peak; internal atol={_fmt(search.atol)}"),
        ("Last tested L2 change", None if last is None else last.relative_l2_change),
        ("Last tested max change", None if last is None else last.absolute_linf_change),
        ("Boundary test", "PASS" if boundary.get("passed") else
         ("FAIL" if boundary.get("tested") else
          "NOT APPLICABLE" if boundary.get("status") == "not_applicable_fixed_input"
          else "NOT TESTED")),
    ]
    if boundary.get("tested"):
        rows += [("Boundary L2 change", boundary.get("relative_l2_change")),
                 ("Boundary L2 limit", boundary.get("boundary_rtol")),
                 ("Boundary max change", boundary.get("absolute_linf_change")),
                 ("Boundary max limit", float(boundary["boundary_atol"]) +
                  float(boundary["boundary_rtol"]) * float(boundary["reference_peak"]))]
    if not result.converged:
        failure = result.failure
        rows += [("", None), ("Problem", None), ("  " + failure.explanation, None)]
        for action in failure.actions:
            rows += [("Try", f"{action.location}({action.parameter}=...)"),
                     ("  Why", action.instruction)]
    if level == "full":
        rows += [("", None), ("Refinement history", None)]
        for i, step in enumerate(search.steps):
            params = ", ".join(f"{k}={_fmt(v)}" for k, v in step.parameters.items())
            rows.append((f"Step {i+1}", params))
            rows.append(("  relative L2", step.relative_l2_change))
            rows.append(("  absolute max", step.absolute_linf_change))
        rows += [("", None), ("Provenance", None)]
        for key, value in result.provenance.items():
            if key in {"field1", "field2"}:
                rows.append((key, f"q={value.get('q_grid', {}).get('size', '?')} points; "
                                   f"m={_fmt(value.get('m_values', '?'))}"))
            elif key == "software":
                for name, v in value.items():
                    rows.append((name + " version", v))
            elif isinstance(value, dict):
                for k, v in value.items():
                    rows.append((f"{key}.{k}", v if not isinstance(v, (dict, list))
                                 else f"{len(v)} entries (see to_dict())"))
            else:
                rows.append((key, value))
        rows += [("", None), ("Scope", "Internal refinement of represented inputs"),
                 ("Reference accuracy", "Not tested")]
    return _lines("Interaction convergence", result.converged, rows)


def format_lowlevel(result: Any, *, level: str = "summary") -> str:
    """Compact view of standalone Hankel/interaction refinement searches."""
    if level not in {"summary", "full"}:
        raise ValueError("level must be 'summary' or 'full'.")
    last = next((s for s in reversed(result.steps)
                 if s.relative_l2_change is not None), None)
    rows: list[tuple[str, Any]] = [
        ("Method", result.method),
        ("Selected parameters", None),
        *_human_params(result.selected_parameters),
        ("", None),
        ("Resolution test", "PASS" if result.resolution_converged else "FAIL"),
        ("Backend stability", "PASS" if result.hyperparameter_robust else "FAIL"),
        ("Relative L2 tolerance", result.rtol),
        ("Absolute maximum rule", f"atol + rtol × peak; atol={_fmt(result.atol)}"),
        ("Last measured L2 change", None if last is None else last.relative_l2_change),
        ("Last absolute max change", None if last is None else last.absolute_linf_change),
    ]
    if not result.converged:
        rows += [("", None),
                 ("Problem", None),
                 ("  The tested refinements did not establish convergence.", None),
                 ("  Review selected parameters and recorded steps.", None)]
    if level == "full":
        rows += [("", None), ("Refinement history", None)]
        for i, step in enumerate(result.steps):
            rows.append((f"Step {i+1}", ", ".join(
                f"{k}={_fmt(v)}" for k, v in step.parameters.items())))
            rows.append(("  Relative L2 change", step.relative_l2_change))
            rows.append(("  Absolute max change", step.absolute_linf_change))
            rows.append(("  Runtime (s)", step.runtime_seconds))
        rows += [("", None), ("Scope", "Self-convergence only"),
                 ("Reference accuracy", "Not tested")]
    return _lines("Numerical refinement", result.converged, rows)
