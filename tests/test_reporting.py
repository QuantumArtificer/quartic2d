"""Tests for data-preserving, concise and actionable reporting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from quartic2d import (
    ConvergenceAction,
    ConvergenceError,
    ConvergenceFailureReport,
    HarmonicConvergenceResult,
    InteractionConvergenceResult,
    ProgressEvent,
)
from quartic2d._convergence import ConvergenceResult, ConvergenceStep
from quartic2d._sampling import QInterpolationStep, QSamplingResult, QTailModeResult
from quartic2d.reporting import decomposition_provenance, emit_progress


def test_failure_reports_keep_evidence_without_dumping_dictionaries():
    report = ConvergenceFailureReport(
        code="Q_RANGE_LIMITED_BY_RADIAL_GRID",
        stage="hankel_q_range",
        explanation="The sampled radial grid cannot test the required higher momenta.",
        observed={"radial-grid q limit": 1.88},
        requested={"q_tail_rtol": 0.001},
        limiting_harmonics=(-1, 1),
        actions=(ConvergenceAction("PolarDecomposition", "Nr", "Refine the radial grid."),),
        evidence={"harmonics": {str(m): {"samples": list(range(100))} for m in (-1, 1)}},
    )
    err = ConvergenceError(report)
    assert isinstance(err, RuntimeError)
    assert "PolarDecomposition(Nr=...)" in str(err)
    assert "radial-grid q limit" in str(err)
    assert "samples" not in str(err)
    assert "2 entries" in report.format(level="full")
    assert report.to_dict()["limiting_harmonics"] == [-1, 1]
    assert report.to_dict()["evidence"]["harmonics"]["-1"]["samples"][-1] == 99
    json.dumps(report.to_dict())


def _harmonic_result():
    sampling = QSamplingResult(
        q_ceiling=6.0, q_max=3.0, n_q=9, support_radius=4.0,
        q_tail_rtol=1.0e-3, interpolation_rtol=1.0e-4,
        interpolation_atol=1.0e-12, tail_pilot_oversampling=8.0,
        interpolation_oversampling=4.0,
        tail_modes={-1: QTailModeResult(-1, 2.5, 2e-4, 1e-7, True),
                    1: QTailModeResult(1, 2.5, 3e-4, 1e-7, True)},
        q_grid=np.linspace(0, 3, 9),
        interpolation_steps=[QInterpolationStep(4.0, 9,
            {-1: {"relative_l2": 1e-5, "absolute_linf": 1e-6},
             1: {"relative_l2": 2e-5, "absolute_linf": 1e-6}}, True)],
    )
    quadrature = ConvergenceResult(
        method="simpson", rtol=1e-4, atol=1e-12,
        selected_parameters={"subdivisions": 4},
        selected_values=np.array([1.0]),
        steps=[ConvergenceStep({"subdivisions": 4}, 0.1, relative_l2_change=1e-5)],
        resolution_converged=True,
        metadata={"harmonics": {"-1": {"steps": [
                        {"relative_l2_change": 3e-5, "absolute_linf_change": 2e-6}]}}},
    )
    return HarmonicConvergenceResult(
        sampling=sampling, quadrature=quadrature, method="simpson",
        interpolator="cubic", rtol=1e-4, atol=1e-12, q_tail_rtol=1e-3,
        provenance={"software": {"quartic2d": "0.1.0"},
                    "decomposition": {"Nr": 151, "Ntheta": 256}},
    )


def test_harmonic_success_reports_measured_errors_and_reproducible_metadata():
    result = _harmonic_result()
    brief = str(result)
    full = result.report(level="full")
    assert result.converged and result.failure is None
    assert "PASS" in brief
    assert "Max measured tail L2" in brief
    assert "Max q interpolation L2" in brief
    assert "Max radial quadrature L2" in brief
    assert "m=-1 Parseval mismatch" in full
    assert "Nr" in full and "Ntheta" in full
    assert "Not tested" in full  # independent reference accuracy
    archived = result.to_dict()
    assert archived["provenance"]["decomposition"]["Nr"] == 151
    assert archived["sampling"]["q_grid"][-1] == 3.0
    json.dumps(archived)
    with pytest.raises(ValueError):
        result.report(level="unknown")


def test_interaction_boundary_failure_distinct_from_quadrature_failure():
    search = ConvergenceResult(
        method="gl4", rtol=5e-5, atol=5e-13,
        selected_parameters={"subdivisions": 16},
        selected_values=np.array([1.0]),
        steps=[ConvergenceStep({"subdivisions": 16}, 0.1,
                               relative_l2_change=1e-5, absolute_linf_change=1e-6)],
        resolution_converged=True, hyperparameter_robust=False,
        metadata={"status": "upstream_q_boundary_not_robust",
                  "q_boundary_robustness": {
                      "tested": True, "passed": False,
                      "relative_l2_change": 0.00786, "boundary_rtol": 0.005,
                      "absolute_linf_change": 0.002,
                      "boundary_atol": 1e-12, "reference_peak": 1.0}},
    )
    result = InteractionConvergenceResult(search, "gl4", "cubic", 0.01, 1e-12)
    assert result.failure.code == "Q_BOUNDARY_NOT_ROBUST"
    assert result.failure.stage == "interaction_boundary"
    assert "HarmonicTransform(q_max=...)" in str(result)
    assert "0.00786" in str(result)
    assert "0.005" in str(result)
    with pytest.raises(ConvergenceError) as err:
        result.interaction(np.zeros((1, 2)), object(), object(), lambda q: q)
    assert err.value.report.code == "Q_BOUNDARY_NOT_ROBUST"
    assert result.to_dict()["failure"]["stage"] == "interaction_boundary"
    json.dumps(result.to_dict())


def test_interaction_success_is_quantitative_and_full_report_retains_history():
    search = ConvergenceResult(
        method="gl4", rtol=5e-5, atol=5e-13,
        selected_parameters={"subdivisions": 8},
        selected_values=np.array([1.0]),
        steps=[ConvergenceStep({"subdivisions": 4}, 0.1, relative_l2_change=2e-4),
               ConvergenceStep({"subdivisions": 8}, 0.1,
                               relative_l2_change=1e-5, absolute_linf_change=1e-6)],
        resolution_converged=True,
        metadata={"q_boundary_robustness": {"status": "not_applicable_fixed_input"}},
    )
    result = InteractionConvergenceResult(search, "gl4", "cubic", 1e-4, 1e-12,
        provenance={"software": {"quartic2d": "0.1.0"},
                    "user": {"kernel": "screened Coulomb"}})
    assert result.converged and result.failure is None
    assert "NOT APPLICABLE" in str(result)
    assert "Last tested L2 change" in str(result)
    full = result.report(level="full")
    assert "Step 1" in full and "Step 2" in full
    assert "quartic2d version" in full
    assert "screened Coulomb" in full


def test_upstream_provenance_does_not_invent_units():
    dec = SimpleNamespace(
        x=np.linspace(-5, 5, 11), y=np.linspace(-2, 2, 5),
        r=np.linspace(0, 3, 9), theta=np.linspace(0, 2, 6),
        m_sorted=[-1, 1], cutoff_radius={-1: 2.0, 1: 2.0},
        recon_error_measured=1e-3, _recon_err_tol=0.01,
    )
    meta = decomposition_provenance(dec)
    assert meta["r_grid"]["size"] == 9
    assert meta["recon_err_tol"] == 0.01
    assert not any("Å" in key or "meV" in key for key in meta)
    json.dumps(meta)


def test_progress_callback_receives_structured_events_and_is_opt_in():
    events = []
    emit_progress(events.append, "interaction_quadrature", "evaluated",
                  parameters={"subdivisions": 2}, completed=1)
    emit_progress(None, "interaction_quadrature", "evaluated")
    assert len(events) == 1
    assert isinstance(events[0], ProgressEvent)
    assert events[0].parameters == {"subdivisions": 2}
    assert events[0].completed == 1


def test_harmonic_calibration_preserves_sampling_failure(monkeypatch):
    import quartic2d._sampling as sampling
    from quartic2d import HarmonicTransform

    report = ConvergenceFailureReport(
        code="Q_RANGE_LIMITED_BY_RADIAL_GRID", stage="hankel_q_range",
        explanation="The radial grid cannot test whether enough of the transform is included.",
        observed={"radial-grid q limit": 2.0},
        actions=(ConvergenceAction("PolarDecomposition", "Nr", "Check radial resolution."),),
    )

    def fail(*args, **kwargs):
        raise ConvergenceError(report)

    monkeypatch.setattr(sampling, "converge_q_sampling", fail)
    events = []
    with pytest.raises(ConvergenceError) as exc:
        HarmonicTransform.converge_parameters(object(), verbose=False,
                                              progress=events.append)
    assert exc.value.report is report
    assert events[0].stage == "hankel_q_sampling" and events[0].status == "started"
    assert events[-1].stage == "hankel_q_range" and events[-1].status == "failed"


def test_standalone_refinement_result_is_short_but_has_full_history():
    search = ConvergenceResult(
        method="simpson", rtol=1e-4, atol=1e-10,
        selected_parameters={"subdivisions": 4},
        selected_values=np.array([1.0]),
        steps=[ConvergenceStep({"subdivisions": 1}, 0.01),
               ConvergenceStep({"subdivisions": 4}, 0.01,
                               relative_l2_change=1e-5)],
        resolution_converged=False,
    )
    assert "FAIL" in str(search)
    assert "Refinement history" not in str(search)
    assert "Step 2" in search.report(level="full")
    assert "Not tested" in search.report(level="full")
    json.dumps(search.to_dict())
