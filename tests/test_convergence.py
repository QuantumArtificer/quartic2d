import numpy as np

from quartic2d._convergence import (
    ConvergenceResult,
    converge_sequence,
    relative_l2_error,
    relative_linf_error,
)
from quartic2d.interaction import HankelTransform


def test_relative_errors_zero_for_identical_arrays():
    x = np.array([1.0, 2.0, 3.0])
    assert relative_l2_error(x, x) == 0.0
    assert relative_linf_error(x, x) == 0.0


def test_converge_sequence_selects_first_stable_refinement():
    def evaluator(p):
        n = p["n"]
        return np.array([1.0 + 1.0 / n**4])

    result = converge_sequence(
        "demo",
        [{"n": 2}, {"n": 4}, {"n": 8}, {"n": 16}],
        evaluator,
        rtol=1.0e-3,
        atol=0.0,
    )
    assert isinstance(result, ConvergenceResult)
    assert result.resolution_converged
    assert result.converged
    assert result.selected_parameters["n"] in (8, 16)


def test_converge_sequence_reports_unconverged_when_range_is_insufficient():
    def evaluator(p):
        n = p["n"]
        return np.array([1.0 + 1.0 / n])

    result = converge_sequence(
        "demo",
        [{"n": 2}, {"n": 4}, {"n": 8}],
        evaluator,
        rtol=1.0e-6,
    )
    assert not result.resolution_converged
    assert not result.converged
    assert result.selected_parameters == {"n": 8}


def test_hankel_transform_convergence_and_apply_for_gl4():
    r = np.linspace(0.0, 6.0, 101)
    q = np.linspace(0.0, 5.0, 41)
    transform = HankelTransform(
        0, q, np.exp(-r * r), r, method="gl4", subdivisions=1
    )
    result = transform.converge(
        rtol=1.0e-3,
        subdivisions=(1, 2, 4),
        apply=True,
    )
    assert result.selected_parameters["subdivisions"] in (2, 4)
    assert np.all(np.isfinite(transform.F_q))


def test_converge_ogata_reports_get_h_maxiter_as_nonconverged(monkeypatch):
    from quartic2d import _convergence as convergence_module

    def fail_get_h(*args, **kwargs):
        raise Exception("Maxiter reached while checking convergence")

    from quartic2d import _numerics as numerics_module
    from types import SimpleNamespace

    monkeypatch.setattr(
        numerics_module, "_require_hankel", lambda: SimpleNamespace(get_h=fail_get_h)
    )
    result = convergence_module.converge_ogata(
        lambda x: np.exp(-np.asarray(x) ** 2),
        0,
        K=np.array([0.1, 1.0]),
        rtol=1.0e-4,
        atol=1.0e-12,
    )
    assert not result.resolution_converged
    assert not result.converged
    assert result.selected_parameters == {}
    assert result.metadata["status"] == "maxiter_reached"
    assert result.steps[0].metadata["status"] == "maxiter_reached"


def test_converge_ogata_does_not_hide_unrelated_errors(monkeypatch):
    from quartic2d import _convergence as convergence_module

    def fail_get_h(*args, **kwargs):
        raise ValueError("bad callable")

    from quartic2d import _numerics as numerics_module
    from types import SimpleNamespace

    monkeypatch.setattr(
        numerics_module, "_require_hankel", lambda: SimpleNamespace(get_h=fail_get_h)
    )
    import pytest

    with pytest.raises(ValueError, match="bad callable"):
        convergence_module.converge_ogata(lambda x: x, 0)


def test_fixed_order_convergence_selects_cheapest_verified_level():
    from quartic2d._convergence import converge_fixed_order_sequence

    exact = np.array([1.0, -0.25])

    def evaluator(parameters):
        s = parameters["subdivisions"]
        return exact + np.array([0.6, -0.2]) / s**4

    loose = converge_fixed_order_sequence(
        "simpson",
        "subdivisions",
        (1, 2, 4, 8),
        evaluator,
        expected_order=4,
        rtol=1.0,
    )
    assert loose.converged
    assert loose.selected_parameters == {"subdivisions": 1}

    tight = converge_fixed_order_sequence(
        "simpson",
        "subdivisions",
        (1, 2, 4, 8, 16),
        evaluator,
        expected_order=4,
        rtol=1.0e-3,
    )
    assert tight.converged
    assert tight.selected_parameters["subdivisions"] >= 4
    assert tight.metadata["last_verification"]["asymptotic_regime_verified"]


def test_fftlog_convergence_rejects_resolution_stable_bias_sensitive_floor():
    from quartic2d._convergence import converge_fftlog_bias_resolution

    def evaluator(parameters):
        n = parameters["n"]
        bias = parameters["bias"]
        # n converges rapidly, but the plateau varies strongly with bias.
        return np.array([1.0 + 1.0 / n**2 + 0.2 * (bias + 0.5)])

    result = converge_fftlog_bias_resolution(
        evaluator,
        n_values=(128, 256, 512, 1024),
        bias_values=(-0.6, -0.55, -0.5, -0.45, -0.4),
        preferred_bias=-0.5,
        rtol=1.0e-3,
    )
    assert result.resolution_converged
    assert not result.hyperparameter_robust
    assert not result.converged
    assert result.metadata["status"] == "bias_robustness_not_established"


def test_fftlog_convergence_accepts_resolution_and_bias_robust_plateau():
    from quartic2d._convergence import converge_fftlog_bias_resolution

    def evaluator(parameters):
        n = parameters["n"]
        bias = parameters["bias"]
        return np.array([1.0 + 1.0 / n**2 + 1.0e-4 * (bias + 0.5) ** 2])

    result = converge_fftlog_bias_resolution(
        evaluator,
        n_values=(128, 256, 512, 1024),
        bias_values=(-0.6, -0.55, -0.5, -0.45, -0.4),
        preferred_bias=-0.5,
        rtol=1.0e-3,
    )
    assert result.converged
    assert result.hyperparameter_robust
    assert result.selected_parameters["bias"] == -0.5


def test_interaction_converge_parameters_builds_reusable_finite_result():
    from quartic2d import Interaction

    class Field:
        def __init__(self):
            self.q = np.linspace(0.0, 8.0, 129)
            self.m_values = np.array([0])
            self.F_q = {0: np.exp(-0.25 * self.q**2).astype(np.complex128)}

        def __call__(self, m, q):
            return np.interp(q, self.q, self.F_q[int(m)].real)

    field = Field()
    deltas = np.column_stack((np.geomspace(0.1, 10.0, 12), np.zeros(12)))

    def kernel(q):
        q = np.asarray(q, dtype=float)
        return np.exp(-0.1 * q * q)

    result = Interaction.converge_parameters(
        deltas,
        field,
        field,
        kernel,
        method="simpson",
        rtol=1.0e-4,
        atol=1.0e-12,
        subdivisions=(1, 2, 4, 8, 16),
        verbose=False,
    )
    assert result.converged
    assert result.parameters["method"] == "simpson"
    production = result.interaction(deltas, field, field, kernel)
    assert production.convergence is result
    assert np.all(np.isfinite(production.V))


def test_ogata_coupled_convergence_uses_N_and_h_lookahead():
    from quartic2d._convergence import converge_ogata_coupled

    exact = np.array([1.0, 0.25, -0.4], dtype=np.complex128)

    def evaluator(parameters):
        N = float(parameters["N"])
        h = float(parameters["h"])
        return (
            exact
            + np.array([0.8, -0.2, 0.1]) / N**2
            + np.array([0.3, 0.1, -0.2]) * h**2
        )

    result = converge_ogata_coupled(
        evaluator,
        n_values=(32, 64, 128, 256, 512, 1024),
        hstart=0.1,
        hdecrement=2.0,
        maxiter=8,
        rtol=1.0e-4,
        atol=1.0e-12,
    )

    assert result.converged
    assert result.metadata["kind"] == "ogata_coupled_N_h"
    assert result.metadata["status"] == "converged"
    assert {"N", "h"}.issubset(result.selected_parameters)
    assert len(result.metadata["per_h"]) >= 3
    assert any(step.converged for step in result.steps)


def test_ogata_coupled_convergence_reports_N_limit():
    from quartic2d._convergence import converge_ogata_coupled

    def evaluator(parameters):
        # Intentionally remains strongly N-dependent across the tested ladder.
        N = float(parameters["N"])
        h = float(parameters["h"])
        return np.array([1.0 + 20.0 / N + h])

    result = converge_ogata_coupled(
        evaluator,
        n_values=(16, 32, 64),
        hstart=0.05,
        hdecrement=2.0,
        maxiter=4,
        rtol=1.0e-6,
        atol=0.0,
    )

    assert not result.converged
    assert result.metadata["status"] == "n_limit_reached"


def test_ogata_coupled_can_converge_below_old_h_floor():
    from quartic2d._convergence import converge_ogata_coupled

    exact = np.array([1.0, -0.3], dtype=np.complex128)

    def evaluator(parameters):
        N = float(parameters["N"])
        h = float(parameters["h"])
        # The h term is deliberately large enough that the old 15-level
        # h ladder cannot certify this tolerance, while a deeper ladder can.
        return exact + np.array([1.0, 0.4]) / N**2 + np.array([4.0, -2.0]) * h

    result = converge_ogata_coupled(
        evaluator,
        n_values=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
        hstart=0.05,
        hdecrement=2.0,
        maxiter=20,
        rtol=1.0e-5,
        atol=1.0e-12,
    )

    assert result.converged
    assert result.selected_parameters["h"] < 0.05 / 2.0**14
    assert result.metadata["h_min_tested"] <= result.selected_parameters["h"]
    assert result.metadata["terminal_h_trend"] == "converged"
    assert result.metadata["n_max_available"] == 8192


def test_ogata_coupled_reports_improving_h_limit():
    from quartic2d._convergence import converge_ogata_coupled

    exact = np.array([1.0], dtype=np.complex128)

    def evaluator(parameters):
        N = float(parameters["N"])
        h = float(parameters["h"])
        return exact + np.array([1.0 / N**4 + h])

    result = converge_ogata_coupled(
        evaluator,
        n_values=(64, 128, 256, 512, 1024, 2048),
        hstart=0.05,
        hdecrement=2.0,
        maxiter=5,
        rtol=1.0e-8,
        atol=0.0,
    )

    assert not result.converged
    assert result.metadata["status"] == "h_limit_reached"
    assert result.metadata["terminal_h_trend"] in {"monotonic_improving", "improving"}
    assert result.metadata["terminal_h_change_metric"] is not None


def test_fftlog_convergence_evaluates_only_admissible_bias_windows_once():
    from quartic2d._convergence import converge_fftlog_bias_resolution

    calls = []

    def evaluator(parameters):
        key = (int(parameters["n"]), float(parameters["bias"]))
        calls.append(key)
        n, bias = key
        return np.array([1.0 + 1.0 / n**2 + 1.0e-4 * (bias + 0.5) ** 2])

    result = converge_fftlog_bias_resolution(
        evaluator,
        n_values=(128, 256, 512, 1024),
        bias_values=(-0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35),
        preferred_bias=-0.5,
        rtol=1.0e-3,
    )

    assert result.converged
    assert result.metadata["admissible_bias_values"] == [-0.6, -0.55, -0.5, -0.45, -0.4]
    assert result.metadata["tested_bias_values"] == [-0.55, -0.5, -0.45]
    assert not result.metadata["fallback_used"]
    assert all(bias not in {-0.65, -0.60, -0.40, -0.35} for _, bias in calls)
    assert len(calls) == len(set(calls))
    assert len(calls) == 9


def test_fixed_order_convergence_falls_back_to_verified_stability_when_order_is_wrong():
    from quartic2d._convergence import converge_fixed_order_sequence

    exact = np.array([1.0])

    def evaluator(parameters):
        # Deliberately first-order rather than the nominal GL4 eighth order.
        # The Richardson order gate must reject it, but two consecutive small
        # changes at the requested budget are still a valid conservative
        # self-convergence certificate.
        s = parameters["subdivisions"]
        return exact + np.array([1.0e-2 / s])

    result = converge_fixed_order_sequence(
        "gl4",
        "subdivisions",
        (1, 2, 4, 8, 16, 32, 64),
        evaluator,
        expected_order=8,
        rtol=5.0e-4,
    )

    assert result.converged
    assert result.metadata["fallback_verification"]["kind"] == "verified_sequence_fallback"
    assert "verified_by_stable_refinement" in next(
        step.metadata for step in result.steps if step.converged
    )


def test_interaction_autoconvergence_rejects_q_boundary_sensitive_sampled_field():
    from quartic2d import Interaction

    class AutomaticallySampledField:
        def __init__(self, *, fast_decay: bool):
            self.q = np.linspace(0.0, 8.0, 129)
            self.m_values = np.array([0])
            exponent = -0.25 * self.q**2 if fast_decay else -0.02 * self.q**2
            self.F_q = {0: np.exp(exponent).astype(np.complex128)}
            # The concrete QSamplingResult contents are not needed by the
            # downstream probe; non-None marks an automatically certified
            # HarmonicTransform input.
            self.sampling_convergence = object()

    deltas = np.column_stack((np.geomspace(10.0, 100.0, 8), np.zeros(8)))

    def kernel(q):
        return np.ones_like(np.asarray(q, dtype=float))

    unsafe = AutomaticallySampledField(fast_decay=False)
    rejected = Interaction.converge_parameters(
        deltas,
        unsafe,
        unsafe,
        kernel,
        method="gl4",
        rtol=1.0e-3,
        atol=1.0e-12,
        subdivisions=(1, 2, 4, 8, 16, 32),
        verbose=False,
    )
    boundary = rejected.search.metadata["q_boundary_robustness"]
    assert rejected.search.resolution_converged
    assert not rejected.converged
    assert boundary["tested"]
    assert not boundary["passed"]
    assert boundary["status"] == "upstream_q_boundary_not_robust"
    assert rejected.search.metadata["status"] == "upstream_q_boundary_not_robust"

    safe = AutomaticallySampledField(fast_decay=True)
    accepted = Interaction.converge_parameters(
        deltas,
        safe,
        safe,
        kernel,
        method="gl4",
        rtol=1.0e-3,
        atol=1.0e-12,
        subdivisions=(1, 2, 4, 8, 16, 32),
        verbose=False,
    )
    assert accepted.converged
    assert accepted.search.metadata["q_boundary_robustness"]["passed"]


def test_interaction_autoconvergence_keeps_fixed_input_semantics_without_sampling_certificate():
    from quartic2d import Interaction

    class FixedField:
        def __init__(self):
            self.q = np.linspace(0.0, 8.0, 129)
            self.m_values = np.array([0])
            self.F_q = {0: np.exp(-0.02 * self.q**2).astype(np.complex128)}

    field = FixedField()
    deltas = np.column_stack((np.geomspace(10.0, 100.0, 8), np.zeros(8)))
    result = Interaction.converge_parameters(
        deltas,
        field,
        field,
        lambda q: np.ones_like(np.asarray(q, dtype=float)),
        method="gl4",
        rtol=1.0e-3,
        atol=1.0e-12,
        subdivisions=(1, 2, 4, 8, 16, 32),
        verbose=False,
    )
    assert result.converged
    boundary = result.search.metadata["q_boundary_robustness"]
    assert not boundary["tested"]
    assert boundary["status"] == "not_applicable_fixed_input"
