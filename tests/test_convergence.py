from typing import ClassVar

import numpy as np
from scipy.integrate import simpson

from quartic2d import HarmonicTransform
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

# ---- q-sampling core behavior ----
from quartic2d._sampling import radial_q_ceiling


def make_anisotropic_decomposition():
    anisotropy = 0.35
    r = np.linspace(0.0, 6.0, 256)
    rho0 = np.exp(-r * r) / np.pi
    rho2 = anisotropy * r * r * np.exp(-r * r) / (2.0 * np.pi)

    def cutoff(profile, tail=1.0e-8, amplitude=1.0e-6):
        power = r * np.abs(profile) ** 2
        total = simpson(power, x=r)
        power_radius = r[-1]
        for i in range(2, r.size):
            inside = simpson(power[: i + 1], x=r[: i + 1])
            if max(total - inside, 0.0) <= tail * total:
                power_radius = r[i]
                break
        threshold = amplitude * np.max(np.abs(profile))
        indices = np.flatnonzero(np.abs(profile) >= threshold)
        amplitude_radius = r[indices[-1]]
        return float(max(power_radius, amplitude_radius))

    R0 = cutoff(rho0)
    R2 = cutoff(rho2)
    p0 = simpson(r * np.abs(rho0) ** 2, x=r)
    p2 = simpson(r * np.abs(rho2) ** 2, x=r)
    total = p0 + 2.0 * p2

    class Decomposition:
        def __init__(self):
            self.r = r
            self.rho = {0: rho0, 2: rho2, -2: rho2.copy()}
            self.m_sorted = [0, 2, -2]
            self.cutoff_radius = {0: R0, 2: R2, -2: R2}
            self.power_fracs = {0: p0 / total, 2: p2 / total, -2: p2 / total}

        def __getitem__(self, m):
            return self.rho[int(m)]

    return Decomposition()


def test_radial_q_ceiling_matches_sampling_bound():
    dec = make_anisotropic_decomposition()
    expected = np.pi / np.max(np.diff(dec.r))
    assert radial_q_ceiling(dec) == expected


def test_default_harmonic_transform_is_one_pass_and_has_fast_diagnostics():
    dec = make_anisotropic_decomposition()
    field = HarmonicTransform(dec)

    assert field.sampling_convergence is None
    assert field.quadrature_convergence is None
    assert field.diagnostics is not None
    assert field.q[-1] < field.diagnostics.q_ceiling
    assert 32 <= field.q.size <= 512
    assert field.diagnostics.q_sampling_factor >= 5.0 - 1.0e-12


def test_converge_parameters_returns_reusable_verified_parameters():
    dec = make_anisotropic_decomposition()
    result = HarmonicTransform.converge_parameters(dec, verbose=False)

    assert result.converged
    assert 5.0 < result.parameters["q_max"] < 8.0
    assert result.parameters["n_q"] < 128
    assert result.parameters["subdivisions"] in {1, 2, 4, 8, 16, 32}

    field = result.transform(dec, check=False)
    assert field.sampling_convergence is result.sampling
    assert field.quadrature_convergence is result.quadrature



def test_sampling_calibration_is_independent_of_production_backend():
    dec = make_anisotropic_decomposition()
    simpson_result = HarmonicTransform.converge_parameters(
        dec, method="simpson", verbose=False
    )
    gl4_result = HarmonicTransform.converge_parameters(
        dec, method="gl4", verbose=False
    )

    assert simpson_result.converged
    assert gl4_result.converged
    assert simpson_result.sampling.q_max == gl4_result.sampling.q_max
    assert simpson_result.sampling.n_q == gl4_result.sampling.n_q
    assert simpson_result.parameters["method"] == "simpson"
    assert gl4_result.parameters["method"] == "gl4"


def test_q_support_certification_uses_requested_tail_budget(monkeypatch):
    """Do not reject a converged pilot because of an arbitrary sub-budget."""
    import quartic2d._sampling as sampling

    class Decomposition:
        r = np.linspace(0.0, 4.0, 129)
        m_sorted: ClassVar[list[int]] = [0]
        cutoff_radius: ClassVar[dict[int, float]] = {0: 4.0}
        power_fracs: ClassVar[dict[int, float]] = {0: 1.0}
        _rho = np.ones_like(r)

        def __getitem__(self, m):
            return self._rho

    # Isolate certification logic from Hankel quadrature.  A constant fake
    # form factor makes cumulative Simpson integration exact on every pilot
    # grid.  The represented power is chosen so the full-domain Parseval
    # mismatch is exactly ~4e-7: below the requested 1e-6 tail-power budget,
    # but above the old arbitrary 2.5e-7 quarter-budget that rejected the real
    # PETAL2D example.
    def fake_grouped(supports, q, **kwargs):
        return {0: np.ones_like(np.asarray(q), dtype=float)}

    q_ceiling = np.pi / (Decomposition.r[1] - Decomposition.r[0])
    q_power = 0.5 * q_ceiling**2
    represented_power = q_power / (1.0 - 4.0e-7)

    monkeypatch.setattr(sampling, "_grouped_hankel_transforms", fake_grouped)
    import quartic2d._diagnostics as diagnostics
    monkeypatch.setattr(
        diagnostics,
        "default_q_grid",
        lambda decomposition: (q_ceiling, 33, {}),
    )
    monkeypatch.setattr(
        sampling,
        "_represented_radial_power",
        lambda *args, **kwargs: represented_power,
    )

    _, q_max, pilot_s, modes = sampling.estimate_q_support(
        Decomposition(),
        q_tail_rtol=1.0e-3,
        pilot_oversampling=(16.0, 32.0, 64.0),
    )

    assert pilot_s > 0.0
    assert modes[0].resolved
    assert 2.5e-7 < modes[0].parseval_relative_error < 1.0e-6
    assert q_max <= q_ceiling
