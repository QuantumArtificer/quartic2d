import sys
from pathlib import Path

import numpy as np
import pytest

from quartic2d import HarmonicTransform

# ``paper`` is intentionally not an installed package (setuptools discovers
# only ``src``), so tests that reuse manuscript reference workloads must add
# the benchmark helper directory explicitly rather than importing ``paper``.
_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "paper" / "benchmarks"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from run_harmonic_transform import (  # noqa: E402
    SyntheticDecomposition,
    q_tail_l2_analytic,
    signed_exact,
    workloads,
)


_WORKLOADS = {item.name: item for item in workloads()}
_VALIDATED_NR = {
    "gaussian_isotropic": 160,
    "gaussian_odd_pair": 160,
    "gaussian_anisotropic_m2": 160,
    "gaussian_high_order_m4": 160,
    "nodal_mixed": 256,
    "complex_mixed": 160,
    "exponential_cusp": 1536,
    "oscillatory_exponential": 2048,
    "algebraic_mixed": 3072,
}


@pytest.mark.parametrize("name", tuple(_VALIDATED_NR))
def test_adaptive_sampling_certifies_diverse_analytic_profiles(name):
    """Sampling convergence is profile-agnostic across the analytic suite."""
    workload = _WORKLOADS[name]
    decomposition = SyntheticDecomposition(workload, _VALIDATED_NR[name])

    result = HarmonicTransform.converge_parameters(
        decomposition,
        rtol=1.0e-4,
        atol=1.0e-12,
        q_tail_rtol=1.0e-3,
        method="simpson",
        verbose=False,
    )

    assert result.converged
    assert result.sampling.grid_kind == "adaptive"
    q_grid = np.asarray(result.sampling.q_grid)
    assert q_grid.ndim == 1
    assert q_grid.size == result.sampling.n_q
    assert q_grid[0] == 0.0
    assert np.all(np.diff(q_grid) > 0.0)

    field = result.transform(decomposition, check=False)
    q_dense = np.linspace(0.0, result.sampling.q_max, 3001)
    for mode in workload.modes:
        exact = signed_exact(mode, q_dense)
        approx = field(mode.m, q_dense)
        relative_l2 = np.linalg.norm(approx - exact) / max(
            np.linalg.norm(exact), np.finfo(float).tiny
        )
        # This is an external analytic check, not the self-convergence metric.
        assert relative_l2 < 1.0e-4
        assert q_tail_l2_analytic(mode, result.sampling.q_max) <= 1.0e-3


@pytest.mark.parametrize("name", ("top_hat_stress", "annulus_stress"))
def test_discontinuous_slow_tail_profiles_refuse_false_q_support_certificate(name):
    """A finite radial grid must not falsely certify unresolved algebraic q tails."""
    workload = _WORKLOADS[name]
    decomposition = SyntheticDecomposition(workload, 256)

    with pytest.raises(RuntimeError, match="Could not certify q-space support"):
        HarmonicTransform.converge_parameters(
            decomposition,
            rtol=1.0e-4,
            atol=1.0e-12,
            q_tail_rtol=1.0e-3,
            method="simpson",
            verbose=False,
        )
