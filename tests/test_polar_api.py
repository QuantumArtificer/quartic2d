"""QUARTIC2D's PETAL2D wrapper must not alter numerical decomposition results."""

import json
from pathlib import Path

import numpy as np
import pytest
from petal2d import PolarDecomposition as UpstreamPolarDecomposition

from quartic2d import HarmonicTransform, PolarDecomposition


def gaussian(x, y):
    return np.exp(-(x * x + y * y)) / np.pi


@pytest.fixture(scope="module")
def decomposition_pair():
    x = np.linspace(-5.0, 5.0, 65)
    kwargs = {
            "Nr": 65,
            "Ntheta": 96,
            "rmax": 4.0,
            "origin": (0.0, 0.0),
            "recon_err_tol": 1e-4,
        }
    return (PolarDecomposition(gaussian, x, x, **kwargs),
            UpstreamPolarDecomposition(gaussian, x, x, **kwargs))


def test_polar_wrapper_preserves_upstream_numerics(decomposition_pair):
    wrapped, upstream = decomposition_pair
    assert isinstance(wrapped, UpstreamPolarDecomposition)
    assert wrapped.m_sorted == upstream.m_sorted
    np.testing.assert_allclose(wrapped.r, upstream.r, rtol=0, atol=0)
    assert wrapped.recon_error_measured == pytest.approx(upstream.recon_error_measured)
    for m in wrapped.m_sorted:
        np.testing.assert_allclose(wrapped.rho[m], upstream.rho[m], rtol=0, atol=0)


def test_polar_reporting_is_compact_and_serializable(decomposition_pair):
    wrapped, _ = decomposition_pair
    summary = wrapped.report()
    detailed = wrapped.report(level="full")
    assert "QUARTIC2D · Polar decomposition" in summary
    assert "Measured reconstruction" in summary
    assert "Harmonic selection target" in summary
    assert "Input and decomposition settings" not in summary
    assert "Input and decomposition settings" in detailed
    assert "radial cutoffs" in detailed.lower() or "Radial cutoffs" in detailed
    assert str(wrapped) == summary
    record = wrapped.to_dict()
    assert record["stage"] == "polar_decomposition"
    assert record["parameters_and_diagnostics"]["r_grid"]["size"] == 65
    assert record["reconstruction_error_percent"] == pytest.approx(
        wrapped.recon_error_measured
    )
    assert "quartic2d" in record["software"]
    json.dumps(record)
    with pytest.raises(ValueError, match="level"):
        wrapped.report(level="unknown")


def test_legacy_and_wrapped_inputs_both_work(decomposition_pair):
    wrapped, upstream = decomposition_pair
    a = HarmonicTransform(wrapped, q_max=3.0, n_q=32, check=False)
    b = HarmonicTransform(upstream, q_max=3.0, n_q=32, check=False)
    assert np.array_equal(a.m_values, b.m_values)
    for m in a.m_values:
        np.testing.assert_allclose(a.F_q[int(m)], b.F_q[int(m)], rtol=0, atol=0)


def test_public_examples_do_not_import_petal2d_directly():
    root = Path(__file__).resolve().parents[1] / "examples"
    for example in root.glob("*.py"):
        assert "from petal2d import" not in example.read_text()
