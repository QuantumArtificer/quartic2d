import numpy as np
import pytest
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform


def gaussian(x, y):
    return np.exp(-(x**2 + y**2)) / np.pi


def make_decomposition():
    x = np.linspace(-5.0, 5.0, 81)
    y = np.linspace(-5.0, 5.0, 81)
    return PolarDecomposition(
        gaussian,
        x,
        y,
        Nr=81,
        Ntheta=64,
        rmax=4.5,
        recon_err_tol=1e-4,
        radial_power_tail_fraction=1e-10,
        radial_relative_amplitude_threshold=1e-6,
        origin=(0.0, 0.0),
    )


def test_petal2d_harmonics_and_evaluation_shapes():
    transformed = HarmonicTransform(
        make_decomposition(), q_max=8.0, n_q=80, method="simpson"
    )

    assert transformed.m_values.tolist() == [0]
    assert transformed.q.shape == (80,)
    assert transformed[0].shape == (80,)
    assert np.isscalar(transformed(0, 1.0))
    assert transformed(0, np.array([0.5, 1.0])).shape == (2,)
    assert transformed(0, np.ones((2, 3))).shape == (2, 3)


def test_simpson_is_default_harmonic_method():
    transformed = HarmonicTransform(make_decomposition(), q_max=8.0, n_q=80)
    assert transformed.method == "simpson"


def test_petal2d_decomposition_is_not_modified():
    dec = make_decomposition()
    before = {m: values.copy() for m, values in dec.rho.items()}

    HarmonicTransform(dec, q_max=8.0, n_q=80, method="gl4")

    for m, values in before.items():
        np.testing.assert_array_equal(dec.rho[m], values)


def test_harmonic_setters_recompute():
    transformed = HarmonicTransform(
        make_decomposition(), q_max=8.0, n_q=80, method="trapezoid"
    )
    before = transformed.F_q[0].copy()

    assert transformed.set_method("gl8") is transformed
    assert transformed.method == "gl8"
    assert np.linalg.norm(transformed.F_q[0] - before) > 0.0

    before = transformed.F_q[0].copy()
    assert transformed.set_interpolator("pchip") is transformed
    assert transformed.interpolator == "pchip"
    assert np.linalg.norm(transformed.F_q[0] - before) > 0.0


def test_missing_harmonic_raises_key_error():
    transformed = HarmonicTransform(
        make_decomposition(), q_max=8.0, n_q=80, method="simpson"
    )

    with pytest.raises(KeyError, match="not available"):
        transformed(4, 1.0)


def test_harmonics_reject_invalid_grid_parameters():
    dec = make_decomposition()

    with pytest.raises(ValueError, match="q_max"):
        HarmonicTransform(dec, q_max=0.0)

    with pytest.raises(ValueError, match="n_q"):
        HarmonicTransform(dec, q_max=4.0, n_q=1)


def test_cutoff_radius_is_required():
    class IncompleteDecomposition:
        r = np.linspace(0.0, 1.0, 5)
        rho = {0: np.ones(5)}
        m_sorted = [0]
        power_fracs = {0: 1.0}

        def __getitem__(self, m):
            return self.rho[m]

    with pytest.raises(AttributeError, match="cutoff_radius"):
        HarmonicTransform(IncompleteDecomposition(), q_max=4.0, n_q=16)


def test_roundtrip_error_returns_relative_fraction():
    transformed = HarmonicTransform(make_decomposition(), q_max=8.0, n_q=80)
    error = transformed.roundtrip_error()
    assert np.isfinite(error)
    assert error >= 0.0
    assert error < 1.0
