# ---- test_helpers.py ----
from typing import ClassVar

import numpy as np
import pytest
from petal2d import PolarDecomposition

from quartic2d import HankelTransform, HarmonicTransform
from quartic2d._numerics import Interpolator1D, composite_gauss_nodes_weights


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_interpolators_support_complex_data_and_zero_exterior(method):
    x = np.linspace(0.0, 2.0, 21)
    y = np.exp(1j * x)
    f = Interpolator1D(x, y, method)

    values = f(np.array([0.3, 1.1]))
    assert values.shape == (2,)
    assert np.iscomplexobj(values)
    assert f(-1.0) == 0.0
    assert f(3.0) == 0.0


def test_composite_gauss_nodes_are_inside_intervals():
    x = np.array([0.0, 0.5, 2.0])
    for order in (4, 8):
        nodes, weights = composite_gauss_nodes_weights(x, order)
        assert nodes.size == (x.size - 1) * order
        assert weights.size == nodes.size
        assert np.all(nodes > x[0])
        assert np.all(nodes < x[-1])
        assert np.all(weights > 0.0)


def test_not_a_knot_cubic_is_available_for_q_space_boundary_conditions():
    x = np.linspace(0.0, 7.0, 39)
    y = np.exp(-x * x / 4.0)
    probe = np.linspace(0.0, 0.5, 101)
    exact = np.exp(-probe * probe / 4.0)

    natural = Interpolator1D(x, y, "cubic")
    q_space = Interpolator1D(x, y, "cubic", cubic_bc_type="not-a-knot")

    natural_error = np.linalg.norm(natural(probe) - exact)
    q_space_error = np.linalg.norm(q_space(probe) - exact)
    assert q_space_error < natural_error

# ---- test_hankel.py ----


def gaussian_profile(r):
    return np.exp(-r**2) / np.pi


@pytest.mark.parametrize("method", ["trapezoid", "simpson", "gl4", "gl8"])
def test_gaussian_hankel_finite_quadratures(method):
    r = np.linspace(0.0, 6.0, 401)
    q = np.linspace(0.0, 6.0, 121)
    transform = HankelTransform(
        0,
        q,
        gaussian_profile(r),
        r,
        r_cutoff=5.5,
        interpolator="cubic",
        method=method,
    )

    exact = np.exp(-q**2 / 4.0) / (2.0 * np.pi)
    relative_l2 = np.linalg.norm(transform.F_q.real - exact) / np.linalg.norm(exact)

    assert relative_l2 < 5e-4
    assert np.max(np.abs(transform.F_q.imag)) < 1e-10



def test_simpson_is_default_radial_method():
    r = np.linspace(0.0, 6.0, 301)
    q = np.linspace(0.0, 5.0, 81)
    transform = HankelTransform(0, q, gaussian_profile(r), r, r_cutoff=5.5)

    assert transform.method == "simpson"

def test_ogata_backend_is_public():
    pytest.importorskip("hankel")
    r = np.linspace(0.0, 6.0, 401)
    q = np.linspace(0.0, 6.0, 65)
    transform = HankelTransform(
        0,
        q,
        gaussian_profile(r),
        r,
        N=512,
        r_cutoff=5.5,
        interpolator="cubic",
        method="ogata",
    )
    assert transform.method == "ogata"
    assert transform.F_q.shape == q.shape
    assert np.all(np.isfinite(transform.F_q))


def test_legacy_experimental_ogata_import_remains_compatible():
    pytest.importorskip("hankel")
    from quartic2d._experimental.ogata import (
        HankelTransform as LegacyOgataHankelTransform,
    )

    assert issubclass(LegacyOgataHankelTransform, HankelTransform)


def test_set_method_recomputes_transform():
    r = np.linspace(0.0, 6.0, 301)
    q = np.linspace(0.0, 5.0, 81)
    transform = HankelTransform(
        0, q, gaussian_profile(r), r, r_cutoff=5.5, method="trapezoid"
    )
    before = transform.F_q.copy()

    returned = transform.set_method("gl8")

    assert returned is transform
    assert transform.method == "gl8"
    assert np.linalg.norm(transform.F_q - before) > 0.0


def test_set_interpolator_recomputes_gl_transform():
    r = np.linspace(0.0, 6.0, 101)
    q = np.linspace(0.0, 5.0, 81)
    transform = HankelTransform(
        0,
        q,
        gaussian_profile(r),
        r,
        r_cutoff=5.5,
        method="gl4",
        interpolator="linear",
    )
    before = transform.F_q.copy()

    transform.set_interpolator("pchip")

    assert transform.interpolator == "pchip"
    assert np.linalg.norm(transform.F_q - before) > 0.0


def test_q_zero_matches_radial_integral():
    r = np.linspace(0.0, 6.0, 401)
    q = np.linspace(0.0, 4.0, 81)
    transform = HankelTransform(
        0, q, gaussian_profile(r), r, r_cutoff=6.0, method="gl8"
    )

    assert transform.F_q[0].real == pytest.approx(1.0 / (2.0 * np.pi), rel=1e-7)
    assert transform.F_q[0].imag == pytest.approx(0.0, abs=1e-12)


def test_negative_integer_order_sign():
    r = np.linspace(0.0, 6.0, 301)
    q = np.linspace(0.0, 5.0, 91)
    profile = r * np.exp(-r**2)

    positive = HankelTransform(1, q, profile, r, r_cutoff=5.5, method="simpson")
    negative = HankelTransform(-1, q, profile, r, r_cutoff=5.5, method="simpson")

    np.testing.assert_allclose(negative.F_q, -positive.F_q, rtol=1e-12, atol=1e-12)


def test_input_profile_is_not_modified():
    r = np.linspace(0.0, 6.0, 301)
    profile = gaussian_profile(r)
    before = profile.copy()

    HankelTransform(
        0,
        np.linspace(0.0, 5.0, 81),
        profile,
        r,
        r_cutoff=5.5,
        method="gl4",
    )

    np.testing.assert_array_equal(profile, before)


def test_zero_profile_transforms_to_zero():
    r = np.linspace(0.0, 3.0, 101)
    q = np.linspace(0.0, 4.0, 81)
    transform = HankelTransform(
        0, q, np.zeros_like(r), r, r_cutoff=3.0, method="simpson"
    )

    np.testing.assert_array_equal(transform.F_q, 0.0)


def test_transform_call_returns_zero_outside_q_grid():
    r = np.linspace(0.0, 4.0, 201)
    q = np.linspace(0.0, 4.0, 81)
    transform = HankelTransform(
        0, q, gaussian_profile(r), r, r_cutoff=4.0, method="simpson"
    )

    outside = transform(np.array([5.0, 8.0]))
    np.testing.assert_array_equal(outside, 0.0)


def test_hankel_transform_rejects_invalid_backends():
    r = np.linspace(0.0, 4.0, 101)
    profile = gaussian_profile(r)
    q = np.linspace(0.0, 4.0, 41)

    with pytest.raises(ValueError, match="method"):
        HankelTransform(0, q, profile, r, r_cutoff=4.0, method="romberg")

    with pytest.raises(ValueError, match="interpolator"):
        HankelTransform(0, q, profile, r, r_cutoff=4.0, interpolator="nearest")

    with pytest.raises(ValueError, match="r_cutoff"):
        HankelTransform(0, q, profile, r, r_cutoff=0.0)


def test_batched_finite_hankel_matches_individual_transforms():
    from quartic2d._numerics import _finite_hankel_many, hankel_transform_sampled

    r = np.linspace(0.0, 6.0, 241)
    q = np.linspace(0.0, 8.0, 73)
    profiles = np.stack(
        (
            np.exp(-r**2),
            (1.0 - 0.3 * r**2) * np.exp(-0.7 * r**2),
        ),
        axis=0,
    )

    batched = _finite_hankel_many(
        r,
        profiles,
        q,
        0,
        "simpson",
        "cubic",
        subdivisions=2,
    )
    individual = np.stack(
        [
            hankel_transform_sampled(
                r,
                profile,
                q,
                0,
                method="simpson",
                interpolator="cubic",
                subdivisions=2,
            )
            for profile in profiles
        ],
        axis=0,
    )
    assert np.allclose(batched, individual, rtol=1.0e-13, atol=1.0e-13)


def test_batched_fftlog_matches_individual_transforms():
    from quartic2d._numerics import (
        _fftlog_hankel_transform_many,
        fftlog_hankel_transform,
    )

    deltas = np.geomspace(0.05, 30.0, 21)

    def values(q):
        return np.stack((np.exp(-q**2), (1.0 + 0.2 * q) * np.exp(-0.5 * q**2)))

    batched, _ = _fftlog_hankel_transform_many(
        values,
        2,
        deltas,
        0,
        native_q_max=8.0,
        native_q_step=0.02,
        n=512,
        bias=-0.5,
        interpolator="cubic",
    )
    individual = np.stack(
        [
            fftlog_hankel_transform(
                lambda q, index=index: values(q)[index],
                deltas,
                0,
                native_q_max=8.0,
                native_q_step=0.02,
                n=512,
                bias=-0.5,
                interpolator="cubic",
            )[0]
            for index in range(2)
        ],
        axis=0,
    )
    assert np.allclose(batched, individual, rtol=1.0e-13, atol=1.0e-13)

# ---- test_harmonics.py ----


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
        rho: ClassVar[dict[int, np.ndarray]] = {0: np.ones(5)}
        m_sorted: ClassVar[list[int]] = [0]
        power_fracs: ClassVar[dict[int, float]] = {0: 1.0}

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
