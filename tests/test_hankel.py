import numpy as np
import pytest

from quartic2d import HankelTransform


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
    from quartic2d._experimental.ogata import HankelTransform as LegacyOgataHankelTransform

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
