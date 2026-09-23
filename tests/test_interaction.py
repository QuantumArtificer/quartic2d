import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from quartic2d import Interaction


class AnalyticHarmonics:
    def __init__(self, functions, q_max=10.0, n_q=401):
        self._functions = {int(m): fn for m, fn in functions.items()}
        self._m_values = np.array(list(self._functions), dtype=int)
        self.q = np.linspace(0.0, q_max, n_q)
        self.F_q = {
            int(m): self._functions[int(m)](self.q) for m in self._m_values
        }

    @property
    def m_values(self):
        return self._m_values.copy()

    def __call__(self, m, q):
        return self._functions[int(m)](np.asarray(q))


def gaussian_transform(q):
    return np.exp(-q**2 / 4.0) / (2.0 * np.pi)


def test_interaction_rejects_invalid_displacements():
    h = AnalyticHarmonics({0: gaussian_transform})

    with pytest.raises(ValueError, match=r"shape \(D, 2\)"):
        Interaction(np.array([1.0, 2.0]), h, h, lambda q: 1.0)

    with pytest.raises(ValueError, match=r"shape \(D, 2\)"):
        Interaction(np.ones((3, 3)), h, h, lambda q: 1.0)

    with pytest.raises(ValueError, match="at least one"):
        Interaction(np.empty((0, 2)), h, h, lambda q: 1.0)


def test_interaction_requires_callable_kernel():
    h = AnalyticHarmonics({0: gaussian_transform})

    with pytest.raises(TypeError, match="callable"):
        Interaction(np.array([[0.0, 0.0]]), h, h, 1.0)




def test_ogata_interaction_backend_is_public():
    pytest.importorskip("hankel")
    h = AnalyticHarmonics({0: gaussian_transform})
    result = Interaction(
        np.array([[1.0, 0.0]]),
        h,
        h,
        lambda q: np.exp(-q),
        method="ogata",
        N=128,
        h=0.025,
    )
    assert result.method == "ogata"
    assert np.all(np.isfinite(result.V))


def test_phase_factors_include_absolute_order_bessel_parity():
    h1 = AnalyticHarmonics({0: gaussian_transform, 2: gaussian_transform})
    h2 = AnalyticHarmonics({0: gaussian_transform, -1: gaussian_transform})
    result = Interaction(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        h1,
        h2,
        lambda q: np.exp(-q),
        method="gl4",
    )

    assert result.Phi_mm.shape == (2, 2, 2)
    np.testing.assert_allclose(
        result.Phi_mm[:, :, 0],
        np.array([[1.0, -1.0], [1.0, -1.0]], dtype=complex),
    )
    assert result.Phi_mm[1, 0, 1] == pytest.approx(np.exp(1j * 2 * np.pi / 2.0))
    assert result.Phi_mm[0, 1, 1] == pytest.approx(-np.exp(1j * np.pi / 2.0))


def _direct_q_phi_reference(m, mp, f1, f2, kernel, delta, angle, q_max):
    """Direct momentum-space reference with no Bessel reduction."""
    q_nodes, q_weights = leggauss(120)
    phi_nodes, phi_weights = leggauss(160)

    q = 0.5 * q_max * (q_nodes + 1.0)
    wq = 0.5 * q_max * q_weights
    phi = np.pi * (phi_nodes + 1.0)
    wphi = np.pi * phi_weights

    q2d = q[:, None]
    phi2d = phi[None, :]
    rho1 = ((-1j) ** int(m)) * np.exp(1j * int(m) * phi2d) * f1(q2d)
    rho2 = ((-1j) ** int(mp)) * np.exp(1j * int(mp) * phi2d) * f2(q2d)
    translation = np.exp(
        -1j * q2d * float(delta) * np.cos(phi2d - float(angle))
    )
    integrand = (
        q2d
        * kernel(q2d)
        * rho1
        * np.conj(rho2)
        * translation
    )
    return np.sum(wq[:, None] * wphi[None, :] * integrand)


@pytest.mark.parametrize(
    "m, mp", [(1, 0), (0, 1), (-1, 0), (0, -1), (3, 0), (0, 3)]
)
def test_odd_harmonic_difference_matches_direct_q_phi_reference(m, mp):
    q_max = 8.0
    f1 = lambda q: (q ** abs(m)) * np.exp(-0.35 * q**2)
    f2 = lambda q: (q ** abs(mp)) * np.exp(-0.45 * q**2)
    kernel = lambda q: np.exp(-0.2 * q**2)
    delta = 1.3
    angle = 0.37

    h1 = AnalyticHarmonics({m: f1}, q_max=q_max, n_q=1025)
    h2 = AnalyticHarmonics({mp: f2}, q_max=q_max, n_q=1025)
    result = Interaction(
        np.array([[delta * np.cos(angle), delta * np.sin(angle)]]),
        h1,
        h2,
        kernel,
        method="gl8",
        subdivisions=2,
    )
    reference = _direct_q_phi_reference(
        m, mp, f1, f2, kernel, delta, angle, q_max
    )

    assert result.V[0] == pytest.approx(reference, rel=2e-8, abs=2e-10)


@pytest.mark.parametrize("method", ["trapezoid", "simpson", "gl4", "gl8"])
def test_zero_displacement_gaussian_unit_kernel(method):
    h = AnalyticHarmonics({0: gaussian_transform}, q_max=12.0, n_q=513)
    result = Interaction(
        np.array([[0.0, 0.0]]),
        h,
        h,
        lambda q: np.ones_like(np.asarray(q), dtype=float),
        method=method,
        interpolator="cubic",
    )

    exact = 1.0 / (2.0 * np.pi)
    assert result.V[0].real == pytest.approx(exact, rel=1e-4)
    assert result.V[0].imag == pytest.approx(0.0, abs=1e-12)


def test_coulomb_kernel_is_never_required_at_zero_for_gl():
    h = AnalyticHarmonics({0: gaussian_transform}, q_max=12.0, n_q=257)

    def coulomb(q):
        q = np.asarray(q)
        if np.any(q == 0.0):
            raise RuntimeError("endpoint evaluated")
        return 2.0 * np.pi / q

    result = Interaction(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        h,
        h,
        coulomb,
        method="gl4",
        interpolator="pchip",
    )
    assert np.all(np.isfinite(result.V))


def test_zero_displacement_removes_unequal_harmonic_pairs():
    h = AnalyticHarmonics(
        {0: gaussian_transform, 1: lambda q: q * gaussian_transform(q)}
    )
    result = Interaction(
        np.array([[0.0, 0.0]]), h, h, lambda q: np.ones_like(q), method="gl4"
    )

    assert result.H_mm.shape == (2, 2, 1)
    assert result.H_mm[0, 1, 0] == pytest.approx(0.0, abs=1e-14)
    assert result.H_mm[1, 0, 0] == pytest.approx(0.0, abs=1e-14)


def test_interaction_output_shapes_and_displacements():
    h1 = AnalyticHarmonics({0: gaussian_transform, 2: gaussian_transform})
    h2 = AnalyticHarmonics({0: gaussian_transform})
    deltas = np.array([[0.0, 0.0], [1.0, 0.0]])

    result = Interaction(deltas, h1, h2, lambda q: np.exp(-q), method="gl4")

    assert result.Phi_mm.shape == (2, 1, 2)
    assert result.H_mm.shape == (2, 1, 2)
    assert result.V_mm.shape == (2, 1, 2)
    assert result.V.shape == (2,)
    np.testing.assert_allclose(result.V, result.V_mm.sum(axis=(0, 1)))
    np.testing.assert_array_equal(result.deltas, deltas)


def test_interaction_setters_switch_methods():
    h = AnalyticHarmonics({0: gaussian_transform}, q_max=12.0, n_q=257)
    deltas = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = Interaction(deltas, h, h, lambda q: np.exp(-q), method="gl4")

    assert result.set_method("gl8") is result
    assert result.method == "gl8"

    assert result.set_method("fftlog", n=256, bias=-0.5) is result
    assert result.method == "fftlog"

    assert result.set_interpolator("pchip") is result
    assert result.interpolator == "pchip"


def test_ogata_reuses_transform_objects_by_order(monkeypatch):
    hankel = pytest.importorskip("hankel")
    real = hankel.HankelTransform
    calls = []

    def counting_transform(*args, **kwargs):
        calls.append(int(kwargs.get("nu", args[0] if args else 0)))
        return real(*args, **kwargs)

    monkeypatch.setattr(hankel, "HankelTransform", counting_transform)

    h = AnalyticHarmonics(
        {
            0: gaussian_transform,
            2: lambda q: q**2 * gaussian_transform(q),
            -2: lambda q: q**2 * gaussian_transform(q),
        },
        q_max=10.0,
        n_q=257,
    )
    Interaction(
        np.array([[0.5, 0.0], [1.0, 0.0]]),
        h,
        h,
        lambda q: np.exp(-q),
        N=64,
        method="ogata",
    )

    assert len(calls) == 3
    assert set(calls) == {0, 2, 4}

def test_gl4_is_default_interaction_method():
    h = AnalyticHarmonics({0: gaussian_transform}, q_max=12.0, n_q=513)
    result = Interaction(
        np.array([[0.1, 0.0], [1.0, 0.0], [10.0, 0.0]]),
        h,
        h,
        lambda q: 2.0 * np.pi / np.asarray(q),
    )
    assert result.method == "gl4"
    assert np.all(np.isfinite(result.V))


def test_fftlog_preserves_unsorted_displacement_order():
    h = AnalyticHarmonics({0: gaussian_transform}, q_max=12.0, n_q=1025)
    sorted_deltas = np.array([0.1, 1.0, 10.0, 100.0])
    order = np.array([2, 0, 1, 3])
    unsorted_deltas = sorted_deltas[order]

    sorted_result = Interaction(
        np.column_stack((sorted_deltas, np.zeros_like(sorted_deltas))),
        h,
        h,
        lambda q: 2.0 * np.pi / np.asarray(q),
        method="fftlog",
    )
    unsorted_result = Interaction(
        np.column_stack((unsorted_deltas, np.zeros_like(unsorted_deltas))),
        h,
        h,
        lambda q: 2.0 * np.pi / np.asarray(q),
        method="fftlog",
    )

    np.testing.assert_allclose(
        unsorted_result.V, sorted_result.V[order], rtol=1e-13, atol=1e-13
    )


def test_fftlog_gaussian_coulomb_accuracy():
    from scipy.special import i0e

    h = AnalyticHarmonics({0: gaussian_transform}, q_max=12.0, n_q=1025)
    deltas = np.array([0.1, 1.0, 10.0, 100.0])
    result = Interaction(
        np.column_stack((deltas, np.zeros_like(deltas))),
        h,
        h,
        lambda q: 2.0 * np.pi / np.asarray(q),
        method="fftlog",
        n=512,
    )
    exact = np.sqrt(np.pi / 2.0) * i0e(deltas**2 / 4.0)
    error = np.abs(result.V.real - exact)
    significant = exact >= 0.05 * np.max(exact)

    np.testing.assert_allclose(
        result.V.real[significant], exact[significant], rtol=2e-3, atol=2e-8
    )
    assert np.max(error) / np.max(np.abs(exact)) < 2e-4
    np.testing.assert_allclose(result.V.imag, 0.0, atol=1e-10)
