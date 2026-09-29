"""Long-range Coulomb interaction of a normalized localized Gaussian state."""

import numpy as np
from petal2d import PolarDecomposition
from scipy.special import i0e

from quartic2d import HarmonicTransform, Interaction


def density(x, y):
    return np.exp(-(x**2 + y**2)) / np.pi


def coulomb(q):
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(q > 0.0, 2.0 * np.pi / q, np.inf)


def exact_coulomb(delta):
    delta = np.asarray(delta, dtype=float)
    return np.sqrt(np.pi / 2.0) * i0e(delta**2 / 4.0)


def relative_l2(values, reference):
    return np.linalg.norm(values.real - reference) / np.linalg.norm(reference)


def main():
    from _docs_artifacts import save_plot

    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)
    dec = PolarDecomposition(
        density,
        x,
        y,
        Nr=181,
        Ntheta=256,
        rmax=5.5,
        origin=(0.0, 0.0),
        recon_err_tol=1.0e-10,
    )

    hcal = HarmonicTransform.converge_parameters(
        dec,
        rtol=1.0e-4,
        atol=1.0e-12,
        q_tail_rtol=1.0e-3,
        method="simpson",
        verbose=False,
    )
    field = hcal.transform(dec)
    save_plot(hcal.plot_convergence, "large_harmonic_convergence.svg")
    exact_transform = np.exp(-field.q**2 / 4.0) / (2.0 * np.pi)
    transform_error = np.linalg.norm(field.F_q[0].real - exact_transform) / np.linalg.norm(exact_transform)

    print("HarmonicTransform")
    print("self-convergence satisfied:", hcal.converged)
    print(f"analytic relative L2 error: {transform_error:.3e}")

    delta = np.geomspace(1.0e2, 1.0e4, 81)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    reference = exact_coulomb(delta)

    gl4_cal = Interaction.converge_parameters(
        vectors,
        field,
        field,
        coulomb,
        rtol=1.0e-3,
        atol=1.0e-12,
        method="gl4",
        subdivisions=(32, 64, 128, 256, 512, 1024),
        verbose=False,
    )
    gl4 = gl4_cal.interaction(vectors, field, field, coulomb)
    save_plot(gl4_cal.plot_convergence, "large_interaction_convergence.svg")
    fftlog = Interaction(vectors, field, field, coulomb, method="fftlog", n=256, bias=0.0)

    print("\nLarge-distance interaction")
    print("GL4 self-convergence satisfied:", gl4_cal.converged)
    print(f"GL4 independent relative L2 error: {relative_l2(gl4.V, reference):.3e}")
    print(f"FFTLog independent relative L2 error: {relative_l2(fftlog.V, reference):.3e}")

    try:
        ogata = Interaction(vectors, field, field, coulomb, method="ogata", N=256, h=0.025)
    except ImportError:
        print("Ogata: optional 'hankel' dependency not installed")
    else:
        print(f"Ogata independent relative L2 error: {relative_l2(ogata.V, reference):.3e}")

    for value in (100.0, 1000.0, 10000.0):
        result = exact_coulomb(np.array([value]))[0]
        print(f"delta={value:7.0f}  delta*V_exact={value * result:.9f}")


if __name__ == "__main__":
    main()
