"""End-to-end isotropic Gaussian interaction with automatic convergence."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction


def density(x, y):
    return np.exp(-(x**2 + y**2)) / np.pi


def coulomb(q):
    return 2.0 * np.pi / q


def rytova_keldysh(q, r0=1.0):
    return 2.0 * np.pi / (q * (1.0 + r0 * q))


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

    print("PETAL2D")
    print("retained modes:", dec.m_sorted)
    print(f"reconstruction error [%]: {dec.recon_error_measured:.3e}")
    print(f"domain consistency: {dec.domain_consistency:.8f}")

    hcal = HarmonicTransform.converge_parameters(
        dec,
        rtol=1.0e-4,
        atol=1.0e-12,
        q_tail_rtol=1.0e-3,
        method="simpson",
        verbose=False,
    )
    field = hcal.transform(dec)
    save_plot(hcal.plot_convergence, "isotropic_harmonic_convergence.svg")

    exact = np.exp(-field.q**2 / 4.0) / (2.0 * np.pi)
    transform_error = np.linalg.norm(field.F_q[0].real - exact) / np.linalg.norm(exact)

    print("\nHarmonicTransform")
    print("self-convergence satisfied:", hcal.converged)
    print(f"analytic relative L2 error: {transform_error:.3e}")
    print(f"selected q_max: {hcal.parameters['q_max']:.6f}")
    print("selected n_q:", hcal.parameters["n_q"])

    calibration_delta = np.geomspace(1.0e-2, 1.0e2, 32)
    calibration_vectors = np.column_stack(
        (calibration_delta, np.zeros_like(calibration_delta))
    )
    ccal = Interaction.converge_parameters(
        calibration_vectors,
        field,
        field,
        coulomb,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    rkcal = Interaction.converge_parameters(
        calibration_vectors,
        field,
        field,
        rytova_keldysh,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    save_plot(ccal.plot_convergence, "isotropic_interaction_convergence.svg")

    delta = np.linspace(0.0, 4.0, 81)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    bare = ccal.interaction(vectors, field, field, coulomb)
    rk = rkcal.interaction(vectors, field, field, rytova_keldysh)

    print("\nInteraction")
    print("Coulomb self-convergence satisfied:", ccal.converged)
    print("Rytova-Keldysh self-convergence satisfied:", rkcal.converged)
    for index in (0, 20, 40, 80):
        print(
            f"delta={delta[index]:3.1f}  "
            f"Coulomb={bare.V[index].real:.6f}  "
            f"RK={rk.V[index].real:.6f}"
        )


if __name__ == "__main__":
    main()
