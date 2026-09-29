"""Reuse calibrated parameters across a Rytova-Keldysh screening family."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction


def orbital_s(x, y):
    r2 = x**2 + y**2
    return np.exp(-0.5 * r2) / np.sqrt(np.pi)


def orbital_h(x, y):
    r2 = x**2 + y**2
    z = x + 1j * y
    zbar = x - 1j * y
    return (
        np.exp(-0.5 * r2)
        / (2.0 * np.sqrt(np.pi))
        * (np.sqrt(2.0) * (1.0 - r2) + 1j * z + zbar**2 / np.sqrt(2.0))
    )


def rho_ss(x, y):
    psi = orbital_s(x, y)
    return np.conj(psi) * psi


def rho_sh(x, y):
    return np.conj(orbital_s(x, y)) * orbital_h(x, y)


def rho_hs(x, y):
    return np.conj(orbital_h(x, y)) * orbital_s(x, y)


def rytova_keldysh(r0):
    def kernel(q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(q > 0.0, 2.0 * np.pi / (q * (1.0 + r0 * q)), np.inf)

    return kernel


def decompose(field, x, y):
    return PolarDecomposition(
        field,
        x,
        y,
        Nr=181,
        Ntheta=256,
        rmax=5.5,
        origin=(0.0, 0.0),
        recon_err_tol=1.0e-4,
    )


def main():
    from _docs_artifacts import save_plot

    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)
    decompositions = {
        "ss": decompose(rho_ss, x, y),
        "sh": decompose(rho_sh, x, y),
        "hs": decompose(rho_hs, x, y),
    }
    hcal = {
        name: HarmonicTransform.converge_parameters(
            dec,
            rtol=1.0e-4,
            atol=1.0e-12,
            q_tail_rtol=1.0e-3,
            method="simpson",
            verbose=False,
        )
        for name, dec in decompositions.items()
    }
    fields = {name: hcal[name].transform(decompositions[name]) for name in hcal}

    channels = {
        "exchange": (fields["sh"], fields["sh"]),
        "pair hopping": (fields["sh"], fields["hs"]),
        "correlated hopping": (fields["ss"], fields["hs"]),
    }
    representative_r0 = (0.1, 1.0, 10.0)
    calibration_delta = np.linspace(0.0, 4.0, 25)
    calibration_vectors = np.column_stack(
        (calibration_delta, np.zeros_like(calibration_delta))
    )

    calibrations = {
        (name, r0): Interaction.converge_parameters(
            calibration_vectors,
            first,
            second,
            rytova_keldysh(r0),
            rtol=1.0e-4,
            atol=1.0e-12,
            method="gl4",
            verbose=False,
        )
        for name, (first, second) in channels.items()
        for r0 in representative_r0
    }
    family_subdivisions = max(
        result.parameters["subdivisions"] for result in calibrations.values()
    )
    save_plot(
        calibrations[("exchange", 1.0)].plot_convergence,
        "screening_interaction_convergence.svg",
    )

    print("Calibration")
    print("all harmonic searches satisfied:", all(result.converged for result in hcal.values()))
    print("all representative interaction searches satisfied:", all(result.converged for result in calibrations.values()))
    print("representative screening lengths:", representative_r0)
    print("common GL4 subdivisions:", family_subdivisions)

    r0_values = np.geomspace(0.1, 10.0, 31)
    delta = np.linspace(0.0, 4.0, 41)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    production = {
        name: np.empty((r0_values.size, delta.size), dtype=np.complex128)
        for name in channels
    }
    for i, r0 in enumerate(r0_values):
        kernel = rytova_keldysh(r0)
        for name, (first, second) in channels.items():
            production[name][i] = Interaction(
                vectors,
                first,
                second,
                kernel,
                method="gl4",
                interpolator="cubic",
                subdivisions=family_subdivisions,
            ).V

    print("\nProduction sweep")
    print("screening lengths:", r0_values.size)
    print("displacements:", delta.size)
    print("channels:", len(channels))
    print("matrix elements evaluated:", r0_values.size * delta.size * len(channels))
    mid = len(delta) // 2
    for name in channels:
        weak = production[name][0, mid]
        strong = production[name][-1, mid]
        print(
            f"{name:18s} V(delta=2,r0=0.1)={weak.real:+.5f}{weak.imag:+.5f}j  "
            f"V(delta=2,r0=10)={strong.real:+.5f}{strong.imag:+.5f}j"
        )


if __name__ == "__main__":
    main()
