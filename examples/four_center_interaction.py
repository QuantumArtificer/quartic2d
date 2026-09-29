"""Direct and exchange four-center terms from normalized s and p_x orbitals."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction


def orbital_s(x, y):
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)


def orbital_px(x, y):
    return np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * (x**2 + y**2))


def rho_ss(x, y):
    psi = orbital_s(x, y)
    return np.conj(psi) * psi


def rho_pp(x, y):
    psi = orbital_px(x, y)
    return np.conj(psi) * psi


def rho_sp(x, y):
    return np.conj(orbital_s(x, y)) * orbital_px(x, y)


def yukawa(q, kappa=0.35):
    return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)


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
        "pp": decompose(rho_pp, x, y),
        "sp": decompose(rho_sp, x, y),
    }

    print("PETAL2D")
    for name, dec in decompositions.items():
        print(
            f"rho_{name}: modes={dec.m_sorted}, "
            f"reconstruction error={dec.recon_error_measured:.3e}%"
        )

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
    save_plot(hcal["sp"].plot_convergence, "direct_exchange_harmonic_convergence.svg")

    calibration_delta = np.linspace(0.0, 4.0, 25)
    calibration_vectors = np.column_stack(
        (calibration_delta, np.zeros_like(calibration_delta))
    )
    direct_cal = Interaction.converge_parameters(
        calibration_vectors,
        fields["ss"],
        fields["pp"],
        yukawa,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    exchange_cal = Interaction.converge_parameters(
        calibration_vectors,
        fields["sp"],
        fields["sp"],
        yukawa,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    save_plot(exchange_cal.plot_convergence, "direct_exchange_interaction_convergence.svg")

    delta = np.linspace(0.0, 4.0, 81)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    direct = direct_cal.interaction(vectors, fields["ss"], fields["pp"], yukawa)
    exchange = exchange_cal.interaction(vectors, fields["sp"], fields["sp"], yukawa)

    print("\nAutomatic convergence")
    print("all harmonic searches satisfied:", all(result.converged for result in hcal.values()))
    print("direct interaction satisfied:", direct_cal.converged)
    print("exchange interaction satisfied:", exchange_cal.converged)
    for index in (0, 20, 40):
        print(
            f"delta={delta[index]:3.1f}  "
            f"direct={direct.V[index].real:+.6f}  "
            f"exchange={exchange.V[index].real:+.6f}"
        )


if __name__ == "__main__":
    main()
