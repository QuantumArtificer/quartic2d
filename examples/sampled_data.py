"""Complex transition fields supplied as sampled Cartesian arrays."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction


def yukawa(q, kappa=0.5):
    return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)


def main():
    from _docs_artifacts import save_plot

    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r2 = X**2 + Y**2

    psi_s = np.exp(-0.5 * r2) / np.sqrt(np.pi)
    psi_p_plus = (X + 1j * Y) * np.exp(-0.5 * r2) / np.sqrt(np.pi)
    rho_forward = np.conj(psi_s) * psi_p_plus
    rho_reverse = np.conj(psi_p_plus) * psi_s

    decompositions = {
        "forward": PolarDecomposition(
            rho_forward,
            x,
            y,
            Nr=181,
            Ntheta=256,
            rmax=5.5,
            origin=(0.0, 0.0),
            interp_method="cubic",
            recon_err_tol=1.0e-2,
        ),
        "reverse": PolarDecomposition(
            rho_reverse,
            x,
            y,
            Nr=181,
            Ntheta=256,
            rmax=5.5,
            origin=(0.0, 0.0),
            interp_method="cubic",
            recon_err_tol=1.0e-2,
        ),
    }

    print("Sampled PETAL2D input")
    print("array shape:", rho_forward.shape)
    for name, dec in decompositions.items():
        print(
            f"{name}: modes={dec.m_sorted}, "
            f"reconstruction error={dec.recon_error_measured:.3e}%, "
            f"domain consistency={dec.domain_consistency:.8f}"
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
    save_plot(hcal["forward"].plot_convergence, "sampled_harmonic_convergence.svg")

    q = np.asarray(fields["forward"].q, dtype=float)
    numerical = np.asarray(fields["forward"].F_q[1])
    exact = q * np.exp(-q**2 / 4.0) / (4.0 * np.pi)
    relative_l2 = np.linalg.norm(numerical - exact) / np.linalg.norm(exact)
    peak_normalized = np.max(np.abs(numerical - exact)) / np.max(np.abs(exact))

    print("\nHarmonicTransform")
    print("all self-convergence criteria satisfied:", all(result.converged for result in hcal.values()))
    print(f"forward analytic relative L2 error: {relative_l2:.3e}")
    print(f"forward peak-normalized max error: {peak_normalized:.3e}")

    calibration_delta = np.linspace(0.0, 4.0, 25)
    calibration_vectors = np.column_stack(
        (calibration_delta, np.zeros_like(calibration_delta))
    )
    ical = Interaction.converge_parameters(
        calibration_vectors,
        fields["forward"],
        fields["reverse"],
        yukawa,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    save_plot(ical.plot_convergence, "sampled_interaction_convergence.svg")

    delta = np.linspace(0.0, 4.0, 41)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    interaction = ical.interaction(
        vectors,
        fields["forward"],
        fields["reverse"],
        yukawa,
    )

    print("\nInteraction")
    print("self-convergence satisfied:", ical.converged)
    for index in (0, 10, 20, 40):
        value = interaction.V[index]
        print(f"delta={delta[index]:3.1f}  V={value.real:+.6f}{value.imag:+.2e}j")


if __name__ == "__main__":
    main()
