"""Orientation-dependent interaction of a quadrupolar localized transition field."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction

ANISOTROPY = 0.35
KAPPA = 0.4


def transition_field(x, y):
    r2 = x**2 + y**2
    return np.exp(-r2) * (1.0 + ANISOTROPY * (x**2 - y**2)) / np.pi


def yukawa(q):
    return 2.0 * np.pi / np.sqrt(q**2 + KAPPA**2)


def make_decomposition():
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)
    return PolarDecomposition(
        transition_field,
        x,
        y,
        Nr=181,
        Ntheta=256,
        rmax=5.5,
        recon_err_tol=1.0e-4,
        radial_power_tail_fraction=1.0e-10,
        radial_relative_amplitude_threshold=1.0e-6,
        origin=(0.0, 0.0),
    )


def main():
    from _docs_artifacts import save_plot

    dec = make_decomposition()
    print("PETAL2D")
    print("retained harmonics:", dec.m_sorted)
    for m in dec.m_sorted:
        print(f"m={int(m):+d}: power fraction={dec.power_fracs[int(m)]:.6f}")
    print("PETAL2D target reached:", dec.target_reached)

    hcal = HarmonicTransform.converge_parameters(
        dec,
        rtol=1.0e-4,
        atol=1.0e-12,
        q_tail_rtol=1.0e-3,
        method="simpson",
        verbose=False,
    )
    field = hcal.transform(dec)
    save_plot(hcal.plot_convergence, "anisotropic_harmonic_convergence.svg")
    print("\nHarmonicTransform")
    print("harmonic transform converged:", hcal.converged)
    print("harmonic quadrature:", field.method)
    print("retained harmonics:", field.m_values.tolist())
    print(f"F_0(0) = {field.F_q[0][0].real:.8f}")

    phi_delta = np.linspace(0.0, 2.0 * np.pi, 181)
    radii = np.array([0.75, 1.50, 2.50])
    vectors = np.vstack(
        [
            np.column_stack(
                (
                    radius * np.cos(phi_delta),
                    radius * np.sin(phi_delta),
                )
            )
            for radius in radii
        ]
    )

    ical = Interaction.converge_parameters(
        vectors,
        field,
        field,
        yukawa,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    interaction = ical.interaction(vectors, field, field, yukawa)
    values = interaction.V.real.reshape(len(radii), len(phi_delta))
    save_plot(ical.plot_convergence, "anisotropic_interaction_convergence.svg")

    print("\nInteraction")
    print("interaction converged:", ical.converged)
    print("interaction quadrature:", ical.method)
    for radius, curve in zip(radii, values):
        print(
            f"delta={radius:.2f}: "
            f"V(0)={curve[0]:.6f}, "
            f"V(pi/2)={curve[45]:.6f}"
        )

    nphi = len(phi_delta)
    mid = slice(nphi, 2 * nphi)
    grouped = {
        0: np.zeros(nphi, dtype=complex),
        2: np.zeros(nphi, dtype=complex),
        4: np.zeros(nphi, dtype=complex),
    }
    for i, m in enumerate(field.m_values):
        for j, mp in enumerate(field.m_values):
            grouped[abs(int(m - mp))] += interaction.V_mm[i, j, mid]

    pair_residual = np.max(np.abs(sum(grouped.values()) - interaction.V[mid]))
    scale = max(float(np.max(np.abs(interaction.V[mid]))), 1.0e-15)
    print(f"harmonic-pair reconstruction relative max residual: {pair_residual / scale:.3e}")


if __name__ == "__main__":
    main()
