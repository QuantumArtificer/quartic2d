"""Complex localized orbitals and off-diagonal four-center interactions."""

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


def dual_gate(q, d=1.0):
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    small = np.abs(q) < 1.0e-12
    out[small] = 2.0 * np.pi * d
    out[~small] = 2.0 * np.pi * np.tanh(q[~small] * d) / q[~small]
    return out


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
    save_plot(hcal["sh"].plot_convergence, "complex_harmonic_convergence.svg")

    q = fields["sh"].q
    envelope = np.exp(-q**2 / 4.0)
    exact = {
        0: q**2 * envelope / (8.0 * np.pi * np.sqrt(2.0)),
        1: 1j * q * envelope / (8.0 * np.pi),
        -2: q**2 * envelope / (16.0 * np.pi * np.sqrt(2.0)),
    }

    print("\nHarmonicTransform")
    print("all self-convergence criteria satisfied:", all(result.converged for result in hcal.values()))
    for m in fields["sh"].m_values:
        m = int(m)
        scale = max(float(np.max(np.abs(exact[m]))), 1.0e-15)
        error = np.max(np.abs(fields["sh"].F_q[m] - exact[m])) / scale
        print(f"m={m:+d} analytic peak-normalized max error: {error:.3e}")

    channels = {
        "exchange": (fields["sh"], fields["sh"]),
        "pair hopping": (fields["sh"], fields["hs"]),
        "correlated hopping": (fields["ss"], fields["hs"]),
    }
    calibration_delta = np.linspace(0.0, 4.0, 25)
    calibration_vectors = np.column_stack(
        (calibration_delta, np.zeros_like(calibration_delta))
    )
    ical = {
        name: Interaction.converge_parameters(
            calibration_vectors,
            first,
            second,
            dual_gate,
            rtol=1.0e-4,
            atol=1.0e-12,
            method="gl4",
            verbose=False,
        )
        for name, (first, second) in channels.items()
    }
    save_plot(ical["exchange"].plot_convergence, "complex_interaction_convergence.svg")

    delta = np.linspace(0.0, 4.0, 81)
    vectors = np.column_stack((delta, np.zeros_like(delta)))
    production = {
        name: ical[name].interaction(vectors, first, second, dual_gate)
        for name, (first, second) in channels.items()
    }

    print("\nInteraction")
    print("all self-convergence criteria satisfied:", all(result.converged for result in ical.values()))
    for index in (0, 20, 40):
        print(f"delta={delta[index]:.1f}")
        for name in channels:
            value = production[name].V[index]
            print(f"  {name:18s} {value.real:+.5f}{value.imag:+.5f}j")


if __name__ == "__main__":
    main()
