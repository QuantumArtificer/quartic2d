"""Inspect angular dependence and harmonic-resolved interaction terms."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction


def density(x, y):
    r2 = x**2 + y**2
    return np.exp(-r2) * (1.0 + 0.35 * (x**2 - y**2)) / np.pi


def screened_coulomb(q):
    return 2.0 * np.pi / np.sqrt(q**2 + 0.4**2)


def make_decomposition():
    x = np.linspace(-6.0, 6.0, 181)
    y = np.linspace(-6.0, 6.0, 181)
    return PolarDecomposition(
        density,
        x,
        y,
        Nr=181,
        Ntheta=256,
        rmax=5.5,
        recon_err_tol=1e-4,
        radial_power_tail_fraction=1e-10,
        radial_relative_amplitude_threshold=1e-6,
        origin=(0.0, 0.0),
    )


def main():
    dec = make_decomposition()
    transformed = HarmonicTransform(dec, q_max=12.0, n_q=512)

    radius = 1.5
    angles = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    deltas = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    interaction = Interaction(deltas, transformed, transformed, screened_coulomb)

    print("first-stage quadrature:", transformed.method)
    print("interaction transform:", interaction.method)
    print("retained harmonics:", transformed.m_values.tolist())
    for angle, value in zip(angles, interaction.V):
        print(f"phi={angle:.6f}: V={value.real:.8f}")

    largest = np.unravel_index(np.argmax(np.abs(interaction.V_mm)), interaction.V_mm.shape)
    print("largest |V_mm| index:", largest)
    print("largest V_mm:", interaction.V_mm[largest])


if __name__ == "__main__":
    main()
