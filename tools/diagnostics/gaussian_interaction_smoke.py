"""Evaluate a screened interaction for an isotropic Gaussian density."""

import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform, Interaction

x = np.linspace(-6.0, 6.0, 161)
y = np.linspace(-6.0, 6.0, 161)


def density(x, y):
    return np.exp(-(x**2 + y**2)) / np.pi


def screened_coulomb(q):
    kappa = 0.25
    return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)


dec = PolarDecomposition(
    density,
    x,
    y,
    Nr=161,
    Ntheta=256,
    rmax=5.5,
    origin=(0.0, 0.0),
)

transformed = HarmonicTransform(dec)

deltas = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
interaction = Interaction(deltas, transformed, transformed, screened_coulomb)

for delta, value in zip(deltas, interaction.V):
    print(f"delta={delta}: V={value.real:.8f}")
