"""Hankel transforms of harmonics retained by PETAL2D."""

import matplotlib.pyplot as plt
import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform

x = np.linspace(-6.0, 6.0, 161)
y = np.linspace(-6.0, 6.0, 161)


def density(x, y):
    return np.exp(-(x**2 + y**2)) * (1.0 + 0.32 * (x**2 - y**2))


dec = PolarDecomposition(
    density,
    x,
    y,
    Nr=256,
    Ntheta=256,
    recon_err_tol=1.0e-4,
)

transformed = HarmonicTransform(dec)

transformed.plot_harmonics()
plt.show()
