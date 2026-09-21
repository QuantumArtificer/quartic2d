"""Parameter convergence for PETAL2D harmonic transforms."""

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

result = HarmonicTransform.converge_parameters(
    dec,
    rtol=1.0e-4,
    atol=1.0e-12,
    q_tail_rtol=1.0e-3,
)

transformed = result.transform(dec)
transformed.plot_convergence()
plt.show()
