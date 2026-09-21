"""Compare a numerical Hankel transform with an analytic Gaussian result."""

import numpy as np

from quartic2d import HankelTransform


r = np.linspace(0.0, 6.0, 401)
q = np.linspace(0.0, 6.0, 121)
rho = np.exp(-r**2) / np.pi

transform = HankelTransform(
    0,
    q,
    rho,
    r,
    r_cutoff=5.5,
    interpolator="cubic",
)

exact = np.exp(-q**2 / 4.0) / (2.0 * np.pi)
relative_l2 = np.linalg.norm(transform.F_q.real - exact) / np.linalg.norm(exact)

print(f"relative L2 error: {relative_l2:.6e}")
