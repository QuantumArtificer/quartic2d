"""Compare the available first-stage numerical methods."""

import numpy as np

from quartic2d import HankelTransform

METHODS = ("trapezoid", "simpson", "gl4", "gl8", "ogata")

r = np.linspace(0.0, 8.0, 401)
q = np.linspace(0.0, 8.0, 257)
profile = np.exp(-r)

transform = HankelTransform(0, q, profile, r, r_cutoff=r[-1])
print("default method:", transform.method)
print("interpolator:", transform.interpolator)

reference = transform.F_q.copy()
for method in METHODS:
    try:
        if method == "ogata":
            transform.set_method(method, N=1024, h=0.002)
        else:
            transform.set_method(method)
    except ImportError as exc:
        print(f"{method:10s} unavailable: {exc}")
        continue
    difference = np.linalg.norm(transform.F_q - reference) / np.linalg.norm(reference)
    print(f"{method:10s} relative difference from default Simpson: {difference:.6e}")
