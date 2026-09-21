#!/usr/bin/env python3
"""Validate QUARTIC2D against an analytic Gaussian/Coulomb reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from petal2d import PolarDecomposition
from scipy.special import i0e

from quartic2d import HarmonicTransform, Interaction


def gaussian_density(x, y):
    return np.exp(-(x * x + y * y)) / np.pi


def gaussian_hankel_exact(q):
    return np.exp(-(q * q) / 4.0) / (2.0 * np.pi)


def coulomb_kernel(q):
    return 2.0 * np.pi / q


def gaussian_coulomb_exact(delta):
    delta = np.asarray(delta, dtype=float)
    x = delta * delta / 4.0
    return np.sqrt(np.pi / 2.0) * i0e(x)


def run(quick: bool) -> dict:
    if quick:
        nxy, ntheta, nq = 101, 128, 256
    else:
        nxy, ntheta, nq = 181, 256, 768

    x = np.linspace(-6.0, 6.0, nxy)
    y = np.linspace(-6.0, 6.0, nxy)
    dec = PolarDecomposition(
        gaussian_density,
        x,
        y,
        Nr=nxy,
        Ntheta=ntheta,
        rmax=5.5,
        recon_err_tol=1e-10,
        radial_power_tail_fraction=1e-12,
        radial_relative_amplitude_threshold=1e-8,
        origin=(0.0, 0.0),
        interp_method="cubic",
    )

    transformed = HarmonicTransform(
        dec,
        q_max=12.0,
        n_q=nq,
        interpolator="cubic",
    )

    if transformed.m_values.tolist() != [0]:
        raise RuntimeError(
            "The isotropic Gaussian validation expected only m=0; found "
            f"{transformed.m_values.tolist()}."
        )

    q = transformed.q
    numerical_f = np.real(transformed.F_q[0])
    exact_f = gaussian_hankel_exact(q)
    hankel_l2_percent = 100.0 * np.linalg.norm(numerical_f - exact_f) / np.linalg.norm(exact_f)

    deltas = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    interaction = Interaction(
        deltas,
        transformed,
        transformed,
        coulomb_kernel,
    )
    numerical_v = np.real(interaction.V)
    exact_v = gaussian_coulomb_exact(np.linalg.norm(deltas, axis=1))
    interaction_rel_percent = 100.0 * np.abs(numerical_v - exact_v) / np.abs(exact_v)

    result = {
        "quick": quick,
        "parameters": {
            "nxy": nxy,
            "ntheta": ntheta,
            "nq": nq,
            "radial_method": transformed.method,
            "interaction_method": interaction.method,
        },
        "retained_harmonics": transformed.m_values.tolist(),
        "hankel_l2_percent": float(hankel_l2_percent),
        "interaction_max_percent": float(np.max(interaction_rel_percent)),
        "interaction_by_distance": [
            {
                "distance": float(np.linalg.norm(delta)),
                "numerical": float(num),
                "exact": float(exact),
                "relative_error_percent": float(err),
            }
            for delta, num, exact, err in zip(
                deltas, numerical_v, exact_v, interaction_rel_percent
            )
        ],
    }

    result["pass"] = bool(
        result["hankel_l2_percent"] <= 5.0
        and result["interaction_max_percent"] <= 5.0
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use CI-sized grids.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.quick)
    text = json.dumps(result, indent=2)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if not result["pass"]:
        raise SystemExit("Gaussian validation exceeded the 5% error threshold.")


if __name__ == "__main__":
    main()
