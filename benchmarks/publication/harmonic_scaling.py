#!/usr/bin/env python3
"""Paper benchmark: Step-2 HarmonicTransform computational scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

from quartic2d import HarmonicTransform

from benchmarks._common import environment_metadata, timed_call, write_json

BATCH_SIZE = 256


class SyntheticDecomposition:
    """Minimal PETAL2D-compatible smooth input used only for scaling tests."""

    def __init__(self, *, n_r: int, m_values: list[int], r_max: float = 6.0):
        self.r = np.linspace(0.0, float(r_max), int(n_r))
        self.m_sorted = [int(m) for m in m_values]
        self.rho = {}
        self.cutoff_radius = {}
        powers = {}
        for m in self.m_sorted:
            a = abs(m)
            values = self.r**a * np.exp(-self.r**2)
            self.rho[m] = values
            # Diagnostics are computed by the public constructor even when
            # warning emission is disabled.  Keeping a fixed support makes the
            # timed operation identical across scaling sweeps except for the
            # dimension being varied.
            self.cutoff_radius[m] = float(r_max)
            powers[m] = float(simpson(self.r * np.abs(values) ** 2, x=self.r))
        total = max(sum(powers.values()), np.finfo(float).tiny)
        self.power_fracs = {m: powers[m] / total for m in self.m_sorted}

    def __getitem__(self, m):
        return self.rho[int(m)]


def symmetric_modes(n_m: int) -> list[int]:
    """Return 0, +/-1, +/-2, ... for a positive odd harmonic count."""
    n_m = int(n_m)
    if n_m < 1 or n_m % 2 == 0:
        raise ValueError("n_m must be a positive odd integer.")
    out = [0]
    for m in range(1, (n_m - 1) // 2 + 1):
        out.extend([m, -m])
    return out


def n_sample_nodes(n_r: int, subdivisions: int) -> int:
    """Effective Simpson/trapezoid radial integration-grid size."""
    return int(subdivisions) * (int(n_r) - 1) + 1


def fit_loglog(x, y) -> dict:
    """Fit y = C x^alpha in log space and return alpha, C, and R^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log(x)
    ly = np.log(y)
    alpha, log_c = np.polyfit(lx, ly, 1)
    pred = alpha * lx + log_c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "alpha": float(alpha),
        "prefactor": float(np.exp(log_c)),
        "r2_log": float(r2),
    }


def scaling_fit(rows: list[dict], *, tail_points: int) -> dict:
    rr = sorted(rows, key=lambda row: row["value"])
    n_tail = min(max(3, int(tail_points)), len(rr))
    tail = rr[-n_tail:]
    return {
        "all_points": fit_loglog(
            [row["fit_x"] for row in rr],
            [row["median_seconds"] for row in rr],
        ),
        "asymptotic_tail": {
            "n_points": n_tail,
            "value_range": [int(tail[0]["value"]), int(tail[-1]["value"])],
            **fit_loglog(
                [row["fit_x"] for row in tail],
                [row["median_seconds"] for row in tail],
            ),
        },
    }


def time_transform(*, n_r, n_q, n_m, subdivisions, warmups, repeats):
    dec = SyntheticDecomposition(n_r=n_r, m_values=symmetric_modes(n_m))
    _, timing = timed_call(
        lambda: HarmonicTransform(
            dec,
            q_max=8.0,
            n_q=n_q,
            method="simpson",
            interpolator="cubic",
            subdivisions=subdivisions,
            check=False,
        ),
        warmups=warmups,
        repeats=repeats,
    )
    n_s = n_sample_nodes(n_r, subdivisions)
    timing.update(
        {
            "N_r": int(n_r),
            "N_q": int(n_q),
            "N_m": int(n_m),
            "s_r": int(subdivisions),
            "N_s": int(n_s),
            "B_eff": int(min(BATCH_SIZE, int(n_q))),
            "work_units": int(n_m) * int(n_q) * int(n_s),
            "temporary_matrix_elements": int(min(BATCH_SIZE, int(n_q))) * int(n_s),
            "persistent_output_elements": int(n_m) * int(n_q),
        }
    )
    timing["seconds_per_work_unit"] = (
        float(timing["median_seconds"]) / float(timing["work_units"])
    )
    return timing


def sweep(*, axis, values, baseline, args):
    rows = []
    for value in values:
        params = dict(baseline)
        params[axis] = int(value)
        timing = time_transform(
            n_r=params["n_r"],
            n_q=params["n_q"],
            n_m=params["n_m"],
            subdivisions=params["subdivisions"],
            warmups=args.warmups,
            repeats=args.repeats,
        )
        if axis == "n_r":
            fit_x = timing["N_s"]
            label = "N_r"
        elif axis == "subdivisions":
            fit_x = timing["N_s"]
            label = "s_r"
        elif axis == "n_q":
            fit_x = timing["N_q"]
            label = "N_q"
        elif axis == "n_m":
            fit_x = timing["N_m"]
            label = "N_m"
        else:  # pragma: no cover
            raise ValueError(axis)
        row = {
            "axis": label,
            "value": int(value),
            "fit_x": int(fit_x),
            **timing,
        }
        rows.append(row)
        print(
            f"{label:>3s}={int(value):5d}  "
            f"N_s={timing['N_s']:5d}  "
            f"W={timing['work_units']:10d}  "
            f"t={1e3*timing['median_seconds']:9.3f} ms"
        )
    return rows


def run(args):
    baseline = {
        "n_r": int(args.base_nr),
        "n_q": int(args.base_nq),
        "n_m": int(args.base_nm),
        "subdivisions": int(args.base_subdivisions),
    }

    rows = []
    print("[HarmonicTransform] N_q scaling")
    rows.extend(
        sweep(
            axis="n_q",
            values=(32, 64, 128, 256, 512, 1024, 2048, 4096),
            baseline=baseline,
            args=args,
        )
    )
    print("\n[HarmonicTransform] N_r scaling")
    rows.extend(
        sweep(
            axis="n_r",
            values=(64, 128, 256, 512, 1024, 2048, 4096),
            baseline=baseline,
            args=args,
        )
    )
    print("\n[HarmonicTransform] N_m scaling")
    rows.extend(
        sweep(
            axis="n_m",
            values=(1, 3, 5, 7, 9, 13, 17),
            baseline=baseline,
            args=args,
        )
    )
    print("\n[HarmonicTransform] s_r scaling")
    rows.extend(
        sweep(
            axis="subdivisions",
            values=(1, 2, 4, 8, 16),
            baseline=baseline,
            args=args,
        )
    )

    fits = {}
    for axis in ("N_q", "N_r", "N_m", "s_r"):
        rr = [row for row in rows if row["axis"] == axis]
        fits[axis] = {
            "expected_alpha": 1.0,
            "fit_variable": "N_s" if axis in ("N_r", "s_r") else axis,
            **scaling_fit(rr, tail_points=args.fit_tail_points),
        }

    # Product-complexity collapse for sweeps with a fixed harmonic set.
    # The N_m sweep is verified separately because adding harmonics also raises
    # the Bessel orders |m| and therefore changes the per-element prefactor.
    collapse_rows = [row for row in rows if row["axis"] != "N_m"]
    collapse_tail = sorted(collapse_rows, key=lambda row: row["work_units"])
    n_tail = min(max(6, int(args.collapse_tail_points)), len(collapse_tail))
    collapse_fit = {
        "expected_alpha": 1.0,
        "all_points": fit_loglog(
            [row["work_units"] for row in collapse_rows],
            [row["median_seconds"] for row in collapse_rows],
        ),
        "asymptotic_tail": {
            "n_points": n_tail,
            "work_range": [
                int(collapse_tail[-n_tail]["work_units"]),
                int(collapse_tail[-1]["work_units"]),
            ],
            **fit_loglog(
                [row["work_units"] for row in collapse_tail[-n_tail:]],
                [row["median_seconds"] for row in collapse_tail[-n_tail:]],
            ),
        },
    }

    return {
        "schema": 2,
        "benchmark": "HarmonicTransform computational scaling",
        "scope": {
            "stage": "multi-harmonic real-space to momentum-space transform",
            "timed_operation": "public HarmonicTransform constructor with explicit q_max, n_q, Simpson subdivisions, and check=False",
            "constructor_note": "check=False suppresses warning emission; the public constructor still computes its inexpensive default-grid metadata and post-transform diagnostics, so those costs are included",
            "excluded": "synthetic input construction; plotting; accuracy/convergence search",
        },
        "environment": environment_metadata(),
        "timing": {
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "statistic": "median",
            "spread": "interquartile range",
        },
        "baseline": {
            "N_r": baseline["n_r"],
            "N_q": baseline["n_q"],
            "N_m": baseline["n_m"],
            "s_r": baseline["subdivisions"],
            "q_max": 8.0,
            "method": "simpson",
            "interpolator": "cubic",
            "B": BATCH_SIZE,
        },
        "theory": {
            "effective_radial_nodes": "N_s = s_r (N_r - 1) + 1",
            "time_full": "Theta(N_m [N_r + N_q + N_q N_s])",
            "time_dominant": "Theta(N_m N_q N_s) = Theta(N_m N_q s_r N_r)",
            "memory": "Theta(N_m N_q + min(B,N_q) N_s + N_r)",
            "assumption": "special-function evaluation cost is treated as O(1) per matrix element for fixed harmonic order",
        },
        "rows": rows,
        "fits": fits,
        "work_collapse_axes": ["N_q", "N_r", "s_r"],
        "work_collapse_fit": collapse_fit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmarks/results/publication/harmonic_scaling.json")
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--base-nr", type=int, default=256)
    ap.add_argument("--base-nq", type=int, default=256)
    ap.add_argument("--base-nm", type=int, default=3)
    ap.add_argument("--base-subdivisions", type=int, default=2)
    ap.add_argument("--fit-tail-points", type=int, default=4)
    ap.add_argument("--collapse-tail-points", type=int, default=10)
    args = ap.parse_args()
    data = run(args)
    write_json(args.output, data)
    print("\n=== asymptotic fits ===")
    for axis, fit in data["fits"].items():
        tail = fit["asymptotic_tail"]
        print(f"{axis:>3s}: alpha={tail['alpha']:.4f}, R2_log={tail['r2_log']:.5f}")
    cfit = data["work_collapse_fit"]["asymptotic_tail"]
    print(f"  W: alpha={cfit['alpha']:.4f}, R2_log={cfit['r2_log']:.5f}")
    print(f"Saved: {Path(args.output)}")


if __name__ == "__main__":
    main()
