#!/usr/bin/env python3
"""Empirical computational scaling of the public ``Interaction`` constructor."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from quartic2d import HarmonicTransform, Interaction

from benchmarks._common import environment_metadata
from benchmarks.harmonic_transform import SyntheticDecomposition, workloads

class FieldView:
    def __init__(self, source, modes=None, n_q=None):
        self._source = source
        self._m_values = np.array(source.m_values if modes is None else modes, dtype=int)
        qmax = float(source.q[-1])
        self.q = source.q if n_q is None else np.linspace(0.0, qmax, int(n_q))
        self.F_q = {int(m): np.asarray(source(int(m), self.q), dtype=np.complex128) for m in self._m_values}

    @property
    def m_values(self):
        return self._m_values.copy()

    def __call__(self, m, q):
        # Interaction uses F_q when available; this fallback preserves the field API.
        return self._source(int(m), q)


def base_field():
    w = {x.name: x for x in workloads()}["gaussian_high_order_m4"]
    decomp = SyntheticDecomposition(w, 384)
    return HarmonicTransform(
        decomp,
        q_max=8.364234227675926,
        n_q=128,
        method="simpson",
        interpolator="cubic",
        subdivisions=1,
        check=False,
    )


def kernel(q):
    q = np.asarray(q, dtype=float)
    return 2.0 * np.pi / (q + 1.0)


def deltas(n):
    r = np.geomspace(1.0e-2, 1.0e2, int(n))
    a = np.linspace(0.0, 1.5 * np.pi, int(n))
    return np.column_stack((r * np.cos(a), r * np.sin(a)))


def timed(fn, warmups=2, repeats=7):
    for _ in range(int(warmups)):
        fn()
    samples = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_seconds": float(statistics.median(samples)),
        "q25_seconds": float(np.quantile(samples, 0.25)),
        "q75_seconds": float(np.quantile(samples, 0.75)),
        "samples_seconds": [float(x) for x in samples],
    }


def fit_power(rows, xkey, tail=4):
    x = np.array([float(r[xkey]) for r in rows], dtype=float)
    y = np.array([float(r["median_seconds"]) for r in rows], dtype=float)
    def fit(xs, ys):
        c = np.polyfit(np.log(xs), np.log(ys), 1)
        pred = np.polyval(c, np.log(xs))
        ss_res = np.sum((np.log(ys) - pred) ** 2)
        ss_tot = np.sum((np.log(ys) - np.mean(np.log(ys))) ** 2)
        return {"alpha": float(c[0]), "prefactor": float(np.exp(c[1])), "r2_log": float(1.0 - ss_res / ss_tot)}
    return {"all_points": fit(x, y), "asymptotic_tail": fit(x[-tail:], y[-tail:]), "tail_n": int(tail)}


def fit_linear(rows, xkey, tail=None):
    rr = rows if tail is None else rows[-tail:]
    x = np.array([float(r[xkey]) for r in rr])
    y = np.array([float(r["median_seconds"]) for r in rr])
    c = np.polyfit(x, y, 1)
    pred = np.polyval(c, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {"slope": float(c[0]), "intercept": float(c[1]), "r2": float(1.0 - ss_res / ss_tot)}


def run_one(field1, field2, dxy, method, *, n=512, subdivisions=1, warmups=2, repeats=7):
    stats = timed(
        lambda: Interaction(
            dxy,
            field1,
            field2,
            kernel,
            method=method,
            interpolator="cubic",
            n=int(n),
            bias=-0.5,
            subdivisions=int(subdivisions),
        ),
        warmups=warmups,
        repeats=repeats,
    )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("benchmarks/results/interaction_scaling.json"))
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="Reduced smoke-test ranges; not for paper results.")
    args = ap.parse_args()

    source = base_field()
    modes1 = [0]
    modes3 = [0, 2, -2]
    modes5 = [0, 2, -2, 4, -4]
    f3 = FieldView(source, modes3)

    rows = []

    nf_values = (128, 512, 2048) if args.quick else (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
    nd_fft_values = (16, 256, 4096) if args.quick else (16, 64, 256, 1024, 4096, 16384, 65536)
    nq_values = (32, 128, 512) if args.quick else (32, 64, 128, 256, 512, 1024)
    nd_finite_values = (16, 256, 1024) if args.quick else (16, 64, 256, 1024, 4096)
    sq_values = (1, 4, 16) if args.quick else (1, 2, 4, 8, 16)

    # FFTLog transform length N_F.
    for nf in nf_values:
        stat = run_one(f3, f3, deltas(256), "fftlog", n=nf, warmups=args.warmups, repeats=args.repeats)
        row = {"branch": "fftlog", "axis": "N_F", "value": nf, "N_F": nf, "N_D": 256, "N_p": 9, **stat}
        rows.append(row)
        print(f"[fftlog N_F] {nf:5d}: {1e3*stat['median_seconds']:.2f} ms")

    # FFTLog requested displacements N_D.
    for nd in nd_fft_values:
        stat = run_one(f3, f3, deltas(nd), "fftlog", n=512, warmups=args.warmups, repeats=args.repeats)
        row = {"branch": "fftlog", "axis": "N_D", "value": nd, "N_F": 512, "N_D": nd, "N_p": 9, **stat}
        rows.append(row)
        print(f"[fftlog N_D] {nd:5d}: {1e3*stat['median_seconds']:.2f} ms")

    # Pair count using subsets of one validated five-harmonic field.
    pair_specs = [
        (modes1, modes1),
        (modes1, modes3),
        (modes3, modes3),
        (modes3, modes5),
        (modes5, modes5),
    ]
    for a, b in pair_specs:
        f1, f2 = FieldView(source, a), FieldView(source, b)
        npair = len(a) * len(b)
        stat = run_one(f1, f2, deltas(256), "fftlog", n=512, warmups=args.warmups, repeats=args.repeats)
        rows.append({"branch": "fftlog", "axis": "N_p", "value": npair, "N_F": 512, "N_D": 256, "N_p": npair, **stat})
        print(f"[fftlog N_p] {npair:3d}: {1e3*stat['median_seconds']:.2f} ms")

    # Finite GL4 backbone: native q-grid size.
    for nq in nq_values:
        fq = FieldView(source, modes3, n_q=nq)
        stat = run_one(fq, fq, deltas(256), "gl4", subdivisions=1, warmups=args.warmups, repeats=args.repeats)
        rows.append({"branch": "finite", "axis": "N_q", "value": nq, "N_q": nq, "s_q": 1, "N_D": 256, "N_p": 9, **stat})
        print(f"[finite N_q] {nq:4d}: {1e3*stat['median_seconds']:.2f} ms")

    for nd in nd_finite_values:
        stat = run_one(f3, f3, deltas(nd), "gl4", subdivisions=1, warmups=args.warmups, repeats=args.repeats)
        rows.append({"branch": "finite", "axis": "N_D", "value": nd, "N_q": 128, "s_q": 1, "N_D": nd, "N_p": 9, **stat})
        print(f"[finite N_D] {nd:4d}: {1e3*stat['median_seconds']:.2f} ms")

    for a, b in pair_specs:
        f1, f2 = FieldView(source, a), FieldView(source, b)
        npair = len(a) * len(b)
        stat = run_one(f1, f2, deltas(256), "gl4", subdivisions=1, warmups=args.warmups, repeats=args.repeats)
        rows.append({"branch": "finite", "axis": "N_p", "value": npair, "N_q": 128, "s_q": 1, "N_D": 256, "N_p": npair, **stat})
        print(f"[finite N_p] {npair:3d}: {1e3*stat['median_seconds']:.2f} ms")

    for sq in sq_values:
        stat = run_one(f3, f3, deltas(256), "gl4", subdivisions=sq, warmups=args.warmups, repeats=args.repeats)
        nqs = sq * (128 - 1) + 1
        rows.append({"branch": "finite", "axis": "s_q", "value": sq, "N_q": 128, "N_qs": nqs, "s_q": sq, "N_D": 256, "N_p": 9, **stat})
        print(f"[finite s_q] {sq:3d}: {1e3*stat['median_seconds']:.2f} ms")

    def select(branch, axis):
        return [r for r in rows if r["branch"] == branch and r["axis"] == axis]

    fft_nf = select("fftlog", "N_F")
    tail4 = min(4, len(fft_nf))
    # Fit against N_F log2 N_F for the asymptotic FFT work.
    for r in fft_nf:
        r["N_F_log2_N_F"] = float(r["N_F"] * math.log2(r["N_F"]))

    fits = {
        "fftlog_N_F_work": {
            "expected": "linear in N_F log2(N_F)",
            "linear_tail": fit_linear(fft_nf, "N_F_log2_N_F", tail=min(5, len(fft_nf))),
            "power_vs_work": fit_power(fft_nf, "N_F_log2_N_F", tail=min(5, len(fft_nf))),
        },
        "fftlog_N_D": {"expected_alpha": 1.0, **fit_power(select("fftlog", "N_D"), "N_D", tail=min(4, len(select("fftlog", "N_D"))))},
        "fftlog_N_p": {"expected_alpha": 1.0, **fit_power(select("fftlog", "N_p"), "N_p", tail=min(4, len(select("fftlog", "N_p"))))},
        "finite_N_q": {"expected_alpha": 1.0, **fit_power(select("finite", "N_q"), "N_q", tail=min(4, len(select("finite", "N_q"))))},
        "finite_N_D": {"expected_alpha": 1.0, **fit_power(select("finite", "N_D"), "N_D", tail=min(4, len(select("finite", "N_D"))))},
        "finite_N_p": {"expected_alpha": 1.0, **fit_power(select("finite", "N_p"), "N_p", tail=min(4, len(select("finite", "N_p"))))},
        "finite_s_q": {"expected_alpha": 1.0, **fit_power(select("finite", "s_q"), "s_q", tail=min(4, len(select("finite", "s_q"))))},
    }

    result = {
        "schema": 1,
        "benchmark": "Interaction computational scaling",
        "scope": {
            "timed_operation": "public Interaction constructor",
            "fixed_input": "one previously validated five-harmonic HarmonicTransform, with mode subsets/resampling used only to vary independent complexity dimensions",
            "finite_backend": "GL4, matching the general Interaction backbone selected by the accuracy/convergence benchmarks",
            "excluded": "HarmonicTransform construction; accuracy convergence; plotting",
        },
        "environment": environment_metadata(),
        "timing": {"warmups": args.warmups, "repeats": args.repeats, "statistic": "median", "spread": "IQR"},
        "theory": {
            "N_p": "N_m1 N_m2",
            "fftlog_time": "Theta(N_p [N_F log N_F + N_D])",
            "finite_effective_q_nodes": "N_qs = s_q (N_q - 1) + 1",
            "finite_time": "Theta(N_p N_D N_qs) = Theta(N_p N_D s_q N_q)",
            "fftlog_memory": "Theta(N_p N_D + N_F + N_q)",
            "finite_memory": "Theta(N_p N_D + N_D N_qs + N_q)",
        },
        "rows": rows,
        "fits": fits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("\n=== interaction scaling fits ===")
    print("FFTLog N_F work R2:", fits["fftlog_N_F_work"]["linear_tail"]["r2"])
    for k, v in fits.items():
        if "asymptotic_tail" in v:
            print(f"{k:16s}: alpha={v['asymptotic_tail']['alpha']:.4f}, R2={v['asymptotic_tail']['r2_log']:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
