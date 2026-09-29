#!/usr/bin/env python3
"""Paper benchmark: peak-RSS scaling of HarmonicTransform."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

from benchmarks._common import environment_metadata, write_json
from quartic2d import HarmonicTransform

BATCH_SIZE = 256
MIB = 1024.0**2


class SyntheticDecomposition:
    """Minimal smooth PETAL2D-compatible input for memory measurements."""

    def __init__(self, *, n_r: int, m_values: list[int], r_max: float = 6.0):
        self.r = np.linspace(0.0, float(r_max), int(n_r))
        self.m_sorted = [int(m) for m in m_values]
        self.rho = {}
        self.cutoff_radius = {}
        powers = {}
        phase = np.exp(0.31j * self.r)
        for m in self.m_sorted:
            a = abs(m)
            values = self.r**a * np.exp(-self.r**2) * phase
            self.rho[m] = np.asarray(values, dtype=np.complex128)
            self.cutoff_radius[m] = float(r_max)
            powers[m] = float(simpson(self.r * np.abs(values) ** 2, x=self.r))
        total = max(sum(powers.values()), np.finfo(float).tiny)
        self.power_fracs = {m: powers[m] / total for m in self.m_sorted}

    def __getitem__(self, m):
        return self.rho[int(m)]


def symmetric_modes(n_m: int) -> list[int]:
    n_m = int(n_m)
    if n_m < 1 or n_m % 2 == 0:
        raise ValueError("n_m must be a positive odd integer.")
    out = [0]
    for m in range(1, (n_m - 1) // 2 + 1):
        out.extend([m, -m])
    return out


def n_sample_nodes(n_r: int, subdivisions: int) -> int:
    return int(subdivisions) * (int(n_r) - 1) + 1


def read_rss_bytes(pid: int) -> int:
    """Read Linux VmRSS for ``pid``."""
    with open(f"/proc/{int(pid)}/status", "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"VmRSS not found for pid={pid}.")


def worker(args) -> int:
    dec = SyntheticDecomposition(
        n_r=args.n_r,
        m_values=symmetric_modes(args.n_m),
        r_max=args.r_max,
    )
    print("READY", flush=True)
    command = sys.stdin.readline().strip()
    if command != "GO":
        raise RuntimeError(f"Expected GO, received {command!r}.")

    transformed = HarmonicTransform(
        dec,
        q_max=args.q_max,
        n_q=args.n_q,
        method="simpson",
        interpolator="cubic",
        subdivisions=args.subdivisions,
        check=False,
    )
    # Touch the stored arrays so the object must remain live through DONE.
    checksum = float(sum(np.sum(np.abs(v)) for v in transformed.F_q.values()))
    print(json.dumps({"status": "DONE", "checksum": checksum}), flush=True)
    sys.stdin.readline()
    return 0


def sample_one(*, n_r, n_q, n_m, subdivisions, r_max, q_max, poll_seconds):
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.harmonic_memory",
        "--worker",
        "--n-r", str(int(n_r)),
        "--n-q", str(int(n_q)),
        "--n-m", str(int(n_m)),
        "--subdivisions", str(int(subdivisions)),
        "--r-max", str(float(r_max)),
        "--q-max", str(float(q_max)),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    ready = proc.stdout.readline().strip()
    if ready != "READY":
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        proc.kill()
        raise RuntimeError(f"Memory worker failed before READY: {ready!r}\n{stderr}")

    baseline = read_rss_bytes(proc.pid)
    samples = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append(read_rss_bytes(proc.pid))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                break
            time.sleep(poll_seconds)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    proc.stdin.write("GO\n")
    proc.stdin.flush()

    done_line = proc.stdout.readline().strip()
    if not done_line:
        stop.set()
        thread.join()
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        proc.kill()
        raise RuntimeError(f"Memory worker exited without DONE.\n{stderr}")
    done = json.loads(done_line)
    if done.get("status") != "DONE":
        stop.set()
        thread.join()
        proc.kill()
        raise RuntimeError(f"Unexpected worker response: {done_line}")

    # Retain the completed object briefly so the sampler records persistent RSS.
    time.sleep(max(0.01, 4.0 * poll_seconds))
    try:
        samples.append(read_rss_bytes(proc.pid))
    except (FileNotFoundError, ProcessLookupError, RuntimeError):
        pass
    stop.set()
    thread.join()

    proc.stdin.write("EXIT\n")
    proc.stdin.flush()
    _, stderr = proc.communicate(timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"Memory worker returned {proc.returncode}.\n{stderr}")

    peak = max(samples)
    return {
        "baseline_rss_bytes": int(baseline),
        "peak_rss_bytes": int(peak),
        "incremental_peak_rss_bytes": int(max(0, peak - baseline)),
        "rss_samples": len(samples),
    }


def summarize_repeats(samples: list[dict]) -> dict:
    out = {"samples": samples}
    for key in ("baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes"):
        values = np.asarray([row[key] for row in samples], dtype=float)
        out[key.replace("_bytes", "_median_bytes")] = int(np.median(values))
        out[key.replace("_bytes", "_q25_bytes")] = int(np.percentile(values, 25))
        out[key.replace("_bytes", "_q75_bytes")] = int(np.percentile(values, 75))
    return out


def linear_fit(x, y) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "slope_bytes_per_x": float(slope),
        "intercept_bytes": float(intercept),
        "r2": float(r2),
    }


def memory_point(*, axis, value, n_r, n_q, n_m, subdivisions, args):
    repetitions = []
    for _ in range(args.repeats):
        repetitions.append(
            sample_one(
                n_r=n_r,
                n_q=n_q,
                n_m=n_m,
                subdivisions=subdivisions,
                r_max=args.r_max,
                q_max=args.q_max,
                poll_seconds=args.poll_ms / 1000.0,
            )
        )
    summary = summarize_repeats(repetitions)
    n_s = n_sample_nodes(n_r, subdivisions)
    b_eff = min(BATCH_SIZE, n_q)
    row = {
        "axis": axis,
        "value": int(value),
        "N_r": int(n_r),
        "N_q": int(n_q),
        "N_m": int(n_m),
        "s_r": int(subdivisions),
        "N_s": int(n_s),
        "B_eff": int(b_eff),
        "temporary_matrix_elements": int(b_eff * n_s),
        "persistent_output_elements": int(n_m * n_q),
        "source_memory_proxy_elements": int(b_eff * n_s + n_m * n_q + n_r),
        **summary,
    }
    print(
        f"{axis:>3s}={int(value):5d}  B_eff={b_eff:3d}  N_s={n_s:5d}  "
        f"DeltaRSS={row['incremental_peak_rss_median_bytes']/MIB:8.2f} MiB"
    )
    return row


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def run(args) -> dict:
    nq_values = parse_int_list(args.nq_values)
    nr_values = parse_int_list(args.nr_values)

    rows = []
    print("[memory] N_q")
    for n_q in nq_values:
        rows.append(
            memory_point(
                axis="N_q",
                value=n_q,
                n_r=args.base_nr,
                n_q=n_q,
                n_m=args.base_nm,
                subdivisions=args.base_subdivisions,
                args=args,
            )
        )

    print("\n[memory] N_r")
    for n_r in nr_values:
        rows.append(
            memory_point(
                axis="N_r",
                value=n_r,
                n_r=n_r,
                n_q=args.base_nq,
                n_m=args.base_nm,
                subdivisions=args.base_subdivisions,
                args=args,
            )
        )

    nq_rows = [row for row in rows if row["axis"] == "N_q"]
    nr_rows = [row for row in rows if row["axis"] == "N_r"]
    pre = [row for row in nq_rows if row["N_q"] <= BATCH_SIZE]
    post = [row for row in nq_rows if row["N_q"] >= BATCH_SIZE]

    if not nq_rows or not nr_rows:
        raise ValueError("nq-values and nr-values must each contain at least one value")

    def optional_linear_fit(selected, *, x_key):
        if len(selected) < 2:
            return None
        return {
            "x": x_key,
            "range": [int(selected[0][x_key]), int(selected[-1][x_key])],
            **linear_fit(
                [row[x_key] for row in selected],
                [row["incremental_peak_rss_median_bytes"] for row in selected],
            ),
        }

    fits = {
        "N_q_pre_batch": optional_linear_fit(pre, x_key="N_q"),
        "N_q_post_batch": optional_linear_fit(post, x_key="N_q"),
        "N_r": optional_linear_fit(nr_rows, x_key="N_s"),
    }
    pre_fit = fits["N_q_pre_batch"]
    post_fit = fits["N_q_post_batch"]
    if pre_fit is None or post_fit is None:
        fits["batching_slope_ratio_post_over_pre"] = None
    else:
        pre_slope = pre_fit["slope_bytes_per_x"]
        post_slope = post_fit["slope_bytes_per_x"]
        fits["batching_slope_ratio_post_over_pre"] = (
            float(post_slope / pre_slope) if pre_slope != 0 else None
        )

    result = {
        "schema": 1,
        "benchmark": "HarmonicTransform peak-memory scaling",
        "scope": {
            "metric": "Linux VmRSS sampled externally while the public HarmonicTransform constructor executes",
            "reported_primary": "baseline-subtracted incremental peak RSS",
            "input_dtype": "complex128",
            "timed": False,
            "diagnostics": "check=False",
        },
        "environment": environment_metadata(),
        "settings": {
            "repeats": int(args.repeats),
            "poll_ms": float(args.poll_ms),
            "q_max": float(args.q_max),
            "r_max": float(args.r_max),
            "method": "simpson",
            "interpolator": "cubic",
            "B": BATCH_SIZE,
            "baseline": {
                "N_r": int(args.base_nr),
                "N_q": int(args.base_nq),
                "N_m": int(args.base_nm),
                "s_r": int(args.base_subdivisions),
            },
            "N_q_values": list(nq_values),
            "N_r_values": list(nr_values),
        },
        "theory": {
            "N_s": "s_r (N_r - 1) + 1",
            "B_eff": "min(B, N_q)",
            "memory": "Theta(N_m N_q + B_eff N_s + N_r)",
            "batching_prediction": "temporary Bessel/integrand workspace stops growing with N_q once N_q >= B",
        },
        "rows": rows,
        "fits": fits,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results/harmonic_memory.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-ms", type=float, default=1.0)
    parser.add_argument("--base-nr", type=int, default=1024)
    parser.add_argument("--base-nq", type=int, default=256)
    parser.add_argument("--base-nm", type=int, default=3)
    parser.add_argument("--base-subdivisions", type=int, default=2)
    parser.add_argument("--nq-values", default="32,64,128,192,256,384,512,1024,2048,4096")
    parser.add_argument("--nr-values", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--q-max", type=float, default=8.0)

    # Internal subprocess protocol.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--n-r", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--n-q", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--n-m", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--subdivisions", type=int, default=2, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args))

    if sys.platform != "linux" or not Path("/proc/self/status").exists():
        raise RuntimeError("This peak-RSS benchmark currently requires Linux /proc.")
    if args.repeats < 1:
        raise ValueError("repeats must be positive.")
    if args.poll_ms <= 0:
        raise ValueError("poll-ms must be positive.")

    result = run(args)
    write_json(args.output, result)
    print("\n=== memory fits ===")
    for name in ("N_q_pre_batch", "N_q_post_batch", "N_r"):
        fit = result["fits"][name]
        if fit is None:
            print(f"{name:>16s}: insufficient sweep points")
            continue
        print(
            f"{name:>16s}: slope={fit['slope_bytes_per_x']/MIB:.6g} MiB/x, "
            f"R2={fit['r2']:.6f}"
        )
    ratio = result["fits"]["batching_slope_ratio_post_over_pre"]
    ratio_text = "--" if ratio is None else f"{ratio:.5g}"
    print("post/pre N_q slope ratio = " + ratio_text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
