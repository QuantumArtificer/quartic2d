#!/usr/bin/env python3
"""Peak-RSS scaling of the public ``Interaction`` constructor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from quartic2d import Interaction

from benchmarks._common import environment_metadata
from benchmarks.interaction_scaling import FieldView, base_field, deltas, kernel

MIB = 1024.0**2
MODES = {
    1: [0],
    3: [0, 2, -2],
    5: [0, 2, -2, 4, -4],
}
PAIR_SPECS = {
    1: (1, 1),
    3: (1, 3),
    9: (3, 3),
    15: (3, 5),
    25: (5, 5),
}


def read_rss_bytes(pid):
    with open(f"/proc/{int(pid)}/status", "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS unavailable")


def make_fields(n_p: int, n_q: int):
    source = base_field()
    na, nb = PAIR_SPECS[int(n_p)]
    return FieldView(source, MODES[na], n_q=n_q), FieldView(source, MODES[nb], n_q=n_q)


def worker(args):
    f1, f2 = make_fields(args.n_p, args.n_q)
    dxy = deltas(args.n_d)
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise RuntimeError("Expected GO")
    obj = Interaction(
        dxy,
        f1,
        f2,
        kernel,
        method=args.method,
        interpolator="cubic",
        n=args.n_f,
        bias=-0.5,
        subdivisions=args.s_q,
    )
    checksum = float(np.sum(np.abs(obj.V)))
    print(json.dumps({"status": "DONE", "checksum": checksum}), flush=True)
    sys.stdin.readline()
    return 0


def sample_one(*, method, n_p, n_d, n_f, n_q, s_q, poll_s):
    cmd = [
        sys.executable, "-m", "benchmarks.interaction_memory", "--worker",
        "--method", method,
        "--n-p", str(int(n_p)), "--n-d", str(int(n_d)),
        "--n-f", str(int(n_f)), "--n-q", str(int(n_q)), "--s-q", str(int(s_q)),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, env=os.environ.copy())
    assert proc.stdin is not None and proc.stdout is not None
    ready = proc.stdout.readline().strip()
    if ready != "READY":
        err = proc.stderr.read() if proc.stderr is not None else ""
        proc.kill(); raise RuntimeError(f"worker failed: {ready!r}\n{err}")
    baseline = read_rss_bytes(proc.pid)
    samples = [baseline]
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try: samples.append(read_rss_bytes(proc.pid))
            except Exception: break
            time.sleep(poll_s)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    proc.stdin.write("GO\n"); proc.stdin.flush()
    done = proc.stdout.readline().strip()
    if not done:
        stop.set(); thread.join();
        err = proc.stderr.read() if proc.stderr is not None else ""
        proc.kill(); raise RuntimeError(err)
    time.sleep(max(0.01, 4 * poll_s))
    try: samples.append(read_rss_bytes(proc.pid))
    except Exception: pass
    stop.set(); thread.join()
    proc.stdin.write("EXIT\n"); proc.stdin.flush()
    _, err = proc.communicate(timeout=30)
    if proc.returncode != 0: raise RuntimeError(err)
    peak = max(samples)
    return {"baseline_rss_bytes": baseline, "peak_rss_bytes": peak,
            "incremental_peak_rss_bytes": max(0, peak-baseline), "rss_samples": len(samples)}


def summarize(samples):
    out = {"samples": samples}
    for k in ("baseline_rss_bytes", "peak_rss_bytes", "incremental_peak_rss_bytes"):
        v = np.array([x[k] for x in samples], float)
        root = k.replace("_bytes", "")
        out[root + "_median_bytes"] = int(np.median(v))
        out[root + "_q25_bytes"] = int(np.percentile(v,25))
        out[root + "_q75_bytes"] = int(np.percentile(v,75))
    return out


def linear_fit(rows, xkey):
    x=np.array([r[xkey] for r in rows],float); y=np.array([r["incremental_peak_rss_median_bytes"] for r in rows],float)
    c=np.polyfit(x,y,1); pred=np.polyval(c,x)
    ssr=float(np.sum((y-pred)**2)); sst=float(np.sum((y-y.mean())**2))
    return {"slope_bytes_per_x":float(c[0]),"intercept_bytes":float(c[1]),"r2":float(1-ssr/sst if sst else 1)}


def point(axis, value, *, method, n_p, n_d, n_f, n_q, s_q, args):
    smp=[sample_one(method=method,n_p=n_p,n_d=n_d,n_f=n_f,n_q=n_q,s_q=s_q,poll_s=args.poll_ms/1000) for _ in range(args.repeats)]
    row={"branch":method,"axis":axis,"value":int(value),"N_p":int(n_p),"N_D":int(n_d),"N_F":int(n_f),"N_q":int(n_q),"s_q":int(s_q),**summarize(smp)}
    print(f"[{method:7s} {axis}] {int(value):6d}: {row['incremental_peak_rss_median_bytes']/MIB:8.2f} MiB")
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path("benchmarks/results/interaction_memory.json"))
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--poll-ms",type=float,default=1.0)
    ap.add_argument("--worker",action="store_true")
    ap.add_argument("--method",default="fftlog"); ap.add_argument("--n-p",type=int,default=9); ap.add_argument("--n-d",type=int,default=256)
    ap.add_argument("--n-f",type=int,default=512); ap.add_argument("--n-q",type=int,default=128); ap.add_argument("--s-q",type=int,default=1)
    args=ap.parse_args()
    if args.worker: return worker(args)
    rows=[]
    for nd in (16,64,256,1024,4096,16384,65536): rows.append(point("N_D",nd,method="fftlog",n_p=9,n_d=nd,n_f=512,n_q=128,s_q=1,args=args))
    for nf in (128,256,512,1024,2048,4096,8192,16384): rows.append(point("N_F",nf,method="fftlog",n_p=9,n_d=256,n_f=nf,n_q=128,s_q=1,args=args))
    for npair in (1,3,9,15,25): rows.append(point("N_p",npair,method="fftlog",n_p=npair,n_d=4096,n_f=512,n_q=128,s_q=1,args=args))
    for nd in (16,64,256,1024,4096): rows.append(point("N_D",nd,method="gl4",n_p=9,n_d=nd,n_f=512,n_q=128,s_q=1,args=args))
    for nq in (32,64,128,256,512,1024): rows.append(point("N_q",nq,method="gl4",n_p=9,n_d=1024,n_f=512,n_q=nq,s_q=1,args=args))
    def sel(branch,axis): return [r for r in rows if r['branch']==branch and r['axis']==axis]
    fits={
      "fftlog_N_D":linear_fit(sel('fftlog','N_D'),'N_D'),
      "fftlog_N_F":linear_fit(sel('fftlog','N_F'),'N_F'),
      "fftlog_N_p":linear_fit(sel('fftlog','N_p'),'N_p'),
      "finite_N_D":linear_fit(sel('gl4','N_D'),'N_D'),
      "finite_N_q":linear_fit(sel('gl4','N_q'),'N_q'),
    }
    result={"schema":1,"benchmark":"Interaction peak-memory scaling","scope":{"metric":"Linux VmRSS sampled externally during the public Interaction constructor","reported_primary":"baseline-subtracted incremental peak RSS","fixed_fields":"constructed before baseline sampling and excluded from incremental memory"},"environment":environment_metadata(),"settings":{"repeats":args.repeats,"poll_ms":args.poll_ms},"theory":{"fftlog_memory":"Theta(N_p N_D + N_F + N_q)","finite_memory":"Theta(N_p N_D + N_D s_q N_q + N_q)"},"rows":rows,"fits":fits}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)); print(f"Saved: {args.output}")

if __name__=='__main__': main()
