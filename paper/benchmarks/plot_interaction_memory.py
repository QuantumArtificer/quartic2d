#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
try:
    from _style import save_figure
except ImportError:
    from paper.benchmarks._style import save_figure

MIB=1024.0**2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('paper/benchmarks/results/interaction_memory.json')); ap.add_argument('--output-dir',type=Path,default=Path('paper/benchmarks/results/figures')); args=ap.parse_args()
    d=json.loads(args.input.read_text()); fig,axes=plt.subplots(2,2,figsize=(7.0,5.0))
    specs=[('fftlog','N_D',r'$N_D$'),('fftlog','N_F',r'$N_F$'),('fftlog','N_p',r'$N_p$'),('simpson','N_q',r'$N_q$')]
    for ax,(branch,axis,label) in zip(axes.ravel(),specs):
        rows=[r for r in d['rows'] if r['branch']==branch and r['axis']==axis]; x=np.array([r[axis] for r in rows]); y=np.array([r['incremental_peak_rss_median_bytes']/MIB for r in rows]); q1=np.array([r['incremental_peak_rss_q25_bytes']/MIB for r in rows]); q3=np.array([r['incremental_peak_rss_q75_bytes']/MIB for r in rows]); ax.plot(x,y,marker='o'); ax.fill_between(x,q1,q3,alpha=.15); ax.set_xlabel(label); ax.set_ylabel(r'$\Delta M_{\rm RSS}$ (MiB)'); ax.text(.05,.92,'FFTLog' if branch=='fftlog' else 'Simpson',transform=ax.transAxes,va='top')
    fig.tight_layout(); args.output_dir.mkdir(parents=True,exist_ok=True); save_figure(fig,args.output_dir/'interaction_memory_scaling'); plt.close(fig)
if __name__=='__main__': main()
