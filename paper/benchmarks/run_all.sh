#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

python paper/benchmarks/run_harmonic_transform.py \
  --output paper/benchmarks/results/harmonic_transform.json

python paper/benchmarks/run_step2_scaling.py \
  --output paper/benchmarks/results/step2_scaling.json

python paper/benchmarks/run_harmonic_memory.py \
  --output paper/benchmarks/results/harmonic_memory.json

python paper/benchmarks/run_interaction_benchmarks.py \
  --output paper/benchmarks/results/interaction_benchmarks.json

python paper/benchmarks/run_interaction_scaling.py \
  --output paper/benchmarks/results/interaction_scaling.json

python paper/benchmarks/run_interaction_memory.py \
  --output paper/benchmarks/results/interaction_memory.json

python paper/benchmarks/plot_step2_scaling.py \
  --input paper/benchmarks/results/step2_scaling.json \
  --output-dir paper/benchmarks/results/figures

python paper/benchmarks/plot_harmonic_memory.py \
  --input paper/benchmarks/results/harmonic_memory.json \
  --output-dir paper/benchmarks/results/figures

python paper/benchmarks/plot_interaction_benchmarks.py \
  --accuracy paper/benchmarks/results/interaction_benchmarks.json \
  --scaling paper/benchmarks/results/interaction_scaling.json \
  --output-dir paper/benchmarks/results/figures

python paper/benchmarks/plot_interaction_memory.py \
  --input paper/benchmarks/results/interaction_memory.json \
  --output-dir paper/benchmarks/results/figures
