#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE=()
if [[ "${1:-}" == "--quick" ]]; then
  MODE=(--quick)
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--quick]" >&2
  exit 2
fi

python tools/clean_generated.py
mkdir -p validation/results benchmarks/results/paper_numerics

python validation/run_gaussian_validation.py \
  "${MODE[@]}" \
  --output validation/results/gaussian.json

python benchmarks/run_numerical_benchmarks.py \
  "${MODE[@]}" \
  --output-dir benchmarks/results/paper_numerics
