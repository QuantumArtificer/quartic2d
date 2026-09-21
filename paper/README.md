# Paper assets

This directory contains analysis and benchmark material used to support the
Quartic2D methods paper. It is intentionally separate from package validation,
unit tests, examples, and release-performance benchmarks.

- `benchmarks/`: paper-specific numerical experiments and figure generation.
- `benchmarks/results/`: generated benchmark data (not intended for source
  distributions).
- `benchmarks/figures/`: generated publication figures.
- `benchmarks/reference/`: frozen benchmark inputs/results used for a submitted
  or archived manuscript when those datasets have been run on the designated
  reference machine.

The top-level `benchmarks/` and `validation/` directories remain the package
engineering/validation layer. Results are promoted into `paper/benchmarks/`
only when they support a specific manuscript claim.
