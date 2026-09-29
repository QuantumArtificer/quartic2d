# Numerical diagnostic scripts

These scripts are useful during development and numerical inspection, but they
are not part of the guided public example sequence.

- `petal2d_harmonics.py`: compact PETAL2D-to-`HarmonicTransform` plotting check.
- `petal2d_full_workflow.py`: extended diagnostic report for transform sampling,
  quadrature, and convergence records.
- `quadrature_methods.py`: low-level radial-method comparison against the
  package default for a single synthetic profile. It is not an independent
  accuracy benchmark.
- `gaussian_interaction_smoke.py`: minimal interaction smoke calculation.

Publication validation and performance evidence belongs under `benchmarks/`.
