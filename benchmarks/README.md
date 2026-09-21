# Numerical benchmarks

`run_numerical_benchmarks.py` is the release benchmark for QUARTIC2D's numerical backends. It separates the sampled radial-transform problem from the interaction-transform problem and reports calibration, held-out validation, requested-tolerance behavior, runtime scaling, large-displacement behavior, and a PETAL2D-connected interaction workload.

The suite compares:

- sampled radial quadratures: trapezoidal, Simpson, GL4, GL8, and Ogata;
- interaction transforms: FFTLog, GL4, GL8, Simpson, trapezoidal, and Ogata;
- analytic or independently integrated references across smooth, higher-order, nodal, algebraic, oscillatory, and discontinuous cases.

Timings use warmups followed by repeated `perf_counter` measurements and report the median and interquartile range. Calibration and validation cases are disjoint.

## Quick run

```bash
python benchmarks/run_numerical_benchmarks.py \
    --quick \
    --output-dir benchmarks/results/quick
```

## Full run

```bash
python benchmarks/run_numerical_benchmarks.py \
    --output-dir benchmarks/results/full
```

The runner writes JSON/CSV data and generates publication-oriented figures with `plot_numerical_benchmarks.py`.

## Curated release result

`reference/paper_numerics_v0.1.0.json` is the complete schema-6 full benchmark used to assess the v0.1.0 numerical defaults. It was generated on an Intel Core i5-8250U with Python 3.12.9, NumPy 1.26.4, SciPy 1.15.1, hankel 1.2.2, PETAL2D 0.1.0, and one computational thread per numerical runtime library.

The PETAL2D-connected interaction comparison in that benchmark deliberately fixes the first transform stage to GL4 so that differences among interaction backends are isolated. The package default for newly constructed sampled radial transforms is Simpson; GL4 and GL8 remain available to users and for verification.

Generated files under `benchmarks/results/` are ignored. The curated JSON under `benchmarks/reference/` is tracked for GitHub/Zenodo reproducibility but excluded from the PyPI source distribution.

## Step-2 q-support and q-sampling benchmark

`run_q_sampling_benchmark.py` validates the automatic momentum-space support and
sampling used by the opt-in `HarmonicTransform.converge_parameters()` helper. The normal `HarmonicTransform(decomposition)` path is benchmarked separately to ensure the fast defaults remain accurate without running convergence.

It tests four separate claims:

- the one-pass default q-range/q-grid rule and the cheap fault detectors that
  inspect only the already-computed form factors;
- the radial sampling ceiling `q_ceiling = pi / max(diff(r))` and the
  per-harmonic Parseval-tail criterion used by explicit convergence to choose
  `q_max`;
- the theoretical q-grid scaling `n_q = 1 + ceil(s q_max R_max / pi)` followed
  by direct off-grid transform checks during explicit convergence;
- the use of not-a-knot cubic boundary conditions for q-space interpolation,
  while retaining the existing natural cubic interpolation for radial profiles.

Run it with

```bash
python benchmarks/run_q_sampling_benchmark.py \
    --output benchmarks/results/q_sampling.json
```

The benchmark also propagates the selected q support through representative
interaction kernels and compares the full automatic anisotropic workload with
the previous fixed `q_max=12`, `n_q=512` setup.

`reference/q_sampling_v0.1.0.json` is the corresponding reference result for
this Step-2 sampling study.
