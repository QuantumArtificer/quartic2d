# Benchmarking

Unit tests protect mathematical identities and public behavior. Benchmarks establish numerical accuracy, automatic-convergence reliability, performance, scaling, memory use, and end-to-end behavior on declared workload families.

## Quick analytic check

```bash
python -m benchmarks.gaussian_validation --quick
```

A passing quick run prints its analytic-versus-numerical error summary and exits with status 0. This check is suitable for local development and CI; it is not a replacement for the publication suite.

## Publication suite

List the full plan with

```bash
python -m benchmarks.run_suite publication --list
```

The output begins with

```text
Harmonic transform
------------------
  HarmonicTransform reference accuracy
  HarmonicTransform automatic convergence
  HarmonicTransform finite-quadrature matrix
```

Run the suite with

```bash
python -m benchmarks.run_suite publication
```

A successful run ends with

```text
BENCHMARK SUITE COMPLETE
Status  : PASS
Results : <repository>/benchmarks/results
```

The publication suite is intentionally expensive. Run it only when the corresponding evidence needs to be regenerated. The exact workload and kernel equations are documented in {doc}`../validation/benchmark_matrix`.

Canonical result files are generated under `benchmarks/results/` and are ignored by Git. See {doc}`../validation/publication_suite` for the result layout and provenance rules.

The ignored result directory is the numerical source of truth only while the run is present locally. For archival publication/release evidence, preserve the complete consolidated bundle externally. Tracked documentation tables and validation SVGs are regenerated from that bundle, not edited by hand:

```bash
python docs/scripts/regenerate_publication_evidence.py \
    --results-dir benchmarks/results
```

A fresh checkout can still regenerate tutorial figures without any publication JSON via `python docs/scripts/generate_figures.py examples`.

## Benchmark changes

When changing a numerical method or benchmark definition:

- keep reference calculations independent of automatic parameter selection
- state which error source is being measured
- record method capability, automatic acceptance, and independent reference accuracy separately
- preserve conservative refusals instead of converting them into successes
- separate calibration time from fixed-parameter production time
- update `benchmarks/CLAIMS.md`, `benchmarks/FIGURES.md`, or `benchmarks/COMPLEXITY.md` when their contracts change

Generate manuscript artifacts from an already completed canonical result set with

```bash
python -m benchmarks.manuscript_artifacts
```

The command writes `benchmarks/results/manuscript_artifacts/` and exits nonzero if the required canonical inputs are missing or inconsistent.
