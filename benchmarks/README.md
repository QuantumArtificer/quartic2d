# QUARTIC2D benchmark suite

The benchmark directory contains the reproducible numerical evidence used by the
methods paper and release validation. Runners are intentionally kept in one flat
namespace so the repository does not mirror development history in its folder tree.

## Layout

- `run_suite.py` — canonical launcher for manuscript-grade and quick validation jobs.
- `harmonic_*.py`, `interaction_*.py`, `cross_stage.py`, `end_to_end.py` — publication benchmark runners.
- `autoconvergence_performance.py`, `fixed_configuration_performance.py` — calibration and fixed-production timing.
- `gaussian_validation.py` — inexpensive analytic workflow validation used by CI/docs.
- `manuscript_artifacts.py` — generates the manuscript/supplement figures and tables from canonical JSON results.
- `_common.py`, `_interaction_suite.py` — shared benchmark infrastructure.
- `CLAIMS.md` — numerical claim-to-evidence registry.
- `FIGURES.md` — manuscript figure/table contract.
- `COMPLEXITY.md` — source-derived complexity model.
- `results/` — generated JSON and manuscript artifacts; ignored by Git.

Exploratory screens, profiling scripts, legacy plotting programs, and historical
one-off regression matrices are intentionally excluded from the release source tree.
They belong in development history or archived release artifacts, not in the package
repository.

The benchmark suite is distinct from unit tests. Unit tests protect stable public
and numerical behavior; benchmarks establish accuracy, convergence, performance,
scaling, memory, and end-to-end claims against explicit references.

## Evidence policy

Publication results use global relative L2 error together with peak-normalized
absolute error. Pointwise relative error near zeros is not used as a primary metric.
Input-representation error is kept separate from QUARTIC2D-controlled numerical
error.

The tolerance hierarchy is:

- `1e-3`: practical/throughput tier;
- `1e-4`: primary publication target;
- `1e-5`: stringent limit/verification tier.

Large-displacement validation uses the predeclared range
$10^2\leq\delta\leq10^4$. Cross-stage results distinguish backend refusal from the
upstream `upstream_q_boundary_not_robust` safeguard; an unsupported upstream field
is never converted into a successful Interaction certificate.

## Canonical commands

List the manuscript-grade suite in execution order:

```bash
python -m benchmarks.run_suite publication --list
```

Run the full publication suite with the fixed single-thread policy:

```bash
python -m benchmarks.run_suite publication
```

A full publication run begins by clearing the generated `benchmarks/results/`
tree. Native runner JSON is written only to a temporary staging directory while
the suite is running. After every benchmark passes, the staging data are promoted
losslessly into the canonical result set:

- `manifest.json` -- repository state, execution policy, grouped benchmark status,
  elapsed times, and structured runner arguments;
- `harmonic.json` -- HarmonicTransform accuracy, automatic convergence, and the
  finite-quadrature method matrix;
- `interaction_accuracy.json` -- Interaction reference accuracy, production timing, and the primary-tolerance public-method matrix;
- `interaction_convergence.json` -- automatic-certification reliability and bounded capability at the primary and practical large-`delta` targets;
- `pipeline.json` -- standard/large-`delta` cross-stage and true end-to-end
  validation;
- `performance.json` -- automatic-calibration cost and independently qualified
  fixed-configuration production timing;
- `scaling.json` -- HarmonicTransform and Interaction runtime scaling;
- `memory.json` -- fresh-process peak-memory scaling.

The numerical objects emitted by each runner are embedded unchanged under named
`sections` in these files. Consolidation does not round, rename, or recompute any
benchmark value. This makes the JSON easier to inspect while preserving exact
figure/table inputs and provenance.

Subset runs never overwrite the canonical result set. They are written under
`benchmarks/results/subsets/<selection>/`.

Run the inexpensive analytic workflow check separately with:

```bash
python -m benchmarks.run_suite documentation
```

Generate manuscript figures and tables from the consolidated publication JSON:

```bash
python -m benchmarks.manuscript_artifacts
```

The figure generator reads the named sections in the six evidence files above.
Its numerical inputs are identical to the native runner objects used before result
consolidation.

## Result promotion

A manuscript/Zenodo snapshot is created only after all required publication runners
complete, independent reference-stability checks pass, no false-positive automatic
certificates are observed, cross-stage and end-to-end validation pass, and the exact
Git/environment metadata is recorded. Archived datasets should be stored with the
release/Zenodo artifact rather than accumulated in the source tree.
