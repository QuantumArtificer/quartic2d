# QUARTIC2D benchmark suite

The benchmark directory contains the reproducible numerical evidence used to validate the QUARTIC2D four-center method, its public implementation, and release-level numerical claims. Runners are kept in one flat namespace so the benchmark surface remains easy to inspect.

## Layout

- `run_suite.py` — canonical launcher for publication-grade and quick validation jobs.
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
maximum absolute error. Pointwise relative error near zeros is not used as a primary
metric. A plotted pointwise peak-normalized residual is a diagnostic curve; it is not
the quantity controlled point by point by `rtol`. Input-representation error is kept
separate from QUARTIC2D-controlled numerical error.

The publication suite uses calculation-specific numerical targets:

- `HarmonicTransform.rtol = 1e-3`, `1e-4`, and `1e-5` are used for the transform tolerance-response study; `HarmonicTransform.rtol = 1e-4` is the primary transform target.
- `HarmonicTransform.q_tail_rtol = 1e-3` controls the separate omitted-q-support budget in the automatic transform calibration.
- `Interaction.rtol = 1e-3`, `1e-4`, and `1e-5` are used for the interaction tolerance-response study; `Interaction.rtol = 1e-4` is the primary standard-domain target.
- `Interaction.rtol = 1e-3` is the practical large-displacement throughput tier.
- `HarmonicTransform.atol = 1e-12` and `Interaction.atol = 1e-12` are the absolute floors used by the automatic selectors.

The parameter name is part of every numerical claim. The same numerical value attached to `HarmonicTransform.rtol`, `HarmonicTransform.q_tail_rtol`, or `Interaction.rtol` controls a different source of error.

The requested `rtol` values are internal self-convergence criteria. They are not definitions of the independent reference errors and they do not imply a pointwise relative-error guarantee. When one scalar reference error is needed for a figure or table, the suite uses `max(relative_L2, peak_normalized_max)`. The two components remain available separately in the result files.

Large-displacement validation uses the predeclared range
$10^2\leq\delta\leq10^4$, where $\delta=|\boldsymbol\Delta|$ is measured in the same length unit as the input coordinates. Cross-stage results distinguish backend refusal from the
upstream `upstream_q_boundary_not_robust` safeguard; an unsupported upstream field
is never converted into an accepted downstream Interaction result.

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
- `interaction_convergence.json` -- automatic-convergence reliability and bounded capability at the primary and practical large-`delta` targets;
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

Regenerate the tracked documentation evidence snapshot from the same consolidated publication JSON:

```bash
python docs/scripts/regenerate_publication_evidence.py --results-dir benchmarks/results
```

This writes validation figures plus generated Markdown fragments and checks that every managed artifact is declared in `docs/artifacts.json`. The source tree does not track the JSON bundle itself; archive the complete bundle with the paper/release.

Generate manuscript figures and tables from the consolidated publication JSON:

```bash
python -m benchmarks.manuscript_artifacts
```

The figure generator reads the named sections in the seven evidence files above.
Its numerical inputs are identical to the native runner objects used before result
consolidation.

## Result promotion

A manuscript/Zenodo snapshot is created only after all required publication runners
complete, independent reference-stability checks pass, no automatically accepted result fails its independent reference check, cross-stage and end-to-end validation pass, and the exact
Git/environment metadata is recorded. Archived datasets should be stored with the
release/Zenodo artifact rather than accumulated in the source tree.
