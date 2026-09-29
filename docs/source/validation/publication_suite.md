# Publication benchmark suite

The canonical publication suite generates the numerical evidence used by the validation pages and manuscript figures. It is separate from the unit tests because several jobs are intentionally expensive.

## List the suite

```bash
python -m benchmarks.run_suite publication --list
```

The command prints the grouped benchmark plan. The publication suite contains separate jobs for analytic/reference accuracy, automatic self-convergence, finite-rule comparisons, Interaction accuracy and method coverage, large-displacement validation, performance and reuse, cross-stage and end-to-end checks, runtime scaling, and peak-memory scaling. Keeping these jobs separate prevents a timing result or an internal convergence result from being used as a substitute for independent accuracy evidence.

## Run from an exact source state

Check the repository first:

```bash
git status --short
git rev-parse HEAD
```

For a clean tree, the first command prints nothing. The second prints the commit SHA, for example

```text
6bcdac3ee02bc8687d341c05049a3d024a2bf4b5
```

Run the suite with

```bash
python -m benchmarks.run_suite publication
```

A successful run ends with

```text
====================================================================================================
BENCHMARK SUITE COMPLETE
====================================================================================================
Status  : PASS
Results : <repository>/benchmarks/results
Files   :
  - harmonic.json
  - interaction_accuracy.json
  - interaction_convergence.json
  - pipeline.json
  - performance.json
  - scaling.json
  - memory.json
```

The exact output order follows the consolidator.

If terminal output is being recorded, write the log outside the repository or to an ignored path:

```bash
python -m benchmarks.run_suite publication 2>&1 | \
    tee ~/Downloads/quartic2d_publication_run.log
```

The terminal still ends with the same `BENCHMARK SUITE COMPLETE` summary, while the log is written to `~/Downloads`. Do not stream benchmark output into a tracked repository file because provenance is sampled during execution.

## Canonical result files

A full run promotes the staged native outputs to

```text
benchmarks/results/
  manifest.json
  harmonic.json
  interaction_accuracy.json
  interaction_convergence.json
  pipeline.json
  performance.json
  scaling.json
  memory.json
```

The manifest records the git commit, branch, source state, execution policy, runner arguments, elapsed times, and job status. Numerical result objects are embedded under named sections in the consolidated JSON without rounding or recomputation.

Generated benchmark result files are ignored by Git. Archive publication-grade data with the paper or software release; generated result snapshots do not belong in source control.

## Subset runs

Subset runs are useful while changing one numerical component. Available job names are shown with `--list`. For example,

```bash
python -m benchmarks.run_suite publication \
    --only interaction-convergence
```

A successful subset run ends with the same PASS footer and stores its native data below

```text
benchmarks/results/subsets/
```

Subset runs do not overwrite the canonical full-suite result collection.

## Lightweight validation

The inexpensive analytic check used during documentation work is

```bash
python -m benchmarks.gaussian_validation --quick
```

It prints a short analytic-versus-numerical Gaussian validation summary and exits nonzero if the declared quick thresholds are violated. It is not a substitute for the publication suite.

## Documentation evidence snapshot

The repository does not track `benchmarks/results/`. Publication-grade JSON should be restored from the paper/release archive when the evidence needs to be regenerated. Keep the entire consolidated bundle together; do not mix files from different runs.

Refresh validation figures and generated numerical prose/tables from one explicit bundle with

```bash
python docs/scripts/regenerate_publication_evidence.py \
    --results-dir /path/to/canonical-results
```

The command regenerates only the publication-dependent documentation artifacts and then verifies `docs/artifacts.json`. Generated Markdown fragments and validation SVGs carry the same publication-bundle fingerprint, so accidental mixing of evidence from different runs is detected. The command does not rerun the numerical benchmarks.

Tutorial/example figures are independent of publication JSON and use a separate path:

```bash
python docs/scripts/generate_figures.py examples
```

## Manuscript artifacts

Generate manuscript figures and tables with

```bash
python -m benchmarks.manuscript_artifacts
```

The command reads the canonical benchmark results and writes the manuscript artifact directory. It does not rerun the numerical benchmarks.
