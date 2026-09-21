# QUARTIC2D benchmark suite

All numerical benchmarking lives here. The suite is organized by **evidence role**, not by development chronology.

## Directory layout

- `publication/` — manuscript-grade numerical experiments. These are the only runners from which paper claims may be taken directly.
- `documentation/` — small analytic/workflow checks used to support documentation and examples. They are intentionally cheap and are not manuscript evidence.
- `development/` — screening, profiling, parameter-search diagnostics, and legacy method studies. These may motivate changes but are never cited as final evidence.
- `figures/` — plotting code and the shared publication style. Numerical runners never create paper figures implicitly.
- `reference/` — frozen, versioned historical datasets. A reference dataset is immutable once archived and is never silently substituted for a current publication run.
- `results/` — generated outputs. `results/publication/`, `results/documentation/`, and `results/development/` are ignored by Git.
- `CLAIMS.md` — claim-to-evidence registry for the methods paper.
- `FIGURES.md` — figure contract and manuscript/supplement allocation.
- `COMPLEXITY.md` — source-derived complexity model used by scaling benchmarks.
- `run_suite.py` — canonical benchmark launcher.

The benchmark suite is separate from unit tests. Unit tests protect implementation invariants; publication benchmarks establish accuracy, convergence, performance, scaling, and end-to-end claims against explicit references.

## Evidence classes

Publication evidence is divided into four classes:

1. **Reference / validation / accuracy** — comparison with analytic or independently converged references. Input-representation error is reported separately from QUARTIC2D-controlled error.
2. **Precision / convergence** — response to requested tolerance, standalone HarmonicTransform and Interaction automatic-certification reliability, difficult regimes, and conservative refusal.
3. **Performance / complexity** — repeated wall time, automatic-convergence overhead, scaling, and peak memory. Performance runners do not make accuracy claims on their own.
4. **Composition / end-to-end** — broad QUARTIC2D cross-stage tests start from qualified analytic radial harmonics, while true end-to-end tests start from sampled 2D functions and exercise `PETAL2D PolarDecomposition -> HarmonicTransform -> Interaction` against independent analytic-form-factor references.

Pointwise relative error is not the primary metric near zeros. Publication runners use global relative L2 error together with peak-normalized absolute error, and explicitly separate q-tail error where applicable.

The canonical tolerance hierarchy is `1e-4` as the primary publication target, `1e-3` as a practical looser target, and `1e-5` as a stringent limit/stress sweep. Publication gating therefore requires the `1e-3` and `1e-4` HarmonicTransform rows to complete; `1e-5` limitations are retained in the canonical dataset and reported rather than silently discarded.

## Canonical runs

List the suite without executing it:

```bash
python -m benchmarks.run_suite --list
```

Run all manuscript-grade benchmarks with the fixed single-thread policy:

```bash
python -m benchmarks.run_suite publication
```

Run documentation checks:

```bash
python -m benchmarks.run_suite documentation
```

Run development-only screens/profilers explicitly:

```bash
python -m benchmarks.run_suite development
```

`publication` is the canonical command for the final manuscript dataset. It refuses a dirty Git working tree by default and writes `results/publication/manifest.json` with the exact commit, commands, timestamps, and job status. Development outputs must never be promoted by copying or renaming them into `results/publication/`.

## Result promotion

A manuscript/Zenodo reference snapshot is created only after:

1. all publication runners complete;
2. independent reference-stability checks pass;
3. zero false-positive automatic certificates are observed;
4. both cross-stage QUARTIC2D composition and true PETAL2D -> QUARTIC2D end-to-end benchmarks pass;
5. environment metadata and the Git commit are recorded;
6. numerically backed claims are extracted from the JSON outputs and cross-checked against `CLAIMS.md`.

Only then should `results/publication/` be frozen into a versioned archive under `reference/` or the Zenodo artifact.
