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

The canonical tolerance hierarchy has three application roles rather than a single pass/fail ladder. `1e-3` is the practical/throughput tier for scans, fitting, optimization, and other workloads that do not require tighter numerical accuracy; `1e-4` is the primary publication validation target; and `1e-5` is a stringent limit/verification tier. Publication gating therefore requires the `1e-3` and `1e-4` HarmonicTransform rows to complete; `1e-5` limitations are retained in the canonical dataset and reported rather than silently discarded. Interaction results distinguish automatic certification from bounded backend capability so a fast method can be useful in a qualified operating regime without being presented as universally reliable.

Large-displacement stage-specific and cross-stage correctness jobs use 12 logarithmically spaced displacement magnitudes across $10^2\leq\delta\leq10^4$, with the deterministic angle sweep used by the benchmark harness. This is the prevalidated oracle grid for the strict reference-stability budget. The more expensive true end-to-end large-displacement benchmark uses four representative logarithmically spaced magnitudes spanning the same interval; its role is full-stack composition validation rather than repeating the denser stage-specific sweep. Standard-domain correctness and performance timing retain 32 displacement samples.

The large-displacement cross-stage job exercises Simpson, GL4, FFTLog, and the public Ogata backend. Its correctness gate is safety-oriented: every emitted automatic certificate must pass the independent dense-field reference. Safe backend refusals are allowed, and automatically sampled HarmonicTransform inputs are additionally subjected to the production q-boundary robustness probe. Per-case coverage is reported separately so an unsupported upstream representation cannot be hidden by a backend self-convergence certificate.

A dedicated practical-tier large-displacement publication set complements the primary `1e-4` study. `interaction-large-practical-convergence` repeats the canonical four-case large-delta capability/certification matrix at `1e-3`; `interaction-large-practical-performance` measures automatic calibration and selected production timing on the same cases; and `interaction-large-qualified-fixed-performance` measures fixed production cost for every configuration already shown by the independent reference to satisfy `1e-3`, including conservative automatic refusals whose terminal parameters pass. Accuracy and timing remain separate evidence classes.

## Canonical runs

List the suite without executing it:

```bash
python -m benchmarks.run_suite --list
```

Run all manuscript-grade benchmarks with the fixed single-thread policy:

```bash
python -m benchmarks.run_suite publication
```

Run only the practical-tier large-displacement extension after committing the benchmark-plan patch:

```bash
python -m benchmarks.run_suite publication --only interaction-large-practical-convergence,interaction-large-practical-performance,interaction-large-qualified-fixed-performance
```

Subset runs write a separate `manifest_<job...>.json` and do not overwrite the canonical full-suite `manifest.json`.

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
