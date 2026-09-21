# Quartic2D paper benchmarks

These benchmarks provide manuscript evidence. They are deliberately separated
from package tests, package validation, and release benchmarks.

## Scope separation

The Quartic2D paper benchmark measures Quartic2D-controlled numerical error.
For `HarmonicTransform`, the independent reference is the high-accuracy Hankel
transform of the **same sampled radial input** passed to Quartic2D. PETAL2D
sampling, finite-domain, and cutoff errors are therefore not included in the
reported Quartic2D error.

The analytic profiles are still used to qualify the synthetic input. Before a
case is admitted, its sampled radial representation must approximate the
analytic benchmark profile to a stricter external tolerance and its radial
spacing must provide a safety margin below

\[
q_{\rm Nyq}=\pi/\Delta r.
\]

The JSON records the first tested `N_r`, `r_max`, `dr`, and per-harmonic input
qualification errors that satisfy those preconditions. These are **input
requirements**, not Quartic2D errors. PETAL2D's ability/cost to produce radial
harmonics satisfying those requirements belongs in the PETAL2D supplemental
benchmarks, not in the Quartic2D performance claim.

Runtime diagnostics are disabled (`check=False`) in analytic paper timing runs
so synthetic inputs are not misidentified as PETAL2D outputs and fault-detector
cost is not mixed into transform-kernel performance. Connected PETAL2D ->
Quartic2D tests will be reported separately.

## Quartic2D error budget

For requested relative L2 tolerance `eps_req`, the harmonic-transform benchmark
uses

\[
\epsilon_{q,\mathrm{tail}} \le \epsilon_{\mathrm{req}}/\sqrt{2},
\qquad
\epsilon_{\mathrm{in}} \le \epsilon_{\mathrm{req}}/\sqrt{2}.
\]

Because the represented and omitted momentum domains are disjoint,

\[
\epsilon_{\mathrm{Q2D}}
=
\sqrt{\epsilon_{q,\mathrm{tail}}^2+\epsilon_{\mathrm{in}}^2}.
\]

The q-tail is evaluated for the actual sampled radial input using Hankel
Parseval and an independently converged Gauss-Legendre integration in q.
In-domain errors are evaluated against adaptive `quad_vec` Hankel references of
the same sampled cubic interpolants. The uniform q-grid interpolation is
converged first, then radial quadrature is refined and the combined in-domain
error is independently rechecked.

Failure/status labels distinguish:

- `input_precondition_not_met`: tested radial input is insufficient; external
  to Quartic2D accuracy;
- `q_grid_not_converged`: Quartic2D q sampling did not reach its allocated
  numerical budget;
- `quadrature_not_converged`: the selected radial quadrature candidates did not
  reach the in-domain target;
- `verification_failed`: selected parameters failed the final independent
  Quartic2D verification;
- `complete`: final Quartic2D tolerance was reached.

## Existing evidence reused from the package benchmark suite

`benchmarks/reference/paper_numerics_v0.1.0.json` already contains:

- disjoint calibration/validation for single-harmonic sampled Hankel kernels;
- radial-backend accuracy/runtime tradeoffs;
- interaction-backend accuracy/runtime tradeoffs;
- requested-tolerance behavior;
- large-displacement stability;
- interaction runtime scaling with the number of requested separations;
- a PETAL2D-connected interaction workload with first-stage numerics fixed.

`benchmarks/reference/q_sampling_v0.1.0.json` additionally contains the
q-support/q-grid theory tests used while developing `HarmonicTransform`.

These data are useful inputs to the paper, but they do **not** establish the
end-to-end runtime of the multi-harmonic `HarmonicTransform` constructor.

## Paper-specific experiments

1. **Harmonic-transform accuracy and performance** (`run_harmonic_transform.py`)

   Time the actual multi-harmonic `HarmonicTransform(...)` constructor against
   the sampled-input reference. Include smooth, nodal, anisotropic,
   mixed-parity, oscillatory, cusp-like, and algebraic cases. Discontinuous
   cases remain explicit stress tests.

2. **Step-2 asymptotic/empirical scaling** (`run_step2_scaling.py`)

   Verify the source-derived `HarmonicTransform` complexity by varying one
   independent dimension at a time: `N_q`, `N_r`, `N_m`, and radial
   subdivisions `s_r`.  A second analysis collapses all measurements against
   `W = N_m N_q N_s`, with `N_s = s_r (N_r - 1) + 1`.

   Step-3 interaction scaling is intentionally deferred until the Step-3
   benchmark is built from the validated Step-2 outputs.

3. **Step-2 scaling figures** (`plot_step2_scaling.py`)

   Build vector PDF/SVG and high-resolution PNG figures using concise standard
   mathematical labels. Figure scripts read JSON benchmark outputs; numerical
   runs and plotting remain separated for reproducibility.

## Run

From the repository root:

```bash
python paper/benchmarks/run_harmonic_transform.py \
    --output paper/benchmarks/results/harmonic_transform.json

python paper/benchmarks/run_step2_scaling.py \
    --output paper/benchmarks/results/step2_scaling.json

python paper/benchmarks/plot_step2_scaling.py \
    --input paper/benchmarks/results/step2_scaling.json \
    --output-dir paper/benchmarks/results/figures
```

Use full single-threaded runs for manuscript numbers. Quick/development runs
must never be promoted to `reference/`.

### Interaction-transform paper benchmarks

The interaction benchmark holds previously validated `HarmonicTransform`
outputs fixed and measures only the second numerical transform.  Its direct
reference integrates the same momentum-space interpolants over the same finite
support, so upstream transform and support errors are not counted as
`Interaction` error.

```bash
python3 paper/benchmarks/run_interaction_benchmarks.py \
  --output paper/benchmarks/results/interaction_benchmarks.json

python3 paper/benchmarks/run_interaction_scaling.py \
  --output paper/benchmarks/results/interaction_scaling.json

python3 paper/benchmarks/plot_interaction_benchmarks.py \
  --accuracy paper/benchmarks/results/interaction_benchmarks.json \
  --scaling paper/benchmarks/results/interaction_scaling.json \
  --output-dir paper/benchmarks/results/figures
```

Generated figures are intentionally kept below `results/` and ignored by git.
Only the final manuscript-selected figures should be copied into a permanent
paper figure directory during the final repository cleanup pass.

Peak-memory scaling is measured independently with fresh subprocesses:

```bash
python3 paper/benchmarks/run_interaction_memory.py \
  --output paper/benchmarks/results/interaction_memory.json

python3 paper/benchmarks/plot_interaction_memory.py \
  --input paper/benchmarks/results/interaction_memory.json \
  --output-dir paper/benchmarks/results/figures
```

### Automatic-convergence method screen

`run_autoconvergence_screen.py` is a fast pre-screen used before the repeated
performance benchmark.  It performs one convergence run and one production
evaluation per method, without warmups, timing repeats, or profiling.  The
HarmonicTransform screen fixes `q_max` and `n_q` to values already selected by
the completed HarmonicTransform benchmark and therefore measures only the
method-specific radial-quadrature convergence cost.  The Interaction screen
uses one representative case and a reduced displacement set.

The screen is not publication performance evidence and does not establish
accuracy.  Its purpose is to identify methods that merit the full repeated
benchmark.

```bash
python3 paper/benchmarks/run_autoconvergence_screen.py \
  --output paper/benchmarks/results/autoconvergence_screen.json
```

### Automatic-convergence performance and profiling

`run_autoconvergence_performance.py` measures the computational cost of
automatic numerical-parameter selection separately from numerical validation.
For `HarmonicTransform` and `Interaction`, it records the convergence wall time,
production time at the selected parameters, the lowest tested production-cost
floor, the highest tested resolution candidate, and convergence-search work.
Optional `cProfile` runs are executed separately from the reported timing
samples.

The highest tested resolution is not treated as a reference solution and is not
assumed to be numerically sufficient.  Numerical sufficiency belongs to the
reference/validation and precision/convergence benchmark classes.

Representative single-threaded run:

```bash
python3 paper/benchmarks/run_autoconvergence_performance.py \
  --output paper/benchmarks/results/autoconvergence_performance.json \
  --profile
```

The experimental Ogata implementation can be included when the experimental
dependency is installed:

```bash
python3 paper/benchmarks/run_autoconvergence_performance.py \
  --output paper/benchmarks/results/autoconvergence_performance_with_ogata.json \
  --profile \
  --include-ogata
```

The `standard` displacement domain spans `1e-2 <= delta <= 1e2`; the `large`
domain spans `1e2 <= delta <= 1e4`.  These domains quantify performance under
increasing oscillatory demand.  Precision/convergence as a function of
`delta` remains a separate benchmark class.
