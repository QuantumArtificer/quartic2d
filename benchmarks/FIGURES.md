# Manuscript figures and tables

This file defines the QUARTIC2D manuscript evidence layout. The main figures follow the scientific argument

`four-center reduction -> correctness -> error control -> cost at controlled accuracy -> scaling and reuse`.

The quadrature backends implement the same reduced interaction formula. Their comparisons support accuracy, robustness, and performance claims for specified workload classes; they do not define the scientific scope of the method. Interfunction separation is denoted by `delta` (\(\delta\)) throughout the manuscript figures and tables.

The plotting contract follows conventions used in peer-reviewed computational-physics and numerical-method papers, in particular:

- Hickstein *et al.*, Rev. Sci. Instrum. **90**, 065115 (2019), DOI `10.1063/1.5092635`: direct analytic validation and runtime scaling;
- Barnett, Magland, and af Klinteberg, SIAM J. Sci. Comput. **41**, C479 (2019), DOI `10.1137/18M120885X`: achieved error versus requested tolerance and runtime versus achieved accuracy;
- Diehl and Grocholski, Eur. Phys. J. C **84**, 858 (2024), DOI `10.1140/epjc/s10052-024-13230-6`: actual versus estimated integration error and error versus numerical work;
- Reischke, JOSS **10**, 8618 (2025), DOI `10.21105/joss.08618`: direct transform/reference overlays with a residual panel and explicit timing comparisons.

`benchmarks/manuscript_artifacts.py` generates the complete figure/table set from publication JSON files. It does not rerun numerical benchmarks.

## Publication style

All manuscript figures use the same restrained computational-physics style:

- double-column width: 7.0 in;
- approximately 8.5--9 pt text at final size;
- serif text and matching math fonts;
- inward ticks on all four sides;
- logarithmic axes for errors, runtimes, and scaling variables when appropriate;
- panel labels `(a)`, `(b)`, ...;
- method identity kept consistent across figures;
- legends are used only when they improve decoding, are placed outside measured structure or in genuinely empty plot regions, and are replaced by direct labels when a sparse figure does not need a legend;
- peak-normalized absolute residuals rather than pointwise relative errors near zeros;
- error bars report measured repeat dispersion where available;
- reference and requested-target guide lines remain visually secondary to measured data and are used only when the plotted metric is directly comparable;
- no points are placed outside the data axes as categorical annotations;
- vector PDF is the manuscript source; raster review/preview exports are generated at 600 dpi.

One scalar cross-method error is

`epsilon_ref = max(epsilon_L2, epsilon_peak)`.

This scalar is used only when a single ordinate is needed. Component errors remain separate in stage-resolved tables and figures.

---

# Main-text figures

## Figure 1 - QUARTIC2D method and representative calculation

**Purpose:** define the numerical method and show one complete calculation before presenting aggregate benchmarks.

**Format:** 2 x 2 panels.

- **(a)** representative anisotropic transition field `rho(x,y)`;
- **(b)** retained angular-harmonic radial profiles `rho_m(r)`;
- **(c)** corresponding Fourier-Bessel form factors `F_m(q)`, analytic/reference and numerical curves overlaid;
- **(d)** final screened interaction `V(delta)`, QUARTIC2D and independently integrated direct-`q` reference overlaid.

The panel order carries the method sequence

`rho(x,y) -> rho_m(r) -> F_m(q) -> V(delta)`.

No separate workflow banner is placed above the panels; this keeps the four data panels at useful size and prevents panel-label/title collisions.

The frozen end-to-end anisotropic Rytova-Keldysh benchmark supplies the numerical parameters. This figure illustrates the method; quantitative claims come from Figs. 2--5 and the tables.

**Generated file:** `main/figure01_method_workflow.pdf`.

## Figure 2 - Direct numerical accuracy

**Purpose:** show that the numerical transforms reproduce known or independently integrated reference solutions on representative smooth, anisotropic, screened, and nodal problems.

**Format:** 2 x 2 benchmark blocks. Each block uses a main numerical/reference curve and a smaller residual axis below it, following the direct-comparison format used in pylevin and related transform papers.

- **(a)** isotropic Gaussian Hankel transform;
- **(b)** anisotropic `m=2` Gaussian Hankel component;
- **(c)** anisotropic strongly screened Rytova-Keldysh interaction;
- **(d)** nodal 2DEG-RPA interaction.

Residual curves use

`|numerical - reference| / max|reference|`.

This is a pointwise peak-normalized absolute error. Its maximum equals the peak-normalized reference error, while zeros of the reference do not generate divergent pointwise relative errors. No requested `rtol` line is drawn on these residual panels because `rtol` is an internal self-convergence threshold, not a pointwise bound.

**Generated file:** `main/figure02_direct_accuracy.pdf`.

## Figure 3 - Automatic error control

**Purpose:** evaluate the reliability of automatic self-convergence against independent reference calculations rather than merely show agreement at hand-selected parameters.

**Format:** four panels.

- **(a) Requested target versus independent reference error.** `epsilon_req` on the horizontal axis and independent `epsilon_ref` on the vertical axis, both logarithmic. The diagonal is `epsilon_ref = epsilon_req`. All automatically accepted runs for which an independent reference is evaluated are shown; large markers denote per-method medians. This comparison uses global aggregate errors and must not be described as a pointwise guarantee or as equality of the selector and reference metrics.
- **(b) Estimated versus independent error.** The finite-rule Richardson estimate is plotted against the independently measured relative `L2` error for automatically accepted Simpson and GL4 calculations. The diagonal denotes exact numerical agreement between these two distinct quantities.
- **(c,d) Case-level selector outcomes.** Standard- and large-displacement matrices show the four canonical cases against the four public methods. Symbols distinguish automatic acceptance followed by an independent-reference pass, reference-qualified points demonstrated after automatic refusal, unresolved automatic refusals, and tested search boxes that did not demonstrate a capable point. This replaces aggregate stacked counts so the figure exposes which workload produced each outcome.

This is the principal evidence for the reliability and limitations of automatic refinement over the tested domain.

**Generated file:** `main/figure03_automatic_error_control.pdf`.

## Figure 4 - Accuracy versus computational cost

**Purpose:** compare public numerical backends at controlled independently measured accuracy, without turning backend selection into the narrative of the paper.

**Format:** four workload-specific panels; horizontal axis is achieved `epsilon_ref`, vertical axis is fixed-parameter production time.

- **(a)** standard-domain isotropic Coulomb;
- **(b)** standard-domain nodal RPA;
- **(c)** large-`delta` isotropic Coulomb;
- **(d)** large-`delta` complex dual-gate screening.

Standard-domain panels use the requested-tolerance sweep (`1e-3`, `1e-4`, `1e-5`) and the primary method matrix. Large-`delta` (\(\delta\)) panels use independently reference-qualified fixed configurations at the practical `1e-3` target. A conservative automatic refusal may appear only when an independent reference has separately qualified the fixed parameter set; the fixed configuration is not relabeled as an automatically selected result.

No aggregate backend winner is reported. Comparisons are conditional on workload and achieved error. A method is plotted only when an independently reference-qualified production timing exists for that workload. Public Ogata is absent from the standard-domain panels because it was not included in the older canonical broad timing sweep. Trapezoid and GL8 are absent from the large-\(\delta\) panels because the practical fixed-configuration timing benchmark was defined for Simpson, GL4, FFTLog, and Ogata. FFTLog is additionally absent from the large-\(\delta\) complex dual-gate panel because no configuration in the declared tested box met the target. These absences must be stated explicitly in the caption/discussion and must not be interpreted as zero runtime or omitted successful data.

**Generated file:** `main/figure04_accuracy_vs_cost.pdf`.

## Figure 5 - Computational scaling and reusable calibration

**Purpose:** test source-derived computational work models and separate one-time calibration cost from repeated fixed-parameter production.

**Format:** 2 x 2 panels.

- **(a)** HarmonicTransform runtime versus dominant work `N_m N_q N_s`, with measured sweeps collapsed onto one work axis and a proportional guide;
- **(b)** finite-grid Interaction runtime versus `N_p N_D N_{q,s}`, with a proportional guide;
- **(c)** FFTLog runtime versus `N_p [N_F log_2(N_F) + N_D]`; the dashed line is a source-derived work guide, **not** a claim that the measured finite-range data have reached a linear asymptote;
- **(d)** effective repeated-evaluation cost

  `t_eff(M) = t_prod + t_cal/M`

  for two independently qualified workload families: smooth isotropic Coulomb (FFTLog and GL4) and complex dual-gate screening (Ogata and GL4). The measured GL4/Ogata crossover is marked when it lies in the displayed range.

**Generated file:** `main/figure05_scaling_and_reuse.pdf`.

---

# Main-text tables

## Table 1 - True end-to-end validation

For each representative standard- and large-separation case report separately:

- PETAL2D relative `L2` error;
- HarmonicTransform analytic relative `L2` error;
- final Interaction relative `L2` error;
- final Interaction peak-normalized error.

This table supports the cumulative `PETAL2D -> HarmonicTransform -> Interaction` claim without hiding which stage contributes the error.

**Generated files:** `tables/table01_end_to_end_accuracy.{tex,csv}`.

## Table 2 - Automatic Interaction convergence at the primary target

For each public method in the standard and large separation domains report:

- automatically accepted and independently reference-passing cases out of four;
- automatically accepted cases that fail the independent reference;
- demonstrated selector misses where the bounded capability test found a reference-passing configuration.

The table is a reliability/coverage summary at `1e-4`, not a method ranking.

**Generated files:** `tables/table02_automatic_convergence_outcomes.{tex,csv}`.

---

# Supplementary figures

## Figure S1 - HarmonicTransform validation by workload and requested tolerance

Three panels for `1e-3`, `1e-4`, and `1e-5`, showing independently measured global error for Simpson and GL4 over the non-stress analytic generality suite. Each workload/method result is an independent categorical scatter point; points are not connected because the workload index has no progression semantics. Incomplete stringent-limit cases remain missing rather than being assigned artificial error values.

**Generated file:** `supplement/figureS01_harmonic_validation.pdf`.

## Figure S2 - Broad Interaction accuracy distributions

Three cumulative panels, one for each requested tolerance, show the fraction of the 74-case fixed-field suite that has an independently reference-qualified result at or below a given `epsilon_ref`. The vertical dotted line is the requested numerical level. Curves that plateau below one expose incomplete qualification in the declared search without inventing values for missing cases. The requested-level guide is compared with the global `epsilon_ref` scalar, not with a pointwise residual.

**Generated file:** `supplement/figureS02_interaction_accuracy_distributions.pdf`.

## Figure S3 - Numerical-method parameter sensitivity

- FFTLog `(N, q_bias)` bounded capability maps for a standard nodal workload and a hostile large-`delta` complex workload;
- Ogata coupled `h`-`N` refinement paths for the large complex and large nodal cases.

This figure documents non-monotone specialist-parameter behavior. It does not imply that increasing FFTLog `N` or Ogata `N` alone guarantees convergence or accuracy.

**Generated file:** `supplement/figureS03_numerical_method_parameter_sensitivity.pdf`.

## Figure S4 - Cross-stage validation and safe refusal

- standard-domain cross-stage error distributions for Simpson and GL4;
- successful large-`delta` cross-stage errors by method/workload;
- the explicit `q`-boundary robustness probe.

The complex mixed-parity dual-gate case is shown as the intentional `upstream_q_boundary_not_robust` refusal. The rejected automatic `q_max`-growth strategy is not used.

**Generated file:** `supplement/figureS04_cross_stage_and_safe_refusal.pdf`.

## Figure S5 - Stage-resolved end-to-end errors

Standard and large separation panels show PETAL2D, HarmonicTransform, and final Interaction errors separately for the representative end-to-end cases.

**Generated file:** `supplement/figureS05_end_to_end_stage_errors.pdf`.

## Figure S6 - Detailed runtime scaling sweeps

Eight compact panels expose the individual independent-variable sweeps underlying the work-collapse representation in Fig. 5:

- HarmonicTransform: `N_q`, `N_r`, `N_m`, `s_r`;
- Interaction: finite `N_q`, finite `N_D`, FFTLog `N_F`, FFTLog `N_D`.

Measured medians and IQRs are shown directly.

**Generated file:** `supplement/figureS06_detailed_scaling.pdf`.

## Figure S7 - Peak-memory scaling

Fresh-process baseline-subtracted peak RSS for representative HarmonicTransform and Interaction size sweeps. This figure is independent of runtime scaling.

**Generated file:** `supplement/figureS07_memory_scaling.pdf`.

## Figure S8 - Large-displacement reference stability

- convergence of the independent phase-resolved large-`delta` reference under reference refinement;
- large-`delta` GL4 `q`-boundary changes relative to the declared boundary budget.

The failed complex dual-gate boundary probe is visible rather than converted into a successful point.

**Generated file:** `supplement/figureS08_reference_stability.pdf`.

---

# Supplementary tables

- **Table S1:** broad 74-case Interaction accuracy/timing summary at `1e-3`, `1e-4`, and `1e-5`;
- **Table S2:** primary `1e-4` public-method matrix including trapezoid, Simpson, GL4, GL8, and FFTLog;
- **Table S3:** case-resolved independently qualified large-`delta` fixed-production timing at `1e-3`, including public Ogata;
- **Table S4:** standard and large cross-stage validation summary;
- **Table S5:** finite-range runtime scaling fits, with measured exponents separated from source-derived expectations;
- **Table S6:** fresh-process peak-memory fits.

Every table is emitted as both a `booktabs` LaTeX fragment and CSV under `tables/`.

---

# Generation

From a clean repository containing the canonical publication JSONs:

```bash
python -m benchmarks.manuscript_artifacts
```

Default output:

```text
benchmarks/results/manuscript_artifacts/
  main/
  supplement/
  tables/
  CAPTIONS.md
  manifest.json
```

`manifest.json` records the benchmark JSON source files used by each artifact. The figure generator never substitutes the stale `interaction_convergence_preflight.json` file for a canonical publication result.
