# Paper figure contract

Paper figures use standard academic nomenclature and mathematical variable names. Axis labels are concise mathematical labels, never sentence-like descriptions. Benchmark-internal language such as "step 2", implementation narration, or chat-specific terminology does not appear in manuscript figures.

The numerical story is a **method portfolio**, not a winner-take-all backend comparison. Main-text figures should make the accuracy--cost--coverage tradeoff visible and should distinguish automatic certification, independently demonstrated capability, conservative refusal, and true reference failure.

## Style and export

The shared style is `benchmarks/figures/_style.py`.

- single column: 3.35 in wide;
- double column: 7.0 in wide;
- double-column tall: 7.0 x 5.0 in;
- manuscript font sizes: approximately 8.5--9 pt;
- inward ticks on all sides;
- unobtrusive major grid only;
- shared legends where panels compare the same methods;
- panel labels `(a)`, `(b)`, ...;
- vector PDF and SVG plus 600-dpi PNG;
- methods use consistent markers across all figures;
- requested tolerances use consistent line styles/marker fills across all figures;
- Ogata is shown as a public specialist backend, with the same method identity used consistently across figures;
- no decorative titles when axis labels and caption carry the meaning.

Unless a panel explicitly studies a signed quantity, errors and runtimes use logarithmic axes. Error panels use the common scalar

`epsilon_ref = max(epsilon_L2, epsilon_peak)`

when one ordinate is required for cross-method comparison. Stage-resolved panels may additionally show the two components separately. Pointwise relative error is not used near zeros.

Figures state broad conclusions; exact benchmark parameters and full numerical tables belong in captions, tables, or the supplement.

## Main-text figure set

The final figures are generated only after the publication-grade datasets required by `CLAIMS.md` have been completed.

### Figure 1 — Accuracy--cost landscape

**Format:** double column, three panels.

**Panel (a): production cost versus achieved reference error**

- x axis: `t_prod` [ms], logarithmic;
- y axis: `epsilon_ref`, logarithmic;
- color/marker identity: method;
- marker fill or line style: requested tolerance (`1e-3`, `1e-4`, `1e-5`);
- horizontal guides: `1e-3`, `1e-4`, `1e-5`;
- standard-domain broad matrix is the primary dataset;
- only reference-evaluated points are plotted as accuracy points;
- automatic refusals are not silently plotted as successful calculations.

The purpose is to expose Pareto structure: some methods trade broad certification for low production cost, while others retain broad accuracy at tighter targets.

**Panel (b): coverage/capability versus requested tolerance**

- x axis: `epsilon_req`, logarithmic and ordered `1e-3 -> 1e-5`;
- y axis: fraction of tested cases, linear `[0, 1]`;
- for each method distinguish stacked or adjacent fractions for:
  1. automatically certified + reference pass;
  2. backend-capable/reference-pass inside the declared search box but not automatically certified;
  3. tested search box not reference-capable at the requested tolerance.

Do not merge categories 2 and 3 into a single "failure" fraction.

**Panel (c): tolerance response**

- x axis: `epsilon_req`, logarithmic;
- y axis: achieved `epsilon_ref`, logarithmic;
- show median and distribution (IQR/whisker or compact violin/box representation) by method;
- include a `y=x` guide;
- stage-isolated Interaction and HarmonicTransform results remain distinct.

### Figure 2 — Method operating-regime map

**Format:** double column, matrix/heatmap plus compact legend.

Rows are quadrature backends:

`trapezoid`, `simpson`, `gl4`, `gl8`, `fftlog`, and public `ogata` where available.

Columns represent physically/numerically distinct workload regimes rather than individual case names, for example:

- standard / smooth isotropic;
- standard / anisotropic;
- standard / nodal or cusp;
- large-delta / smooth;
- large-delta / anisotropic;
- large-delta / complex;
- large-delta / nodal/cusp.

Each cell reports one of:

- achieved `log10(epsilon_ref)` for a declared tolerance, or
- a categorical state: certified, capable-but-selector-refused, tested-box-not-capable, upstream unsupported, not tested.

If both `1e-3` and `1e-4` are shown, use separate subpanels rather than encoding two tolerances in one cell. The figure must make clear that an operating regime is conditional on requested accuracy.

### Figure 3 — Automatic-certification reliability

**Format:** double column, two panels.

**Panel (a): estimated versus actual error**

- x axis: automatic convergence/error estimate when defined, logarithmic;
- y axis: independent `epsilon_ref`, logarithmic;
- diagonal `y=x` guide;
- requested-tolerance boundaries;
- method marker identity;
- dangerous false-positive quadrant visually identifiable;
- certified and refused cases distinguished.

If a common estimator cannot be placed on one axis for every backend, use backend-specific subpanels with identical axis limits rather than inventing a synthetic estimator.

**Panel (b): classification counts**

Show, separately:

- correct automatic certificates;
- false-positive certificates;
- conservative finite-rule misses;
- FFTLog selector misses with demonstrated bounded capability;
- specialist-method conservative refusals;
- true tested-box incapability.

This panel exists to demonstrate reliability policy, not to rank methods.

### Figure 4 — Repeated-evaluation throughput and amortization

**Format:** double column, two panels.

For repeated evaluations of a prequalified workload family define

`t_eff(M) = t_prod + t_cal / M`.

**Panel (a): standard-domain practical tier**

- x axis: number of repeated evaluations `M`, logarithmic;
- y axis: effective time per evaluation `t_eff` [ms], logarithmic;
- methods: at minimum Simpson, GL4, FFTLog where the corresponding accuracy benchmark demonstrates capability at the stated tolerance;
- target accuracy stated explicitly in the panel/caption (normally `1e-3`).

**Panel (b): large-delta practical tier**

- same axes and construction;
- use the dedicated `1e-3`, `1e2 <= delta <= 1e4` convergence and performance datasets;
- include only method/workload combinations that independently satisfy the target;
- if a method is only useful for a subset of workload families, show the subset explicitly rather than averaging it together with failed families.

Calibration and production measurements must come from the same benchmark configuration and machine environment. For conservative automatic refusals, fixed-parameter production timing comes from `interaction-large-qualified-fixed-performance`: finite rules use the minimum tested oracle-proven passing subdivision count, while other backends use a terminal configuration only when that terminal point independently passes the reference. Those points must be labeled as independently qualified rather than automatically certified. The plot may illustrate amortization; it must not imply that new workload families can skip qualification.

### Figure 5 — Computational scaling

**Format:** double-column multi-panel figure.

HarmonicTransform panels use the independent variables justified in `COMPLEXITY.md`, including `N_q`, `N_r`, `N_m`, and radial quadrature work. Interaction panels include the finite-grid and FFTLog dimensions such as transform/grid size, requested separations, harmonic-pair count, and subdivision work.

- x and y axes: logarithmic;
- points: measured medians;
- uncertainty: IQR or equivalent repeat dispersion;
- fitted finite-range slope: solid line with reported fit interval;
- source-derived asymptotic guide: dashed reference where justified;
- measured slopes are not described as asymptotic laws when the fitted range is still overhead/vectorization dominated.

## Supplementary figures and tables

The supplement contains:

- full workload/kernel validation tables at each requested tolerance;
- full method-selection table with certified fraction, bounded capability fraction, worst/median error, production time, and calibration time;
- FFTLog `(N, q_bias)` sensitivity maps for at least one favorable and one hostile workload, with `1e-3` and `1e-4` contours where available;
- Ogata `(N, h)` parameter sensitivity / convergence maps, including its high-cost nodal/cusp limit;
- difficult/stress inputs and explicit refusals;
- peak-memory scaling;
- detailed large-delta oracle stability;
- PETAL2D input-resolution/performance benchmarks needed to meet the QUARTIC2D input requirements;
- stage-resolved PETAL2D, HarmonicTransform, and final Interaction errors from the true end-to-end pipeline;
- any development sensitivity study required to justify a frozen search box, but not development timing as manuscript performance evidence.

## Main-text numerical tables

### Table 1 — Method-selection guide

A compact application-facing table with columns such as:

- use case / workload character;
- typical requested accuracy;
- suitable backend(s);
- main advantage;
- principal limitation or fallback.

The entries must be populated from measured benchmark regimes, not from generic expectations about the algorithms.

### Table 2 — Quantitative backend summary

For each method and requested tolerance report, where applicable:

- number/fraction automatically certified;
- number/fraction reference-capable inside the tested search box;
- worst and median `epsilon_ref` among the stated population;
- median `t_prod` and `t_cal`;
- displacement regime and population size.

Never quote one aggregate timing across cases that include reference failures without identifying the population used for the timing statistic.
