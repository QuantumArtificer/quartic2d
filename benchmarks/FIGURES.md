# Paper figure contract

Paper figures use standard academic nomenclature and mathematical variable names. Axis labels are concise mathematical labels, never sentence-like descriptions. Benchmark-internal language such as “step 2”, implementation narration, or chat-specific terminology does not appear in manuscript figures.

## Style

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
- method comparisons use consistent markers across figures;
- no decorative titles when the axis labels and caption carry the meaning.

Figures state broad conclusions; exact benchmark parameters and full numerical tables belong in captions, tables, or the supplement.

## Main-text figure set

The final figures are generated only after the canonical publication suite has been rerun and the claim registry has been populated.

### Figure 1 — Accuracy and tolerance response

Two or four compact panels combining:

- requested tolerance `epsilon_req` versus achieved `epsilon_Q2D`;
- q-tail and in-domain contributions where informative;
- production runtime versus `epsilon_req`;
- representative true end-to-end error from `PETAL2D -> HarmonicTransform -> Interaction`.

This figure must keep stage-isolated QUARTIC2D-controlled transform error separate from PETAL2D qualification error, while any end-to-end panel must be explicitly labeled as cumulative pipeline error.

### Figure 2 — Automatic convergence reliability and cost

Panels report:

- certified/reference-pass fractions by method and displacement regime;
- false-positive count explicitly;
- calibration time `t_conv` versus selected production time `t_prod`;
- conservative refusal/selector-miss counts separately from wrong certificates.

Near-zero observables use global relative L2 and peak-normalized absolute metrics rather than pointwise relative error.

### Figure 3 — Backend accuracy/runtime tradeoff

Accuracy-versus-runtime or Pareto presentation for applicable methods, with standard and large-delta regimes distinguished. Simpson, GL4, and FFTLog are always shown where applicable; private Ogata may appear in the supplement or as a clearly identified benchmark-only comparison.

### Figure 4 — Computational scaling

Double-column multi-panel figure with standard mathematical labels for the independent dimensions, including the HarmonicTransform work collapse and Interaction scaling. Use measured medians/IQR and source-derived asymptotic guides where justified.

## Supplement

The supplement contains:

- full workload/kernel validation tables;
- difficult/stress inputs and explicit refusals;
- memory scaling;
- detailed large-delta oracle stability;
- specialist-method parameter studies, including Ogata;
- PETAL2D input-resolution/performance benchmarks needed to meet the QUARTIC2D input requirements;
- stage-resolved PETAL2D, HarmonicTransform, and final Interaction errors from the true end-to-end pipeline;
- any development sensitivity study required to justify a frozen search box, but not development timing as manuscript performance evidence.
