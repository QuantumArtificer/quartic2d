# Accuracy and automatic convergence

## Reference hierarchy

The validation suite separates the numerical decision made by QUARTIC2D from the reference used to evaluate that decision.

1. `HankelTransform` is compared with analytic transforms where a closed form is available.
2. `HarmonicTransform` separates omitted-q support, interpolation on the represented q domain, and radial quadrature error using the analytic field families in {doc}`benchmark_matrix`.
3. Fixed-field `Interaction` results are compared with high-order direct-q quadrature of the same transformed fields.
4. Large-displacement references use a phase-resolved composite Gauss-Legendre calculation whose own refinement stability must pass before a production configuration is evaluated.
5. End-to-end tests begin with sampled real-space fields and compare the final result with analytic momentum-space form factors.

The automatic selectors never receive the independent interaction reference. They accept or refuse a calculation from internal refinement comparisons. The independent reference is evaluated afterward. This distinction is essential: automatic self-convergence and independent accuracy are related validation questions, not the same error metric.

## Error metrics

Reference comparisons use two global quantities. For numerical values $f_i$ and reference values $f_i^{\mathrm{ref}}$,

```{math}
:label: eq-reference-validation-l2

\epsilon_{L^2}
=\frac{\|f-f^{\mathrm{ref}}\|_2}{\|f^{\mathrm{ref}}\|_2},
```

and

```{math}
:label: eq-reference-validation-peak

\epsilon_{\mathrm{peak}}
=\frac{\|f-f^{\mathrm{ref}}\|_\infty}{\|f^{\mathrm{ref}}\|_\infty}.
```

When one scalar is needed for a plot or table, the benchmark uses

```{math}
:label: eq-validation-reference-scalar

\epsilon_{\mathrm{ref}}=\max(\epsilon_{L^2},\epsilon_{\mathrm{peak}}).
```

A residual curve of the form $|f_i-f_i^{\mathrm{ref}}|/\|f^{\mathrm{ref}}\|_\infty$ is a **pointwise peak-normalized absolute error**. Its maximum is $\epsilon_{\mathrm{peak}}$. It is not a pointwise relative-error tolerance. The requested `rtol` is likewise not a guarantee that every pointwise residual lies below that numerical value.

## HarmonicTransform validation

The nine non-stress field workloads defined in {ref}`benchmark-gaussian-family`, {ref}`benchmark-nodal-complex`, and {ref}`benchmark-difficult-fields` are evaluated at requested `HarmonicTransform.rtol` values of `1e-3`, `1e-4`, and `1e-5`. Omitted-q support and in-domain transform accuracy have separate budgets.

At the primary `HarmonicTransform.rtol = 1e-4` target, the finite-rule comparison gives:

```{include} ../_generated/validation/harmonic_primary.md
```

These measurements support the Simpson default described in {doc}`../user_guide/numerical_methods` for the tested smooth localized workloads. They do not establish a universal method ranking.

```{figure} ../_static/validation/harmonic_method_accuracy_cost.svg
:class: q2d-figure q2d-figure-compact
:alt: Harmonic-transform timing and independent-reference error for four finite quadrature methods

Finite-rule HarmonicTransform timing versus independently measured in-domain relative $L^2$ error. Each marker is one workload/method calculation, so timing and error belong to the same evaluation. The horizontal guide marks $10^{-4}$ as a reference-comparison level, not the selector's internal acceptance metric.
```

The automatic-convergence study separately tests nine non-stress workloads and two discontinuous stress profiles. Simpson and GL4 accept every non-stress workload in the declared search and those accepted results pass the analytic reference checks. The top-hat and annulus stress cases are refused at the declared settings rather than being reported as resolved smooth inputs.

## Broad 74-case Interaction matrix

The case construction is given in {ref}`benchmark-case-matrix`. At `Interaction.rtol = 1e-4`:

```{include} ../_generated/validation/interaction_primary.md
```

The FFTLog {cite:p}`Hamilton2000` row shows partial coverage in the declared parameter box; this is a coverage result, not a global grade.

```{include} ../_generated/validation/interaction_tolerance_response.md
```

## Interpreting the FFTLog coverage

At `Interaction.rtol = 1e-4`, the declared FFTLog search has the following field- and kernel-family coverage:

```{include} ../_generated/validation/fftlog_coverage.md
```

These counts motivate workload-dependent FFTLog guidance. They should not be extrapolated to untested field families, kernels, parameter boxes, or tolerances.

```{figure} ../_static/validation/fftlog_workload_coverage.svg
:class: q2d-figure q2d-figure-wide
:alt: FFTLog automatic-acceptance counts by field workload in the declared search box

FFTLog automatic-acceptance coverage by field workload at the primary requested tolerance. Each marker is annotated as accepted/tested; zero-coverage rows remain visible rather than disappearing as empty bars. The workload order follows the benchmark definition and is not sorted by acceptance fraction.
```

```{figure} ../_static/validation/fftlog_kernel_coverage.svg
:class: q2d-figure q2d-figure-wide
:alt: FFTLog automatic-acceptance counts by kernel family in the declared search box

FFTLog automatic-acceptance coverage by kernel family. Families retain a physics-based ordering instead of being sorted by acceptance rate, so the figure describes the tested boundary without presenting a ranking.
```

## Canonical automatic-convergence matrix

The four canonical cases are defined in {doc}`benchmark_matrix`. Their `Interaction.rtol = 1e-4` outcomes distinguish selector behavior from demonstrated numerical-method capability.

```{include} ../_generated/validation/canonical_outcomes.md
```

None of the automatic acceptances in these 32 method/case/domain rows fails the independent reference check.

```{figure} ../_static/validation/canonical_method_status.svg
:class: q2d-figure q2d-figure-wide
:alt: Case-level automatic-convergence outcomes for four methods in standard and large displacement domains

Case-level selector outcomes for the canonical matrix in the standard and large-displacement domains. Filled circles indicate automatic acceptance followed by an independent-reference pass. Open symbols distinguish automatic refusals for which an independently qualified point was subsequently demonstrated from unresolved refusals. Crosses mark cases for which the declared search box did not demonstrate a capable point.
```

For finite rules, validation also determines the minimum tested subdivision count that independently satisfies both {eq}`eq-reference-validation-l2` and {eq}`eq-reference-validation-peak`. This reference-qualified minimum is distinct from the subdivision count chosen by the automatic selector.

## Mixed-parity angular content

The complex workload {eq}`eq-benchmark-complex-mixed` contains $m\in\{0,1,-2\}$. It therefore exercises odd differences $m-m'$ and the signed-order translation Bessel parity. Unit tests also compare selected odd harmonic pairs with an unreduced $(q,\phi_q)$ quadrature.

This coverage matters because zero displacement and even-only harmonic sets cannot expose an incorrect odd-order translation sign.

## Reference stability at large displacement

Large-$\delta$ references are refined independently before they are used to evaluate a production method.

```{include} ../_generated/validation/reference_stability.md
```

## Cross-stage and end-to-end checks

```{include} ../_generated/validation/cross_stage.md
```

At large displacement, an upstream q-boundary refusal remains a refusal. The validation suite does not extend q support beyond the information supported by the radial sampling merely to force a downstream result.

The true end-to-end suite begins with sampled Cartesian fields and reports errors at each layer separately:

```{include} ../_generated/validation/end_to_end.md
```

This accounting keeps the sampled-field representation error, transform error, and final interaction error distinguishable. Agreement in the final scalar interaction is not used to hide a poorly resolved upstream field.

## Unit tests

Unit tests are software-regression evidence rather than publication numerical evidence. The current test command and CI expectations are maintained in {doc}`../development/testing`. Tests cover transform conventions, negative harmonic order, mixed-parity translation phases, zero-momentum and zero-displacement limits, numerical-method behavior, convergence interfaces, and public API checks.
