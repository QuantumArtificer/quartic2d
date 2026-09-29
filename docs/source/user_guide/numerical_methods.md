# Choosing numerical methods

QUARTIC2D exposes several numerical methods because the best practical choice
depends on the represented field, radial kernel, displacement range, and
production pattern. This page is operational guidance. Detailed validation
coverage, timings, and workload definitions are kept in
{doc}`../validation/index` rather than duplicated here.

## Start from the physical/numerical problem

Before changing methods, inspect:

- the PETAL2D reconstruction and retained harmonics.
- `HarmonicTransform` diagnostics and transformed profiles.
- the required momentum support.
- the displacement range used by `Interaction`.
- whether the kernel is smooth, singular, or nonanalytic in q.
- whether the calculation is one-off or part of a large repeated sweep.

Then use the automatic convergence helper for the candidate method and inspect
its public convergence plot. A method that is fast but does not satisfy its own
configured refinement checks is not a production result.

## HarmonicTransform methods

### Simpson

Simpson quadrature is the ordinary `HarmonicTransform` default. It is a finite
rule with a simple ordered refinement parameter, `subdivisions`, and is a useful
starting point for smooth and moderately structured localized fields.

```python
hcal = HarmonicTransform.converge_parameters(
    decomposition,
    method="simpson",
    rtol=1e-4,
    q_tail_rtol=1e-3,
)
fig, axes = hcal.plot_convergence()
```

The convergence plot should be read together with the q-support and q-grid
panels. A refined radial quadrature cannot repair inadequate momentum support.

### Gauss--Legendre finite rules

`gl4` and `gl8` use composite Gauss--Legendre quadrature on the represented
radial intervals. They are useful finite-rule alternatives when higher order is
helpful for a particular radial profile.

Use the same calibration workflow rather than choosing a subdivision count by
analogy with another problem.

### Trapezoid

`trapezoid` is a deliberately simple low-order finite rule. It is useful as a
transparent baseline and for cross-checking numerical behavior, but it should
not be treated as a universal production preference.

### Ogata radial transform

The optional Ogata transform uses a Bessel-zero quadrature {cite:p}`Ogata2005` and is intended for cases where direct Bessel
quadrature benefits from its specialized representation. Its numerical controls
are coupled and should be calibrated rather than copied from an unrelated
example.

## Interaction methods

### GL4

GL4 is the ordinary `Interaction` default. It provides a systematically
refinable finite-rule calculation and is a sensible first method for new
problems.

```python
ical = Interaction.converge_parameters(
    deltas,
    field_13,
    field_42,
    U_q,
    method="gl4",
    rtol=1e-4,
)
fig, axes = ical.plot_convergence()
```

For finite rules, the plot shows successive changes and the observed asymptotic
order used by the selector. The true error is not assumed to decrease
monotonically at every refinement level.

### Simpson, GL8, and trapezoid

These finite interaction rules use the same represented transformed fields and
an ordered subdivision ladder. They are useful alternatives and cross-checks
when the target problem benefits from a different quadrature order or when a
method comparison is part of the numerical study.

Do not rank them from a single benchmark count. Calibrate the target field,
kernel, and displacement domain.

### FFTLog

FFTLog is a logarithmically sampled fast Hankel-transform method {cite:p}`Hamilton2000`. It can be attractive when many displacements are evaluated from a
representation that is well behaved on its logarithmic grid. Its automatic
search checks more than transform length: the result must also be sufficiently
stable across a local bias window.

```python
ical = Interaction.converge_parameters(
    deltas,
    field_13,
    field_42,
    U_q,
    method="fftlog",
    rtol=1e-4,
)
fig, axes = ical.plot_convergence()
```

A result that stabilizes with resolution but remains sensitive to the FFTLog
bias is intentionally not accepted by the automatic search.

### Ogata

Ogata quadrature {cite:p}`Ogata2005` is a specialist option for oscillatory interaction integrals. The
interaction-level helper calibrates the coupled $(N,h)$ representation on the
assembled observable:

```python
ical = Interaction.converge_parameters(
    deltas,
    field_13,
    field_42,
    U_q,
    method="ogata",
    rtol=1e-4,
)
fig, axes = ical.plot_convergence()
```

The useful question is not whether Ogata is globally faster or slower. It is
whether a calibrated Ogata representation is effective for the target
field/kernel/displacement family.

## Large displacement

Increasing $\delta$ increases the oscillation rate of the translation Bessel
functions even when the transition fields and kernel are unchanged. A method
that is inexpensive over a short displacement interval can require much more
work over a substantially larger one.

Calibrate on a displacement set that covers production. If large-separation
finite quadrature becomes expensive, compare a calibrated finite rule with
calibrated FFTLog or Ogata rather than changing methods on timing alone.

## Nonanalytic kernels and difficult fields

Cusps, long algebraic tails, radial oscillations, nodes, and nonanalytic kernel
features can alter both support requirements and convergence behavior. Treat a
new qualitative feature as a new numerical problem until demonstrated
otherwise.

The automatic plotters are useful here because they expose which part of the
search is limiting: support, q-grid interpolation, finite-rule refinement,
FFTLog bias stability, or coupled Ogata refinement.

## A practical method-selection sequence

For a new calculation:

1. Inspect the PETAL2D representation and retained harmonics.
2. Construct the default `HarmonicTransform` and inspect
   `field.plot_harmonics()` plus the fast diagnostics.
3. If a quantitative transform is required, calibrate it and inspect
   `hcal.plot_convergence()`.
4. Construct the default `Interaction` to establish the physical scale and
   structure of the result.
5. Calibrate the interaction over the intended displacement domain and inspect
   `ical.plot_convergence()`.
6. Consider another method when the convergence record, production pattern, or
   target numerical structure gives a concrete reason to do so.
7. For repeated families, form and validate a common parameter envelope as
   described in {ref}`family-calibration`.

The benchmark matrix in {doc}`../validation/benchmark_matrix` documents which
field/kernel/displacement combinations have actually been tested. The accuracy
and automatic-selection evidence is in {doc}`../validation/accuracy`, and the
measured runtime/scaling evidence is in {doc}`../validation/performance`.
