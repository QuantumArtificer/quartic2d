# Convergence and diagnostics

QUARTIC2D separates inexpensive exploratory diagnostics from explicit numerical
refinement. The ordinary constructors help inspect a new model quickly.
`converge_parameters(...)` is the supported path when a calculation needs a
stated self-convergence criterion and a recorded parameter search.

The two main convergence workflows are deliberately parallel:

```python
hcal = HarmonicTransform.converge_parameters(
    decomposition, rtol=1e-4, q_tail_rtol=1e-3
)
fig, axes = hcal.plot_convergence()
field = hcal.transform(decomposition)

ical = Interaction.converge_parameters(
    deltas, field_13, field_42, U_q, rtol=1e-4
)
fig, axes = ical.plot_convergence()
interaction = ical.interaction(deltas, field_13, field_42, U_q)
```

The returned result objects preserve both the selected production parameters and
the numerical history used to select them.

(convergence-tolerance-table)=
## Tolerance vocabulary

The tolerances act on different numerical objects and should be named explicitly
when a result is reported.

| Parameter | Numerical object | Meaning |
| --- | --- | --- |
| `recon_err_tol` | PETAL2D angular representation | target reconstruction error of the represented real-space field, expressed in percent |
| `HarmonicTransform.rtol` | transform refinement | relative $L^2$ change required between successive represented transforms |
| `HarmonicTransform.atol` | transform refinement | absolute floor in the peak-scaled maximum-change condition |
| `HarmonicTransform.q_tail_rtol` | momentum support | relative $L^2$ budget for the estimated q-space norm omitted beyond the selected `q_max` |
| `Interaction.rtol` | interaction refinement | relative $L^2$ change required between successive assembled interaction evaluations |
| `Interaction.atol` | interaction refinement | absolute floor in the peak-scaled maximum-change condition |
| independent reference threshold | validation only | external comparison criterion used when a separately refined or analytic reference exists |

For ordered refinement, successive numerical results $f^{(n-1)}$ and $f^{(n)}$
are accepted only when

$$
\frac{\|f^{(n)}-f^{(n-1)}\|_2}
     {\|f^{(n-1)}\|_2}
\le \mathrm{rtol}
$$

and

$$
\|f^{(n)}-f^{(n-1)}\|_\infty
\le \mathrm{atol}
+\mathrm{rtol}\,\|f^{(n-1)}\|_\infty.
$$

The code also records relative $L^\infty$ change as a diagnostic, but it is not
an additional acceptance condition. Finite rules can add known-order and
Richardson extrapolation/order checks {cite:p}`Richardson1911`. None of these self-convergence quantities is a pointwise
error guarantee against an unknown exact solution.

## Fast diagnostics for exploratory work

The ordinary `HarmonicTransform` constructor runs inexpensive checks on the
representation it has just built:

```python
field = HarmonicTransform(decomposition)
print("diagnostics healthy:", field.diagnostics.healthy)
```

If the summary is unhealthy, inspect the details:

```python
print(field.diagnostics.warning_message())
for issue in field.diagnostics.issues:
    print(issue)
```

For machine-readable provenance,

```python
diagnostics = field.diagnostics.to_dict()
```

`diagnostics.healthy=True` means that no fast support or sampling fault was
detected. It does not demonstrate a particular `rtol` or an independent
reference error.

## HarmonicTransform calibration

The harmonic selector answers three separate questions:

1. how far in q the represented transform must extend;
2. how densely the q-space form factors must be tabulated;
3. how finely the sampled radial Hankel integral must be evaluated.

A converged radial quadrature does not compensate for insufficient q support,
and a large `q_max` does not compensate for an under-resolved q grid.

```python
hcal = HarmonicTransform.converge_parameters(
    decomposition,
    rtol=1e-4,
    atol=1e-12,
    q_tail_rtol=1e-3,
    method="simpson",
)
```

### Inspect the search visually

```python
fig, axes = hcal.plot_convergence()
```

The plot shows:

- required q support for each retained harmonic and the selected `q_max`.
- q-grid interpolation refinement.
- radial quadrature refinement.

The q-support panel uses `q_tail_rtol`. The refinement panels show internal
changes controlled by `rtol`; final acceptance also includes the peak-scaled
absolute maximum-change condition controlled by `atol`.

### Inspect the search programmatically

```python
print(hcal.converged)
print(hcal.parameters)
print(hcal.sampling)
print(hcal.quadrature)
```

Build the production field without repeating calibration:

```python
field = hcal.transform(decomposition)
```

The production object keeps the structured records:

```python
sampling = field.sampling_convergence
quadrature = field.quadrature_convergence
fig, axes = field.plot_convergence()
```

For persistent provenance,

```python
record = hcal.to_dict()
```

is JSON-serializable.

## Interaction calibration

`Interaction.converge_parameters(...)` refines the assembled matrix element over
the supplied displacement vectors:

```python
ical = Interaction.converge_parameters(
    deltas,
    field_13,
    field_42,
    U_q,
    rtol=1e-4,
    atol=1e-12,
    method="gl4",
)
```

The displacement set is part of the numerical problem. A calibration restricted
to one range of separations does not automatically justify a later production
calculation over a much larger range.

### Inspect the method-specific search

```python
fig, axes = ical.plot_convergence()
```

The plot adapts to the selected method:

- finite rules show the $L^2$ error over ordered refinements.
- FFTLog shows transform-length refinement and bias robustness.
- Ogata shows the coupled $(N,h)$ search.

This is the interaction analogue of `HarmonicConvergenceResult.plot_convergence()`.
It visualizes the internal search rather than an independent reference error.

Construct production directly from the result:

```python
interaction = ical.interaction(
    deltas,
    field_13,
    field_42,
    U_q,
)
```

The resulting object retains the calibration:

```python
print(interaction.convergence is ical)
fig, axes = interaction.plot_convergence()
```

```text
True
```

The search record can be stored compactly:

```python
record = ical.to_dict()
```

Use `include_values=True` only when the interaction arrays from every refinement
step are needed in addition to the search metadata.

## Forward/inverse consistency

A calibrated or exploratory `HarmonicTransform` can report a numerical
round-trip consistency error:

```python
error = field.roundtrip_error()
```

This compares represented radial harmonics with a forward/inverse numerical
Hankel cycle. It is useful for detecting inconsistencies in the represented
transform, but it is not an external accuracy estimate because both directions
share the same input representation and numerical conventions.

## What should be recalibrated

Recalibrate the numerical objects that actually changed.

| Change in the calculation | Recalibrate | Reason |
| --- | --- | --- |
| identical transformed fields, kernel, and displacement set | nothing | represented numerical problem is unchanged |
| same transition field reused in several four-center channels | reuse its `HarmonicTransform` | the transform depends on the field, not the many-body label |
| tighter or longer-tailed transition field | `HarmonicTransform` | q support and q-grid density can change |
| different retained angular content | transform and interaction | harmonic pairs and translation Bessel orders change |
| changed kernel parameters within a smooth family | normally `Interaction` | q weighting changes while the field transform can often be reused |
| new singularity or nonanalytic structure in the kernel | `Interaction` and possibly method choice | numerical regularity changes qualitatively |
| larger maximum displacement | `Interaction` | translation Bessel oscillations become faster |
| changed PETAL2D radial grid or support | transform and interaction | the represented input itself changed |

(family-calibration)=
## Calibrate families, not every point

Repeated calculations should separate **calibration** from **production**.
Calibrate a small envelope of representative problems and reuse a common
parameter set only within the numerical family that envelope actually covers.

### Define the field family

Group transition fields according to features that control their transforms:

- radial extent and tail behavior.
- smooth, cusped, oscillatory, or algebraic radial structure.
- nodal structure.
- retained angular harmonics.
- real versus complex mixed-parity content.
- PETAL2D radial spacing and represented support.

A smooth localized Gaussian and a long algebraic tail should not automatically
share one calibration. A set of localized orbitals with comparable extent and
the same angular structure may be a reasonable candidate family, but the
assumption should be checked.

### Define the kernel family

Group kernels according to q-space structure rather than physical naming alone.
A smooth sweep of one screening parameter can often share a calibration
envelope. Introducing a cusp, pole, or nonanalytic point changes the numerical
problem and can require a separate family.

### Include the production displacement envelope

The calibration displacement set should cover the range and angular geometry
that production will use. Large separation increases the oscillation rate of
the translation Bessel functions even when the local fields and kernel are
unchanged.

### Form and check the production envelope

For a finite interaction rule, a conservative family setting can be formed from
the largest qualified subdivision requirement among representative members.
For `HarmonicTransform`, retain q support and q-grid resolution sufficient for
the most demanding qualified field.

Then run production with fixed settings and spot-check representative interior
and boundary points with `converge_parameters(...)`. If the checks require
stricter settings, enlarge the envelope. If one member repeatedly behaves
differently, split the family instead of over-resolving every calculation.

## Upstream q-boundary robustness

A downstream interaction quadrature can self-converge while remaining sensitive
to the finite q support of an automatically sampled transformed field. When the
necessary metadata are available, `Interaction.converge_parameters(...)` probes
that sensitivity as part of the search.

A failed q-boundary robustness check means the represented transformed field
needs attention. Do not repair it by extending `q_max` beyond the radial
sampling ceiling

$$
q_{\mathrm{Nyquist}}=\frac{\pi}{\Delta r}.
$$

Instead refine or enlarge the upstream real-space representation as needed.

## Reading an unsuccessful search

An unsuccessful automatic search is useful numerical information. Inspect the
result object and its convergence plot before changing parameters blindly.
Common causes include:

- the declared search ladder ending before the refinement criteria are met.
- insufficient upstream q support.
- FFTLog resolution that stabilizes while the local bias window remains
  sensitive;
- an Ogata $(N,h)$ search that does not stabilize over the tested range.
- large-displacement oscillations requiring a different or larger search box.

Benchmark-specific classification labels are intentionally kept in the
Validation documentation rather than exposed as ordinary user concepts. See
{doc}`../validation/accuracy` when interpreting the behavior of the automatic
selectors against independent references.

## Self-convergence and independent validation

Self-convergence compares numerical resolutions generated by the same method.
Independent validation compares against a separately derived analytic result or
a separately refined numerical reference. Agreement between successive
resolutions is necessary evidence for a quantitative calculation, but it is not
by itself proof of absolute accuracy.

When an independent reference is available, report its error separately. The
validation suite uses global relative $L^2$ and peak-normalized maximum errors
because the latter remains meaningful for interactions that cross zero. The
reference hierarchy and benchmark classification policy are documented in
{doc}`../validation/accuracy`.
