# Validation and benchmarks

QUARTIC2D separates mathematical checks, numerical accuracy, automatic-convergence reliability, end-to-end validation, timing, and memory. The benchmark definitions are public and are part of the documentation because the practical recommendations in the User Guide are based on them.

::::{grid} 2
:::{grid-item-card} Benchmark matrix
:link: benchmark_matrix
:link-type: doc
Exact field profiles, analytic transforms, kernel equations, displacement domains, and construction of the 74-case interaction matrix.
:::
:::{grid-item-card} Accuracy and automatic convergence
:link: accuracy
:link-type: doc
Independent references, automatic self-convergence, numerical-method capability, mixed-parity coverage, and end-to-end errors.
:::
:::{grid-item-card} Performance and scaling
:link: performance
:link-type: doc
Calibration cost, qualified fixed-production timing, runtime scaling, and peak-memory measurements.
:::
:::{grid-item-card} Publication suite
:link: publication_suite
:link-type: doc
Canonical commands, result files, provenance requirements, and subset runs.
:::
::::

## What is tested

The broad interaction study uses 74 field/kernel combinations at the primary `Interaction.rtol = 1e-4` target. Its inputs include smooth Gaussian harmonics, odd angular pairs, anisotropy through $m=\pm2$ and $m=\pm4$, nodal fields, a complex mixed-parity field, a radial cusp, an oscillatory exponential, and algebraic tails. Kernel families include Coulomb, three Rytova-Keldysh screening lengths, single and dual gates, Thomas-Fermi, three Yukawa scales, a strictly two-dimensional Helmholtz/Yukawa Green function, interlayer and gated-interlayer Coulomb, and the nonanalytic static 2DEG RPA/Stern kernel {cite:p}`Stern1967`.

The exact equations and parameter values are collected in {doc}`benchmark_matrix` and are referenced throughout the User Guide.

## Independent accuracy and selector behavior

The end-to-end benchmark begins with analytic two-dimensional fields sampled on a Cartesian grid, decomposes those arrays with PETAL2D, calibrates the momentum-space representation, and evaluates the final interaction against analytic momentum-space form factors.

```{figure} ../_static/validation/end_to_end_accuracy.svg
:class: q2d-figure q2d-figure-compact
:alt: Worst relative errors at PETAL2D, HarmonicTransform, and final Interaction stages in standard and large displacement tests

Case-resolved relative $L^2$ reference errors for the three end-to-end pipelines in the standard and large-displacement domains. PETAL2D profile error, HarmonicTransform error, and final Interaction error are shown as separate markers for each physical case; no line connects independently selected suite maxima.
```

Automatic selection is evaluated separately from independent accuracy. The four-case canonical matrix records whether each selector accepted, refused conservatively, missed a demonstrably capable point, or exhausted the declared search box without finding one. See {doc}`accuracy` for the case-level status matrix and the broader 74-case reference study.

```{toctree}
:maxdepth: 2
:hidden:

benchmark_matrix
accuracy
performance
publication_suite
```
