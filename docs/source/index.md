# QUARTIC2D

QUARTIC2D: *Quadrature for Radial Tetra-center Interaction Coefficients in 2D*

Four-center interactions for localized states in two dimensions

Localized-orbital models of two-dimensional materials often require Coulomb
matrix elements between Wannier functions, defect states, quantum-dot states,
or other localized wavefunctions. These matrix elements set the interaction
parameters of Hubbard-like and multiorbital Hamiltonians, including direct
terms, exchange, pair hopping, and correlated hopping. The calculation becomes
more demanding when the orbitals are anisotropic or complex, when dielectric
screening is nonlocal, or when many separations and orbital combinations must
be evaluated.

QUARTIC2D evaluates the corresponding four-center interaction matrix elements

$$
U_{1234}=\iint d^2\mathbf r\,d^2\mathbf r'\,
\phi_1^*(\mathbf r)\phi_2^*(\mathbf r')
U(|\mathbf r-\mathbf r'|)
\phi_3(\mathbf r)\phi_4(\mathbf r')
$$

for localized two-dimensional orbitals and translationally invariant radial kernels. The four orbital indices enter through two transition fields,

$$
\rho_{13}(\mathbf r)=\phi_1^*(\mathbf r)\phi_3(\mathbf r),
\qquad
\rho_{42}(\mathbf r)=\phi_4^*(\mathbf r)\phi_2(\mathbf r).
$$

The transition fields may be real or complex. Their index pattern determines the physical matrix element. Direct interactions, exchange, pair hopping, correlated hopping, and other four-index channels use the same numerical formulation.

QUARTIC2D expands each transition field in circular harmonics, Hankel-transforms the radial coefficients, and evaluates the remaining momentum integrals for the requested separation vectors. The factorization separates orbital structure, relative geometry, and the radial interaction kernel while retaining anisotropic and complex orbital products explicitly.

```{figure} _static/examples/gaussian_interaction_end_to_end.svg
:class: q2d-figure q2d-figure-compact
:alt: Bare Coulomb and Rytova-Keldysh interactions of an isotropic Gaussian transition field

Bare Coulomb and Rytova--Keldysh matrix elements {cite:p}`Rytova1967,Keldysh1979` for the isotropic Gaussian calculation in {doc}`getting_started`. Blue denotes the Coulomb kernel and orange denotes the Rytova--Keldysh kernel. Solid curves show the analytic matrix elements, open circles show the default QUARTIC2D calculation, and open squares show the self-converged QUARTIC2D calculation. Screening primarily reduces the short-distance matrix element.
```

::::{grid} 2
:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc
Install QUARTIC2D and evaluate an isotropic Gaussian with bare Coulomb and Rytova--Keldysh kernels.
:::
:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Four-center inputs, transition fields, displacements, transformed harmonics, convergence, diagnostics, and numerical-method guidance.
:::
:::{grid-item-card} API reference
:link: reference/index
:link-type: doc
Object-oriented API reference with separate pages for public classes, convergence results, and callable methods.
:::
:::{grid-item-card} Guided examples
:link: examples/index
:link-type: doc
Analytic transforms, direct and exchange channels, anisotropy, sampled Cartesian data, complex transition fields, dielectric screening, and long-range interactions.
:::
:::{grid-item-card} Theory
:link: theory/index
:link-type: doc
Four-center reduction, Fourier conventions, angular harmonics, signed Bessel orders, and the final interaction formula.
:::
:::{grid-item-card} Limitations and scope
:link: limitations
:link-type: doc
Mathematical, implemented, validated, and recommended scope, including the assumptions on radial kernels and represented momentum support.
:::
:::{grid-item-card} Validation and benchmarks
:link: validation/index
:link-type: doc
Independent reference accuracy, automatic-refinement behavior, end-to-end tests, timing, scaling, and memory measurements.
:::
:::{grid-item-card} References
:link: references
:link-type: doc
Primary references for the physical kernels and numerical methods named throughout the documentation.
:::
::::

## What QUARTIC2D is for

The package is intended for localized 2D orbitals, Wannier functions, bound states, defect states, moiré states, and other scalar fields for which the interaction kernel depends only on the in-plane separation. A calculation may use identical fields, two different densities, or complex transition fields built from different orbitals.

For example, the same four-center expression contains

$$
U_{ijij}
$$

for a direct density-density term and

$$
U_{ijji}
$$

for exchange. Pair hopping and correlated hopping follow from other choices of the four orbital indices. QUARTIC2D does not restrict the transition fields to positive densities.

The current spatial reduction assumes a scalar radial kernel $U(q)$ in momentum space. Genuinely anisotropic, tensor-valued, or non-translationally-invariant spatial kernels require a different reduction. See {doc}`limitations` for the precise package scope.

## Numerical control

Ordinary constructors provide practical defaults and inexpensive diagnostics. When a calculation requires an explicit numerical criterion, `HarmonicTransform.converge_parameters(...)` and `Interaction.converge_parameters(...)` refine the represented transform and assembled interaction and record the selected parameters. Both convergence-result objects provide `plot_convergence()` for inspecting the search, and objects constructed from those results retain the same record. Their `rtol` values are self-convergence criteria, not pointwise error bounds against an unknown exact solution. For repeated calculations, settings may be reused after the intended field, kernel, and displacement family has been checked.

See {doc}`user_guide/numerical_methods` for default and specialist-method guidance, {doc}`user_guide/convergence_and_diagnostics` for automatic convergence and parameter reuse, and {doc}`validation/benchmark_matrix` for the exact equations behind the validation suite.

```{toctree}
:maxdepth: 2
:hidden:

getting_started
user_guide/index
reference/index
examples/index
theory/index
limitations
validation/index
references
development/index
```
