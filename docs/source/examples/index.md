# Guided examples

The guided examples follow interaction problems that recur in localized-state
models of two-dimensional systems: screened direct interactions, exchange
between distinct orbitals, orientation dependence, numerically sampled
wavefunctions, complex transition fields, dielectric-environment sweeps, and
long-range matrix elements. Each example keeps the field definitions simple
enough that the physical content remains visible while exercising the same
PETAL2D → QUARTIC2D workflow used for numerical orbitals.

```{toctree}
:maxdepth: 1

isotropic_interaction
direct_exchange
anisotropic_interaction
sampled_data
complex_transition_field
screening_family_sweep
large_separation
```

The progression is deliberate:

1. {doc}`isotropic_interaction` establishes the end-to-end workflow and the
   distinction between self-convergence and an analytic reference.
2. {doc}`direct_exchange` introduces distinct orbital products and a
   sign-changing exchange transition field.
3. {doc}`anisotropic_interaction` resolves orientation dependence into harmonic
   pair contributions.
4. {doc}`sampled_data` replaces analytic callables by sampled Cartesian arrays
   and carries them through a complete four-center interaction.
5. {doc}`complex_transition_field` treats genuinely complex off-diagonal
   transition fields and multiorbital channels.
6. {doc}`screening_family_sweep` demonstrates calibrated parameter reuse across
   a bounded family of radial kernels.
7. {doc}`large_separation` examines a strongly oscillatory displacement regime
   against an analytic long-distance reference.

Benchmark stress fields, publication-sized sweeps, method coverage, and timing
measurements are defined separately in {doc}`../validation/index`. They are
evidence for validated capability, not the identity of the guided examples.
