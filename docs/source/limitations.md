# Limitations

QUARTIC2D currently supports scalar two-dimensional fields and radially symmetric momentum-space kernels.

The accuracy of an interaction calculation depends on both the PETAL2D decomposition and the QUARTIC2D transforms. Finite radial support, finite momentum support, interpolation, and numerical resolution must be checked for the application of interest.

`q_cutoff` and `q_cutoffs` are sampled power-support diagnostics. They are not error bounds for the final interaction.

The convergence interfaces report numerical self-convergence with respect to the parameter being refined. They do not include every possible representation error. In particular, FFTLog convergence in `n` is distinct from sensitivity to the logarithmic domain, endpoint periodicity, interpolation, and `bias`.

The public Ogata backend requires the optional `hankel` dependency. Its cost can grow steeply for difficult nodal or cusp-like workloads, so `N` and `h` should be calibrated for the application rather than treated as monotonic resolution knobs. `Interaction.converge_parameters(..., method="ogata")` performs the coupled interaction-level calibration; for radial transforms, Ogata `N`/`h` calibration is available through the fixed-q-grid `converge()` helper rather than the full q-support `HarmonicTransform.converge_parameters(...)` path.

General non-radial kernels, tensor-valued fields, and a public API for arbitrary continuous radial callables are outside the current scope.
