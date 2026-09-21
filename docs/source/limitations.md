# Limitations

QUARTIC2D currently supports scalar two-dimensional fields and radially symmetric momentum-space kernels.

The accuracy of an interaction calculation depends on both the PETAL2D decomposition and the QUARTIC2D transforms. Finite radial support, finite momentum support, interpolation, and numerical resolution must be checked for the application of interest.

`q_cutoff` and `q_cutoffs` are sampled power-support diagnostics. They are not error bounds for the final interaction.

The convergence interfaces report numerical self-convergence with respect to the parameter being refined. They do not include every possible representation error. In particular, FFTLog convergence in `n` is distinct from sensitivity to the logarithmic domain, endpoint periodicity, interpolation, and `bias`.

The Ogata backend is retained for numerical benchmarking but is excluded from the supported public API while its automatic parameter selection is under evaluation.

General non-radial kernels, tensor-valued fields, and a public API for arbitrary continuous radial callables are outside the current scope.
