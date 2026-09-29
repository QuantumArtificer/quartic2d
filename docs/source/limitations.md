# Limitations and scope

QUARTIC2D has a deliberately specific spatial reduction. The mathematical formulation, the implemented API, the validation evidence, and the recommended numerical use are related but are not interchangeable statements.

## Spatial kernel

The current reduction assumes a scalar two-body kernel that is translationally invariant and radial in the two spatial coordinates. In momentum space the interaction is supplied as

$$
U(q),\qquad q=|\mathbf q|.
$$

A genuinely anisotropic $U(\mathbf q)$, tensor-valued spatial kernel, or interaction with explicit dependence on both absolute positions requires a different spatial reduction.

Internal labels such as spin, valley, layer, or orbital flavor can enter through the orbitals and through coefficients outside the spatial integral. QUARTIC2D does not provide a many-body operator algebra for those labels.

## Transition fields

The two transition fields may be different, complex, sign-changing, and built from different orbitals. The mathematical reduction therefore applies to general four-index channels within the radial-kernel assumption.

The validation portfolio is narrower than that mathematical statement. It contains analytic and synthetic isotropic, anisotropic, odd-harmonic, mixed-parity, nodal, cusp-like, oscillatory, and algebraic fields together with several radial kernel families. It does not independently test every orbital-index permutation or every material-specific transition field.

## Four distinct scope statements

| Scope level | Meaning |
| --- | --- |
| Mathematical | What follows from the four-center reduction for sufficiently well-behaved two-dimensional transition fields and a scalar radial kernel. |
| Implemented | What the current public API can represent and evaluate. |
| Validated | The field, kernel, displacement, and tolerance families exercised by tests and benchmarks. |
| Recommended use | The numerical methods and parameter-selection strategies supported by that evidence for a particular problem class. |

A calculation can lie within the mathematical formulation without having an identical case in the validation matrix. Conversely, a method being implemented does not make it the recommended choice for every input.

## Real-space representation is upstream

A final interaction depends on the quality of the input fields. Finite Cartesian extent, PETAL2D interpolation, radial sampling, angular truncation, and orbital normalization are separate from the downstream QUARTIC2D radial quadrature. No transform setting can reconstruct information absent from the original sampled field.

## Momentum support is finite

`q_max` defines the represented momentum interval. The radial sample spacing imposes

$$
q_{\mathrm{Nyquist}}=\frac{\pi}{\Delta r}.
$$

QUARTIC2D rejects a requested support beyond that ceiling. Large-displacement calculations can also be sensitive to the outer q boundary of an automatically sampled field. The interaction convergence interface reports that condition as `upstream_q_boundary_not_robust` rather than extending the representation beyond the information contained in the radial input.

## FFTLog is conditional on the input

FFTLog is a logarithmically sampled fast Hankel-transform method {cite:p}`Hamilton2000`. Its accuracy depends on transform length, bias, endpoint behavior, interpolation, ringing, and aliasing. A sequence that is stable only as the transform length increases can still remain bias-sensitive. QUARTIC2D therefore requires local bias robustness in addition to resolution stability before an FFTLog search is accepted.

The declared FFTLog search box does not produce an accepted result for every benchmark case at every tolerance. A refusal remains a refusal rather than being replaced by an extrapolated parameter choice.

## Ogata is a specialist method

The optional Ogata quadrature uses a Bessel-zero-based integration formula {cite:p}`Ogata2005` and requires the optional `hankel` dependency. Its useful $(N,h)$ region is workload-dependent, and difficult nodal or oscillatory integrands can be expensive. Calibrate it on the assembled observable rather than selecting a large `N` by inspection.

## Named physical screening models

The Rytova--Keldysh interaction used in the examples and validation suite follows the thin-film screening model associated with Rytova and Keldysh {cite:p}`Rytova1967,Keldysh1979`. The static two-dimensional electron-gas response used in the RPA benchmark is based on Stern's zero-temperature polarizability {cite:p}`Stern1967`. These kernels are validation inputs within the radial-kernel formulation; QUARTIC2D does not derive the material parameters entering them.

## Numerical convergence is local to the represented problem

A calibration result refers to the supplied fields, kernel, displacement set, tolerances, and tested parameter box. It establishes self-consistency of the implemented refinement checks for that represented problem. It is not a universal pointwise error guarantee. Reusing parameters for a related calculation should be justified by checks on the new problem class, and benchmark results should not be interpreted as universal parameter choices for future inputs.
