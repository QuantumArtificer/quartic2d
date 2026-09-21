# Computational complexity

This file records the source-level complexity model used for the manuscript.
Step 2 is complete here. Step 3 is intentionally deferred until its interaction
benchmark is constructed from the validated Step-2 outputs.

## Step 2: `HarmonicTransform`

Notation:

- `N_m`: retained angular harmonics;
- `N_r`: radial samples in the supplied profile;
- `s_r`: finite-quadrature subdivision factor;
- `N_s = s_r (N_r - 1) + 1`: effective radial integration nodes for
  Simpson/trapezoid;
- `N_q`: tabulated momentum samples;
- `B`: internal q-batch size, currently 256;
- `B_eff = min(B, N_q)`.

For each harmonic, Simpson/trapezoid forms Bessel-function blocks with
`B_eff x N_s` entries and integrates all `N_q` rows. Treating one fixed-order
Bessel evaluation as constant cost per matrix element gives

`Theta(N_q N_s)`

per retained harmonic and therefore

`T_H = Theta(N_m N_q N_s)`

for the dominant transform kernel. Because

`N_s = s_r (N_r - 1) + 1 = Theta(s_r N_r)`,

this is equivalently

`T_H = Theta(N_m N_q s_r N_r)`.

The public constructor also performs scale-aware default-grid metadata even
when explicit q parameters are supplied, builds interpolation objects, and
computes inexpensive post-transform diagnostics. Their leading work is

`Theta(N_m N_r + N_m N_q)`.

Hence the full constructor has

`T_H,full = Theta(N_m [N_r + N_q + N_q N_s])`,

which reduces to the transform-kernel expression in the nontrivial asymptotic
regime.

For composite Gauss-Legendre order `p`, the sampled quadrature has
approximately `p s_r (N_r - 1)` nodes. Thus

`T_H,GLp = Theta(p N_m N_q s_r N_r)`.

Because `p` is fixed (`p=4` or `8`), GL4 and GL8 have the same Big-Theta class
as Simpson/trapezoid but larger method-dependent prefactors.

### Step-2 memory

The transform processes harmonics sequentially. Previously computed outputs
remain stored, while the dominant temporary Bessel block contains
`B_eff N_s` elements. Excluding the input decomposition, the leading storage is

`M_H = Theta(N_m N_q + B_eff N_s + N_r)`.

The first term is persistent transformed output, the second is the batched
Bessel/integrand workspace, and the last covers radial/interpolation arrays.
The batch therefore prevents the temporary matrix from scaling as
`N_q N_s` once `N_q > B`.

### Assumption on angular order

The dimensional analysis treats a Bessel evaluation as `O(1)` for fixed order.
The empirical `N_m` sweep necessarily includes progressively larger `|m|`, so
small changes in the measured prefactor with harmonic order are expected; the
predicted dependence on the number of retained harmonics remains linear.

## Step-2 empirical verification

`run_step2_scaling.py` varies one independent dimension at a time while holding
all others fixed:

- `N_q`;
- `N_r`;
- `N_m`;
- `s_r`.

The `N_r` and `s_r` fits use the exact effective-node count `N_s`. Each sweep
stores the asymptotic log-log exponent `alpha` and `R^2` over the largest
points, as well as an all-point fit to expose finite-size overhead.

A second test collapses the `N_q`, `N_r`, and `s_r` sweeps (fixed harmonic set) against the source-derived work variable

`W = N_m N_q N_s`.

The `N_m` sweep is kept separate because increasing the harmonic count also
increases the Bessel orders and can change the special-function prefactor. If
the dominant dimensional model is correct, the fixed-order-set collapse should approach

`t proportional to W`

with log-log slope `alpha -> 1`.

The benchmark times the complete public `HarmonicTransform` constructor.
`check=False` suppresses warning emission but does not bypass the constructor's
cheap diagnostic calculation, so the reported wall time is an end-to-end Step-2
production cost rather than an isolated Bessel-kernel microbenchmark.

## Step 3


## Interaction transform

Let

- \(N_p=N_m^{(1)}N_m^{(2)}\) be the number of harmonic pairs,
- \(N_D\) the number of requested displacement vectors,
- \(N_F\) the FFTLog sequence length,
- \(N_q\) the native common momentum-grid size, and
- \(N_{q,s}=s_q(N_q-1)+1\) the refined finite-quadrature grid size.

For FFTLog, each harmonic pair performs one logarithmic transform and then
interpolates the result to the requested displacements.  Treating FFT work as
\(\Theta(N_F\log N_F)\),

\[
T_{\mathrm{FFTLog}}
=\Theta\!\left[N_p\left(N_F\log N_F+N_D\right)\right].
\]

The pair-resolved phase, radial-integral and contribution arrays are all
\(N_p\times N_D\), while the active FFTLog workspace is independent of
\(N_p\) because pairs are processed sequentially.  Thus

\[
M_{\mathrm{FFTLog}}
=\Theta\!\left(N_pN_D+N_F+N_q\right),
\]

apart from storage already owned by the input fields.

For finite Simpson/trapezoid quadrature, every pair forms a Bessel matrix of
shape \(N_D\times N_{q,s}\).  Therefore

\[
T_{\mathrm{finite}}
=\Theta\!\left(N_pN_DN_{q,s}\right)
=\Theta\!\left(N_pN_Ds_qN_q\right),
\]

and, because pairs are processed sequentially,

\[
M_{\mathrm{finite}}
=\Theta\!\left(N_pN_D+N_DN_{q,s}+N_q\right).
\]

Fixed-order Gauss--Legendre rules preserve these asymptotic classes and change
only the quadrature-node prefactor.
