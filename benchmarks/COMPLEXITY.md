# Computational complexity

This note records the source-derived runtime and memory models used by the scaling benchmarks and manuscript. The asymptotic expressions describe the implemented numerical kernels. Fitted benchmark slopes are finite-range measurements and are reported separately from these source-derived expectations.

## HarmonicTransform

Let

- $N_m$ be the number of retained angular harmonics;
- $N_r$ the number of radial samples supplied by the decomposition;
- $s_r$ the radial subdivision factor used by a sampled finite quadrature;
- $N_s=s_r(N_r-1)+1$ the effective sampled radial-grid size for Simpson and trapezoid;
- $N_q$ the number of momentum samples;
- $B$ the internal q-batch size, currently 256;
- $B_{\mathrm{eff}}=\min(B,N_q)$.

### Sampled finite quadrature

For one harmonic, Simpson and trapezoid evaluate Bessel functions on batches containing $B_{\mathrm{eff}}N_s$ matrix elements and integrate all $N_q$ momentum points. Treating evaluation of a fixed-order Bessel function as constant work per matrix element gives

\[
T_{H,m}=\Theta(N_qN_s).
\]

For $N_m$ retained harmonics,

\[
T_H=\Theta(N_mN_qN_s)
=\Theta(N_mN_qs_rN_r).
\]

The constructor also reads the radial profiles, constructs interpolation objects, stores the transformed harmonics, and evaluates diagnostics. Their leading work is

\[
\Theta(N_mN_r+N_mN_q),
\]

so a more explicit constructor-level model is

\[
T_{H,\mathrm{full}}
=\Theta\!\left[N_m\left(N_r+N_q+N_qN_s\right)\right].
\]

The $N_qN_s$ term dominates in the nontrivial sampled-quadrature regime.

For a fixed-order composite Gauss-Legendre rule of order $p$, the quadrature uses approximately $p s_r(N_r-1)$ nodes. Therefore

\[
T_{H,\mathrm{GL}p}
=\Theta(pN_mN_qs_rN_r).
\]

Because $p=4$ or $8$ is fixed, GL4 and GL8 have the same asymptotic dependence as Simpson and trapezoid, with method-dependent prefactors.

### Memory

Harmonics are transformed sequentially. Previously computed form factors remain stored, while the dominant temporary sampled-quadrature object is the batched Bessel/integrand block. Excluding storage already owned by the input decomposition,

\[
M_H
=\Theta(N_mN_q+B_{\mathrm{eff}}N_s+N_r).
\]

The first term is the persistent transformed output, the second is the active Bessel/integrand workspace, and the final term represents radial/interpolation arrays. Batching prevents the temporary matrix from growing as $N_qN_s$ once $N_q>B$.

### Angular-order assumption

The dimensional model treats one Bessel evaluation as $O(1)$ at fixed order. The empirical $N_m$ sweep necessarily introduces progressively larger $|m|$, which can change the special-function prefactor. The source-derived prediction is therefore linear in the **number** of retained harmonics at fixed-order cost; deviations caused by the order distribution are a prefactor effect rather than a different array-dimensional scaling law.

## HarmonicTransform scaling benchmark

`harmonic_scaling.py` varies one independent dimension at a time:

- $N_q$;
- $N_r$;
- $N_m$;
- $s_r$.

The $N_r$ and $s_r$ analyses use the exact effective node count $N_s$. Each sweep stores a finite-range log-log fit over the largest measured points and an all-point fit that exposes fixed overhead at small problem sizes.

A separate work-collapse test combines the $N_q$, $N_r$, and $s_r$ sweeps at fixed harmonic content using

\[
W_H=N_mN_qN_s.
\]

If the dominant sampled-transform work controls the measured regime, runtime should approach proportionality to $W_H$. The $N_m$ sweep is kept separate because increasing the number of harmonics also changes the represented Bessel orders.

The benchmark times the complete public `HarmonicTransform` constructor. `check=False` suppresses warning emission but does not bypass the ordinary diagnostic calculation, so the measured time is constructor-level production cost rather than an isolated Bessel microbenchmark.

## Interaction

Let

- $N_m^{(1)}$ and $N_m^{(2)}$ be the retained harmonic counts of the two transformed fields;
- $N_p=N_m^{(1)}N_m^{(2)}$ the number of harmonic pairs;
- $N_D$ the number of requested displacement vectors;
- $N_q$ the common native momentum-grid size;
- $s_q$ the finite-quadrature subdivision factor;
- $N_{q,s}=s_q(N_q-1)+1$ the refined sampled momentum-grid size;
- $N_F$ the FFTLog sequence length.

### Finite quadrature

For each harmonic pair, a finite sampled method evaluates a translation-Bessel block over $N_D$ displacements and $N_{q,s}$ momentum nodes. Therefore

\[
T_{\mathrm{finite}}
=\Theta(N_pN_DN_{q,s})
=\Theta(N_pN_Ds_qN_q).
\]

Pairs are processed sequentially. The leading storage is

\[
M_{\mathrm{finite}}
=\Theta(N_pN_D+N_DN_{q,s}+N_q).
\]

The $N_pN_D$ term stores pair-resolved outputs, while $N_DN_{q,s}$ is the active translation/quadrature workspace. Fixed-order Gauss-Legendre rules preserve the same asymptotic dependence and change the quadrature-node prefactor.

### FFTLog

For each harmonic pair, FFTLog performs one logarithmic transform and evaluates/interpolates the transformed representation at the requested displacements. Treating the transform work as $\Theta(N_F\log N_F)$ gives

\[
T_{\mathrm{FFTLog}}
=\Theta\!\left[N_p\left(N_F\log N_F+N_D\right)\right].
\]

The active FFTLog workspace is pair-local, while pair-resolved interaction arrays scale with $N_pN_D$. Thus

\[
M_{\mathrm{FFTLog}}
=\Theta(N_pN_D+N_F+N_q),
\]

apart from storage already owned by the two transformed input fields.

## Interaction scaling benchmark

`interaction_scaling.py` measures finite-grid and FFTLog dependence on their independent problem dimensions. The manuscript uses source-derived work variables

\[
W_{\mathrm{finite}}=N_pN_DN_{q,s},
\]

and

\[
W_{\mathrm{FFTLog}}
=N_p\left[N_F\log_2(N_F)+N_D\right].
\]

The base of the logarithm changes only a constant prefactor. The plotted work guides express the source-derived dimensional models. They are not assertions that every finite benchmark range has reached its asymptotic regime.

## Interpretation of fitted slopes

A measured log-log slope is evidence about the finite range covered by a particular sweep. It should be reported with the swept variable, fixed dimensions, fit range, and goodness of fit. It must not replace the source-derived complexity expression, and a slope close to one in a collapsed work variable should not be generalized beyond the measured regime without additional evidence.

Runtime and memory are benchmarked separately. Neither scaling result establishes numerical accuracy; the configurations used for performance figures must be qualified by the corresponding accuracy/reference benchmarks.
