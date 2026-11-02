# QUARTIC2D manuscript notes

This file is the internal scientific companion for the forthcoming QUARTIC2D paper. It records the canonical mathematical formulation, the intended manuscript argument, the numerical evidence supporting that argument, the limits of the present validation, and the claims that should or should not appear in the paper.

It is not a user guide, API reference, benchmark log, or manuscript draft. Public documentation should remain organized around user tasks. Numerical values in this file are summaries of a specific publication benchmark snapshot and must be refreshed when that snapshot is replaced.

## 1. Scientific identity

QUARTIC2D is a method for evaluating general four-center two-body matrix elements between localized states in two dimensions,

$$
U_{1234}=\iint d^2\mathbf r\,d^2\mathbf r'\,
\phi_1^*(\mathbf r)\phi_2^*(\mathbf r')
U(|\mathbf r-\mathbf r'|)
\phi_3(\mathbf r)\phi_4(\mathbf r').
$$

The organizing observation is that the four orbital indices enter through two transition fields,

$$
\rho_{13}(\mathbf r)=\phi_1^*(\mathbf r)\phi_3(\mathbf r),
\qquad
\rho_{42}(\mathbf r')=\phi_4^*(\mathbf r')\phi_2(\mathbf r').
$$

The fields may be real or complex, positive or sign-changing. Density-density interactions are one special case. The same spatial formulation covers direct terms, exchange, pair hopping, correlated hopping, and other four-index channels obtained from different orbital choices.

The present method assumes a scalar, translationally invariant radial kernel. In momentum space the user supplies $U(q)$ with $q=|\mathbf q|$.

The paper should be framed around this four-center reduction and its numerical realization. It should not be framed as a quadrature comparison, backend benchmark, API paper, or package announcement. QUARTIC2D is the validated implementation of the method, not the intellectual definition of the method.

## 2. Scope hierarchy

Four levels of capability must remain separate throughout the manuscript.

### Mathematical scope

The reduction applies to sufficiently well-behaved two-dimensional transition fields and a scalar radial two-body kernel. The fields need not be densities, need not be real, and need not correspond to the same pair of orbitals.

### Implemented scope

The public implementation accepts PETAL2D angular-harmonic decompositions, transforms their retained radial harmonics, and evaluates interactions for one or many displacement vectors. Supported radial-transform methods are trapezoid, Simpson, GL4, GL8, and optional Ogata quadrature. Supported Interaction methods are trapezoid, Simpson, GL4, GL8, FFTLog, and optional Ogata quadrature.

### Validated scope

The publication benchmark portfolio tests a finite set of analytic and synthetic transition-field families, radial kernels, displacement ranges, and numerical targets. That portfolio is deliberately diverse, but it is not equivalent to the full mathematical scope.

### Recommended use

Recommendations are conditional on workload, displacement range, desired accuracy, and the evidence available for that regime. No backend should be described as universally best.

The paper must not narrow the mathematical statement merely because every four-index physical channel has not been benchmarked separately. Conversely, mathematical applicability must not be presented as numerical validation outside the tested domain.

## 3. Canonical derivation

### 3.1 Fourier conventions

Use

$$
f(\mathbf q)=\frac{1}{2\pi}\int d^2\mathbf r\,
f(\mathbf r)e^{-i\mathbf q\cdot\mathbf r},
$$

and

$$
U(\mathbf R)=\int\frac{d^2\mathbf q}{(2\pi)^2}
U(q)e^{i\mathbf q\cdot\mathbf R}.
$$

The callable passed to `Interaction` is the momentum-space radial kernel $U(q)$ in this convention.

### 3.2 Circular harmonics

For either transition field,

$$
\rho(r,\theta)=\sum_m \rho_m(r)e^{im\theta},
$$

with

$$
\rho_m(r)=\frac{1}{2\pi}\int_0^{2\pi}
\rho(r,\theta)e^{-im\theta}\,d\theta.
$$

Define

$$
F_m(q)=\int_0^\infty r\,dr\,\rho_m(r)J_m(qr).
$$

Then

$$
\rho(q,\phi_q)=\sum_m(-i)^m e^{im\phi_q}F_m(q).
$$

The orbital or transition-field structure is therefore isolated in the one-dimensional functions $F_m(q)$.

### 3.3 Relative displacement

Let the centers of the two localized transition fields be separated by

$$
\boldsymbol\delta=\mathbf R_1-\mathbf R_2
=\delta(\cos\phi_\delta,\sin\phi_\delta).
$$

The momentum-space interaction contains the translation phase

$$
e^{-i\mathbf q\cdot\boldsymbol\delta}.
$$

The angular identity

$$
\int_0^{2\pi}d\phi_q\,
e^{in\phi_q}
e^{-iq\delta\cos(\phi_q-\phi_\delta)}
=2\pi(-i)^n e^{in\phi_\delta}J_n(q\delta)
$$

reduces the four-center integral to

$$
\boxed{
V(\boldsymbol\delta)
=2\pi\sum_{m,m'}e^{i(m-m')\phi_\delta}
\int_0^\infty q\,dq\,
U(q)F_m^{(1)}(q)F_{m'}^{(2)*}(q)
J_{m'-m}(q\delta)
}.
$$

This is the central equation of the paper.

### 3.4 Nonnegative numerical Bessel order

The implementation evaluates $J_{|m-m'|}$ and carries the signed-order parity separately. For $k\ge0$,

$$
J_{-k}(x)=(-1)^kJ_k(x).
$$

Define

$$
\eta_{mm'}=
\begin{cases}
(-1)^{m-m'}, & m>m',\\
1, & m\le m',
\end{cases}
$$

and

$$
H_{mm'}(\delta)=
\int_0^\infty q\,dq\,
U(q)F_m^{(1)}(q)F_{m'}^{(2)*}(q)
J_{|m-m'|}(q\delta),
$$

$$
\Phi_{mm'}(\phi_\delta)=
\eta_{mm'}e^{i(m-m')\phi_\delta}.
$$

Then

$$
V_{mm'}(\boldsymbol\delta)=2\pi\Phi_{mm'}H_{mm'},
\qquad
V(\boldsymbol\delta)=\sum_{m,m'}V_{mm'}(\boldsymbol\delta).
$$

The notation should match the public `Phi_mm`, `H_mm`, `V_mm`, and `V` arrays wherever implementation details are discussed.

### 3.5 Zero displacement

At $\delta=0$, only $m=m'$ terms survive because $J_n(0)=0$ for nonzero integer $n$. Zero-displacement tests are useful for normalization and radial quadrature, but they do not test the signed-order phase structure of finite-displacement odd $m-m'$ channels.

## 4. Why the reduction is computationally useful

The paper should explain the factorization, not only derive it.

The transition-field dependence is stored in $F_m^{(1)}(q)$ and $F_{m'}^{(2)}(q)$. The kernel enters multiplicatively through $U(q)$. Relative geometry enters through

$$
J_{|m-m'|}(q\delta)e^{i(m-m')\phi_\delta}
$$

plus the signed-order parity. This separation has three practical consequences.

First, the two-dimensional angular dependence is reduced to a discrete set of harmonic channels and one-dimensional radial integrals. Second, transformed fields can be reused across many displacement vectors. Third, when the transition fields are unchanged, their form factors can be reused for different radial interaction kernels.

These reuse properties are part of the method motivation. They should not be confused with a claim that every numerical backend has the same cost or that one backend is optimal in every regime.

## 5. Numerical realization

### 5.1 Upstream representation

PETAL2D supplies the polar decomposition of each transition field. QUARTIC2D begins from the retained radial harmonic profiles and their represented radial domain. Errors introduced by Cartesian sampling, interpolation, finite spatial extent, angular truncation, or orbital normalization are upstream of the QUARTIC2D radial transform.

A downstream convergence test cannot reconstruct information absent from the input representation. End-to-end validation must therefore keep PETAL2D/input error separate from isolated QUARTIC2D transform and interaction error.

### 5.2 Harmonic transform

`HarmonicTransform` constructs $F_m(q)$ for each retained harmonic. Its numerical problem has three distinct resolution questions:

1. radial quadrature of each Hankel transform;
2. represented momentum support $q_{\max}$;
3. interpolation/sampling of the tabulated momentum-space form factor.

Automatic calibration treats the latter two explicitly and respects the radial Nyquist ceiling implied by the PETAL2D radial grid. A transform that cannot establish adequate represented support at the declared settings should be refused rather than extended beyond the information in the input.

### 5.3 Interaction evaluation

`Interaction` assembles the harmonic-pair contributions for all requested $\boldsymbol\delta$. Finite-grid methods refine the represented momentum grid. FFTLog replaces the oscillatory radial integral with a logarithmic transform whose accuracy depends on transform length, bias, endpoint behavior, and interpolation. Ogata quadrature searches a coupled $(N,h)$ space on the assembled observable.

Finite rules, FFTLog, and Ogata are alternative numerical realizations of the same reduced equation. Their comparisons belong in validation and practical guidance, not in the definition of the scientific problem.

## 6. Error and convergence language

This distinction is mandatory in the paper, captions, tables, and documentation.

A requested `rtol` is an internal self-convergence criterion. It is not an independently measured error and it is not a pointwise relative-error guarantee.

For two numerical vectors $a$ and $b$, the refinement machinery uses a relative $L^2$ change together with a peak-scaled absolute $L^\infty$ condition. Schematically,

$$
\epsilon_{L^2}^{\rm change}
=\frac{\|a-b\|_2}{\|b\|_2},
$$

and

$$
\|a-b\|_\infty
\le
\mathrm{atol}+\mathrm{rtol}\,\|b\|_\infty.
$$

A recorded relative $L^\infty$ change is diagnostic; it is not an additional independent acceptance rule.

Independent reference validation uses

$$
\epsilon_{L^2}^{\rm ref}
=\frac{\|f_{\rm num}-f_{\rm ref}\|_2}
{\|f_{\rm ref}\|_2},
$$

and

$$
\epsilon_{\rm peak}^{\rm ref}
=\frac{\|f_{\rm num}-f_{\rm ref}\|_\infty}
{\|f_{\rm ref}\|_\infty}.
$$

When one scalar ordinate is needed,

$$
\epsilon_{\rm ref}
=\max(\epsilon_{L^2}^{\rm ref},\epsilon_{\rm peak}^{\rm ref}).
$$

A plotted pointwise peak-normalized residual,

$$
e_i^{\rm peak}
=\frac{|f_{{\rm num},i}-f_{{\rm ref},i}|}
{\|f_{\rm ref}\|_\infty},
$$

shows where error occurs. `rtol` must not be drawn as a pointwise bound on this curve.

A genuinely pointwise relative error,

$$
e_i^{\rm rel}
=\frac{|f_{{\rm num},i}-f_{{\rm ref},i}|}
{|f_{{\rm ref},i}|},
$$

is a different quantity and can diverge or become visually misleading near zeros of the reference.

### Terminology to use

Use `automatic acceptance`, `self-convergence`, `independent reference agreement`, `reference-passing`, `requested tolerance`, `observed relative L2 error`, and `peak-normalized maximum error`.

Avoid `certified accuracy`, `guaranteed error`, `rigorous error bound`, or equivalent language unless a future analysis establishes such a bound. Raw benchmark metadata may contain legacy field names such as `qualification_kind`; manuscript prose should translate those into the scientific distinction above rather than repeat implementation labels.

## 7. Validation portfolio

The benchmark suite should be presented as evidence that the implementation handles qualitatively different transition-field and kernel structures, not as an attempt to enumerate all possible physical inputs.

### 7.1 Transition-field families

The current analytic/synthetic portfolio contains:

- isotropic Gaussian, one harmonic;
- odd conjugate Gaussian harmonics;
- anisotropic Gaussian with $m=0,\pm2$;
- higher-order Gaussian angular structure through $|m|=4$;
- nodal mixed-harmonic fields;
- complex mixed-parity fields;
- localized exponential cusp;
- oscillatory exponential profile;
- algebraically decaying mixed-harmonic profile.

The guided documentation examples should use cleaner pedagogical fields than the benchmark stress suite where possible. Benchmark-only profiles may remain artificial when their purpose is to isolate numerical difficulty.

### 7.2 Radial kernels

The current portfolio contains:

- bare Coulomb;
- Rytova-Keldysh screening at several screening lengths;
- dual-gate screening;
- single-gate image-charge screening;
- Thomas-Fermi screening;
- Yukawa screening at several inverse lengths;
- strictly two-dimensional Helmholtz/Yukawa Green function;
- interlayer Coulomb;
- gated interlayer Coulomb;
- zero-temperature 2DEG RPA/Stern screening with its $2k_F$ nonanalyticity;
- a smooth Gaussian numerical control kernel.

The manuscript should describe this diversity explicitly. Phrases such as `smooth and difficult cases` are too vague to communicate what was tested.

### 7.3 Displacement regimes

The standard-domain validation and the large-displacement validation are separate evidence classes. The declared large-displacement range is

$$
10^2\le\delta\le10^4,
$$

in the same length units as the input coordinates. Large $\delta$ creates a much more oscillatory translation-Bessel factor and changes both capability and cost.

### 7.4 Numerical targets

The benchmark suite uses three requested relative targets with different roles:

- $10^{-3}$: practical throughput and the dedicated large-displacement timing tier;
- $10^{-4}$: primary publication validation target;
- $10^{-5}$: stringent-limit sweep probing resolution boundaries.

These values are not a quality ranking of methods. A method useful at $10^{-3}$ is not `worse` merely because its declared search box has less coverage at $10^{-5}$.

## 8. Evidence classes

Keep the following evidence classes distinct.

1. Independent analytic or stabilized-reference accuracy.
2. Numerical self-convergence and tolerance response.
3. Automatic-selection reliability.
4. Cross-stage propagation from transformed fields into Interaction.
5. True end-to-end PETAL2D-to-Interaction accuracy.
6. Fixed-production performance at independently qualified accuracy.
7. Calibration cost and reuse/amortization.
8. Runtime scaling.
9. Peak-memory scaling.
10. Stress tests and refusal behavior.

Self-agreement is not an independent reference. Speed is not evidence of accuracy. An automatic acceptance is not a proof of absolute accuracy. A conservative refusal is not evidence that the underlying numerical method is incapable unless an explicit parameter-box capability study establishes that.

## 9. Current publication evidence snapshot

This section records the publication benchmark set currently present in `benchmarks/results/`. It should be refreshed, not manually massaged, when the canonical run is replaced.

### 9.1 Provenance

The present full publication suite ran on 2026-09-23 and reports overall `passed` status. The manifest records commit

```text
6bcdac3ee02bc8687d341c05049a3d024a2bf4b5
```

with a single-thread policy for OMP, MKL, OpenBLAS, and NumExpr.

The same manifest records

```text
source_state = working_tree_modified
reproducibility = working_tree_dependent
```

because `benchmarks_publication_run.log` was modified inside the repository while provenance was sampled. The numerical results are usable for development and manuscript planning, but the final paper/Zenodo dataset should be regenerated from a clean exact source state with logs written outside the repository.

### 9.2 HarmonicTransform

At the required $10^{-3}$ and $10^{-4}$ targets, all 18 required non-stress reference-accuracy rows completed. At the stringent $10^{-5}$ sweep, six of nine rows completed; the remaining three ended through one q-grid search-limit result and two unmet input preconditions. The stringent sweep should therefore be described as a limit study, not as universal $10^{-5}$ coverage.

The automatic-convergence generality study contains 18 regular rows. All 18 automatically accepted rows pass the independent reference, with no missed regular convergence in that study. Four declared stress cases are refused as expected. The current worst independent errors among accepted regular cases are approximately

$$
\epsilon_{L^2}^{\rm ref}=4.21\times10^{-5},
\qquad
\epsilon_{\rm peak}^{\rm ref}=2.79\times10^{-4},
$$

while the separate q-tail measure reaches approximately $9.76\times10^{-4}$ under its own $q$-tail target.

These numbers support reliability over the declared generality suite. They do not turn `HarmonicTransform.rtol` into an absolute or pointwise error bound.

### 9.3 Broad Interaction accuracy

The broad fixed-field study contains 74 cases per method/tolerance combination. Simpson and GL4 have complete independent-reference coverage over this matrix at $10^{-3}$, $10^{-4}$, and $10^{-5}$ in the current run. At the primary $10^{-4}$ target, the worst errors are approximately

| Method | Worst relative $L^2$ | Worst peak-normalized max |
| --- | ---: | ---: |
| Simpson | $3.38\times10^{-5}$ | $4.43\times10^{-5}$ |
| GL4 | $1.32\times10^{-5}$ | $4.97\times10^{-5}$ |

FFTLog has workload-dependent coverage rather than complete coverage. At $10^{-4}$ its broad fixed-field reference-passing fractions include 75% of the kernel group, 33% of the hard group, 22% of the continuity group, and the smooth control case. At $10^{-5}$ the declared FFTLog configuration/search used in this broad study does not supply comparable general coverage. This is evidence for conditional capability, not a reason to characterize FFTLog globally as accurate or inaccurate.

### 9.4 Automatic Interaction convergence

At the primary $10^{-4}$ target, the canonical four-case matrix is evaluated in both standard and large-displacement domains for Simpson, GL4, FFTLog, and Ogata. The current 32-row study reports zero automatically accepted results that fail the independent reference. Every canonical case/domain has at least one automatically accepted, reference-passing method.

The method coverage is deliberately uneven. For example, in the large-displacement primary matrix the complex dual-gate case is automatically accepted by GL4 and Ogata, while the nodal-RPA case is automatically accepted by GL4. This is exactly why the paper should report workload-conditioned coverage rather than an aggregate backend winner.

The practical $10^{-3}$ large-displacement matrix likewise reports zero accepted/reference-failing cases and complete case-level coverage by at least one method. Some finite-rule refusals are conservative: an independently checked parameter set may pass even when the automatic selector refuses within its search logic.

### 9.5 Cross-stage and true end-to-end validation

The pipeline benchmark separately evaluates:

- adaptive `HarmonicTransform -> Interaction` composition in the standard domain;
- representative large-displacement cross-stage calculations;
- true `PETAL2D -> HarmonicTransform -> Interaction` standard-domain calculations;
- true end-to-end calculations through $\delta=10^4$.

The current suite reports these sections as passing. The paper should preserve the stage-resolved errors rather than quote only the final interaction error. In particular, upstream PETAL2D/input representation error must remain visible when it dominates or is comparable to the downstream numerical error.

### 9.6 Large-displacement production cost

The current practical $10^{-3}$ fixed-configuration timing study illustrates why no single backend ranking is scientifically useful. Median qualified production times on the four canonical large-displacement workloads are approximately:

| Workload | Simpson | GL4 | FFTLog | Ogata |
| --- | ---: | ---: | ---: | ---: |
| isotropic Coulomb | 0.118 s | 0.108 s | 0.00187 s | 0.00622 s |
| anisotropic RK | 0.405 s | 0.323 s | 0.00736 s | 0.0241 s |
| complex dual gate | 3.88 s | 1.69 s | not qualified | 0.947 s |
| nodal RPA | 0.772 s | 0.642 s | not qualified | 15.1 s |

The scientific statement is conditional: FFTLog is exceptionally inexpensive on the qualified smooth cases, Ogata is effective for several smooth/complex cases but expensive for the nodal RPA workload, and finite rules provide broader robust coverage at higher cost in the large-$\delta$ regime. These values must not be collapsed into one global ranking.

Some production configurations in this table were independently qualified after an automatic refusal. That status must remain explicit.

### 9.7 Scaling

The source-derived leading work for sampled HarmonicTransform evaluation is

$$
T_H=\Theta(N_mN_qN_s),
$$

where $N_s$ is the effective number of radial quadrature nodes. The current work-collapse fit gives an exponent very close to one over the measured range,

$$
\alpha\approx0.9993,
$$

with log-space $R^2\approx0.9995$ for the all-point fit.

For finite-grid Interaction,

$$
T_{\rm finite}
=\Theta(N_pN_DN_{q,s}),
$$

where $N_p=N_m^{(1)}N_m^{(2)}$. The measured $N_D$ tail exponent is approximately $1.003$ in the current sweep. The finite $N_q$ sweep is likewise close to linear.

For FFTLog,

$$
T_{\rm FFTLog}
=\Theta\!\left[N_p\left(N_F\log N_F+N_D\right)\right].
$$

The current finite benchmark range does not show a simple unit power-law exponent in every isolated FFTLog sweep because fixed transform cost and the two additive work terms remain important. The paper should show the source-derived work model and the measured finite-range behavior separately, rather than forcing an asymptotic interpretation onto the fitted slopes.

### 9.8 Memory

Peak-memory measurements are separate evidence from runtime scaling. HarmonicTransform batching strongly reduces the growth with $N_q$ once the internal q batch is saturated. Interaction retains pair-resolved output proportional to $N_pN_D$; finite rules additionally carry the refined q-space translation/quadrature workspace, while FFTLog carries its pair-local transform workspace.

Do not turn fitted byte slopes from one machine into universal memory constants. The manuscript should emphasize the scaling structure and measured benchmark environment.

## 10. Manuscript argument

The preferred paper sequence is:

1. physical/computational motivation for four-center interactions in localized two-dimensional bases;
2. reduction from four orbitals to two transition fields;
3. circular-harmonic/Hankel derivation and separation of field, kernel, and geometry;
4. numerical realization and error-control strategy;
5. independent correctness validation;
6. automatic self-convergence reliability and refusal behavior;
7. practical capability and cost at controlled independent accuracy;
8. scaling, memory, and reuse;
9. limitations and outlook.

The paper should establish the method before discussing software choices. Performance demonstrates practicality; it is not the novelty claim.

## 11. Suggested section content

### Introduction

The introduction should motivate four-center matrix elements in localized 2D descriptions and the repeated numerical burden created by anisotropic, sign-changing, and complex orbital products. It should state the computational opportunity: a radial kernel plus localized 2D transition fields permit an angular-harmonic reduction that keeps angular structure explicit while reducing the integrations to reusable radial objects.

Do not begin with QUARTIC2D classes, backend names, or a list of software features.

The introduction must not imply that density-density interactions define the scope. A simple density example may motivate the problem, but exchange and other off-diagonal channels should appear early enough to establish why complex transition fields matter.

### Method

Start from $U_{1234}$ and introduce $\rho_{13}$ and $\rho_{42}$. Derive the momentum-space expression, circular-harmonic expansion, Hankel form factors, displacement identity, and final reduced equation. Explain reuse and computational factorization immediately after the derivation.

Only after the mathematical structure is clear should the implementation introduce PETAL2D input, q support, interpolation, radial quadrature, FFTLog, Ogata, and automatic refinement.

### Numerical validation

Open with independent reference validation, not with timing. Then test whether automatic self-convergence agrees with those references over the declared domain. Treat refusals as part of the result rather than hiding them.

The benchmark field and kernel families should be named explicitly so a reader can judge the diversity of the evidence without opening JSON files.

### Performance and scaling

Compare cost only among independently qualified calculations. Keep calibration and fixed-production timing separate. Discuss parameter reuse only after establishing that the reused configuration was independently qualified for the stated workload family.

Source-derived complexity expressions come first. Finite-range fitted slopes are evidence about the measured regime and should not replace the asymptotic model.

### Discussion

Return to the method rather than the software. Discuss what the factorization enables, where it is likely useful, which aspects are limited by the radial-kernel assumption, and how upstream field quality limits the final observable. State that backend suitability is workload-dependent.

The discussion should distinguish extensions of the mathematical method from extensions of benchmark coverage.

## 12. Main figure story

The current figure contract in `benchmarks/FIGURES.md` is appropriate and should remain scientifically ordered.

### Figure 1: method and representative calculation

Show

$$
\rho(x,y)\rightarrow\rho_m(r)\rightarrow F_m(q)\rightarrow V(\delta).
$$

This figure teaches the method. It should not carry the burden of aggregate numerical validation.

### Figure 2: direct numerical accuracy

Overlay numerical and independent analytic/reference curves with residual panels. Residuals should be pointwise peak-normalized absolute error, not pointwise relative error near zeros. Do not draw `rtol` as a pointwise residual bound.

### Figure 3: automatic error control

Compare requested target with independent global reference error, estimated refinement error with independent error where meaningful, and categorical convergence outcomes. Explain that the compared scalars are distinct metrics.

### Figure 4: accuracy versus cost

Use workload-specific panels and achieved independent error. Do not aggregate methods into a winner. Missing points must mean `not independently qualified in this declared study`, not zero cost or an implied failure outside the tested box.

### Figure 5: scaling and reusable calibration

Show source-derived work variables, measured finite-range behavior, and calibration amortization. The FFTLog guide is a work model, not a claim that every measured sweep has reached an asymptotic linear regime.

## 13. Main tables

### Table 1: true end-to-end validation

Report PETAL2D/input error, HarmonicTransform error, and final Interaction errors separately. This table is the clearest defense against conflating upstream representation error with QUARTIC2D-controlled numerical error.

### Table 2: automatic Interaction convergence

Report automatic acceptance, independent reference status, and parameter-box capability as separate columns or categories. Do not collapse them into a single `pass/fail` field.

## 14. Claims that are currently supportable

Subject to the exact benchmark definitions and a final clean provenance rerun, the current evidence supports statements of the following form:

- The four-center problem for general complex transition fields and a scalar radial kernel reduces to a sum of one-dimensional radial harmonic-pair integrals.
- QUARTIC2D implements that reduction and supports direct, exchange, pair-hopping, correlated-hopping, and other channels through the choice of transition fields.
- The implementation agrees with analytic or independently stabilized references over the declared validation portfolio.
- The automatic self-convergence machinery has no observed false acceptances in the current declared validation matrices, while conservative refusals and bounded-capability gaps are retained explicitly.
- The method remains usable over large displacement ranges, but the practical cost and backend coverage change substantially as the translation Bessel factor becomes more oscillatory.
- No one numerical backend is uniformly preferred across the tested workloads.
- Runtime follows the source-derived dimensional work models over the measured ranges to the degree shown by the scaling benchmarks.
- Calibration and fixed-production cost are separable, making reuse of independently qualified parameters useful for repeated calculations within an established workload family.

Every numerical version of these statements must carry the relevant target, population of benchmark cases, displacement regime, and qualification condition.

## 15. Claims that are not currently supportable

Do not claim any of the following without new evidence:

- a universal pointwise error bound set by `rtol`;
- rigorous or certified absolute accuracy from self-convergence alone;
- universal superiority of Simpson, GL4, FFTLog, Ogata, or any other backend;
- numerical validation for every possible four-index orbital channel simply because the mathematical reduction covers it;
- numerical validation for anisotropic, tensor-valued, spin-dependent, or explicitly position-dependent spatial kernels;
- universal accuracy for experimentally reconstructed fields without a separate study of reconstruction and noise/systematic errors;
- universal production parameters for all future workloads;
- asymptotic scaling conclusions outside the measured/source-analyzed regimes;
- novelty or `first method` claims without a dedicated literature review.

## 16. Limitations to state plainly

### Radial-kernel assumption

The current spatial reduction assumes $U(\mathbf q)=U(q)$. A genuinely anisotropic momentum-space kernel changes the angular reduction and is outside the present implementation.

### Input quality

Finite spatial extent, sampled-grid interpolation, radial resolution, PETAL2D harmonic truncation, and orbital normalization can limit the final result. QUARTIC2D cannot restore information absent from the supplied field.

### Finite momentum support

The represented $q$ interval is finite and bounded by the radial input resolution. Large-displacement observables can be sensitive to the outer q boundary. The automatic interface therefore includes an upstream-boundary robustness check for automatically sampled transformed fields.

### Method-specific parameter domains

FFTLog has coupled sensitivity to transform length and bias. Ogata has coupled sensitivity to $(N,h)$. Finite quadrature can become expensive at large displacement because of the increasingly oscillatory Bessel factor. Refusal within a declared search box is a statement about that search, not automatically about the mathematical method.

### Validation is finite

The field/kernel portfolio is intentionally broad but remains finite. Material-specific transition densities and additional singular/nonanalytic structures should be validated when they become target applications.

## 17. Likely referee questions

### What is genuinely general about the method?

Answer from the mathematics: the orbital indices enter through two arbitrary complex transition fields, while the spatial kernel is radial. Generality refers to the four-center orbital structure within that kernel class, not to arbitrary two-body operators.

### Is this only a density-density method?

No. Density-density is obtained when the transition fields reduce to orbital densities. Exchange, pair hopping, correlated hopping, and other off-diagonal channels use complex or sign-changing orbital products in the same reduced equation.

### Why not integrate the original 4D real-space expression directly?

The angular-harmonic reduction exploits radial-kernel symmetry, reduces dimensionality, exposes the angular channels, and permits reuse of transformed fields across displacement vectors and kernels. The paper should support the practical advantage through the measured cost/scaling evidence rather than a rhetorical claim.

### How is accuracy known if convergence is self-referential?

It is not inferred from self-convergence alone. The paper separately compares accepted calculations with analytic or independently stabilized references, reports refusals and misses, and distinguishes global reference metrics from the internal refinement criteria.

### Why can an accepted `rtol=10^{-4}` calculation have a larger pointwise residual?

Because `rtol` is not a pointwise reference-error bound. The selector acts on aggregate changes between numerical refinements, while plotted pointwise residuals are independent diagnostics. Upstream representation error can also contribute to end-to-end discrepancies.

### Why are several quadrature methods needed?

Their numerical behavior differs across smooth, nodal, nonanalytic, and large-displacement workloads. The method does not require a single backend. The benchmark portfolio is used to identify conditional domains of usefulness rather than to select a universal winner.

### Does a conservative automatic refusal mean the backend fails?

No. A refusal means the automatic search did not establish acceptance under its declared criteria/search box. Independent parameter-box studies can show that a reference-passing configuration exists despite the refusal. The manuscript must distinguish these cases.

### How independent are the references?

The answer depends on the benchmark. Analytic form factors are preferred where available. Interaction references use separately stabilized high-order/dense calculations and explicit reference-stability checks. The paper should state the construction for each evidence class rather than use one generic phrase.

### What is the role of PETAL2D?

PETAL2D supplies the polar/angular decomposition of the transition fields. QUARTIC2D begins from those radial harmonics, transforms them to momentum space, and evaluates the interaction. End-to-end benchmarks quantify the upstream PETAL2D/input contribution separately.

### Are the benchmark functions physically realistic?

Some are deliberately analytic or synthetic because they provide exact structure and controlled numerical difficulty. Their role is validation, not material modeling. The portfolio spans properties expected in realistic transition fields: anisotropy, sign changes, complex phase structure, nodes, cusps, oscillations, and long tails. Material-specific examples can be added separately without redefining the validation logic.

### Is the algorithm novel?

Do not answer this categorically until the literature-positioning pass is complete. The paper can state the derived formulation, implementation, validation, and practical capabilities without using `first`, `novel`, or equivalent priority language. A dedicated literature search must determine which part, if any, supports a novelty claim.

## 18. Supplementary material

The supplement should carry the evidence that is important for reproducibility but too detailed for the main scientific narrative:

- complete transition-field and kernel portfolio;
- full tolerance-response tables;
- finite-quadrature method matrix;
- method-specific convergence/capability matrices;
- FFTLog $(N,q_{\rm bias})$ capability maps;
- Ogata parameter-search details where needed;
- large-displacement reference-stability evidence;
- cross-stage matrices;
- scaling fit tables and finite-range fit intervals;
- peak-memory measurements;
- benchmark environment and exact source/provenance record.

The supplement should not become a dump of JSON field names. Tables and captions must explain the scientific meaning of each classification.

## 19. Style and terminology contract for the paper

Use `four-center interaction matrix element` as the central term. Use `transition field` for $\phi_i^*\phi_j$ unless the object is specifically a density. Use `delta` or $\delta$ consistently for interfield displacement.

Prefer ordinary scientific prose. Avoid internal development narration, `Step 1/2/3` language, source-tree descriptions, API-first exposition, slogans, and artificial `not X but Y` constructions. Do not overuse boldface, italics, semicolons, colons, or em dashes.

Figures should use conventional mathematical axis labels and concise legends. Equations defining error metrics belong in the text or caption, not in long axis labels. Captions must state the problem, plotted quantity, reference, numerical target when relevant, and the conclusion supported by the panel.

Do not use `tetra-center`; use `four-center`.

## 20. Literature positioning still required

Before drafting claims of originality, perform a dedicated literature review around:

- four-center Coulomb/integral evaluation in two-dimensional localized bases;
- circular-harmonic and Fourier-Bessel reductions of displaced 2D interactions;
- multipole/angular-momentum decompositions for two-body matrix elements;
- numerical Hankel-transform methods and accuracy control;
- FFTLog and Ogata applications to oscillatory Bessel integrals;
- computational methods used in Wannier/localized-orbital interaction calculations for 2D materials.

The computational-presentation references already used in `benchmarks/FIGURES.md` are useful models for validation/error/performance presentation, but they do not by themselves establish the scientific novelty of the four-center reduction.

## 21. Pre-submission evidence checklist

Before numerical values are frozen into the manuscript:

1. rerun the full publication suite from a clean exact Git state;
2. write terminal logs outside the tracked repository;
3. confirm the manifest reports clean reproducible provenance;
4. archive the exact consolidated JSON files used for every manuscript figure/table;
5. regenerate manuscript artifacts from those frozen JSON files only;
6. verify all numerical statements against `benchmarks/CLAIMS.md`;
7. verify that every requested tolerance is named with its controlling quantity;
8. check that no pointwise plot presents `rtol` as a pointwise bound;
9. ensure calibration cost and fixed-production cost are never mixed;
10. ensure every timed fixed configuration has an independent qualification source;
11. report missing method points as unqualified/unavailable in the declared study, not as zero or generic failure;
12. preserve stage-resolved errors in end-to-end results;
13. confirm source-derived complexity formulas match `benchmarks/COMPLEXITY.md`;
14. perform the literature-positioning review before using novelty language;
15. archive environment, package versions, thread settings, commit SHA, and DOI/release identifiers with the final dataset.

## 22. Relationship to repository documents

Use this file for manuscript reasoning and scientific continuity.

Use `benchmarks/CLAIMS.md` for the claim-to-evidence registry.

Use `benchmarks/FIGURES.md` for the exact figure/table contract.

Use `benchmarks/COMPLEXITY.md` for source-derived runtime and memory models.

Use `benchmarks/README.md` for benchmark execution and result-layout instructions.

Use `docs/source/theory/` for reader-facing derivations and `docs/source/validation/` for reader-facing numerical evidence.

If these documents disagree, resolve the scientific issue rather than documenting both formulations. The manuscript, theory docs, figures, benchmark definitions, and public terminology should converge to one canonical description.
