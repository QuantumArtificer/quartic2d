# Manuscript claim registry

This file defines the evidence required before a numerical statement is used in the paper. It is an evidence registry, not a results summary. Numerical values are taken only from completed publication-grade runs.

The manuscript presents a general four-center interaction method together with its validated numerical implementation. The evidence sequence is: derive the angular-harmonic/Hankel reduction, establish correctness against independent references, evaluate the reliability of automatic self-convergence, measure cost at controlled independent accuracy, and test runtime and memory scaling. The quadrature methods are alternative numerical realizations of the same reduced interaction formula. Their comparison is always conditioned on workload, displacement range, and requested numerical target.

| ID | Claim class | Defensible claim | Required benchmark | Placement |
|---|---|---|---|---|
| A1 | Reference / accuracy | `HarmonicTransform` meets the stated independent global error target for the qualified sampled radial inputs in the tested domain; unresolved stringent-limit cases are reported separately | `harmonic_transform.py` | Main + supplement |
| A2 | Reference / accuracy | Supported finite harmonic-transform quadratures have a documented accuracy/cost tradeoff at the primary target without implying a universal method ranking | `harmonic_transform.py` (`harmonic-method-matrix`) | Supplement; supporting text for Fig. 4 |
| A3 | Reference / accuracy | Interaction methods are compared against independently stabilized fixed-field references at practical (`1e-3`), primary (`1e-4`), and stringent (`1e-5`) targets where applicable; automatic acceptance, demonstrated parameter-box capability, and independent reference failure are reported separately | `interaction_accuracy.py` + `interaction_convergence.py` | Main + supplement |
| A4 | Reference / accuracy | Large-displacement interaction references are independently phase-resolved and refinement-stable over the tested displacement interval before they are used to evaluate production configurations | `interaction_convergence.py` | Main methods text; detailed supplement |
| A5 | Reference / accuracy | The practical large-displacement tier (`1e-3`, `1e2 <= delta <= 1e4`) is benchmarked separately from the primary `1e-4` tier so performance statements identify the accuracy target to which they apply | `interaction-large-practical-convergence` | Fig. 4 + supplement |
| C1 | Automatic convergence | Requested `HarmonicTransform.rtol` values are evaluated against independent global reference errors without folding PETAL2D/input-resolution error into the isolated QUARTIC2D transform metric; `1e-5` is retained as a stringent limit sweep | `harmonic_transform.py` | Fig. S1 + Methods text |
| C2 | Automatic convergence | For the analytic generality suite, automatic `HarmonicTransform` acceptance agrees with the independent reference on accepted non-stress cases, while unresolved discontinuous slow-tail inputs are refused at the declared settings | `harmonic_convergence.py` | Main + supplement |
| C3 | Automatic convergence | In the declared standard- and large-displacement validation matrices, no automatically accepted `Interaction` result fails the independent reference check | `interaction_convergence.py` | Main reliability figure/table |
| C4 | Automatic convergence | Within the declared primary-target matrices, at least one tested `Interaction` method provides an automatically accepted, independently reference-passing result for each benchmark case; any coverage gaps of individual methods are reported explicitly | `interaction_convergence.py` | Main text |
| C5 | Automatic convergence | Conservative refusals, selector misses, demonstrated parameter-box capability without automatic acceptance, tested-box incapability, and independent reference failures are reported as distinct outcomes | `interaction_convergence.py` | Fig. 3 + supplement |
| C6 | Automatic convergence | FFTLog parameter sweeps are interpreted as bounded capability studies in `(N, q_bias)`, not as evidence that increasing transform size monotonically improves accuracy | `interaction_convergence.py` | Fig. S3 + supplement text |
| E1a | Cross-stage composition | Adaptive `HarmonicTransform -> Interaction` composition passes the independent dense reference over the full standard-domain matrix for the principal finite-grid methods, with upstream and downstream errors kept distinguishable | `cross_stage.py` (`cross-stage-standard`) | Fig. S4 + Table S4 |
| E1b | Cross-stage composition | At large displacement, automatic downstream acceptance is checked against the canonical dense reference; numerical-method refusals and upstream q-boundary refusals remain separate, and unsupported upstream representations are not converted into successful downstream results | `cross_stage.py` (`cross-stage-large`) | Figs. S4, S8 + Table S4 |
| E2 | True end-to-end | The complete `PETAL2D PolarDecomposition -> HarmonicTransform -> Interaction` calculation is validated against analytic references for representative isotropic, anisotropic, and nodal sampled inputs in the declared standard and large-displacement tests, with errors reported separately at each layer | `end_to_end.py` (`end-to-end-standard`, `end-to-end-large`) | Main figure/table + supplement |
| P1 | Performance | Automatic calibration cost and fixed-parameter production cost are measured separately | `autoconvergence_performance.py` | Fig. 5(d) + Tables S1--S3 |
| P2 | Performance | Interaction methods are compared only at independently measured accuracy; performance statements are conditioned on requested target, displacement regime, and workload class rather than reduced to one overall ordering | `interaction_accuracy.py` + `interaction_convergence.py` + `autoconvergence_performance.py` | Fig. 4 + Tables S1--S3 |
| P3 | Performance | For repeated evaluations within a previously qualified workload family, calibration cost can be amortized; `t_eff(M) = t_prod + t_cal/M` may be reported using separately measured calibration and production timings, but an unqualified fixed parameter set is never described as converged | `autoconvergence_performance.py` + corresponding reference/capability benchmark | Fig. 5(d) |
| P4 | Performance | The practical large-displacement tier (`1e-3`) has a dedicated timing comparison on the same canonical four-case matrix used for its capability study | `interaction-large-practical-performance` + `interaction-large-practical-convergence` | Fig. 4 + Table S3 |
| P5 | Performance | Independently reference-qualified fixed parameter sets may be timed as production configurations after an automatic refusal, provided the qualification source is stated and the result is not relabeled as automatic acceptance | `interaction-large-qualified-fixed-performance` + `interaction-large-practical-convergence` | Fig. 4 + Table S3 |
| P6 | Complexity | `HarmonicTransform` runtime is compared with the source-derived dependence on $N_q$, $N_r$, $N_m$, and radial quadrature work | `harmonic_scaling.py` | Fig. 5(a) + Fig. S6 |
| P7 | Complexity | `Interaction` runtime is compared with the source-derived finite-grid and FFTLog work variables; theoretical asymptotic forms and finite-range fitted slopes are reported separately | `interaction_scaling.py` | Figs. 5(b,c), S6 + Table S5 |
| P8 | Memory | Peak memory growth is measured independently from runtime | `harmonic_memory.py`, `interaction_memory.py` | Supplement |
| L1 | Limits | Inputs unresolved by the declared automatic transform search are refused rather than assigned an unsupported accuracy claim | `harmonic_convergence.py` stress cases | Supplement |
| L2 | Limits | A numerical method may be useful on a restricted workload even when the declared parameter box has incomplete coverage; method-specific refusals and failures are not converted into package-wide conclusions | `interaction_convergence.py` | Figs. 3--4 + supplement |
| U1 | Upstream separation | Minimum radial-input requirements and PETAL2D representation errors are explicit and are not counted as isolated QUARTIC2D transform error | `harmonic_transform.py` | Methods + supplement |

## Error-language contract

A requested `rtol` is an internal self-convergence threshold. It is not itself an independent reference error and it is not a pointwise relative-error guarantee.

Independent validation uses global relative $L^2$ and peak-normalized maximum errors. When a single scalar is required for a figure or table,

\[
\epsilon_{\mathrm{ref}}
=\max(\epsilon_{L^2},\epsilon_{\mathrm{peak}}).
\]

A pointwise residual normalized by the peak reference magnitude may be plotted to show where error occurs, but the requested `rtol` must not be drawn or described as a pointwise bound on that curve.

## Numerical-target hierarchy

The three requested relative targets have different roles and are not a universal quality ranking:

- `1e-3` is the practical throughput target used for exploratory calculations, dense parameter scans, fitting/optimization loops, and the dedicated large-displacement timing study.
- `1e-4` is the primary validation target for the broad standard-domain accuracy, automatic-convergence, cross-stage, and end-to-end evidence unless a claim explicitly names another value.
- `1e-5` is a stringent-limit sweep used to probe numerical and input-resolution boundaries.

A method that is useful at `1e-3` is not described as inferior because it has less coverage at `1e-5`. Conversely, speed measured at `1e-3` is not extrapolated into an accuracy statement at tighter targets.

## Method-comparison language

Acceptable manuscript language identifies the workload and the evidence being compared. For example:

- "For this workload and requested target, method X has lower qualified production cost, while method Y has broader automatic coverage in the declared search."
- "After independent qualification, a fixed parameter set can be reused for repeated evaluations and the calibration cost amortized."
- "The automatic selector refused this case, while the bounded capability sweep found an independently reference-passing configuration."

Avoid unsupported universal language such as "best method", "fastest method" without a specified workload and accuracy target, or "converged because $N$ is large" for FFTLog.

## Claims that cannot be inferred from a single benchmark

- A performance run cannot establish accuracy.
- Agreement between two QUARTIC2D methods is internal consistency evidence, not an independent validation reference.
- A high-resolution QUARTIC2D calculation is a numerical reference only when its own reference stability has been demonstrated independently of the configuration being evaluated.
- PETAL2D sampling/domain error is not folded into isolated QUARTIC2D transform claims. In the true end-to-end benchmark it is included in the cumulative calculation and also reported separately at the PETAL2D layer.
- Automatic acceptance, bounded parameter-box capability, and independent reference accuracy are distinct quantities and must not be collapsed into one pass rate.
- A reference-passing FFTLog point inside the tested `(N, q_bias)` box does not imply monotone convergence with `N` or validity outside that box.
- Amortized production timing is a throughput statement for a previously qualified workload family; it is not permission to skip qualification for a new family of inputs.
- Archived release datasets cannot be quoted as current results unless the manuscript explicitly identifies the archived release and environment.
