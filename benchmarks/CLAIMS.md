# Manuscript claim registry

This file defines what evidence is required before a numerical statement is allowed into the paper. It is a registry, not a results summary. Numerical values are filled only from completed publication-grade runs.

The Interaction section is intentionally organized around a **portfolio of quadrature backends** rather than a single preferred method. The manuscript may identify operating regimes in accuracy, displacement range, workload character, and throughput, but it must not imply that one backend is uniformly optimal.

| ID | Claim class | Defensible claim | Required benchmark | Placement |
|---|---|---|---|---|
| A1 | Reference / accuracy | `HarmonicTransform` reaches the required publication error budget for qualified sampled radial inputs; more stringent unresolved limits are reported separately | `publication/harmonic_transform.py` | Main + supplement |
| A2 | Reference / accuracy | Supported finite harmonic-transform quadratures have a documented accuracy/runtime tradeoff at the primary tolerance; method choice is reported as an operating-regime tradeoff rather than a universal ranking | `publication/harmonic_transform.py` (`harmonic-method-matrix`) | Supplement + main summary |
| A3 | Reference / accuracy | Interaction backends are compared against independently stabilized fixed-field references across practical (`1e-3`), primary (`1e-4`), and stringent (`1e-5`) accuracy targets where applicable; automatic certification, demonstrated backend capability, and reference failure are reported separately | `publication/interaction_accuracy.py` + `publication/interaction_convergence.py` | Main + supplement |
| A4 | Reference / accuracy | Large-displacement interaction references remain independently phase-resolved through the tested delta range before any backend is judged | `publication/interaction_convergence.py` | Main methods text; detailed supplement |
| A5 | Reference / accuracy | The practical large-displacement tier (`1e-3`, `1e2 <= delta <= 1e4`) is benchmarked separately from the primary `1e-4` tier so high-throughput methods are not evaluated only under a stricter-than-necessary application target | `interaction-large-practical-convergence` | Main method-selection figure/table |
| C1 | Precision / convergence | Tightening the required tolerance from `1e-3` to `1e-4` tracks achieved harmonic-transform error without folding PETAL2D/input-resolution error into the QUARTIC2D metric; `1e-5` is retained as a stringent limit sweep | `publication/harmonic_transform.py` | Main figure |
| C2 | Precision / convergence | Automatic HarmonicTransform convergence certifies the diverse analytic generality suite without false-positive certificates and explicitly refuses unresolved discontinuous slow-tail inputs | `publication/harmonic_convergence.py` | Main + supplement |
| C3 | Precision / convergence | Automatic Interaction convergence produces no false-positive certificates on the standard and large-delta validation suites | `publication/interaction_convergence.py` | Main reliability figure/table |
| C4 | Precision / convergence | At the primary publication tolerance, at least one automatically certified Interaction method covers every tested case and displacement regime | `publication/interaction_convergence.py` | Main text |
| C5 | Precision / convergence | Conservative refusals, selector misses, backend-capable-but-uncertified cases, and true reference failures are reported as distinct outcomes | `publication/interaction_convergence.py` | Main regime map + supplement |
| C6 | Precision / convergence | FFTLog parameter sweeps are interpreted as a bounded capability study in `(N, q_bias)`, not as evidence that increasing transform size monotonically improves accuracy | `publication/interaction_convergence.py` | Main/supplement parameter-study text |
| E1a | Cross-stage composition | Adaptive HarmonicTransform -> Interaction composition preserves the target accuracy against the canonical dense reference on the full standard-domain matrix for the principal finite-grid methods | `publication/cross_stage.py` (`cross-stage-standard`) | Main table + supplement |
| E1b | Cross-stage composition | At large displacement, every automatic cross-stage certificate is checked against the canonical dense reference; backend refusals and upstream q-boundary refusals are reported separately, and unsupported compositions are not emitted as successful certificates | `publication/cross_stage.py` (`cross-stage-large`) | Main table + supplement |
| E2 | True end-to-end | The actual dependency stack `PETAL2D PolarDecomposition -> HarmonicTransform -> Interaction` preserves the publication error target for representative isotropic, anisotropic, and nodal inputs on the standard-domain grid and selected large-displacement points spanning the validated large-distance interval | `publication/end_to_end.py` (`end-to-end-standard`, `end-to-end-large`) | Main figure/table + supplement |
| P1 | Performance | Automatic-convergence/calibration cost and fixed-parameter production cost are quantified separately | `publication/autoconvergence_performance.py` | Main throughput figure |
| P2 | Performance | The Interaction backends occupy distinct accuracy--cost--robustness regimes; performance claims are conditioned on requested accuracy, displacement regime, and workload class rather than reduced to a single overall winner | `publication/interaction_accuracy.py` + `publication/interaction_convergence.py` + `publication/autoconvergence_performance.py` | Main accuracy--cost figure + method-selection table |
| P3 | Performance | For repeated evaluations of a previously qualified workload family, the cost of numerical calibration can be amortized; the paper may report `t_eff(M) = t_prod + t_cal/M` using separately measured calibration and production timings, but must not describe an uncertified fixed parameter set as converged | `publication/autoconvergence_performance.py` + corresponding reference/capability benchmark | Main throughput/amortization figure |
| P4 | Performance | The practical large-displacement tier (`1e-3`) has a dedicated timing comparison on the same canonical four-case matrix used for its capability study | `interaction-large-practical-performance` + `interaction-large-practical-convergence` | Main throughput/regime figure |
| P5 | Performance | Independently reference-qualified fixed parameter sets, including conservative automatic refusals whose terminal configurations pass the reference, may be timed as production configurations without relabeling them as automatic certificates | `interaction-large-qualified-fixed-performance` + `interaction-large-practical-convergence` | Main throughput/regime figure + supplement |
| P6 | Complexity | HarmonicTransform runtime follows the source-derived dependence on N_q, N_r, N_m, and radial quadrature work | `publication/harmonic_scaling.py` | Main or supplement |
| P7 | Complexity | Interaction runtime follows the measured FFTLog/finite-grid scaling with transform size, requested separations, and harmonic count; source-derived asymptotic expectations and finite-range fitted slopes are shown separately | `publication/interaction_scaling.py` | Main or supplement |
| P8 | Memory | Peak memory growth is quantified independently from runtime | `publication/harmonic_memory.py`, `publication/interaction_memory.py` | Supplement |
| L1 | Limits | Discontinuous or otherwise uncertifiable inputs are refused rather than assigned unsupported guarantees | `publication/harmonic_convergence.py` stress cases | Supplement |
| L2 | Limits | A backend may be useful in a restricted operating regime even when it is not universally certifiable; specialist-method failures or conservative refusals are reported without converting them into a package-wide failure | `publication/interaction_convergence.py` | Main method-selection discussion + supplement |
| U1 | Upstream separation | Minimum radial input requirements are explicit and are not counted as QUARTIC2D numerical error | `publication/harmonic_transform.py` | Methods + supplement |


## Tolerance hierarchy

The three accuracy levels have different application roles and must not be presented as a single pass/fail ladder:

- `1e-3` is the **practical / throughput tier**. It is a legitimate target for exploratory calculations, dense parameter scans, fitting/optimization loops, and other workloads where sub-per-mille numerical accuracy is unnecessary. It is also the tier at which low-overhead transform methods may offer the largest practical advantage.
- `1e-4` is the **primary publication validation target**. Broad correctness, automatic certification, cross-stage composition, and end-to-end claims are gated here unless a claim explicitly names another tolerance.
- `1e-5` is the **stringent limit / verification tier**. Failure to certify `1e-5` is reported as a numerical or input-resolution limit and does not invalidate the `1e-4` claim, provided the method refuses safely rather than emitting an unsupported certificate.

A method that is appropriate at `1e-3` is not described as inferior merely because it is less broadly useful at `1e-5`. Conversely, speed at `1e-3` is not extrapolated into an accuracy claim at tighter tolerances.

## Method-selection language

Allowed manuscript language:

- "The tested quadratures occupy complementary operating regimes."
- "For this workload class and requested tolerance, method X provides lower production cost, while method Y provides broader automatic certification."
- "After independent qualification, a fixed parameter set can be reused for repeated evaluations and the calibration cost amortized."
- "The automatic selector conservatively refused this case even though the bounded capability sweep found a reference-passing configuration."

Avoid unsupported universal language such as "best method", "fastest method" without a specified workload/tolerance, or "converged because N is large" for FFTLog.

## Claims that are not allowed from a single benchmark

- A performance run cannot establish accuracy.
- Agreement between two QUARTIC2D methods is precision evidence, not an external validation reference.
- A high-resolution QUARTIC2D calculation is a numerical reference only when independent reference stability has been demonstrated.
- PETAL2D sampling/domain error is not folded into stage-isolated QUARTIC2D transform claims. In the true end-to-end benchmark it is intentionally included in the cumulative final error and is also reported separately at the PETAL2D stage.
- Automatic certification, bounded backend capability, and independent reference accuracy are distinct quantities and must not be collapsed into a single "pass rate".
- A reference-passing FFTLog point inside the tested `(N, q_bias)` box does not imply monotone convergence with `N` or validity outside that box.
- Amortized production timing is a throughput statement for a prequalified workload family; it is not permission to skip qualification for a new family of inputs.
- Historical files under `reference/` cannot be quoted as current results unless the paper explicitly identifies the archived release and environment.
