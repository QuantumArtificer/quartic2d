# Manuscript claim registry

This file defines what evidence is required before a numerical statement is allowed into the paper. It is a registry, not a results summary. Numerical values are filled only from a completed canonical publication run.

| ID | Claim class | Defensible claim | Required benchmark | Placement |
|---|---|---|---|---|
| A1 | Reference / accuracy | `HarmonicTransform` reaches the requested QUARTIC2D error budget for qualified sampled radial inputs | `publication/harmonic_transform.py` | Main + supplement |
| A2 | Reference / accuracy | Supported finite harmonic-transform quadratures have a documented accuracy/runtime tradeoff at the primary tolerance, with Simpson and GL4 as the principal finalists | `publication/harmonic_transform.py` (`harmonic-method-matrix`) | Supplement + main summary |
| A3 | Reference / accuracy | Automatically certified Interaction results are checked against independently converged fixed-field references over representative kernels and difficult fields; backend capability and conservative refusals are reported separately | `publication/interaction_accuracy.py` (`interaction-accuracy`, `interaction-method-matrix`) | Main + supplement |
| A4 | Reference / accuracy | Large-displacement interaction references remain independently phase-resolved through the tested delta range | `publication/interaction_convergence.py` | Main methods text; detailed supplement |
| C1 | Precision / convergence | Requested tolerance tracks achieved harmonic-transform error without folding PETAL2D/input-resolution error into the QUARTIC2D metric | `publication/harmonic_transform.py` | Main figure |
| C2 | Precision / convergence | Automatic HarmonicTransform convergence certifies the diverse analytic generality suite without false-positive certificates and explicitly refuses unresolved discontinuous slow-tail inputs | `publication/harmonic_convergence.py` | Main + supplement |
| C3 | Precision / convergence | Automatic Interaction convergence produces no false-positive certificates on the standard and large-delta validation suites | `publication/interaction_convergence.py` | Main figure/table |
| C4 | Precision / convergence | At least one automatically certified Interaction method covers every tested case and displacement regime at the publication tolerance | `publication/interaction_convergence.py` | Main text |
| C5 | Precision / convergence | Conservative refusals and selector misses are reported separately from incorrect certificates | `publication/interaction_convergence.py` | Supplement |
| E1 | Cross-stage composition | Adaptive HarmonicTransform -> Interaction composition preserves the target accuracy against the canonical dense reference on the full standard-domain matrix and representative large-displacement cases | `publication/cross_stage.py` (`cross-stage-standard`, `cross-stage-large`) | Main table + supplement |
| E2 | True end-to-end | The actual dependency stack `PETAL2D PolarDecomposition -> HarmonicTransform -> Interaction` preserves the publication error target for representative isotropic, anisotropic, and nodal inputs on standard and large displacement domains | `publication/end_to_end.py` (`end-to-end-standard`, `end-to-end-large`) | Main figure/table + supplement |
| P1 | Performance | Automatic-convergence cost and selected production cost are quantified separately | `publication/autoconvergence_performance.py` | Main figure |
| P2 | Performance | Standard-domain finite-grid methods and large-delta FFTLog/finite-grid regimes have distinct performance tradeoffs | `publication/autoconvergence_performance.py` + `publication/interaction_convergence.py` | Main figure |
| P3 | Complexity | HarmonicTransform runtime follows the source-derived dependence on N_q, N_r, N_m, and radial quadrature work | `publication/harmonic_scaling.py` | Main or supplement |
| P4 | Complexity | Interaction runtime follows the measured FFTLog/finite-grid scaling with transform size, requested separations, and harmonic count | `publication/interaction_scaling.py` | Main or supplement |
| P5 | Memory | Peak memory growth is quantified independently from runtime | `publication/harmonic_memory.py`, `publication/interaction_memory.py` | Supplement |
| L1 | Limits | Discontinuous or otherwise uncertifiable inputs are refused rather than assigned unsupported guarantees | `publication/harmonic_convergence.py` stress cases | Supplement |
| U1 | Upstream separation | Minimum radial input requirements are explicit and are not counted as QUARTIC2D numerical error | `publication/harmonic_transform.py` | Methods + supplement |

## Claims that are not allowed from a single benchmark

- A performance run cannot establish accuracy.
- Agreement between two QUARTIC2D methods is precision evidence, not an external validation reference.
- A high-resolution QUARTIC2D calculation is a numerical reference only when independent reference stability has been demonstrated.
- PETAL2D sampling/domain error is not folded into stage-isolated QUARTIC2D transform claims. In the true end-to-end benchmark it is intentionally included in the cumulative final error and is also reported separately at the PETAL2D stage.
- Automatic certification and reference accuracy are distinct: both must be reported when making reliability claims.
- Historical files under `reference/` cannot be quoted as current results unless the paper explicitly identifies the archived release and environment.
