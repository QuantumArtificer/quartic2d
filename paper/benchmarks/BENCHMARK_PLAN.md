# Manuscript benchmark plan and evidence status

| Manuscript claim | Required evidence | Existing evidence | Status |
|---|---|---|---|
| Sampled Hankel backends are accurate and controllable | Analytic/independent references, disjoint calibration/validation, runtime/error tradeoff | `benchmarks/reference/paper_numerics_v0.1.0.json` | Available |
| Interaction backends have a clear accuracy/runtime tradeoff | Held-out interaction references and calibrated methods | `paper_numerics_v0.1.0.json` | Available |
| Requested tolerance predicts achieved Quartic2D error | q-tail + in-domain error budget evaluated against the same sampled input | revised `run_harmonic_transform.py` | Run on reference machine |
| Multi-harmonic `HarmonicTransform` is accurate and fast | End-to-end constructor timing over representative classes | revised `run_harmonic_transform.py` | Run on reference machine |
| Quartic2D benchmark is not contaminated by PETAL2D input error | Sampled-input reference plus separately recorded input qualification | revised `run_harmonic_transform.py` | Implemented |
| Minimum radial input required by each Quartic2D benchmark is explicit | `N_r`, `r_max`, `dr`, q-Nyquist margin, per-mode input L2 qualification | revised `run_harmonic_transform.py` | Implemented; PETAL2D generation cost belongs in supplement |
| Package defaults are robust for routine use | One frozen default policy applied unchanged to the representative suite | Defaults not yet frozen | Pending API/default freeze |
| Automatic convergence selects trustworthy production parameters | Run public convergence helper, verify selected output against independent truth | q-support/q-grid development benchmark exists | Pending final convergence-helper API |
| Automatic convergence is computationally efficient | Screen methods with one-shot convergence timing, then measure finalists with repeated wall time, selected production time, lowest/highest candidate production timing, evaluation count, and cProfile hot spots | `run_autoconvergence_screen.py`; `run_autoconvergence_performance.py` | Screen implemented; finalist benchmark pending |
| Full interaction pipeline preserves accuracy | Several PETAL2D -> harmonic transform -> interaction cases with stage-resolved references | one connected case exists | Partial; supplement/main table after API freeze |
| Difficult inputs remain numerically controllable | cusp, oscillatory, algebraic, discontinuous cases with explicit cost/error | lower-level stress coverage + revised end-to-end runner | Reference-machine run needed |
| Step-2 `HarmonicTransform` follows predicted computational complexity | source-derived asymptotics plus independent `N_q`, `N_r`, `N_m`, `s_r` sweeps and work collapse | source derivation + `run_step2_scaling.py` | Run on reference machine |
| Interaction follows predicted computational complexity | source-derived asymptotics + FFTLog/finite scaling sweeps | `run_interaction_scaling.py` | Implemented; reference-machine run needed |
| Memory growth is predictable | peak-RSS scaling against dominant dimensions | harmonic + interaction memory runners | Harmonic run complete; interaction reference-machine run needed |

## Core paper figures

1. **Harmonic-transform tolerance response**: `epsilon_Q2D` and `t` versus `epsilon_req`.
2. **Interaction backend Pareto front**: calibrated `max epsilon_2` versus `t`.
3. **Step-2 computational scaling**: `N_q`, `N_r`, `N_m`, `s_r`, plus collapse versus `W=N_m N_q N_s`.
4. **Interaction benchmarks/scaling**: fixed-field accuracy/runtime and FFTLog/finite complexity scaling.
5. **Stress/limits**: challenging classes, Quartic2D error, runtime, and required radial-input resolution.

A defaults-versus-tuned panel should be added only after package defaults are frozen. A PETAL2D input-resolution/performance panel belongs in the supplemental PETAL2D benchmark section and should use the `input_requirements` emitted here as its targets.

## Error definitions used in paper figures

- `epsilon_Q2D`: Quartic2D total relative L2 error for the supplied sampled input,
  `sqrt(epsilon_q,tail^2 + epsilon_in^2)`.
- `epsilon_in`: in-domain relative L2 error of Quartic2D against an independent
  transform of the same sampled radial input, normalized by the full sampled
  radial/Hankel norm.
- `epsilon_inf,pk`: maximum in-domain absolute error normalized by peak
  reference magnitude.
- `epsilon_q,tail`: relative L2 norm of the sampled-input transform omitted
  above `q_max`, obtained from Parseval.
- `epsilon_input`: external representation error of the supplied radial input
  against the analytic benchmark class. It qualifies the input and is never
  included in `epsilon_Q2D`.

Pointwise relative error is not the primary metric when the exact result crosses
or approaches zero.

## Numerical method status for the next benchmark cycle

| Role | Methods | Status |
|---|---|---|
| Radial-transform default | Simpson | Frozen |
| Principal finite-grid comparison | GL4 | Frozen |
| Additional finite-grid candidates | trapezoid, GL8 | Retained pending automatic-convergence profiling |
| Interaction throughput method | FFTLog | Retained; accuracy and convergence regime reported separately |
| Experimental specialist method | Ogata | Benchmark-only; excluded from the supported public API |

Method retention is evaluated using reference accuracy where available, convergence reliability, production runtime, and automatic-convergence cost. Trapezoid and GL8 remain in the benchmark matrix until profiling establishes whether either provides a non-dominated operating regime.
