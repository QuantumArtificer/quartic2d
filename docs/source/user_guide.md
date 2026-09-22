# User guide

## PETAL2D input

`HarmonicTransform` accepts a `petal2d.PolarDecomposition`. QUARTIC2D reads its retained radial harmonics, radial grid, power fractions, and `cutoff_radius` values without modifying the PETAL2D object.

## Normal Step-2 use

```python
from quartic2d import HarmonicTransform

transformed = HarmonicTransform(dec)
```

For each retained harmonic $m$, QUARTIC2D evaluates

$$
F_m(q)=\int_0^\infty r\,\rho_m(r)J_{|m|}(qr)\,dr.
$$

The default path performs one transform pass. It does not hide an automatic convergence study inside object construction.

### `q_max`: represented momentum range

`q_max` is the upper end of the momentum interval. QUARTIC2D can estimate it from the momentum scale implied by the radial profiles. The estimate is kept below the radial Nyquist ceiling $\pi/\Delta r$, because interpolation cannot create momentum information beyond the original PETAL2D radial sampling.

A too-small `q_max` can truncate a form factor before it has decayed. A too-large `q_max` wastes work and can approach a momentum range unsupported by the input grid.

### `n_q`: tabulation density

`n_q` is the number of uniformly spaced q samples between zero and `q_max`. It controls how accurately QUARTIC2D can interpolate between the directly transformed q points. It does not refine the radial integral itself.

The default grid is deliberately bounded in size so that normal use has predictable runtime and memory. If a particular profile contains unusually fine q-space structure, the fast diagnostics warn and the explicit convergence helper can determine a larger grid.

### `subdivisions`: radial quadrature refinement

PETAL2D supplies samples on a radial grid. Finite-grid quadratures evaluate the integral on a refined version of that grid. `subdivisions=2` splits each original PETAL2D radial interval once before composite Simpson integration. Increasing it generally improves radial integration accuracy at approximately linear extra cost.

Simpson with `subdivisions=2` is the benchmarked release default for the standard accuracy target. GL4, GL8, and trapezoidal finite-grid backends remain available.

## Fast fault detectors

After the transform, QUARTIC2D inspects quantities already in memory. These checks are designed to be negligible compared with the Hankel calculation itself.

Two q-boundary indicators are reported per harmonic when needed:

**Boundary amplitude ratio**
: The largest $|F_m(q)|$ found in the final 5% of the sampled q interval, divided by that harmonic's peak $|F_m|$. A large value means the form factor has not visibly decayed near `q_max`.

**Boundary power fraction**
: The fraction of the *represented* q-weighted L2 norm, $\int q|F_m(q)|^2dq$, that lies in the final 10% of the sampled q interval. This is a boundary-activity indicator. It is **not** an estimate of the norm omitted beyond `q_max`.

QUARTIC2D also checks whether `q_max` approaches the momentum ceiling implied by the PETAL2D radial spacing, whether the q grid violates the basic sampling bound associated with the retained real-space support, and whether PETAL2D reports radial support reaching the edge of its available radial domain.

A warning means "this result deserves a convergence check", not "the calculation is definitely wrong".

The diagnostics are available programmatically:

```python
transformed.diagnostics.healthy
transformed.diagnostics.to_dict()
```

## Quantitative convergence

Use convergence when a fast warning appears, when you need a documented numerical tolerance, or when you want to optimize a repeated production calculation for speed or memory.

```python
calibration = HarmonicTransform.converge_parameters(
    dec,
    rtol=1e-4,
    atol=1e-12,
    q_tail_rtol=1e-3,
)
```

The helper verifies three separate choices:

1. **q support** -- `q_max` is enlarged until every retained harmonic leaves no more than `q_tail_rtol` relative L2 norm outside the represented q interval.
2. **q-grid interpolation** -- `n_q` is increased until off-grid direct transform checks agree with interpolation to `rtol` / `atol`.
3. **radial quadrature** -- each retained harmonic is independently refined in `subdivisions` until successive radial integrations agree to `rtol` / `atol`.

This distinction is important: `q_tail_rtol` measures truncation of the represented momentum domain, whereas `rtol` and `atol` measure numerical discretization inside that domain.

The helper prints a compact report and returns reusable parameters:

```python
transformed = calibration.transform(dec)
```

No convergence work is repeated by that production transform.

## Plotting Step 2

```python
transformed.plot()
```

plots all retained transformed harmonics. If the transform was built from a convergence result,

```python
transformed.plot(show_convergence=True)
```

also plots the q-support, q-grid interpolation, and radial-quadrature convergence histories.

## Interaction transforms

`Interaction` evaluates a radially symmetric momentum-space kernel `U_q(q)` for one or more displacement vectors. GL4 is the benchmarked general-purpose interaction backend and the public default. Simpson, GL8, and trapezoidal finite-grid methods remain available; FFTLog is the high-throughput option for compatible qualified workloads; and Ogata is a public oscillatory specialist using the optional `hankel` dependency (`pip install "quartic2d[ogata]"`).

```python
from quartic2d import Interaction

interaction = Interaction(deltas, transformed, transformed, U_q)
```

Ogata can be selected directly from the public API:

```python
interaction = Interaction(
    deltas, transformed, transformed, U_q,
    method="ogata", N=32768, h=1.953125e-4,
)
```

For unfamiliar workloads, prefer `Interaction.converge_parameters(..., method="ogata")` over guessing `N` and `h`. The coupled search checks both parameters on the assembled interaction.

For production calculations, use `Interaction.converge_parameters(...)` to
calibrate the interaction backend.  If `transformed` was built from
`HarmonicTransform.converge_parameters(...)`, Interaction also performs a
downstream q-boundary robustness probe: it smoothly tapers only the outer part
of the automatically selected q interval and verifies that the requested
interaction is unchanged within a reserved fraction of the tolerance.  A
backend may therefore be numerically self-converged but still be refused with
`status="upstream_q_boundary_not_robust"` when the finite upstream q support is
not adequate for the requested displacement range.  Fixed user-supplied
momentum fields without an automatic q-sampling certificate retain the legacy
represented-input semantics.

If the first transformed field contains $M_1$ harmonics, the second contains $M_2$ harmonics, and $D$ displacements are supplied, the principal output shapes are

```text
Phi_mm : (M1, M2, D)
H_mm   : (M1, M2, D)
V_mm   : (M1, M2, D)
V      : (D,)
```

The total interaction is

$$
V(\boldsymbol\Delta)=\sum_{m,m'}V_{mm'}(\boldsymbol\Delta).
$$
