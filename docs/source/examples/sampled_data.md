# Sampled transition fields on a Cartesian grid

Electronic-structure and continuum calculations usually provide localized
states numerically rather than as closed-form functions. Wannier functions,
defect states, quantum-dot eigenstates, and continuum-model wavefunctions may
therefore enter an interaction calculation as values on a Cartesian grid.

QUARTIC2D does not require analytic input functions. To represent that workflow
while keeping an exact reference available, sample the normalized states

$$
\phi_s=\frac{e^{-r^2/2}}{\sqrt\pi},
\qquad
\phi_+=\frac{x+iy}{\sqrt\pi}e^{-r^2/2}.
$$

The forward transition field is

$$
\rho_{s+}=\phi_s^*\phi_+
=\frac{r e^{-r^2}}{\pi}e^{i\theta},
$$

so only $m=+1$ is present. The reverse field contains only $m=-1$.

## Supply arrays to PETAL2D

```python
X, Y = np.meshgrid(x, y, indexing="ij")
r2 = X**2 + Y**2
psi_s = np.exp(-0.5 * r2) / np.sqrt(np.pi)
psi_p_plus = (X + 1j * Y) * np.exp(-0.5 * r2) / np.sqrt(np.pi)

rho_forward = np.conj(psi_s) * psi_p_plus
rho_reverse = np.conj(psi_p_plus) * psi_s

dec_forward = PolarDecomposition(
    rho_forward,
    x,
    y,
    Nr=181,
    Ntheta=256,
    rmax=5.5,
    origin=(0.0, 0.0),
    interp_method="cubic",
    recon_err_tol=1.0e-2,
)
```

`recon_err_tol` is expressed in **percent**.


```{figure} ../_static/examples/sampled_data_field.svg
:class: q2d-figure q2d-figure-standard
:alt: Real and imaginary parts of the sampled complex transition field

Cartesian samples of the complex transition field. Both panels use the same
color normalization so the real and imaginary components can be compared
directly.
```

## Transform and inspect the sampled harmonic

The exact forward transform is

$$
F_1(q)=\frac{q}{4\pi}e^{-q^2/4}.
$$

```{figure} ../_static/examples/sampled_data_transform.svg
:class: q2d-figure q2d-figure-compact
:alt: Analytic and numerical transform of the sampled m equals one transition field with independent reference error

The transformed retained harmonic compared with the analytic result. The lower
panel is an independent pointwise comparison normalized by the peak analytic
transform; it is separate from the automatic self-convergence criteria.
```

```python
hcal = HarmonicTransform.converge_parameters(
    dec_forward,
    rtol=1.0e-4,
    atol=1.0e-12,
    q_tail_rtol=1.0e-3,
    method="simpson",
    verbose=False,
)
field_forward = hcal.transform(dec_forward)
hcal.plot_convergence()
```

```{figure} ../_static/examples/sampled_harmonic_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Harmonic-transform self-convergence diagnostics for sampled Cartesian input

Self-convergence diagnostics for the sampled-data harmonic transform.
```


The analytic errors reported below can exceed `rtol=1e-4` because
`rtol` compares successive QUARTIC2D calculations, whereas the analytic comparison
also includes the upstream Cartesian-to-polar representation error.

## Continue to a four-center interaction

The sampled-data path is otherwise identical to the callable-field path. After
calibrating both forward and reverse transforms, evaluate an off-diagonal
Yukawa-screened interaction:

```python
ical = Interaction.converge_parameters(
    calibration_vectors,
    field_forward,
    field_reverse,
    yukawa,
    rtol=1.0e-4,
    atol=1.0e-12,
    method="gl4",
    verbose=False,
)
interaction = ical.interaction(
    vectors,
    field_forward,
    field_reverse,
    yukawa,
)
ical.plot_convergence()
```

```{figure} ../_static/examples/sampled_interaction_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Interaction self-convergence diagnostics for sampled Cartesian input

Self-convergence diagnostics for the sampled-data interaction calibration.
```


The small imaginary parts are numerical roundoff for this symmetry-related
forward/reverse pair. The complete executable calculation is
`examples/sampled_data.py`.

```{literalinclude} ../_generated/examples/sampled_data.txt
:language: text
```
