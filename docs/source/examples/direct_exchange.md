# Direct and exchange couplings in a two-orbital localized basis

Multiorbital effective Hamiltonians require more than an onsite or intersite
density-density repulsion. Distinct orbital products generate exchange and
other off-diagonal four-center terms, and sign changes in those products are
part of the physical matrix element rather than a numerical pathology.

A two-orbital basis already contains this distinction. Consider the normalized
orbitals

$$
\phi_s=\frac{e^{-r^2/2}}{\sqrt{\pi}},
\qquad
\phi_{p_x}=\sqrt{\frac{2}{\pi}}\,x e^{-r^2/2}.
$$

For the direct channel, the transition fields are

$$
\rho_{ss}=|\phi_s|^2,
\qquad
\rho_{pp}=|\phi_{p_x}|^2,
$$

whereas exchange uses

$$
\rho_{sp}=\phi_s^*\phi_{p_x}.
$$

The latter changes sign across the $p_x$ node. The interaction kernel is the
regular two-dimensional Yukawa form

$$
U(q)=\frac{2\pi}{\sqrt{q^2+\kappa^2}},
\qquad \kappa=0.35.
$$

## Angular content

The analytic angular structure is

$$
\rho_{ss}: m=0,
\qquad
\rho_{pp}: m=0,\pm2,
\qquad
\rho_{sp}: m=\pm1.
$$

The executable example constructs all three decompositions and records the
measured PETAL2D reconstruction errors reported below.


For an interactive calculation, PETAL2D and QUARTIC2D provide the relevant
inspection helpers directly:

```python
dec_sp.plot_harmonics_with_hist("Exchange transition field")
hcal_sp.plot_convergence()
field_sp.plot_harmonics()
```

```{figure} ../_static/examples/direct_exchange_harmonic_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Harmonic-transform self-convergence diagnostics for the exchange transition field

Self-convergence diagnostics for the exchange transition-field transform.
```

## Direct and exchange interactions

After separately calibrating the three harmonic transforms, calibrate the two
interaction channels on the production displacement domain:

```python
direct_cal = Interaction.converge_parameters(
    calibration_vectors,
    field_ss,
    field_pp,
    yukawa,
    rtol=1.0e-4,
    atol=1.0e-12,
    method="gl4",
    verbose=False,
)
exchange_cal = Interaction.converge_parameters(
    calibration_vectors,
    field_sp,
    field_sp,
    yukawa,
    rtol=1.0e-4,
    atol=1.0e-12,
    method="gl4",
    verbose=False,
)

exchange_cal.plot_convergence()
```

```{figure} ../_static/examples/direct_exchange_interaction_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Interaction self-convergence diagnostics for the exchange matrix element

Self-convergence diagnostics for the exchange interaction calibration.
```


```{figure} ../_static/examples/direct_exchange.svg
:class: q2d-figure q2d-figure-standard
:alt: Direct and exchange four-center interactions from normalized s and p_x orbitals

Direct and exchange matrix elements from the same localized orbital pair. The upper plot compares independent reference curves with QUARTIC2D values; the lower plot shows the exchange-to-direct ratio. The sign change of the exchange channel follows from the nodal transition field and is a genuine four-center effect rather than a numerical artifact.
```


```{literalinclude} ../_generated/examples/direct_exchange.txt
:language: text
```

The complete script is `examples/four_center_interaction.py`. Field
preparation, transform convergence, and interaction convergence are reported
separately for the two channels.
