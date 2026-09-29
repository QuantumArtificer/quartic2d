# Complex localized orbitals and off-diagonal Coulomb matrix elements

Valley, angular-momentum, magnetic, and other phase-carrying orbital bases
naturally produce complex transition fields. The corresponding off-diagonal
interaction matrix elements cannot be reduced to products of positive real
densities; their relative phases are part of the four-center integral.

The orbitals

$$
\phi_s=\frac{e^{-r^2/2}}{\sqrt\pi}
$$

and

$$
\phi_h=
\frac{e^{-r^2/2}}{2\sqrt\pi}
\left[
\sqrt2(1-r^2)
+i r e^{i\theta}
+\frac{r^2}{\sqrt2}e^{-2i\theta}
\right].
$$

The transition field $\rho_{sh}=\phi_s^*\phi_h$ contains exactly
$m=0,+1,-2$, while $\rho_{hs}=\rho_{sh}^*$ contains $m=0,-1,+2$.

## Transform the complex channels

The analytic transforms are

$$
F_{sh,0}=\frac{q^2}{8\sqrt2\pi}e^{-q^2/4},
\qquad
F_{sh,1}=\frac{i q}{8\pi}e^{-q^2/4},
\qquad
F_{sh,-2}=\frac{q^2}{16\sqrt2\pi}e^{-q^2/4}.
$$

The $ss$, $sh$, and $hs$ transforms are converged independently.


Those are independent analytic errors, not the internal self-convergence
quantity. The selected transform can be inspected directly:

```python
hcal["sh"].plot_convergence()
fields["sh"].plot_harmonics()
```

```{figure} ../_static/examples/complex_harmonic_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Harmonic-transform self-convergence diagnostics for the complex transition field

Self-convergence diagnostics for the $sh$ harmonic transform.
```

```{figure} ../_static/examples/complex_orbital_hankel.svg
:class: q2d-figure q2d-figure-standard
:alt: Harmonic transforms of a complex mixed-angular transition field

The three retained angular channels are transformed separately. Solid curves
show the analytic components and open markers show QUARTIC2D values; the lower
panel gives the independent peak-normalized transform error.
```

## Build exchange, pair-hopping, and correlated-hopping channels

Using a dual-gate kernel,

$$
U(q)=\frac{2\pi}{q}\tanh(qd),
$$

three representative off-diagonal channels are

```python
channels = {
    "exchange": (fields["sh"], fields["sh"]),
    "pair hopping": (fields["sh"], fields["hs"]),
    "correlated hopping": (fields["ss"], fields["hs"]),
}

ical = {
    name: Interaction.converge_parameters(
        calibration_vectors,
        first,
        second,
        dual_gate,
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    for name, (first, second) in channels.items()
}

ical["exchange"].plot_convergence()
```

```{figure} ../_static/examples/complex_interaction_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Interaction self-convergence diagnostics for the exchange channel

Self-convergence diagnostics for the exchange interaction calibration.
```


```{figure} ../_static/examples/complex_orbital_channels.svg
:class: q2d-figure q2d-figure-standard
:alt: Complex off-diagonal interaction channels under dual-gate screening

The top plot compares the bare Coulomb and dual-gate kernels. The remaining plots show the magnitude and unwrapped phase of the exchange, pair-hopping, and correlated-hopping matrix elements. Complex transition fields enter the same four-center harmonic representation as real transition fields.
```


```{literalinclude} ../_generated/examples/complex_transition_field.txt
:language: text
```

The complete script is `examples/complex_transition_field.py`.
