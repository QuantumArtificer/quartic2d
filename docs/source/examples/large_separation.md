# Long-range Coulomb matrix elements between distant localized states

Extended Hubbard and other long-range lattice models require interaction
matrix elements beyond the nearest neighbor. Establishing where those terms can
be truncated means evaluating the interaction over several lattice spacings,
where the matrix element is small but may still influence collective behavior.

Large separations also make the translation factor highly oscillatory in
momentum space. A normalized Gaussian with the bare Coulomb kernel provides an
analytic reference while exposing this numerical regime.

For

$$
\rho(r)=\frac{e^{-r^2}}{\pi},
$$

the exact interaction is

$$
V(\delta)
=
\sqrt{\frac{\pi}{2}}\,
I_0\!\left(\frac{\delta^2}{4}\right)
\exp\!\left(-\frac{\delta^2}{4}\right),
$$

implemented stably with `scipy.special.i0e`. At large separation,

$$
V(\delta)\sim\frac{1}{\delta}.
$$

## Calibrate the represented field first

The harmonic transform is calibrated independently of the later large-$\delta$
interaction:

```python
hcal = HarmonicTransform.converge_parameters(
    dec,
    rtol=1.0e-4,
    atol=1.0e-12,
    q_tail_rtol=1.0e-3,
    method="simpson",
    verbose=False,
)
field = hcal.transform(dec)
hcal.plot_convergence()
```

```{figure} ../_static/examples/large_harmonic_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Harmonic-transform self-convergence diagnostics for the long-range Gaussian example

Self-convergence diagnostics for the represented Gaussian field.
```


Again, the independent analytic error and the internal refinement target are
different quantities.

## Evaluate the oscillatory interaction

The production domain spans

$$
10^2\le\delta\le10^4.
$$

A finite GL4 sequence is calibrated directly on that domain:

```python
gl4_cal = Interaction.converge_parameters(
    vectors,
    field,
    field,
    coulomb,
    rtol=1.0e-3,
    atol=1.0e-12,
    method="gl4",
    subdivisions=(32, 64, 128, 256, 512, 1024),
    verbose=False,
)
gl4 = gl4_cal.interaction(vectors, field, field, coulomb)
gl4_cal.plot_convergence()
```

```{figure} ../_static/examples/large_interaction_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: GL4 self-convergence diagnostics over the large-separation domain

Self-convergence diagnostics for the GL4 large-separation interaction.
```

FFTLog {cite:p}`Hamilton2000` is also evaluated as a specialist oscillatory method. Ogata quadrature {cite:p}`Ogata2005` is attempted
only when the optional `hankel` dependency is installed.


```{figure} ../_static/examples/large_delta_methods.svg
:class: q2d-figure q2d-figure-standard
:alt: Large-separation Coulomb interaction compared with an analytic Gaussian reference

The upper plot compares GL4, FFTLog, and Ogata evaluations with the analytic Gaussian Coulomb interaction and its $1/\delta$ asymptote. The middle plot shows the pointwise absolute residual normalized by the peak analytic interaction. The lower plot shows the finite-separation correction $\delta V_{\mathrm{C}}-1$ and its leading asymptotic form. The purpose is to expose the oscillatory regime and validate specific configurations, not to establish a universal method ranking.
```


```{literalinclude} ../_generated/examples/large_separation.txt
:language: text
```

The complete script is `examples/large_separation.py`. Publication-scale method
coverage and timing studies are kept separately in {doc}`../validation/index`.
