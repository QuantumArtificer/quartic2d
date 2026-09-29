# Dielectric screening of multiorbital interaction channels

In two-dimensional materials, substrate choice, encapsulation, nearby gates,
and dielectric engineering can modify the effective interaction without
changing the underlying localized orbitals. Parameter studies therefore often
require the same orbital matrix elements over a family of screening lengths or
dielectric environments.

The complex-orbital fields from {doc}`complex_transition_field` are retained
while only the radial interaction kernel changes. Once the transition fields
have been transformed, a family of radial kernels can be evaluated without
repeating the PETAL2D decomposition.

For Rytova–Keldysh screening {cite:p}`Rytova1967,Keldysh1979`,

$$
U(q;r_0)=\frac{2\pi}{q(1+r_0q)}.
$$

We vary $r_0$ while keeping the three channels

- exchange,
- pair hopping,
- correlated hopping

fixed.

## Calibrate a bounded kernel family

A production sweep should not silently assume that one kernel calibration is
valid for every parameter value. The example therefore calibrates representative
screening lengths

$$
r_0=0.1,\ 1,\ 10
$$

for each channel and takes the most refined finite-rule setting found within
that bounded family.

```python
representative_r0 = (0.1, 1.0, 10.0)

calibrations = {
    (name, r0): Interaction.converge_parameters(
        calibration_vectors,
        first,
        second,
        rytova_keldysh(r0),
        rtol=1.0e-4,
        atol=1.0e-12,
        method="gl4",
        verbose=False,
    )
    for name, (first, second) in channels.items()
    for r0 in representative_r0
}

# Inspect one representative search directly.
calibrations[("exchange", 1.0)].plot_convergence()
```

```{figure} ../_static/examples/screening_interaction_convergence.svg
:class: q2d-figure q2d-figure-standard
:alt: Interaction self-convergence diagnostics for a representative screened exchange channel

Self-convergence diagnostics for the representative $r_0=1$ exchange calibration.
```


This is a **family-specific reuse decision**, not a universal statement about
all Rytova–Keldysh parameters or all transition fields.

## Reuse the calibrated resolution

The public example uses a modest $31\times41$ screening/displacement grid so it
remains suitable as a tutorial rather than a benchmark workload:

```python
r0_values = np.geomspace(0.1, 10.0, 31)
delta = np.linspace(0.0, 4.0, 41)
```

It evaluates three interaction channels, for 3813 matrix elements in total.


```{figure} ../_static/examples/rk_multichannel_map.svg
:class: q2d-figure q2d-figure-standard
:alt: Screening-length and displacement dependence of three multiorbital interaction channels on a shared magnitude scale

Magnitude of the exchange, pair-hopping, and correlated-hopping channels versus screening length and displacement. All three plots use the same logarithmic color normalization, so color has the same quantitative meaning for every channel. The figure visualizes parameter reuse across the bounded screening family.
```

```{figure} ../_static/examples/rk_multichannel_cuts.svg
:class: q2d-figure q2d-figure-standard
:alt: Representative screening-length cuts of the multiorbital interaction channels

Representative cuts at $\delta=0,1,2,$ and $4$ show how the magnitude of each interaction channel changes with screening length. The same displacement styles are used in all three plots.
```


```{literalinclude} ../_generated/examples/screening_family_sweep.txt
:language: text
```

The complete calculation is `examples/screening_family_sweep.py`. The larger
publication stress sweeps remain under `benchmarks/` and are intentionally not
part of this guided example.
