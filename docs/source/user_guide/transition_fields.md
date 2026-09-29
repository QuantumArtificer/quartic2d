# Preparing transition fields

The four orbital indices enter QUARTIC2D through two localized transition
fields,

$$
\rho_{ij}(x,y)=\phi_i^*(x,y)\phi_j(x,y).
$$

PETAL2D represents each transition field by radial circular harmonics. These
radial functions are the real-space input to `HarmonicTransform`.

## Callable fields

For analytic or callable orbitals, construct the transition field directly. The
following normalized two-dimensional Gaussian orbitals produce an odd
$s$--$p_x$ transition field:

```python
import numpy as np
from petal2d import PolarDecomposition


def orbital_s(x, y):
    return np.exp(-0.5 * (x**2 + y**2)) / np.sqrt(np.pi)


def orbital_px(x, y):
    return np.sqrt(2.0 / np.pi) * x * np.exp(-0.5 * (x**2 + y**2))


def rho_sp(x, y):
    return np.conj(orbital_s(x, y)) * orbital_px(x, y)

x = np.linspace(-6.0, 6.0, 181)
y = np.linspace(-6.0, 6.0, 181)

dec_sp = PolarDecomposition(
    rho_sp,
    x,
    y,
    Nr=181,
    Ntheta=256,
    rmax=5.5,
    origin=(0.0, 0.0),
)

print(dec_sp.m_sorted)
```

For this real field PETAL2D retains the conjugate $m=\pm1$ pair, up to the exact
ordering used by the PETAL2D version.

The chosen origin defines the angular expansion center. It must be consistent
with the centers used when the displacement vectors are defined.

## Inspect the representation before transforming

PETAL2D exposes plotting helpers that should normally be used before moving to
QUARTIC2D. The radial-harmonic view shows which modes were retained and how
their power is distributed:

```python
fig, axes = dec_sp.plot_harmonics_with_hist(
    title=r"$s$--$p_x$ transition field"
)
```

```{figure} ../_static/examples/gaussian_petal2d.svg
:class: q2d-figure q2d-figure-standard
:alt: PETAL2D decomposition of a normalized isotropic Gaussian

A simple retained-harmonic plot from PETAL2D. The same helper should be used on
more complicated transition fields before any QUARTIC2D calculation is
attempted.
```

The original/reconstructed-field view is useful for sampled, anisotropic, or
sign-changing fields:

```python
fig, axes = dec_sp.plot_original_vs_reconstructions()
```

```{figure} ../_static/examples/sampled_data_field.svg
:class: q2d-figure q2d-figure-standard
:alt: Original and reconstructed sampled transition field

A sampled transition field and its PETAL2D reconstruction. This is the plot to
inspect when the input comes from numerical grid data rather than an analytic
callable.
```

These plots are not substitutes for the scalar diagnostics. Record the actual
reconstruction and represented-domain checks as well:

```python
print(f"reconstruction error [%]: {dec_sp.recon_error_measured:.3e}")
print(f"domain consistency: {dec_sp.domain_consistency:.8f}")
print("retained harmonics:", dec_sp.m_sorted)
```

`recon_err_tol` and PETAL2D's reconstruction-error attributes are expressed in
**percent**. For example, `recon_err_tol=1e-2` means a target of
$10^{-2}\%$, not a dimensionless relative error of $10^{-2}$.

## Sampled numerical fields

For orbitals obtained numerically, first form the transition field on a common
uniform Cartesian grid:

```python
rho_ij = np.conj(psi_i) * psi_j

dec_ij = PolarDecomposition(
    rho_ij,
    x,
    y,
    interp_method="cubic",
)
```

`dec_ij.recon_error_measured` measures the retained-harmonic reconstruction on
the polar grid. `dec_ij.domain_consistency` compares represented polar-domain
power with the Cartesian input power. These checks precede QUARTIC2D refinement:
downstream momentum-space calculations cannot reconstruct information absent
from the sampled field.

The PETAL2D documentation gives the exact array-shape, Cartesian-grid,
centering, and interpolation requirements. The QUARTIC2D sampled-data example
in {doc}`../examples/sampled_data` shows the array-to-interaction path, including an
independent analytic transform comparison before the four-center calculation.

## Angular expansion

PETAL2D represents

$$
\rho(r,\theta)=\sum_m\rho_m(r)e^{im\theta}.
$$

A real field has conjugate $+m$ and $-m$ channels. A complex transition field
need not. This is why the two transition fields in a general four-center matrix
element should not be treated as ordinary positive densities.

If the first and second fields contain $N_m^{(1)}$ and $N_m^{(2)}$ retained
harmonics, the final interaction contains

$$
N_p=N_m^{(1)}N_m^{(2)}
$$

harmonic pairs. Angular truncation therefore affects both represented field
accuracy and downstream cost.

## What to check before transforming

Before building `HarmonicTransform`, inspect:

- the requested and measured PETAL2D reconstruction error.
- retained harmonics and their power fractions.
- `domain_consistency`.
- radial support/cutoff diagnostics.
- angular and radial sampling limits.
- the field visually when interpolation, centering, nodes, or complex phase are
  important.

A QUARTIC2D convergence result concerns the numerical transform of the field
PETAL2D supplied. It does not establish the accuracy of the upstream
electronic-structure calculation, finite Cartesian domain, orbital sampling, or
PETAL2D interpolation.

For related orbital families, determine reusable settings from the difficult or
spatially extended members rather than assuming that one field represents the
entire family. See {ref}`family-calibration`.
