# Four-center matrix elements

Projecting an interacting continuum problem onto localized orbitals produces a
four-index interaction tensor. In Hubbard-like and multiorbital models, its
entries become the direct repulsion, exchange, pair-hopping,
correlated-hopping, and longer-range interaction parameters that control the
many-body Hamiltonian. Computing that tensor requires the spatial orbital
products, not only the orbital densities.

QUARTIC2D evaluates

$$
U_{1234}=\iint d^2\mathbf r\,d^2\mathbf r'\,
\phi_1^*(\mathbf r)\phi_2^*(\mathbf r')
U(|\mathbf r-\mathbf r'|)
\phi_3(\mathbf r)\phi_4(\mathbf r').
$$

The orbitals $\phi_i$ are localized scalar functions in two dimensions. They may be real or complex. The kernel $U$ is assumed to be translationally invariant and radial in the in-plane coordinates.

Define two transition fields,

$$
\rho_{13}(\mathbf r)=\phi_1^*(\mathbf r)\phi_3(\mathbf r),
\qquad
\rho_{42}(\mathbf r)=\phi_4^*(\mathbf r)\phi_2(\mathbf r).
$$

The matrix element becomes

$$
U_{1234}=\iint d^2\mathbf r\,d^2\mathbf r'\,
\rho_{13}(\mathbf r)
U(|\mathbf r-\mathbf r'|)
\rho_{42}^*(\mathbf r').
$$

This is the input structure used by `Interaction`. Neither transition field is required to be positive, real, or equal to the other. The many-body label attached to a matrix element follows from the orbital indices, not from a separate numerical mode in QUARTIC2D.

## Common orbital-index choices

| Physical term | Four-index element | First transition field | Second transition field |
| --- | --- | --- | --- |
| direct / density-density | $U_{ijij}$ | $\phi_i^*\phi_i$ | $\phi_j^*\phi_j$ |
| exchange | $U_{ijji}$ | $\phi_i^*\phi_j$ | $\phi_i^*\phi_j$ |
| pair hopping | $U_{iijj}$ | $\phi_i^*\phi_j$ | $\phi_j^*\phi_i$ |
| correlated hopping | e.g. $U_{iiij}$ | $\phi_i^*\phi_i$ | $\phi_j^*\phi_i$ |

The table lists common many-body names. QUARTIC2D receives only the two transition fields, the radial kernel, and the relative displacement; no separate interaction-type switch is required.

## Displaced localized states

The two transition fields may be centered at different sites. QUARTIC2D represents their relative in-plane displacement by

$$
\boldsymbol\delta=(\delta_x,\delta_y)
=\delta(\cos\phi_\delta,\sin\phi_\delta).
$$

`Interaction` accepts an array of Cartesian vectors with shape `(D, 2)`:

```python
deltas = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])
print(deltas.shape)
```

Output:

```text
(3, 2)
```

Each row is evaluated independently. Angular dependence is retained through the harmonic-pair phase factors, so two vectors with the same magnitude can give different values for anisotropic transition fields.

## Kernel convention

The package expects the momentum-space radial kernel $U(q)$, with

$$
U(\mathbf R)=\int\frac{d^2\mathbf q}{(2\pi)^2}
U(q)e^{i\mathbf q\cdot\mathbf R}.
$$

Examples include

$$
U(q)\propto \frac{1}{q}
$$

for a bare 2D Coulomb form, screened kernels such as Rytova-Keldysh {cite:p}`Rytova1967,Keldysh1979`, gate-screened interactions, interlayer kernels, Yukawa forms, and other scalar radial functions.

The callable must accept NumPy arrays:

```python
kappa = 0.4


def U_q(q):
    return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)

print(U_q(np.array([0.0, 1.0])))
```

Output:

```text
[15.70796327  5.83379110]
```

QUARTIC2D does not impose a unit system. The real-space coordinates, momentum coordinates, kernel, and orbital normalization must use mutually consistent units. If real-space coordinates are measured in nanometers, for example, the momentum variable is in inverse nanometers and the kernel must be written in the corresponding convention.

See {doc}`../theory/conventions` for the complete Fourier normalization used by the package.


The benchmark suite contains the explicit kernel families and parameter values in {eq}`eq-benchmark-kernels` and {eq}`eq-benchmark-rpa`. Use those definitions when comparing a production kernel with the numerical regimes discussed in {doc}`numerical_methods`.
