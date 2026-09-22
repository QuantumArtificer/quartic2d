# QUARTIC2D

[![tests](https://github.com/QuantumArtificer/quartic2d/actions/workflows/tests.yml/badge.svg)](https://github.com/QuantumArtificer/quartic2d/actions/workflows/tests.yml)
[![Documentation](https://github.com/QuantumArtificer/quartic2d/actions/workflows/docs.yml/badge.svg)](https://quantumartificer.github.io/quartic2d/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

QUARTIC2D is a Python package for evaluating interaction matrix elements between localized two-dimensional fields with radially symmetric interaction kernels. It is built on top of [PETAL2D](https://github.com/QuantumArtificer/petal2d): PETAL2D supplies the angular-harmonic decomposition of the fields, while QUARTIC2D evaluates the radial Hankel transforms and harmonic-resolved interaction integrals.

For a PETAL2D decomposition

$$
\rho(r,\theta)=\sum_m \rho_m(r)e^{im\theta},
$$

QUARTIC2D computes

$$
F_m(q)=\int_0^\infty r\,\rho_m(r)J_m(qr)\,dr
$$

and combines the transformed harmonics to evaluate displaced interactions for radially symmetric momentum-space kernels $U(q)$.

## Installation

```bash
python -m pip install quartic2d
```

Ogata quadrature is a supported optional backend. Install its dependency with

```bash
python -m pip install "quartic2d[ogata]"
```

For development:

```bash
git clone https://github.com/QuantumArtificer/quartic2d.git
cd quartic2d
python -m pip install -e ".[test,docs,release,ogata]"
```

QUARTIC2D supports Python 3.10--3.13 and depends on NumPy, SciPy, Matplotlib, and PETAL2D. The public Ogata backend uses the optional `hankel` dependency installed by the `ogata` extra.

## Basic usage

```python
import numpy as np
from petal2d import PolarDecomposition
from quartic2d import HarmonicTransform, Interaction

x = np.linspace(-6.0, 6.0, 161)
y = np.linspace(-6.0, 6.0, 161)


def density(x, y):
    return np.exp(-(x**2 + y**2)) / np.pi


dec = PolarDecomposition(density, x, y, Nr=161, Ntheta=256)

# Normal Step 2: one transform pass with benchmarked defaults.
transformed = HarmonicTransform(dec)


def U_q(q):
    kappa = 0.25
    return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)


deltas = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
interaction = Interaction(deltas, transformed, transformed, U_q)
print(interaction.V)
```

The normal `HarmonicTransform(dec)` path does **not** run a convergence study. It chooses a scale-aware momentum range and q grid, evaluates the transform once, and then performs inexpensive checks on the arrays it already computed. If those checks see signs of inadequate q support or sampling, QUARTIC2D emits one concise warning explaining the quantity that triggered it and recommends the explicit convergence helper.

For unusual inputs, or when you want to trade accuracy against runtime or memory, calibrate once and reuse the result:

```python
calibration = HarmonicTransform.converge_parameters(
    dec,
    rtol=1e-4,
    q_tail_rtol=1e-3,
)
transformed = calibration.transform(dec)
```

Here `rtol` controls numerical discretization on the represented q interval, while `q_tail_rtol` controls how much transformed L2 norm may remain beyond the selected `q_max`. The convergence call is intentionally more expensive than normal production use.

## Numerical backends

QUARTIC2D keeps the numerical backend explicit and user-selectable.

For the sampled radial transform $\rho_m(r)\rightarrow F_m(q)$, available quadratures are:

- Simpson (default)
- composite 4-point Gauss--Legendre (`gl4`)
- composite 8-point Gauss--Legendre (`gl8`)
- trapezoidal
- Ogata quadrature through the `hankel` package

For the interaction transform, available backends are:

- `gl4` (default)
- FFTLog (`n=512`, `bias=-0.5`)
- `gl8`
- Simpson
- trapezoidal
- Ogata

The full benchmark suite compares accuracy, convergence, and runtime across smooth, higher-order, nodal, algebraic, oscillatory, and discontinuous reference problems. In the release benchmark, Simpson and GL4 both give high-quality sampled radial transforms; Simpson reaches comparable accuracy at lower cost for the tested sampled PETAL2D-like profiles, while GL4/GL8 remain useful finite-quadrature verification backends. GL4 is the robust general-purpose interaction default. FFTLog provides a high-throughput option for compatible qualified workloads, while public Ogata quadrature provides a complementary oscillatory specialist whose `N` and `h` parameters can be calibrated explicitly.

The defaults are intended for normal use. Use `HarmonicTransform.converge_parameters(...)` when a fast diagnostic warns, when you need a documented numerical tolerance, or when you want to reduce runtime or memory for a repeated production workload. For radial Ogata transforms, `HankelTransform.converge(...)` / `HarmonicTransform.converge(...)` calibrate `N` and `h` on an already chosen q grid; the full q-support `HarmonicTransform.converge_parameters(...)` path currently calibrates the finite-grid radial backends.

## Examples

Executable examples are provided in [`examples/`](examples/):

- [`hankel_gaussian.py`](examples/hankel_gaussian.py) -- analytic single-mode Hankel transform.
- [`petal2d_harmonics.py`](examples/petal2d_harmonics.py) -- PETAL2D decomposition followed by harmonic transforms.
- [`gaussian_interaction.py`](examples/gaussian_interaction.py) -- complete PETAL2D -> QUARTIC2D interaction workflow.
- [`anisotropic_interaction.py`](examples/anisotropic_interaction.py) -- angular dependence and harmonic-resolved contributions.
- [`quadrature_backends.py`](examples/quadrature_backends.py) -- available numerical backends.
- [`convergence.py`](examples/convergence.py) -- opt-in Step-2 calibration and reuse of verified parameters.

Run an example from the repository root, for example:

```bash
python examples/gaussian_interaction.py
```

## Validation and benchmarks

Run the unit tests with

```bash
python -m pytest -q
```

Run the lightweight analytic documentation check with

```bash
python -m benchmarks.documentation.gaussian_validation --quick
```

The complete benchmark organization and manuscript evidence policy live in [`benchmarks/`](benchmarks/README.md). List the canonical manuscript-grade suite with

```bash
python -m benchmarks.run_suite publication --list
```

Generated benchmark output is written under `benchmarks/results/` and is not tracked. Historical frozen datasets live under `benchmarks/reference/`; they are not silently reused as current manuscript evidence.

## Documentation

Documentation is published at [quantumartificer.github.io/quartic2d](https://quantumartificer.github.io/quartic2d/).

Build it locally with

```bash
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

## Public API

The main public classes are:

- `quartic2d.HankelTransform` -- low-level transform of one sampled radial profile.
- `quartic2d.HarmonicTransform` -- Step-2 transform of all retained PETAL2D harmonics.
- `quartic2d.Interaction` -- Step-3 interaction calculation.

See the [API reference](https://quantumartificer.github.io/quartic2d/reference/) for full parameter documentation.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A versioned DOI will be added after the first archived Zenodo release.

## License

QUARTIC2D is distributed under the MIT License. See [`LICENSE`](LICENSE).
