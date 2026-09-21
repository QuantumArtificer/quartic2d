# Examples

The `examples/` directory contains executable scripts covering the main QUARTIC2D workflows.

## Gaussian Hankel transform

[`hankel_gaussian.py`](https://github.com/QuantumArtificer/quartic2d/blob/main/examples/hankel_gaussian.py) evaluates the zeroth-order Hankel transform of a Gaussian and compares it with the analytic result.

```bash
python examples/hankel_gaussian.py
```

## PETAL2D harmonics

[`petal2d_harmonics.py`](https://github.com/QuantumArtificer/quartic2d/blob/main/examples/petal2d_harmonics.py) constructs an anisotropic field with PETAL2D, transforms the retained harmonics, and plots the momentum-space amplitudes.

```bash
python examples/petal2d_harmonics.py
```

## Gaussian interaction

[`gaussian_interaction.py`](https://github.com/QuantumArtificer/quartic2d/blob/main/examples/gaussian_interaction.py) demonstrates the complete PETAL2D -> QUARTIC2D workflow for a screened radial interaction.

```bash
python examples/gaussian_interaction.py
```

## Anisotropic interaction

[`anisotropic_interaction.py`](https://github.com/QuantumArtificer/quartic2d/blob/main/examples/anisotropic_interaction.py) evaluates equal-magnitude displacements at different polar angles and prints harmonic-resolved interaction contributions.

```bash
python examples/anisotropic_interaction.py
```

## Numerical backends

[`quadrature_backends.py`](https://github.com/QuantumArtificer/quartic2d/blob/main/examples/quadrature_backends.py) lists the available interpolation, radial-quadrature, and interaction-transform backends and compares radial quadratures with the default Simpson result.

```bash
python examples/quadrature_backends.py
```

## Convergence

[`convergence.py`](https://github.com/QuantumArtificer/quartic2d/blob/main/examples/convergence.py) applies the sampled radial convergence interface and optionally updates the transform with the converged parameters.

```bash
python examples/convergence.py
```
