# Validation and benchmarking

## Analytic Gaussian validation

The lightweight validation suite uses

$$
\rho(r)=\frac{1}{\pi}e^{-r^2},
$$

with Hankel transform

$$
F_0(q)=\frac{1}{2\pi}e^{-q^2/4}.
$$

For the two-dimensional Coulomb kernel $U(q)=2\pi/q$, the displaced interaction has the closed form

$$
V(d)=\sqrt{\frac{\pi}{2}}
\exp\left(-\frac{d^2}{4}\right)
I_0\left(\frac{d^2}{4}\right).
$$

Run the CI-sized validation with

```bash
python validation/run_gaussian_validation.py --quick
```

## Unit tests

Run

```bash
python -m pytest -q
```

The unit tests cover:

- analytic Hankel-transform values and the $q=0$ limit;
- positive and negative integer harmonic orders;
- scalar and array interpolation;
- supported finite quadrature and FFTLog backends, with Ogata retained in the experimental benchmark suite;
- PETAL2D integration and input immutability;
- displacement validation, arbitrary displacement ordering, and phase factors;
- zero-displacement selection rules and singular-kernel endpoint handling;
- analytic interaction values;
- convergence utilities, public API, and version metadata.

## Numerical-method benchmark

The release benchmark separates the sampled radial transform from the interaction transform. It uses disjoint calibration and validation sets and includes smooth, higher-order, nodal, algebraic, oscillatory, and discontinuous profiles together with multiple interaction kernels.

Run a reduced benchmark with

```bash
python benchmarks/run_numerical_benchmarks.py \
    --quick \
    --output-dir benchmarks/results/quick
```

or the full suite with

```bash
python benchmarks/run_numerical_benchmarks.py \
    --output-dir benchmarks/results/full
```

The runner records warmup/repeated timing statistics, convergence selections, discretization and finite-support errors, requested-tolerance sweeps, large-displacement behavior, output-count scaling, and a PETAL2D-connected workload. It also generates the publication-oriented comparison figures.

The complete full result used for the v0.1.0 numerical assessment is stored in `benchmarks/reference/paper_numerics_v0.1.0.json`.

## Interpreting convergence

Finite quadrature error, finite represented support, interpolation error, and input sampling are separate numerical effects. A method can self-converge while the calculation remains limited by another error source.

For FFTLog in particular, `rtol` measures stability under increasing `n`; it does not guarantee the same absolute error against an external reference. The `bias` parameter controls logarithmic transform conditioning and is not a resolution parameter.
