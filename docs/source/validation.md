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
python -m benchmarks.gaussian_validation --quick
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
- supported finite quadrature, FFTLog, and public Ogata backends;
- PETAL2D integration and input immutability;
- displacement validation, arbitrary displacement ordering, and phase factors;
- zero-displacement selection rules and singular-kernel endpoint handling;
- analytic interaction values;
- convergence utilities, public API, and version metadata.

## Benchmark suite

The repository has one benchmark hierarchy under `benchmarks/`. Publication-grade accuracy, convergence, performance, scaling, memory, and end-to-end experiments are separated from documentation checks and development-only screens/profilers.

List the manuscript-grade suite with

```bash
python -m benchmarks.run_suite publication --list
```

and run the complete canonical dataset with

```bash
python -m benchmarks.run_suite publication
```

The evidence policy and claim registry are documented in `benchmarks/README.md` and `benchmarks/CLAIMS.md`. Frozen release datasets are archival evidence and are never substituted for a current benchmark run.

## Interpreting convergence

Finite quadrature error, finite represented support, interpolation error, and input sampling are separate numerical effects. A method can self-converge while the calculation remains limited by another error source.

For FFTLog in particular, `rtol` measures stability under increasing `n`; it does not guarantee the same absolute error against an external reference. The `bias` parameter controls logarithmic transform conditioning and is not a resolution parameter.
