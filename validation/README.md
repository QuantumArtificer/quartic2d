# Analytic Gaussian validation

`run_gaussian_validation.py` compares the default QUARTIC2D workflow with analytic results for a normalized two-dimensional Gaussian. The current defaults exercised by this script are Simpson for the sampled radial transform and FFTLog for the interaction transform.

Run the CI-sized validation with

```bash
python validation/run_gaussian_validation.py --quick
```

Write the result to JSON with

```bash
python validation/run_gaussian_validation.py \
    --quick \
    --output validation/results/gaussian.json
```

The reported quantities are the relative L2 error of the sampled zeroth-order Hankel transform and the interaction error at each displacement.

The reference density is

$$
\rho(r)=\frac{1}{\pi}e^{-r^2},
$$

with

$$
F_0(q)=\frac{1}{2\pi}e^{-q^2/4}.
$$

For $U(q)=2\pi/q$, the displaced interaction is

$$
V(d)=\sqrt{\frac{\pi}{2}}e^{-d^2/4}I_0(d^2/4).
$$

The broader backend comparison is implemented by `benchmarks/run_numerical_benchmarks.py`.
