# Contributing to QUARTIC2D

QUARTIC2D is a focused scientific package. Contributions should keep mathematical conventions explicit, numerical behavior tested, and the public API narrow.

## Development setup

```bash
python -m pip install -e ".[test,docs,release]"
```

## Before submitting a change

```bash
python -m pytest -q
python -m benchmarks.gaussian_validation --quick
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Changes to transform conventions, phase factors, support handling, interpolation, quadrature defaults, or interaction backends should include an analytic reference test or a convergence study.

Generated files under `benchmarks/results/`, `docs/_build/`, build directories, caches, and local analysis bundles should not be committed. Curated benchmark datasets belong to release/Zenodo archives rather than the source tree.

## Scope

Core changes should support QUARTIC2D's purpose: efficient evaluation of two-dimensional radial-kernel interaction integrals from angular-harmonic radial data. General-purpose electronic-structure file readers, unrelated basis-set machinery, and non-radial interaction solvers are outside the core package unless they directly support that workflow.
