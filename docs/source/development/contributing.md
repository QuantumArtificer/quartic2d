# Contributing

QUARTIC2D is a focused scientific package. Contributions should keep the four-center conventions explicit, public interfaces small, and numerical claims tied to reproducible tests.

## Development setup

```bash
python -m pip install -e ".[test,docs,dev,release,ogata]"
```

## Before submitting a change

```bash
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts tools
python -m benchmarks.gaussian_validation --quick
python docs/scripts/check_artifacts.py
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Public behavior should include tests and documentation. Numerical changes should use analytic or otherwise independent references whenever practical.

The repository-local Ruff policy lives in `pyproject.toml`. The `dev` extra supplies the supported Ruff release range, so local checks and CI use the same rule selection instead of inheriting machine-specific configuration.

Generated benchmark results, documentation builds, package distributions, caches, and local analysis bundles should not be committed.

## Scope

Core additions should support the evaluation, validation, or practical use of localized two-dimensional four-center interactions with radial translationally invariant kernels. General electronic-structure file readers, unrelated many-body solvers, and non-radial interaction engines should remain separate packages unless they directly support the QUARTIC2D calculation model.
