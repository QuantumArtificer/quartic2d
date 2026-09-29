# Contributing to QUARTIC2D

QUARTIC2D is a focused scientific package for localized two-dimensional four-center interactions. Contributions should keep mathematical conventions explicit, numerical behavior tested, and the public API small.

## Development setup

```bash
python -m pip install -e ".[test,docs,dev,release,ogata]"
```

## Before submitting a change

```bash
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts tools

Ruff's selected rules, Python target, and import classification are defined in
`pyproject.toml`; no user-level Ruff configuration is required. Install the
repository's `dev` extra rather than relying on a separately configured Ruff.
Benchmark-wide exception catches are allowed only when the runner must record a
failed configuration and continue, and those sites are annotated locally.
python -m benchmarks.gaussian_validation --quick
python docs/scripts/check_artifacts.py
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Run the affected examples when public behavior changes.

Changes to Fourier phases, Bessel orders, support selection, interpolation, quadrature defaults, automatic convergence, or interaction assembly should include an analytic identity, an unreduced numerical reference, or another check that does not merely repeat the same reduced algebra.

Generated files under `benchmarks/results/`, `docs/_build/`, build directories, caches, and local analysis bundles should not be committed. Curated publication datasets belong in an archived result set with their manifest and source provenance. The tracked files under `docs/source/_generated/validation/` and the managed validation SVGs are an exception: they are derived documentation snapshots generated from one complete canonical publication bundle. Regenerate them only with `docs/scripts/regenerate_publication_evidence.py` so their bundle fingerprints remain synchronized.

## Scope

Core additions should support the evaluation, validation, or practical use of localized two-dimensional four-center interactions with radial translationally invariant kernels. General electronic-structure readers, unrelated many-body solvers, and non-radial interaction engines should remain separate packages unless they directly support the QUARTIC2D calculation model.
