# Development

Install development extras with

```bash
python -m pip install -e ".[test,docs,release]"
```

Before a change is merged, run

```bash
python -m pytest -q
python -m benchmarks.documentation.gaussian_validation --quick
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

Build release artifacts with

```bash
python tools/clean_generated.py
python -m build
python -m twine check dist/*
```

Generated output under `benchmarks/results/`, `docs/_build/`, build directories, and caches should not be committed. `tools/clean_generated.py` removes known generated artifacts while refusing to delete Git-tracked files. The versioned file under `benchmarks/reference/` is curated release data and is intentionally tracked in Git/Zenodo but excluded from the PyPI source distribution.

Numerical changes should state which mathematical convention is affected and include either an analytic reference test or a convergence study.

See the repository-level `RELEASING.md` for the PyPI Trusted Publishing and Zenodo release procedure.
