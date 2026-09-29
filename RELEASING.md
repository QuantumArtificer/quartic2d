# Releasing QUARTIC2D

The complete maintainer procedure is documented in `docs/source/development/releasing.md`.

A release is created from one annotated `vX.Y.Z` tag. The tag starts `.github/workflows/release.yml`, which validates the tagged source, builds and verifies the wheel and source distribution, publishes those exact distributions to PyPI through Trusted Publishing, and only then creates the GitHub Release. If the repository is enabled in Zenodo, that GitHub Release becomes the archived software record.

Before tagging, make the package version, `CITATION.cff`, and dated `CHANGELOG.md` heading agree. Run

```bash
python tools/check_release_metadata.py
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts tools
python -m benchmarks.gaussian_validation --quick
python docs/scripts/generate_figures.py examples
python docs/scripts/generate_example_artifacts.py
python docs/scripts/check_artifacts.py
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
python docs/scripts/check_rendered_docs.py docs/_build/html
python tools/clean_generated.py
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
git status --short
```

Publication benchmark logs must be written outside the repository or to an ignored path. Generated benchmark results remain local or are archived with their manifest; they are not accumulated in the source tree. The tracked documentation evidence snapshot is regenerated from one archived canonical bundle and checked with `docs/scripts/check_artifacts.py`; it is not a substitute for archiving the JSON source bundle itself.

The release lint gate uses the repository-local Ruff policy in `pyproject.toml` through the `dev` extra; do not substitute a user-level Ruff configuration.
