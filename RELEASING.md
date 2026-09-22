# Releasing QUARTIC2D

This checklist is for versioned GitHub, PyPI, and Zenodo releases.

## 1. Prepare the source tree

Confirm that `src/quartic2d/_version.py`, `CITATION.cff`, and `CHANGELOG.md` agree on the release version. Replace `Unreleased` in the changelog with the release date before tagging.

Clean generated artifacts and run the release checks:

```bash
python tools/clean_generated.py
python -m pytest -q
python -m benchmarks.gaussian_validation --quick
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
python -m build
python -m twine check dist/*
```

Inspect the contents of both the wheel and source distribution before publishing.

## 2. Configure PyPI Trusted Publishing

The release workflow is `.github/workflows/release.yml` and uses the GitHub environment `pypi`.

For the first PyPI release, configure a pending Trusted Publisher for:

- PyPI project: `quartic2d`
- GitHub owner: `QuantumArtificer`
- repository: `quartic2d`
- workflow: `release.yml`
- environment: `pypi`

For an existing PyPI project, configure the same values under the project's Publishing settings. No long-lived PyPI token is required by the workflow.

## 3. Enable Zenodo archiving

Connect the GitHub account to Zenodo, synchronize the repository list, and enable `QuantumArtificer/quartic2d` before publishing the GitHub release. Zenodo will ingest enabled GitHub releases automatically.

QUARTIC2D uses `CITATION.cff` as the software metadata source. Do not add `.zenodo.json` unless Zenodo-specific metadata becomes necessary, because Zenodo gives `.zenodo.json` precedence over `CITATION.cff` when both exist.

## 4. Tag and publish

After all checks pass:

```bash
git status
git tag -a v0.1.0 -m "QUARTIC2D v0.1.0"
git push origin main
git push origin v0.1.0
```

Create the corresponding GitHub release from tag `v0.1.0`. Publishing the release triggers the PyPI workflow; the enabled Zenodo integration archives the GitHub release.

After Zenodo creates the DOI, add the DOI badge/link to the repository for the next commit or release as appropriate.
