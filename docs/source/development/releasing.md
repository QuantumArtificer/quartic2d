# Releasing

A QUARTIC2D release has one source artifact, the annotated Git tag, and three publication surfaces: PyPI, a GitHub Release, and a Zenodo software archive. The release automation publishes them in that order so an archived GitHub release is not created before the Python distributions have passed validation and reached PyPI.

## 1. Prepare the release commit

Update

- `src/quartic2d/_version.py`
- `CITATION.cff`
- the matching version heading in `CHANGELOG.md`

For a release commit, `CHANGELOG.md` must use a date in `YYYY-MM-DD` form and `CITATION.cff` must contain the same `date-released` value.

Check the metadata before tagging:

```bash
python tools/check_release_metadata.py
```

For the current unreleased 0.1.0 state, the checker prints

```text
package version : 0.1.0
citation version: 0.1.0
changelog state : Unreleased
release metadata: consistent
```

Install the maintainer dependencies:

```bash
python -m pip install -e ".[test,docs,dev,release,ogata]"
```

## 2. Run the local release checks

```bash
python -m pytest
python -m ruff check src tests examples benchmarks docs/scripts tools
python -m benchmarks.gaussian_validation --quick
python docs/scripts/generate_figures.py examples
python docs/scripts/generate_example_artifacts.py
python docs/scripts/check_artifacts.py
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
python docs/scripts/check_rendered_docs.py docs/_build/html
```

A clean source state gives the following stable end conditions:

```text
pytest: all tests passed
ruff: no diagnostics
gaussian validation: PASS
documentation artifact manifest: PASS
Sphinx: build succeeded.
```

Run the executable examples:

```bash
MPLBACKEND=Agg bash -c '
for example in examples/*.py; do
    echo "==> $example"
    python "$example"
done
'
```

If publication benchmark evidence accompanies the release, run it from the exact release commit:

```bash
python -m benchmarks.run_suite publication 2>&1 | \
    tee ~/Downloads/quartic2d_publication_run.log
```

The log path is deliberately outside the repository. Benchmark provenance samples the Git working tree during execution, so a tracked output log would make the run dirty even when no computational input changed.

A successful run ends with

```text
BENCHMARK SUITE COMPLETE
Status  : PASS
Results : <repository>/benchmarks/results
```

Check the generated manifest before archiving the benchmark dataset. The commit must match the release commit and the recorded source state must be clean.

## 3. Build and inspect the distributions

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
```

The final lines should report successful wheel/sdist creation followed by `PASSED` from `twine check` for both distributions.

Inspect the archives:

```bash
python -m zipfile -l dist/*.whl
tar -tf dist/*.tar.gz
```

Both commands print archive member lists. Confirm that `quartic2d/`, package metadata, license, README, examples, documentation sources, tests, and benchmark sources are present according to `MANIFEST.in`, while generated result/build directories are absent.

The source distribution includes the package metadata, license, documentation sources, examples, tests, benchmark sources, and maintainer/release documentation. Generated benchmark results, documentation builds, caches, and local environments are excluded.

## 4. Test the wheel outside the checkout

An editable installation can hide packaging mistakes. Test the built wheel in a fresh environment:

```bash
python -m venv /tmp/quartic2d-release-check
source /tmp/quartic2d-release-check/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
cd /tmp
python - <<'PY'
import quartic2d
print(quartic2d.__version__)
print(quartic2d.__all__)
PY
deactivate
```

For version 0.1.0 the import block prints

```text
0.1.0
['HankelTransform', 'HarmonicConvergenceResult', 'HarmonicTransform', 'Interaction', 'InteractionConvergenceResult', '__version__']
```

## 5. Enable GitHub Pages

Before the first public release, enable GitHub Pages for the repository from
**Settings → Pages → Build and deployment → Source → GitHub Actions**. The
`Documentation` workflow builds and validates the Sphinx site on every push to
`main` and deploys that verified artifact through the `github-pages`
environment.

`actions/configure-pages` cannot enable Pages with the default `GITHUB_TOKEN`,
so this repository setting must be enabled once by a maintainer. After enabling
it, rerun the Documentation workflow and require both its build and deployment
jobs to pass before tagging a release.

## 6. Configure PyPI Trusted Publishing

The release workflow uses the GitHub environment `pypi` and OIDC Trusted Publishing. Configure the PyPI project or pending publisher with

- owner: `QuantumArtificer`
- repository: `quartic2d`
- workflow: `release.yml`
- environment: `pypi`

No long-lived PyPI token is stored in the repository.

Official guidance: [Publishing package distribution releases](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) and [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/).

## 7. Enable Zenodo before the first tag

Connect the GitHub account to Zenodo, synchronize repositories, and enable `QuantumArtificer/quartic2d` before publishing the first release tag.

QUARTIC2D uses `CITATION.cff` as the release metadata source. Add `.zenodo.json` only if Zenodo-specific metadata are needed. When `.zenodo.json` is present, Zenodo uses it instead of `CITATION.cff` for GitHub-release metadata.

Zenodo guidance: [GitHub integration](https://help.zenodo.org/docs/github/) and [software metadata](https://help.zenodo.org/docs/github/describe-software/).

## 8. Create and push the release tag

Commit the fully checked release state, then create an annotated tag:

```bash
git status --short
python tools/check_release_metadata.py --tag v0.1.0
git tag -a v0.1.0 -m "QUARTIC2D v0.1.0"
git push origin main
git push origin v0.1.0
```

For a clean tree, `git status --short` prints nothing. `check_release_metadata.py --tag ...` must end with `release metadata: consistent` before the tag is created.

The tag push starts `.github/workflows/release.yml`.

The workflow performs the following sequence:

1. checks the tag, package version, `CITATION.cff`, release date, and changelog
2. runs tests, Ruff, analytic validation, documentation, and executable examples
3. builds the wheel and source distribution once and checks their metadata
4. installs the built wheel and imports it outside the repository checkout
5. publishes the verified distributions to PyPI through Trusted Publishing
6. creates the GitHub Release from the same tag and attaches the distributions plus `SHA256SUMS`
7. allows the enabled Zenodo GitHub integration to archive that GitHub Release

A failure before step 5 leaves no PyPI or GitHub release. A PyPI failure prevents creation of the GitHub Release and therefore prevents the Zenodo archive from being triggered by this release path.

## 9. Verify the published release

After the workflow succeeds:

1. install `quartic2d==X.Y.Z` from PyPI in a clean environment
2. confirm the GitHub Release contains the wheel, source distribution, and `SHA256SUMS`
3. confirm the GitHub Release points to the intended annotated tag
4. confirm the Zenodo record contains the correct title, author, version, license, and repository link
5. record the version DOI after Zenodo creates it

The DOI does not exist before the first archive is created. Add it to the next source commit and to subsequent citation metadata. Do not rewrite the already published tag solely to insert an identifier that did not exist when that source state was created.

The release lint gate uses the repository-local Ruff policy in `pyproject.toml` through the `dev` extra. User-level Ruff configuration is not part of the release contract.
