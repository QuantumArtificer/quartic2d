# Testing

Install the development dependencies with

```bash
python -m pip install -e ".[test,docs,dev,release,ogata]"
```

The final pip line should report a successful editable installation of `quartic2d`; exact dependency lines depend on the environment.

## Unit tests

```bash
python -m pytest
```

A clean run exits with status 0 and reports that all collected tests passed. The exact count is intentionally not hard-coded into the documentation.

Numerical changes should include a test that can fail when the underlying mathematical convention is wrong. Useful references include analytic transforms, minimal identities, and unreduced numerical integrals that do not reuse the same reduced algebra as the production implementation.

## Static checks

```bash
python -m ruff check src tests examples benchmarks docs/scripts tools
```

A clean run prints no diagnostics and exits with status 0. The selected rules, Python target, and first-party import classification are defined in `pyproject.toml`; local Ruff configuration outside the repository is not part of the quality contract.

## Documentation

Check the tracked generated-artifact contract before building:

```bash
python docs/scripts/check_artifacts.py
```

Then build Sphinx with warnings as errors:

```bash
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

A clean build ends with

```text
build succeeded.
```

Warnings are treated as errors in CI.

## Examples

Run the repository examples in a noninteractive plotting backend:

```bash
MPLBACKEND=Agg bash -c '
for example in examples/*.py; do
    echo "==> $example"
    python "$example"
done
'
```

The command runs the curated public example set. Developer diagnostics are kept under `tools/diagnostics/` and are not part of this smoke test. Any uncaught exception makes the shell command fail.

The examples are part of the user-facing package surface. Changes to public constructors, defaults, or output conventions should keep them executable.
