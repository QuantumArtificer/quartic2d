# QUARTIC2D documentation

Build the documentation from the repository root with

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs/source docs/_build/html
```

The GitHub Actions documentation workflow builds the same source and deploys the `main` branch to GitHub Pages.
