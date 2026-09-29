from quartic2d import __version__

project = "QUARTIC2D"
author = "Alex Santacruz, 2DQMAT Research @ IF-UNAM"
copyright = "2026, Alex Santacruz"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "description"

html_theme = "pydata_sphinx_theme"
html_title = f"QUARTIC2D {release}"
html_static_path = ["_static"]
html_css_files = ["css/quartic2d.css"]
html_theme_options = {
    "github_url": "https://github.com/QuantumArtificer/quartic2d",
    "show_toc_level": 2,
}

# Scientific bibliography used by MyST citation roles.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
