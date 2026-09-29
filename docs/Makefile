SPHINXOPTS ?= -W --keep-going
SPHINXBUILD ?= python -m sphinx
SOURCEDIR = source
BUILDDIR = _build

.PHONY: examples html qa clean

examples:
	python scripts/generate_figures.py examples
	python scripts/generate_example_artifacts.py

html: examples
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

qa: html
	python scripts/check_rendered_docs.py $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
