SPHINXOPTS ?= -W --keep-going
SPHINXBUILD ?= python -m sphinx
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html clean

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
