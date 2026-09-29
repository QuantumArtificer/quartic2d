HankelTransform
===============

.. currentmodule:: quartic2d

``HankelTransform`` is the low-level sampled radial transform used internally by
``HarmonicTransform``.  It is public for specialist calculations that already
have one radial profile and an explicit momentum grid.  Most four-center
workflows should use :class:`HarmonicTransform`.

.. autoclass:: HankelTransform

Methods
-------

.. autosummary::
   :toctree: generated

   ~HankelTransform.set_method
   ~HankelTransform.set_interpolator
   ~HankelTransform.converge
