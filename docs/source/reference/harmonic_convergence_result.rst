HarmonicConvergenceResult
=========================

.. currentmodule:: quartic2d

``HarmonicConvergenceResult`` is the reusable output of
:meth:`HarmonicTransform.converge_parameters`.  It keeps calibration separate
from production evaluation and preserves the selected parameters together with
sampling and quadrature diagnostics.

.. autoclass:: HarmonicConvergenceResult

Common operations
-----------------

.. autosummary::
   :toctree: generated

   ~HarmonicConvergenceResult.plot_convergence
   ~HarmonicConvergenceResult.transform
   ~HarmonicConvergenceResult.to_dict

The ``parameters`` property can be inspected directly, while ``to_dict()`` is
intended for provenance records and machine-readable output.
