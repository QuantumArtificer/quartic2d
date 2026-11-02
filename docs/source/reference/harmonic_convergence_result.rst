HarmonicConvergenceResult
=========================

.. currentmodule:: quartic2d

``HarmonicConvergenceResult`` is the reusable output of
:meth:`HarmonicTransform.converge_parameters`.  It keeps calibration separate
from production evaluation and preserves the selected parameters together with
sampling and quadrature diagnostics.

.. autoclass:: HarmonicConvergenceResult

Quick reports
-------------

Use ``print(result)`` for a compact summary or
``result.report(level="full")`` for the numerical-method settings and actual
acceptance tests. The ``to_dict()`` method retains machine-readable evidence.
The numerical self-convergence checks must not be mistaken for an independent
accuracy guarantee. See :doc:`convergence_reporting`.

Common operations
-----------------

.. autosummary::
   :toctree: generated

   ~HarmonicConvergenceResult.plot_convergence
   ~HarmonicConvergenceResult.transform
   ~HarmonicConvergenceResult.to_dict

The ``parameters`` property can be inspected directly, while ``to_dict()`` is
intended for provenance records and machine-readable output.
