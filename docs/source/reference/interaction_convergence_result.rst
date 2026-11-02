InteractionConvergenceResult
============================

.. currentmodule:: quartic2d

``InteractionConvergenceResult`` is the reusable output of
:meth:`Interaction.converge_parameters`.  It records the selected
method-specific parameters and the complete interaction-level refinement
history without conflating that self-convergence record with an independent
reference error.

.. autoclass:: InteractionConvergenceResult

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

   ~InteractionConvergenceResult.plot_convergence
   ~InteractionConvergenceResult.interaction
   ~InteractionConvergenceResult.to_dict

The result object keeps interaction calibration separate from production
evaluation and provides the natural public surface for serialization and
convergence diagnostics.
