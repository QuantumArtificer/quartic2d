InteractionConvergenceResult
============================

.. currentmodule:: quartic2d

``InteractionConvergenceResult`` is the reusable output of
:meth:`Interaction.converge_parameters`.  It records the selected
method-specific parameters and the complete interaction-level refinement
history without conflating that self-convergence record with an independent
reference error.

.. autoclass:: InteractionConvergenceResult

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
