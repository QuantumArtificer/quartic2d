Numerical reports and failure diagnostics
=========================================

.. currentmodule:: quartic2d

The decomposition, harmonic-convergence, and interaction-convergence results
provide compact and detailed reports alongside machine-readable metadata.
The numerical calculations and their validation rules are independent of the
terminal formatter.

Reporting interface
-------------------

.. code-block:: python

   print(decomposition.report())
   print(hcal.report())
   print(ical.report())

   print(hcal.report(level="full"))
   print(ical.report(level="full"))

   decomposition_metadata = decomposition.to_dict()
   harmonic_metadata = hcal.to_dict()
   interaction_metadata = ical.to_dict()

``report(level="summary")`` is the default and avoids printing large search
histories. ``report(level="full")`` records the available numerical settings,
measured acceptance criteria, and method-specific diagnostics. Save the
structured records when the complete refinement history is needed.

``converged`` refers to the documented internal numerical criteria. It does
**not** imply independently established accuracy of the original input field
or agreement with an unknown exact result.

Failure handling
----------------

Expected failures from parameter searches can carry a structured diagnostic:

.. code-block:: python

   from quartic2d import ConvergenceError

   try:
       hcal = HarmonicTransform.converge_parameters(decomposition)
   except ConvergenceError as exc:
       print(exc)
       print(exc.report.format(level="full"))
       evidence = exc.report.to_dict()

The default error text states what did not meet the requested criterion,
measured values when known, and an actionable location such as
``PolarDecomposition(Nr=...)`` or
``Interaction.converge_parameters(subdivisions=...)``. A failure report can
also identify situations in which there is insufficient evidence to diagnose
the root cause.

.. autoclass:: ConvergenceError
   :members:

.. autoclass:: ConvergenceFailureReport
   :members:

.. autoclass:: ConvergenceAction
   :members:

.. autoclass:: ProgressEvent

Progress and provenance
-----------------------

The expensive calibration methods accept an optional ``progress=callback``
for structured events. The default is quiet. They also support
``provenance={...}`` for user-supplied physical units, model descriptions, and
other reproducibility metadata. A callable interaction kernel cannot be
reliably reconstructed from its Python identity, so record its mathematical
definition and parameters explicitly when archiving calculations.
