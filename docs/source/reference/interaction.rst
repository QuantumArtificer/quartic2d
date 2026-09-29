Interaction
===========

.. currentmodule:: quartic2d

``Interaction`` evaluates the four-center matrix element after the two
transition fields have been transformed to momentum space.  For transition
fields :math:`\rho_{13}` and :math:`\rho_{42}`, QUARTIC2D evaluates the
radial-kernel reduction of

.. math::

   U_{1234}=\iint d^2\mathbf r\,d^2\mathbf r'\,
   \rho_{13}(\mathbf r)\,U(|\mathbf r-\mathbf r'|)\,
   \rho_{42}^*(\mathbf r').

The ordinary constructor evaluates the requested method once.  Use
:meth:`Interaction.converge_parameters` when the interaction itself requires a
recorded refinement study.

.. autoclass:: Interaction

Typical workflow
----------------

::

   interaction = Interaction(deltas, field_13, field_42, U_q)
   values = interaction.V

For quantitative production work::

   result = Interaction.converge_parameters(
       deltas,
       field_13,
       field_42,
       U_q,
       rtol=1e-4,
   )
   if not result.converged:
       raise RuntimeError("interaction convergence search did not converge")
   fig, axes = result.plot_convergence()
   interaction = result.interaction(deltas, field_13, field_42, U_q)
   interaction.plot_convergence()

The returned :class:`InteractionConvergenceResult` stores the selected
parameters and the method-specific search history.  It can be serialized,
reused to construct an interaction, and inspected with the convergence plots.

Recommended method
------------------

.. autosummary::
   :toctree: generated

   ~Interaction.converge_parameters
   ~Interaction.plot_convergence

Advanced mutation and low-level convergence
-------------------------------------------

.. autosummary::
   :toctree: generated

   ~Interaction.set_method
   ~Interaction.set_interpolator
   ~Interaction.converge
