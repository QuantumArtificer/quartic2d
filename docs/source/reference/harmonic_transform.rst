HarmonicTransform
=================

.. currentmodule:: quartic2d

``HarmonicTransform`` converts the radial angular harmonics retained by PETAL2D
into the momentum-space form factors used by the four-center interaction.  The
ordinary constructor evaluates one explicit numerical representation;
:meth:`HarmonicTransform.converge_parameters` provides an explicit convergence
search when a numerical criterion is required.

.. autoclass:: HarmonicTransform

Typical workflow
----------------

Start from a PETAL2D decomposition, inspect the default transform, then run an
explicit convergence search when the calculation needs a recorded numerical
criterion::

   field = HarmonicTransform(decomposition)
   field.plot_harmonics()

   result = HarmonicTransform.converge_parameters(
       decomposition,
       rtol=1e-4,
       q_tail_rtol=1e-3,
   )
   if not result.converged:
       raise RuntimeError("harmonic-transform convergence search did not converge")
   field = result.transform(decomposition)
   field.plot_convergence()

The convergence plots visualize the internal refinement tests.  They do not
replace an independent reference comparison when one is available.

Recommended methods
-------------------

.. autosummary::
   :toctree: generated

   ~HarmonicTransform.converge_parameters
   ~HarmonicTransform.plot_harmonics
   ~HarmonicTransform.plot_convergence
   ~HarmonicTransform.plot
   ~HarmonicTransform.roundtrip_error

Advanced mutation and low-level convergence
-------------------------------------------

These methods are useful for controlled numerical studies.  They are not a
substitute for the full q-support plus quadrature workflow provided by
:meth:`HarmonicTransform.converge_parameters`.

.. autosummary::
   :toctree: generated

   ~HarmonicTransform.set_method
   ~HarmonicTransform.set_interpolator
   ~HarmonicTransform.converge
