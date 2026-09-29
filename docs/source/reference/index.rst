API reference
=============

The API reference is organized by public objects rather than as a single
member dump.  Each class has an overview page, and callable methods have their
own pages with the exact signature, parameter contract, and a short usage
pattern.  Tutorial calculations remain in the :doc:`../getting_started` and
:doc:`../examples/index` sections.  Physical and numerical boundaries are
documented separately in :doc:`../limitations`.

Core calculation
----------------

:doc:`harmonic_transform`
    Transform the retained PETAL2D angular harmonics to momentum space and
    inspect or calibrate the numerical representation.

:doc:`interaction`
    Assemble four-center matrix elements from two transformed transition
    fields, a radial kernel, and displacement vectors.

Convergence results
-------------------

:doc:`harmonic_convergence_result`
    Reusable result returned by
    :meth:`quartic2d.HarmonicTransform.converge_parameters`.

:doc:`interaction_convergence_result`
    Reusable result returned by
    :meth:`quartic2d.Interaction.converge_parameters`.

Advanced numerical primitive
----------------------------

:doc:`hankel_transform`
    One sampled radial Hankel transform.  Most users should work through
    :class:`quartic2d.HarmonicTransform` instead.


.. toctree::
   :maxdepth: 1
   :hidden:

   harmonic_transform
   interaction
   harmonic_convergence_result
   interaction_convergence_result
   hankel_transform
