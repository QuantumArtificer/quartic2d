API reference
=============

The public API is re-exported from :mod:`quartic2d`.

.. currentmodule:: quartic2d

HankelTransform
---------------

.. autoclass:: HankelTransform
   :members:
   :undoc-members:

HankelofHarmonics
-----------------

.. autoclass:: HankelofHarmonics
   :members:
   :undoc-members:

Interact
--------

.. autoclass:: Interact
   :members:
   :undoc-members:

Convergence results
-------------------

.. autoclass:: ConvergenceStep
   :members:

.. autoclass:: ConvergenceResult
   :members:

Convergence utilities
---------------------

.. autofunction:: converge_sequence


.. autofunction:: relative_l2_error

.. autofunction:: relative_linf_error

.. autofunction:: absolute_linf_error

.. autofunction:: roundtrip_error

Backend registries
------------------

.. autofunction:: available_quadratures

.. autofunction:: available_transforms

.. autofunction:: available_interpolators
