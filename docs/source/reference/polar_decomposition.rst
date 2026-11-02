PolarDecomposition
==================

.. currentmodule:: quartic2d

**Input: callable or sampled 2D transition field.** QUARTIC2D exposes
PETAL2D's polar-harmonic decomposition directly through its public API:

.. code-block:: python

   from quartic2d import PolarDecomposition

   dec = PolarDecomposition(
       rho_xy, x, y, Nr=181, Ntheta=256,
       rmax=5.5, origin=(0.0, 0.0),
       recon_err_tol=1e-4,
   )
   print(dec.report())
   print(dec.report(level="full"))

``recon_err_tol`` is a **percentage**, not a fractional error tolerance.
The decomposition class inherits its numerical implementation, constructor
parameters, reconstruction methods, and plotting from PETAL2D. Its numerical
behavior is unchanged. An object constructed with the original
``petal2d.PolarDecomposition`` remains a valid input to QUARTIC2D transforms.

.. autoclass:: PolarDecomposition

Reporting methods
-----------------

.. automethod:: PolarDecomposition.report

.. automethod:: PolarDecomposition.to_dict

The summary report shows retained harmonics, the harmonic-selection target,
and measured reconstruction error. The full report includes available grid
information, input settings, per-harmonic cutoff radii, and software versions.
``to_dict()`` saves JSON-compatible numerical-method metadata, **not** the
original field or all radial profiles.

The accuracy of an upstream sampled field is outside the scope of the
harmonic-selection test. Downstream convergence is performed separately with
:meth:`HarmonicTransform.converge_parameters`.
