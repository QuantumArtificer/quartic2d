"""QUARTIC2D-facing interface to PETAL2D's real-space harmonic decomposition.

PETAL2D remains the numerical implementation; this subclass only adds concise
reporting and a stable top-level QUARTIC2D import. It does not repeat or alter
polar decomposition calculations.
"""

from __future__ import annotations

from typing import Any

from petal2d import PolarDecomposition as _PetalPolarDecomposition

from .reporting import decomposition_provenance, format_decomposition, software_versions

__all__ = ["PolarDecomposition"]


class PolarDecomposition(_PetalPolarDecomposition):
    """Represent a real or complex 2D field using PETAL2D angular harmonics.

    This class inherits the full :class:`petal2d.PolarDecomposition` constructor
    and numerical API, including callable or sampled-array inputs, reconstruction,
    and plotting. QUARTIC2D adds :meth:`report` and :meth:`to_dict` for concise
    inspection and numerical-method metadata. Existing PETAL2D decomposition
    instances remain valid inputs to :class:`quartic2d.HarmonicTransform`.

    The ``recon_err_tol`` constructor argument is a **percentage**, unlike the
    fractional ``rtol`` used in downstream convergence calibration.
    """

    def report(self, *, level: str = "summary") -> str:
        """Format the decomposition and its reconstruction diagnostics.

        ``level='summary'`` shows the retained harmonics and actual measured
        reconstruction error. ``level='full'`` also includes input-grid ranges,
        decomposition settings, mode cutoffs, and recorded software versions.
        This does not assess the accuracy of the original supplied field.
        """
        return format_decomposition(self, level=level)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable decomposition settings and diagnostics.

        This is a numerical-method metadata record, not an export of the full
        input field or every radial harmonic profile. Those remain accessible
        on the decomposition object itself.
        """
        return {
            "stage": "polar_decomposition",
            "harmonic_target_reached": bool(self.target_reached),
            "reconstruction_error_percent": float(self.recon_error_measured),
            "parameters_and_diagnostics": decomposition_provenance(self),
            "software": software_versions(),
        }

    def __str__(self) -> str:
        return self.report()
