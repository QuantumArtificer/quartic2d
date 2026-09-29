"""Four-center interaction evaluation from angular-harmonic Hankel transforms."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable

import numpy as np
import petal2d
from scipy.integrate import simpson, trapezoid
from scipy.special import jv

from ._diagnostics import (
    DiagnosticIssue,
    Quartic2DNumericalWarning,
    default_q_grid,
    inspect_transform,
    radial_q_ceiling,
)
from ._numerics import (
    Interpolator1D,
    _fftlog_hankel_transform_many,
    _require_hankel,
    _validate_interaction_method,
    _validate_radial_method,
    common_grid,
    composite_gauss_nodes_weights,
    extrapolate_q_times_kernel_zero,
    fftlog_hankel_transform,
    hankel_transform_sampled,
    subdivide_grid,
    validate_interpolator,
)
from .convergence import HarmonicConvergenceResult, InteractionConvergenceResult

__all__ = ["HankelTransform", "HarmonicTransform", "Interaction"]


class HankelTransform:
    r"""Hankel transform of a sampled radial profile.

    The convention is

    .. math::

        F_\nu(q) = \int_0^\infty r f(r) J_\nu(qr)\,dr.

    Parameters
    ----------
    nu : int
        Angular-harmonic order of the transform.  Only ``abs(nu)`` enters the
        Bessel function; the sign is retained so negative PETAL2D harmonics use
        the package's phase convention consistently.
    q : array_like
        Momentum values at which the transform is evaluated directly.  They
        must be finite, non-negative, and strictly increasing.
    f_r : array_like
        Sampled radial profile to transform.  ``f_r[i]`` is the value at
        ``r[i]``.
    r : array_like
        Real-space radial coordinates of ``f_r``.  These are the physical input
        samples; quadrature refinement interpolates between them but does not
        create new input information.
    r_cutoff : float or None, optional
        Largest radius included in the represented profile.  Samples beyond it
        are ignored.  ``None`` uses the entire supplied radial interval.
    method : {'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}, default='simpson'
        Numerical rule used to evaluate the radial integral.  Simpson is the
        release default.  Method-specific convergence and validation behavior
        is documented separately from this constructor contract.
    interpolator : {'linear', 'cubic', 'pchip'}, default='cubic'
        How the sampled radial profile is evaluated between input grid points
        during quadrature.  Cubic interpolation is the benchmarked default.
    subdivisions : int, default=2
        Refinement of each original radial interval for finite-grid methods.
        With ``subdivisions=2``, each original radial interval is split once
        before composite integration.  Larger values increase radial
        integration resolution at approximately linear additional cost.
    N : int, default=2048
        Number of Ogata nodes when ``method='ogata'``.  It is ignored by the
        finite-grid methods.
    h : float or None, optional
        Ogata spacing/resolution parameter.  ``None`` uses ``pi/N``.  It is
        ignored by finite-grid methods.

    Attributes
    ----------
    F_q : ndarray
        Transform values on ``q``.
    """

    _allow_experimental_methods = False

    def __init__(
        self,
        nu: int,
        q: np.ndarray,
        f_r: np.ndarray,
        r: np.ndarray,
        *,
        r_cutoff: float | None = None,
        method: str = "simpson",
        interpolator: str = "cubic",
        subdivisions: int = 2,
        N: int = 2048,
        h: float | None = None,
    ):
        self._nu = int(nu)
        self._q = np.asarray(q, dtype=float)
        self._r_input = np.asarray(r, dtype=float)
        self._f_r_input = np.array(f_r, copy=True)
        self._method = _validate_radial_method(method, allow_experimental=type(self)._allow_experimental_methods)
        self._interpolator = validate_interpolator(interpolator)
        self._subdivisions = self._positive_int(subdivisions, "subdivisions")
        self._N = self._positive_int(N, "N")
        self._h = None if h is None else float(h)

        self._validate_arrays()
        if r_cutoff is None:
            r_cutoff = float(self._r_input[-1])
        self._r_cutoff = float(r_cutoff)
        if not np.isfinite(self._r_cutoff):
            raise ValueError("r_cutoff must be finite.")
        if self._h is not None and (not np.isfinite(self._h) or self._h <= 0.0):
            raise ValueError("h must be finite and positive.")

        self._prepare_support()
        self._compute()

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _validate_arrays(self) -> None:
        if self._q.ndim != 1 or self._q.size < 2:
            raise ValueError("q must be one-dimensional with at least two samples.")
        if np.any(~np.isfinite(self._q)) or np.any(self._q < 0.0):
            raise ValueError("q must contain finite non-negative values.")
        if np.any(np.diff(self._q) <= 0.0):
            raise ValueError("q must be strictly increasing.")

        if self._r_input.ndim != 1 or self._f_r_input.ndim != 1:
            raise ValueError("r and f_r must be one-dimensional.")
        if self._r_input.size != self._f_r_input.size:
            raise ValueError("r and f_r must have the same length.")
        if self._r_input.size < 2:
            raise ValueError("r and f_r must contain at least two samples.")
        if np.any(~np.isfinite(self._r_input)) or np.any(np.diff(self._r_input) <= 0.0):
            raise ValueError("r must be finite and strictly increasing.")

    def _prepare_support(self) -> None:
        mask = self._r_input <= self._r_cutoff
        if np.count_nonzero(mask) < 2:
            raise ValueError("r_cutoff must retain at least two radial samples.")
        self._r = self._r_input[mask].copy()
        self._f_r = self._f_r_input[mask].copy()

    @property
    def nu(self) -> int:
        """Signed angular-harmonic index."""
        return self._nu

    @property
    def q(self) -> np.ndarray:
        """Momentum grid on which ``F_q`` is tabulated."""
        return self._q.copy()

    @property
    def r(self) -> np.ndarray:
        """Retained radial grid."""
        return self._r.copy()

    @property
    def method(self) -> str:
        """Current numerical method."""
        return self._method

    @property
    def interpolator(self) -> str:
        """Current interpolation method."""
        return self._interpolator

    def __call__(self, q):
        """Interpolate the transformed profile at scalar or array momentum ``q``."""
        return self._q_interpolant(q)

    def set_method(
        self,
        method: str,
        *,
        subdivisions: int | None = None,
        N: int | None = None,
        h: float | None = None,
    ) -> HankelTransform:
        """Select the radial quadrature and recompute the transform.

        Parameters
        ----------
        method : {'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}
            Radial integration method. ``trapezoid``, ``simpson``, ``gl4``,
            and ``gl8`` operate on the sampled radial interval. ``ogata`` uses
            the optional ``hankel`` dependency.
        subdivisions : int or None, optional
            Finite-grid refinement. When supplied, it replaces the stored
            subdivision count used by ``trapezoid``, ``simpson``, ``gl4``,
            and ``gl8``.
        N : int or None, optional
            Ogata node count. Used only by ``method='ogata'``.
        h : float or None, optional
            Positive Ogata resolution parameter. Used only by
            ``method='ogata'``.

        Returns
        -------
        HankelTransform
            This instance after recomputing ``F_q``.

        Raises
        ------
        ValueError
            If ``method`` is unsupported or a supplied numerical parameter is
            invalid.
        ImportError
            If ``method='ogata'`` and the optional ``hankel`` dependency is
            unavailable.

        Notes
        -----
        Method-specific parameters that are not supplied retain their current
        values.
        """
        self._method = _validate_radial_method(method, allow_experimental=type(self)._allow_experimental_methods)
        if subdivisions is not None:
            self._subdivisions = self._positive_int(subdivisions, "subdivisions")
        if N is not None:
            self._N = self._positive_int(N, "N")
        if h is not None:
            h = float(h)
            if not np.isfinite(h) or h <= 0.0:
                raise ValueError("h must be finite and positive.")
            self._h = h
        self._compute()
        return self

    def set_interpolator(self, interpolator: str) -> HankelTransform:
        """Select the radial interpolator and recompute the transform.

        Parameters
        ----------
        interpolator : {'linear', 'cubic', 'pchip'}
            Interpolation used between the supplied radial samples.

        Returns
        -------
        HankelTransform
            This instance after recomputing ``F_q``.

        Raises
        ------
        ValueError
            If ``interpolator`` is not one of the supported names.
        """
        self._interpolator = validate_interpolator(interpolator)
        self._compute()
        return self

    def converge(
        self,
        *,
        rtol: float = 1.0e-3,
        atol: float = 0.0,
        subdivisions: Iterable[int] = (1, 2, 4, 8),
        hstart: float = 0.05,
        hdecrement: float = 2.0,
        maxiter: int = 15,
        apply: bool = False,
    ):
        """Refine the active radial quadrature on the current input grids.

        Parameters
        ----------
        rtol : float, default=1e-3
            Relative self-convergence threshold between successive numerical
            refinements.
        atol : float, default=0
            Absolute self-convergence threshold used together with ``rtol``.
        subdivisions : iterable of int, default=(1, 2, 4, 8)
            Candidate finite-grid subdivision counts, tried in order.
        hstart : float, default=0.05
            Initial Ogata resolution parameter.
        hdecrement : float, default=2
            Factor controlling the Ogata resolution search.
        maxiter : int, default=15
            Maximum number of Ogata search iterations.
        apply : bool, default=False
            If ``True``, apply the selected parameters to this object and
            recompute the transform.

        Returns
        -------
        ConvergenceResult
            Internal self-convergence record and selected parameters.

        Raises
        ------
        RuntimeError
            If ``apply=True`` but the search does not produce converged
            parameters.
        ImportError
            If the active method is ``'ogata'`` and the optional ``hankel``
            dependency is unavailable.

        Notes
        -----
        Finite-grid methods refine ``subdivisions``. Ogata searches ``N`` and
        ``h``. The reported error is an internal refinement diagnostic, not an
        independent reference error.
        """
        from ._convergence import converge_ogata, converge_sequence

        if self._method == "ogata":
            f = Interpolator1D(self._r, self._f_r, self._interpolator)
            K = self._representative_positive_values(self._q)
            result = converge_ogata(
                f,
                abs(self._nu),
                K=K,
                rtol=rtol,
                atol=atol,
                hstart=hstart,
                hdecrement=hdecrement,
                maxiter=maxiter,
            )
        else:
            def evaluate(parameters):
                return hankel_transform_sampled(
                    self._r,
                    self._f_r,
                    self._q,
                    abs(self._nu),
                    method=self._method,
                    interpolator=self._interpolator,
                    subdivisions=int(parameters["subdivisions"]),
                    _allow_experimental=type(self)._allow_experimental_methods,
                )

            result = converge_sequence(
                self._method,
                [{"subdivisions": int(value)} for value in subdivisions],
                evaluate,
                rtol=rtol,
                atol=atol,
            )

        if apply:
            if not result.converged or not result.selected_parameters:
                raise RuntimeError("Cannot apply unconverged parameters.")
            if self._method == "ogata":
                self._N = int(result.selected_parameters["N"])
                self._h = float(result.selected_parameters["h"])
            else:
                self._subdivisions = int(result.selected_parameters["subdivisions"])
            self._compute()
        return result

    @staticmethod
    def _representative_positive_values(values: np.ndarray, maximum: int = 9):
        positive = np.asarray(values, dtype=float)
        positive = positive[positive > 0.0]
        if positive.size == 0:
            return None
        if positive.size <= maximum:
            return positive
        indices = np.unique(np.linspace(0, positive.size - 1, maximum, dtype=int))
        return positive[indices]

    def _compute(self) -> None:
        order = abs(self._nu)
        sign = (-1.0) ** order if self._nu < 0 else 1.0
        h = np.pi / self._N if self._h is None else self._h
        values = hankel_transform_sampled(
            self._r,
            self._f_r,
            self._q,
            order,
            method=self._method,
            interpolator=self._interpolator,
            ogata_N=self._N,
            ogata_h=h,
            subdivisions=self._subdivisions,
            _allow_experimental=type(self)._allow_experimental_methods,
        )
        self.F_q = sign * np.asarray(values, dtype=np.complex128)
        self._q_interpolant = Interpolator1D(
            self._q,
            self.F_q,
            self._interpolator,
            cubic_bc_type="not-a-knot",
        )


class HarmonicTransform:
    r"""Transform retained PETAL2D angular harmonics to momentum space.

    In ordinary use only the PETAL2D decomposition is required::

        transformed = HarmonicTransform(decomposition)

    QUARTIC2D then uses benchmarked, scale-aware defaults for the momentum range
    and q-grid density, performs each Hankel transform once, and runs inexpensive
    sanity checks on the result.  These checks never launch a second transform.
    If they detect signs of insufficient support or sampling, a concise warning
    recommends the explicit convergence helper.

    Parameters
    ----------
    decomposition : petal2d.PolarDecomposition
        Polar harmonic decomposition produced by PETAL2D.  QUARTIC2D uses its
        retained radial profiles and per-harmonic ``cutoff_radius`` values.
    q_max : float or None, optional
        Largest momentum represented by the transform.  ``None`` uses a fast,
        scale-aware estimate based on the RMS momentum content of the retained
        radial profiles, capped safely below the momentum Nyquist limit implied
        by the PETAL2D radial spacing.  Supply a value when a downstream model
        requires a specific q range.  The fast diagnostics warn if the selected
        range appears too short; ``converge_parameters`` performs the explicit
        q-support refinement study when needed.
    n_q : int or None, optional
        Number of uniformly spaced momentum samples between 0 and ``q_max``.
        ``None`` chooses a conservative grid from the retained real-space
        support, bounded between 64 and 512 points for predictable cost.  This
        controls how accurately the tabulated form factors can be interpolated;
        it does not change the radial quadrature itself.  Ignored as a grid
        generator when ``q_grid`` is supplied, but if both are supplied it must
        equal ``len(q_grid)``.
    q_grid : array_like or None, optional
        Explicit strictly increasing momentum grid beginning at zero.  This is
        primarily used by ``converge_parameters`` to reuse its directly
        converged adaptive q grid in production.  Supplying an explicit grid
        does not weaken interpolation checks; the convergence helper verifies
        every final interval against direct Hankel evaluations before returning
        it.
    method : {'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}, default='simpson'
        Numerical rule used for the radial Hankel integral.  Simpson is the
        release default.  Method-specific validation and performance evidence
        are documented separately from this API contract.
    interpolator : {'linear', 'cubic', 'pchip'}, default='cubic'
        Interpolation used between the PETAL2D radial samples.  Cubic
        interpolation of the resulting q-space form factors uses not-a-knot end
        conditions, which avoid imposing an artificial zero second derivative
        at q=0.
    subdivisions : int, default=2
        Radial integration refinement for finite-grid methods.  A value of 2
        splits every original PETAL2D radial interval once before composite
        Simpson integration.  Larger values cost approximately linearly and are
        available when a case requires explicit convergence.  Validation of the
        release default is reported separately from this parameter definition.
    N : int, default=2048
        Number of Ogata quadrature nodes when ``method='ogata'``.  Larger values
        provide a denser Ogata representation at higher computational cost.
        This parameter is ignored by finite-grid methods.
    h : float or None, optional
        Ogata spacing/resolution parameter.  ``None`` uses ``pi/N``.  Smaller
        values resolve finer transform structure but generally require more
        effective work.  This parameter is ignored by finite-grid methods.
    check : bool, default=True
        Run fast post-transform fault detectors and emit one concise warning if
        the result shows signs of inadequate q support or sampling.  The checks
        only inspect arrays already computed by the transform; they do not run
        convergence studies or additional Hankel transforms.

    Attributes
    ----------
    F_q : dict[int, ndarray]
        Momentum-space harmonic profiles keyed by angular index ``m``.
    diagnostics : HarmonicTransformDiagnostics
        Results of the inexpensive sanity checks.  ``diagnostics.healthy`` is
        ``True`` when no fault detector was triggered.

    See Also
    --------
    HarmonicTransform.converge_parameters
        Opt-in, quantitative convergence of q support, q-grid interpolation,
        and radial quadrature.
    """

    _allow_experimental_methods = False
    _hankel_transform_type = HankelTransform

    def __init__(
        self,
        decomposition: petal2d.PolarDecomposition,
        q_max: float | None = None,
        *,
        n_q: int | None = None,
        q_grid: np.ndarray | list[float] | None = None,
        method: str = "simpson",
        interpolator: str = "cubic",
        subdivisions: int = 2,
        N: int = 2048,
        h: float | None = None,
        check: bool = True,
    ):
        self._decomposition = decomposition
        self._rho = decomposition.rho
        self._r = np.asarray(decomposition.r, dtype=float)
        self._m_values = np.asarray(decomposition.m_sorted, dtype=int)
        if self._m_values.size == 0:
            raise ValueError("decomposition contains no retained harmonics.")
        if not hasattr(decomposition, "cutoff_radius"):
            raise AttributeError("decomposition must provide cutoff_radius.")

        self._r_cutoffs = {
            int(m): float(decomposition.cutoff_radius[int(m)])
            for m in self._m_values
        }
        suggested_q_max, suggested_n_q, default_info = default_q_grid(decomposition)
        self._default_grid_info = default_info
        self._explicit_q_grid = None

        if q_grid is not None:
            explicit = np.asarray(q_grid, dtype=float)
            if explicit.ndim != 1 or explicit.size < 2:
                raise ValueError("q_grid must be one-dimensional with at least two points.")
            if np.any(~np.isfinite(explicit)) or np.any(explicit < 0.0) or np.any(np.diff(explicit) <= 0.0):
                raise ValueError("q_grid must be finite, non-negative, and strictly increasing.")
            if abs(float(explicit[0])) > 10.0 * np.finfo(float).eps:
                raise ValueError("q_grid must start at q=0.")
            if q_max is not None and not np.isclose(float(q_max), float(explicit[-1]), rtol=1e-12, atol=1e-14):
                raise ValueError("q_max must equal q_grid[-1] when q_grid is supplied.")
            if n_q is not None and int(n_q) != int(explicit.size):
                raise ValueError("n_q must equal len(q_grid) when q_grid is supplied.")
            self._explicit_q_grid = explicit.copy()
            self._q_max = float(explicit[-1])
            self._n_q = int(explicit.size)
            self._used_default_q_max = False
            self._used_default_n_q = False
        else:
            self._used_default_q_max = q_max is None
            self._used_default_n_q = n_q is None
            self._q_max = suggested_q_max if q_max is None else float(q_max)
            if not np.isfinite(self._q_max) or self._q_max <= 0.0:
                raise ValueError("q_max must be finite and positive.")

            if n_q is None:
                if q_max is None:
                    chosen_n_q = suggested_n_q
                else:
                    # Re-evaluate only the cheap q-density rule for a user-supplied
                    # q range.  The 512-point cap keeps normal use predictable; the
                    # fault detector will warn when that cap is visibly inadequate.
                    support_radius = max(self._r_cutoffs.values())
                    predicted = 1 + int(np.ceil(8.0 * self._q_max * support_radius / np.pi))
                    chosen_n_q = min(512, max(64, predicted))
            else:
                chosen_n_q = n_q
            self._n_q = HankelTransform._positive_int(chosen_n_q, "n_q")
            if self._n_q < 2:
                raise ValueError("n_q must be at least 2.")

        q_ceiling = radial_q_ceiling(self._r)
        if self._q_max > q_ceiling:
            raise ValueError(
                f"q_max={self._q_max:.6g} exceeds the momentum Nyquist ceiling "
                f"pi/dr={q_ceiling:.6g} set by the PETAL2D radial spacing. "
                "Increase the real-space radial resolution instead of extending q_max."
            )

        self._method = _validate_radial_method(method, allow_experimental=type(self)._allow_experimental_methods)
        self._interpolator = validate_interpolator(interpolator)
        self._subdivisions = HankelTransform._positive_int(subdivisions, "subdivisions")
        self._N = HankelTransform._positive_int(N, "N")
        self._h = None if h is None else float(h)
        if self._h is not None and (not np.isfinite(self._h) or self._h <= 0.0):
            raise ValueError("h must be finite and positive.")

        self._check = bool(check)
        self._sampling_convergence = None
        self._quadrature_convergence = None
        self._diagnostics = None
        self._compute()

    @classmethod
    def converge_parameters(
        cls,
        decomposition: petal2d.PolarDecomposition,
        *,
        rtol: float = 1.0e-4,
        atol: float = 1.0e-12,
        q_tail_rtol: float = 1.0e-3,
        method: str = "simpson",
        interpolator: str = "cubic",
        subdivisions: Iterable[int] = (1, 2, 4, 8, 16, 32),
        pilot_subdivisions: int = 2,
        verbose: bool = True,
    ) -> HarmonicConvergenceResult:
        r"""Find harmonic-transform parameters that meet explicit numerical tolerances.

        This is the expensive, opt-in calibration path.  Normal transforms do
        not call it.  The routine refines three independent numerical choices:

        1. ``q_max`` -- how far the momentum-space transform must be followed
           before the omitted q-space norm is small;
        2. ``n_q`` -- how many samples are needed to interpolate the transform
           accurately between q-grid points;
        3. radial quadrature resolution -- how finely the PETAL2D radial
           profiles must be integrated.

        Parameters
        ----------
        decomposition : petal2d.PolarDecomposition
            Retained PETAL2D angular harmonics and their radial profiles.
        rtol : float, default=1e-4
            Internal self-convergence threshold for q-grid interpolation and
            radial quadrature on the represented q interval.  Acceptance uses
            both relative L2 change <= ``rtol`` and maximum absolute change <=
            ``atol + rtol * peak``.  Relative Linf change is retained as a
            diagnostic only.  This criterion does not determine how much
            q-space may be omitted and is not an independent reference-error
            guarantee.
        atol : float, default=1e-12
            Absolute floor in the peak-scaled maximum-change condition used
            together with ``rtol``.
        q_tail_rtol : float, default=1e-3
            Maximum relative L2 norm allowed outside the selected ``q_max`` for
            every retained harmonic.  This is a momentum-support/truncation
            tolerance, not a quadrature tolerance.  The default is validated by
            the interaction benchmark; choose a smaller value when the form
            factors themselves require stricter tail control.
        method : {'trapezoid', 'simpson', 'gl4', 'gl8'}, default='simpson'
            Radial quadrature method to calibrate.  Simpson is the release
            default; validation evidence for the tested workload portfolio is
            reported separately in the numerical-method and validation pages.
        interpolator : {'linear', 'cubic', 'pchip'}, default='cubic'
            Interpolation used for the sampled PETAL2D radial profiles and the
            tabulated form factors.  Cubic q-space interpolation uses
            not-a-knot end conditions.
        subdivisions : iterable of int, default=(1, 2, 4, 8, 16, 32)
            Candidate refinements for finite-grid radial quadrature.  A value of
            4, for example, inserts three intermediate integration points inside
            each original PETAL2D radial interval.  Larger values increase the
            radial integration cost approximately linearly.
        pilot_subdivisions : int, default=2
            Initial radial refinement used by the canonical Simpson pilot that
            converges q support and q-grid density.  Sampling is a property of
            the represented transform rather than of the production quadrature
            backend.  If the requested backend requires a finer radial
            refinement, only the canonical sampling study is repeated at that
            finer resolution before returning the selected parameters.
        verbose : bool, default=True
            Print one compact convergence report.  Set to ``False`` for batch
            jobs; the returned result still contains the full diagnostics.

        Returns
        -------
        HarmonicConvergenceResult
            Calibration result containing the selected parameters and full
            refinement diagnostics.  Check ``result.converged`` before using
            ``result.transform(decomposition)`` to build a production transform.

        Notes
        -----
        Convergence is intentionally separate from the ordinary constructor so
        that production transforms pay only for the requested calculation.
        """
        from ._sampling import converge_q_sampling

        method = _validate_radial_method(method, allow_experimental=cls._allow_experimental_methods)
        if method == "ogata":
            raise ValueError(
                "converge_parameters currently calibrates finite-grid methods. "
                "Use the low-level converge() method for Ogata."
            )

        sampling_subdivisions = HankelTransform._positive_int(
            pilot_subdivisions, "pilot_subdivisions"
        )
        sampling = None
        quadrature = None

        # q support and q interpolation density describe the represented
        # momentum-space function, not the production quadrature backend.
        # Calibrate them with the benchmarked Simpson pilot and converge the
        # requested radial quadrature independently on that q grid.  If the
        # requested method shows that the radial representation needs finer
        # refinement, repeat only the canonical Simpson sampling study at that
        # finer ordered resolution.  Candidate refinements increase strictly,
        # so the loop cannot return to a coarser level.
        for _ in range(3):
            sampling = converge_q_sampling(
                decomposition,
                method="simpson",
                interpolator=interpolator,
                subdivisions=sampling_subdivisions,
                q_tail_rtol=q_tail_rtol,
                interpolation_rtol=rtol,
                interpolation_atol=atol,
            )
            transform = cls(
                decomposition,
                q_max=sampling.q_max,
                n_q=sampling.n_q,
                q_grid=sampling.q_grid,
                method=method,
                interpolator=interpolator,
                subdivisions=sampling_subdivisions,
                check=False,
            )
            quadrature = transform.converge(
                rtol=rtol,
                atol=atol,
                subdivisions=subdivisions,
                apply=False,
            )
            if not quadrature.converged or not quadrature.selected_parameters:
                raise RuntimeError(
                    "Radial quadrature did not converge for every retained harmonic. "
                    "Inspect the convergence result or expand the candidate range."
                )
            selected = int(quadrature.selected_parameters["subdivisions"])
            required_sampling_subdivisions = max(sampling_subdivisions, selected)
            if required_sampling_subdivisions == sampling_subdivisions:
                break
            sampling_subdivisions = required_sampling_subdivisions
        else:  # pragma: no cover - monotonic defensive guard
            raise RuntimeError("Harmonic-transform convergence did not stabilize.")

        result = HarmonicConvergenceResult(
            sampling=sampling,
            quadrature=quadrature,
            method=method,
            interpolator=interpolator,
            rtol=float(rtol),
            atol=float(atol),
            q_tail_rtol=float(q_tail_rtol),
        )
        if verbose:
            print(result)
        return result

    @property
    def m_values(self) -> np.ndarray:
        """Retained angular-harmonic indices."""
        return self._m_values.copy()

    @property
    def q(self) -> np.ndarray:
        """Common momentum grid for all transformed harmonics."""
        return self._q.copy()

    @property
    def method(self) -> str:
        """Current numerical method."""
        return self._method

    @property
    def interpolator(self) -> str:
        """Current interpolation method."""
        return self._interpolator

    @property
    def diagnostics(self):
        """Fast post-transform sanity checks for the current numerical grid.

        The diagnostics are indicators, not quantitative convergence results.
        In particular, ``boundary_power_fraction`` describes power *inside the
        sampled interval near its upper edge*; it is not an estimate of the
        omitted tail beyond ``q_max``.
        """
        return self._diagnostics

    @property
    def sampling_convergence(self):
        """q-support/q-grid convergence result when built from convergence output."""
        return self._sampling_convergence

    @property
    def quadrature_convergence(self):
        """Radial-quadrature convergence result when built from convergence output."""
        return self._quadrature_convergence

    def __getitem__(self, m: int) -> np.ndarray:
        """Return tabulated values of transformed harmonic ``m``."""
        return self.F_q[int(m)]

    def __call__(self, m: int, q):
        """Interpolate transformed harmonic ``m`` at scalar or array ``q``."""
        m = int(m)
        if m not in self._q_interpolants:
            raise KeyError(f"Harmonic m={m} is not available.")
        return self._q_interpolants[m](q)

    def set_method(
        self,
        method: str,
        *,
        subdivisions: int | None = None,
        N: int | None = None,
        h: float | None = None,
    ) -> HarmonicTransform:
        """Select the radial quadrature and recompute all retained harmonics.

        Parameters
        ----------
        method : {'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}
            Radial Hankel-transform method.
        subdivisions : int or None, optional
            Finite-grid refinement used by ``trapezoid``, ``simpson``,
            ``gl4``, and ``gl8``.
        N : int or None, optional
            Ogata node count. Used only by ``method='ogata'``.
        h : float or None, optional
            Positive Ogata resolution parameter. Used only by
            ``method='ogata'``.

        Returns
        -------
        HarmonicTransform
            This instance after recomputing all ``F_m(q)`` values.

        Raises
        ------
        ValueError
            If ``method`` is unsupported or a supplied numerical parameter is
            invalid.
        ImportError
            If ``method='ogata'`` and the optional ``hankel`` dependency is
            unavailable.

        Notes
        -----
        Method-specific parameters that are not supplied retain their current
        values. Changing the method clears any attached q-sampling and radial-
        quadrature convergence records because those records refer to the
        previous numerical representation.
        """
        self._method = _validate_radial_method(method, allow_experimental=type(self)._allow_experimental_methods)
        if subdivisions is not None:
            self._subdivisions = HankelTransform._positive_int(subdivisions, "subdivisions")
        if N is not None:
            self._N = HankelTransform._positive_int(N, "N")
        if h is not None:
            h = float(h)
            if not np.isfinite(h) or h <= 0.0:
                raise ValueError("h must be finite and positive.")
            self._h = h
        self._sampling_convergence = None
        self._quadrature_convergence = None
        self._compute()
        return self

    def set_interpolator(self, interpolator: str) -> HarmonicTransform:
        """Select the radial interpolator and recompute all retained harmonics.

        Parameters
        ----------
        interpolator : {'linear', 'cubic', 'pchip'}
            Interpolation used between the PETAL2D radial samples.

        Returns
        -------
        HarmonicTransform
            This instance after recomputing all ``F_m(q)`` values.

        Raises
        ------
        ValueError
            If ``interpolator`` is not one of the supported names.

        Notes
        -----
        Changing the interpolator clears the attached q-sampling and radial-
        quadrature convergence records.
        """
        self._interpolator = validate_interpolator(interpolator)
        self._sampling_convergence = None
        self._quadrature_convergence = None
        self._compute()
        return self

    def converge(
        self,
        *,
        rtol: float = 1.0e-3,
        atol: float = 0.0,
        subdivisions: Iterable[int] = (1, 2, 4, 8),
        hstart: float = 0.05,
        hdecrement: float = 2.0,
        maxiter: int = 15,
        apply: bool = False,
    ):
        """Converge only the radial quadrature on the current q grid.

        This low-level helper leaves ``q_max`` and ``n_q`` unchanged. Finite
        methods repeat each retained harmonic with progressively larger
        ``subdivisions`` and choose a common value sufficient for all modes.
        Use :meth:`converge_parameters` when q support and q-grid interpolation
        must also be included in the convergence search.

        Parameters
        ----------
        rtol : float, default=1e-3
            Relative self-convergence threshold between successive radial
            quadrature refinements.
        atol : float, default=0
            Absolute self-convergence threshold used together with ``rtol``.
        subdivisions : iterable of int, default=(1, 2, 4, 8)
            Candidate finite-grid refinements, tried in order. Each value is
            the number of integration panels placed inside one original radial
            interval.
        hstart : float, default=0.05
            Initial Ogata resolution parameter.
        hdecrement : float, default=2
            Factor controlling the Ogata resolution search.
        maxiter : int, default=15
            Maximum number of Ogata search iterations for each retained mode.
        apply : bool, default=False
            If ``True``, recompute this object with the selected radial
            quadrature parameters.

        Returns
        -------
        ConvergenceResult
            Internal self-convergence record and selected parameters.

        Raises
        ------
        RuntimeError
            If ``apply=True`` but the search does not produce converged
            parameters.
        ImportError
            If the active method is ``'ogata'`` and the optional ``hankel``
            dependency is unavailable.

        Notes
        -----
        The reported error compares numerical refinements. It is not an
        independent reference error.
        """
        from ._convergence import (
            ConvergenceResult,
            ConvergenceStep,
            converge_ogata,
            converge_sequence,
        )

        if self._method == "ogata":
            mode_results = []
            K = HankelTransform._representative_positive_values(self._q)
            for m_raw in self._m_values:
                m = int(m_raw)
                mask = self._r <= self._r_cutoffs[m]
                f = Interpolator1D(
                    self._r[mask],
                    np.asarray(self._decomposition[m])[mask],
                    self._interpolator,
                )
                mode_results.append(
                    (
                        m,
                        converge_ogata(
                            f,
                            abs(m),
                            K=K,
                            rtol=rtol,
                            atol=atol,
                            hstart=hstart,
                            hdecrement=hdecrement,
                            maxiter=maxiter,
                        ),
                    )
                )

            successful = [
                (m, result)
                for m, result in mode_results
                if result.resolution_converged
                and {"N", "h"}.issubset(result.selected_parameters)
            ]
            selected = (
                {
                    "N": max(int(result.selected_parameters["N"]) for _, result in successful),
                    "h": min(float(result.selected_parameters["h"]) for _, result in successful),
                }
                if successful
                else {}
            )
            all_converged = bool(mode_results) and len(successful) == len(mode_results)
            result = ConvergenceResult(
                method="ogata",
                rtol=rtol,
                atol=atol,
                selected_parameters=selected,
                selected_values=np.empty(0, dtype=np.complex128),
                steps=[
                    ConvergenceStep(
                        selected,
                        0.0,
                        converged=all_converged,
                        metadata={"source": "hankel.get_h"},
                    )
                ],
                resolution_converged=all_converged,
                metadata={
                    "kind": "hankel.get_h_across_harmonics",
                    "harmonics": {str(m): item.to_dict() for m, item in mode_results},
                    "n_converged_harmonics": len(successful),
                    "n_harmonics": len(mode_results),
                },
            )
        else:
            mode_results = []
            parameter_sets = [
                {"subdivisions": int(value)} for value in subdivisions
            ]
            for m_raw in self._m_values:
                m = int(m_raw)
                mask = self._r <= self._r_cutoffs[m]

                def evaluate(parameters, m=m, mask=mask):
                    return hankel_transform_sampled(
                        self._r[mask],
                        np.asarray(self._decomposition[m])[mask],
                        self._q,
                        abs(m),
                        method=self._method,
                        interpolator=self._interpolator,
                        subdivisions=int(parameters["subdivisions"]),
                        _allow_experimental=type(self)._allow_experimental_methods,
                    )

                mode_results.append(
                    (
                        m,
                        converge_sequence(
                            self._method,
                            parameter_sets,
                            evaluate,
                            rtol=rtol,
                            atol=atol,
                        ),
                    )
                )

            successful = [
                (m, item) for m, item in mode_results if item.resolution_converged
            ]
            selected = (
                {
                    "subdivisions": max(
                        int(item.selected_parameters["subdivisions"])
                        for _, item in successful
                    )
                }
                if successful
                else {}
            )
            all_converged = bool(mode_results) and len(successful) == len(mode_results)
            result = ConvergenceResult(
                method=self._method,
                rtol=rtol,
                atol=atol,
                selected_parameters=selected,
                selected_values=np.empty(0, dtype=np.complex128),
                steps=[
                    ConvergenceStep(
                        selected,
                        sum(
                            sum(step.runtime_seconds for step in item.steps)
                            for _, item in mode_results
                        ),
                        converged=all_converged,
                        metadata={"source": "per_harmonic_convergence"},
                    )
                ],
                resolution_converged=all_converged,
                metadata={
                    "kind": "finite_quadrature_across_harmonics",
                    "harmonics": {
                        str(m): item.to_dict() for m, item in mode_results
                    },
                    "n_converged_harmonics": len(successful),
                    "n_harmonics": len(mode_results),
                },
            )

        if apply:
            if not result.converged or not result.selected_parameters:
                raise RuntimeError("Cannot apply unconverged parameters.")
            if self._method == "ogata":
                self._N = int(result.selected_parameters["N"])
                self._h = float(result.selected_parameters["h"])
            else:
                self._subdivisions = int(result.selected_parameters["subdivisions"])
            self._compute()
        return result

    def plot_harmonics(self, title: str = "Hankel-transformed radial profiles"):
        """Plot the retained momentum-space harmonic profiles.

        Parameters
        ----------
        title : str, default="Hankel-transformed radial profiles"
            Figure title. Pass an empty string to suppress it.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axes : ndarray of matplotlib.axes.Axes
            Real- and imaginary-part axes.
        """
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.2, 6.0))
        for m_raw in self._m_values:
            m = int(m_raw)
            axes[0].plot(self._q, self.F_q[m].real, label=f"m={m}")
            axes[1].plot(self._q, self.F_q[m].imag, label=f"m={m}")
        axes[0].set_ylabel(r"Re $F_m(q)$")
        axes[1].set_ylabel(r"Im $F_m(q)$")
        axes[1].set_xlabel(r"$q$")
        axes[0].legend()
        axes[1].legend()
        axes[0].grid(True, alpha=0.2)
        axes[1].grid(True, alpha=0.2)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig, axes

    def plot_convergence(self, title: str = "Harmonic-transform convergence"):
        """Plot the attached automatic-convergence record.

        This delegates to the :class:`HarmonicConvergenceResult` associated
        with the calibrated production transform.

        Parameters
        ----------
        title : str, default="Harmonic-transform convergence"
            Figure title. Pass an empty string to suppress it.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axes : ndarray of matplotlib.axes.Axes
            Convergence-diagnostic axes.

        Raises
        ------
        RuntimeError
            If the transform was not constructed from
            ``HarmonicTransform.converge_parameters(...)``.
        """
        if self._sampling_convergence is None or self._quadrature_convergence is None:
            raise RuntimeError(
                "No convergence record is attached. Run "
                "HarmonicTransform.converge_parameters(...) and build the field "
                "with result.transform(decomposition)."
            )
        from .convergence import HarmonicConvergenceResult

        result = HarmonicConvergenceResult(
            sampling=self._sampling_convergence,
            quadrature=self._quadrature_convergence,
            method=self._method,
            interpolator=self._interpolator,
            rtol=float(self._quadrature_convergence.rtol),
            atol=float(self._quadrature_convergence.atol),
            q_tail_rtol=float(self._sampling_convergence.q_tail_rtol),
        )
        return result.plot_convergence(title=title)

    def plot(self, *, title: str = "Hankel-transformed radial profiles", show_convergence: bool = False):
        """Plot transformed harmonics and optionally the convergence record.

        Parameters
        ----------
        title : str, default="Hankel-transformed radial profiles"
            Title for the harmonic-profile figure.
        show_convergence : bool, default=False
            If ``True``, also return the attached convergence figure.

        Returns
        -------
        profiles : tuple
            ``(fig, axes)`` returned by :meth:`plot_harmonics`.
        convergence : tuple, optional
            ``(fig, axes)`` returned by :meth:`plot_convergence` when
            ``show_convergence=True``.

        Raises
        ------
        RuntimeError
            If ``show_convergence=True`` and no convergence record is attached.
        """
        profiles = self.plot_harmonics(title=title)
        if show_convergence:
            return profiles, self.plot_convergence()
        return profiles

    def roundtrip_error(self) -> float:
        """Return the power-weighted relative L2 round-trip error.

        Returns
        -------
        float
            Relative L2 difference between the retained PETAL2D radial
            harmonics and their forward/inverse Hankel reconstruction.

        Notes
        -----
        This is a numerical consistency diagnostic for the represented field;
        it is not an independent reference error for the original Cartesian
        input.
        """
        total_error_sq = 0.0
        for m_raw in self._m_values:
            m = int(m_raw)
            sign = (-1.0) ** abs(m) if m < 0 else 1.0
            reconstructed = sign * hankel_transform_sampled(
                self._q,
                self.F_q[m],
                self._r,
                abs(m),
                method=self._method,
                interpolator=self._interpolator,
                ogata_N=self._N,
                ogata_h=self._h,
                subdivisions=self._subdivisions,
                _allow_experimental=type(self)._allow_experimental_methods,
            )
            reference = np.asarray(self._rho[m])
            norm = float(np.linalg.norm(reference))
            mode_error = (
                0.0
                if norm == 0.0
                else float(np.linalg.norm(reference - reconstructed) / norm)
            )
            total_error_sq += mode_error**2 * float(self._decomposition.power_fracs[m])
        return float(np.sqrt(total_error_sq))

    def _compute(self) -> None:
        self._q = (
            self._explicit_q_grid.copy()
            if self._explicit_q_grid is not None
            else np.linspace(0.0, self._q_max, self._n_q)
        )
        self.F_q: dict[int, np.ndarray] = {}
        self._q_interpolants: dict[int, Interpolator1D] = {}

        for m_raw in self._m_values:
            m = int(m_raw)
            transform = type(self)._hankel_transform_type(
                m,
                self._q,
                self._decomposition[m],
                self._r,
                r_cutoff=self._r_cutoffs[m],
                method=self._method,
                interpolator=self._interpolator,
                subdivisions=self._subdivisions,
                N=self._N,
                h=self._h,
            )
            self.F_q[m] = transform.F_q
            self._q_interpolants[m] = Interpolator1D(
                self._q,
                self.F_q[m],
                self._interpolator,
                cubic_bc_type="not-a-knot",
            )

        self._diagnostics = inspect_transform(
            self._decomposition, self._q, self.F_q
        )
        if (
            self._used_default_q_max
            and self._default_grid_info["q_max_unclipped"] > self._q_max * (1.0 + 1.0e-12)
        ):
            self._diagnostics.issues.insert(
                0,
                DiagnosticIssue(
                    "default_q_range_clipped_by_radial_sampling",
                    "The scale-aware default q range had to be shortened to stay "
                    "below the momentum limit supported by the PETAL2D radial "
                    "spacing. The input radial resolution may be insufficient for "
                    "this profile; consider refining PETAL2D before trusting high-q "
                    "structure.",
                ),
            )
        if self._check and not self._diagnostics.healthy:
            warnings.warn(
                self._diagnostics.warning_message(),
                Quartic2DNumericalWarning,
                stacklevel=2,
            )


class _BoundaryTaperedField:
    """Read-only momentum field with a smooth taper at converged q support.

    This helper is used only as a downstream robustness probe during automatic
    Interaction calibration.  It preserves the native q grid and retained
    harmonics, while multiplying automatically sampled HarmonicTransform data
    by a half-cosine window over the final fraction of represented q support.
    """

    def __init__(self, field, taper_fraction: float):
        taper_fraction = float(taper_fraction)
        if not np.isfinite(taper_fraction) or not (0.0 < taper_fraction < 1.0):
            raise ValueError("taper_fraction must satisfy 0 < value < 1.")

        self._m_values = np.asarray(field.m_values, dtype=int).copy()
        self.q = np.asarray(field.q, dtype=float).copy()
        if self.q.ndim != 1 or self.q.size < 3:
            raise ValueError(
                "boundary robustness requires a one-dimensional q grid with at least three samples."
            )
        q_max = float(self.q[-1])
        if not np.isfinite(q_max) or q_max <= 0.0:
            raise ValueError("boundary robustness requires positive finite q support.")

        q_start = (1.0 - taper_fraction) * q_max
        window = np.ones_like(self.q)
        mask = self.q > q_start
        phase = (self.q[mask] - q_start) / (q_max - q_start)
        window[mask] = 0.5 * (1.0 + np.cos(np.pi * phase))

        self.F_q = {
            int(m): np.asarray(field.F_q[int(m)], dtype=np.complex128) * window
            for m in self._m_values
        }

    @property
    def m_values(self) -> np.ndarray:
        return self._m_values.copy()


class Interaction:
    r"""Evaluate four-center matrix elements from transformed transition fields.

    For transition fields

    .. math::

        \rho_{13}(\mathbf r)=\phi_1^*(\mathbf r)\phi_3(\mathbf r),
        \qquad
        \rho_{42}(\mathbf r)=\phi_4^*(\mathbf r)\phi_2(\mathbf r),

    ``Interaction`` evaluates the radial-kernel reduction of

    .. math::

        U_{1234}(\boldsymbol\delta)=
        \iint d^2\mathbf s\,d^2\mathbf t\,
        \rho_{13}(\mathbf s)
        U(|\mathbf s-\mathbf t+\boldsymbol\delta|)
        \rho_{42}^*(\mathbf t).

    The two fields are supplied as :class:`HarmonicTransform` objects.  The
    ordinary constructor performs one evaluation using explicit numerical
    parameters.  :meth:`converge_parameters` is the recommended opt-in path
    when the assembled interaction requires a recorded refinement study.

    Parameters
    ----------
    deltas : array_like, shape (D, 2)
        Cartesian displacement vectors :math:`\boldsymbol\delta`.
    field1, field2 : HarmonicTransform
        Momentum-space harmonic representations of the two transition fields.
    U_q : callable
        Scalar radial interaction kernel in momentum space.  The callable is
        evaluated at non-negative momentum magnitudes.
    method : {'fftlog', 'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}, default='gl4'
        Numerical method used for the final radial interaction integrals.  The
        methods have different convergence controls; see
        :meth:`converge_parameters` and the numerical-method guide rather than
        interpreting the method name as an accuracy ranking.
    interpolator : {'linear', 'cubic', 'pchip'}, default='cubic'
        Interpolator used for the sampled momentum-space harmonics.
    n : int, default=512
        FFTLog sequence length.  Used only for ``method='fftlog'``.
    bias : float, default=-0.5
        FFTLog power-law bias.  Used only for ``method='fftlog'``.
    subdivisions : int, default=1
        Number of equal finite-rule subdivisions per original momentum
        interval.  Used by ``trapezoid``, ``simpson``, ``gl4``, and ``gl8``.
    N : int, default=1024
        Ogata node count.  Used only for ``method='ogata'``.
    h : float or None, optional
        Ogata resolution parameter.  ``None`` selects the package default for
        the requested ``N``.

    Attributes
    ----------
    Phi_mm : ndarray
        Angular prefactors for every harmonic pair and displacement.
    H_mm : ndarray
        Radial interaction integrals for every harmonic pair and displacement.
    V_mm : ndarray
        Harmonic-pair contributions to the final matrix element.
    V : ndarray, shape (D,)
        Total interaction for each displacement.
    convergence : InteractionConvergenceResult or None
        Calibration record attached when the object is constructed through
        :meth:`InteractionConvergenceResult.interaction`.

    See Also
    --------
    Interaction.converge_parameters
        Recommended interaction-level calibration workflow.
    InteractionConvergenceResult
        Reusable selected parameters and convergence history.
    HarmonicTransform
        Momentum-space representation of one transition field.
    """
    _allow_experimental_methods = False

    def __init__(
        self,
        deltas: np.ndarray,
        field1: HarmonicTransform,
        field2: HarmonicTransform,
        U_q: Callable,
        *,
        method: str = "gl4",
        interpolator: str = "cubic",
        n: int = 512,
        bias: float = -0.5,
        subdivisions: int = 1,
        N: int = 1024,
        h: float | None = None,
    ):
        self._initialize(
            deltas,
            field1,
            field2,
            U_q,
            method=method,
            interpolator=interpolator,
            n=n,
            bias=bias,
            subdivisions=subdivisions,
            N=N,
            h=h,
        )
        self._compute()

    def _initialize(
        self,
        deltas: np.ndarray,
        field1: HarmonicTransform,
        field2: HarmonicTransform,
        U_q: Callable,
        *,
        method: str,
        interpolator: str,
        n: int,
        bias: float,
        subdivisions: int,
        N: int,
        h: float | None,
    ) -> None:
        if not callable(U_q):
            raise TypeError("U_q must be callable.")

        self._field1 = field1
        self._field2 = field2
        self._U_q = U_q
        self._method = _validate_interaction_method(method, allow_experimental=type(self)._allow_experimental_methods)
        self._interpolator = validate_interpolator(interpolator)
        self._fftlog_n = HankelTransform._positive_int(n, "n")
        self._fftlog_bias = float(bias)
        if not np.isfinite(self._fftlog_bias):
            raise ValueError("bias must be finite.")
        self._subdivisions = HankelTransform._positive_int(subdivisions, "subdivisions")
        self._N = HankelTransform._positive_int(N, "N")
        self._h = None if h is None else float(h)
        if self._h is not None and (not np.isfinite(self._h) or self._h <= 0.0):
            raise ValueError("h must be finite and positive.")
        self._ogata_cache: dict[int, object] = {}
        self._mode_cache1: dict[int, Callable] = {}
        self._mode_cache2: dict[int, Callable] = {}
        self._pair_grid_cache: dict[tuple[int, int], np.ndarray] = {}
        self._common_grid_cache: dict[tuple[bytes, bytes], np.ndarray] = {}
        self._evaluation_cache: dict | None = None

        self._deltas = self._validate_deltas(deltas)
        self._delta_mags = np.linalg.norm(self._deltas, axis=1)
        self._delta_angles = np.arctan2(self._deltas[:, 1], self._deltas[:, 0])
        self._m_values1 = self._field_m_values(field1)
        self._m_values2 = self._field_m_values(field2)

        self.Phi_mm = self._phase_factors()
        self._convergence_result = None

    @classmethod
    def converge_parameters(
        cls,
        deltas: np.ndarray,
        field1: HarmonicTransform,
        field2: HarmonicTransform,
        U_q: Callable,
        *,
        rtol: float = 1.0e-4,
        atol: float = 1.0e-12,
        method: str = "gl4",
        interpolator: str = "cubic",
        subdivisions: Iterable[int] = (
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
        ),
        n_values: Iterable[int] = (128, 256, 512, 1024, 2048, 4096),
        bias: float = -0.5,
        bias_values: Iterable[float] = (-0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35),
        hstart: float = 0.05,
        hdecrement: float = 2.0,
        maxiter: int = 20,
        ogata_n_values: Iterable[int] = (
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
            65536, 131072, 262144, 524288,
        ),
        ogata_verification_levels: int = 4,
        q_boundary_check: bool = True,
        q_boundary_taper_fraction: float = 0.1,
        q_boundary_budget_fraction: float = 0.5,
        verbose: bool = True,
    ) -> InteractionConvergenceResult:
        r"""Calibrate the assembled interaction for explicit numerical criteria.

        This is the recommended quantitative convergence entry point for
        ``Interaction``.  It selects method-specific numerical parameters from
        self-convergence tests of the assembled interaction over the supplied
        displacement set.  The requested ``rtol`` is therefore a refinement
        criterion, not an independent reference-error guarantee.

        Finite-grid rules use their known asymptotic order together with a
        three-resolution Richardson check.  FFTLog must stabilize with both
        transform length and a local bias window.  Ogata is calibrated directly
        on the assembled interaction in the coupled ``(N, h)`` space.  When an
        input ``HarmonicTransform`` carries automatic q-sampling metadata, an
        optional downstream q-boundary probe also tests sensitivity to the
        finite represented support.

        Parameters
        ----------
        deltas : array_like, shape (D, 2)
            Cartesian displacement vectors included in the convergence study.
            The selected parameters are justified for this tested displacement
            domain; extending the domain can require recalibration.
        field1, field2 : HarmonicTransform
            Momentum-space transition fields used by the interaction.
        U_q : callable
            Scalar radial interaction kernel in momentum space.
        rtol : float, default=1e-4
            Relative self-convergence threshold applied to successive
            interaction values.  Acceptance is paired with the peak-scaled
            absolute condition controlled by ``atol``.  For finite-grid rules
            the internal search uses a safety margin before reporting success.
        atol : float, default=1e-12
            Absolute floor in the peak-scaled maximum-change condition.
        method : {'fftlog', 'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}, default='gl4'
            Interaction integration method to calibrate.
        interpolator : {'linear', 'cubic', 'pchip'}, default='cubic'
            Interpolator used for the sampled momentum-space harmonics.
        subdivisions : iterable of int, optional
            Ordered candidate finite-rule subdivision counts.  Used by
            ``trapezoid``, ``simpson``, ``gl4``, and ``gl8``.
        n_values : iterable of int, optional
            Ordered FFTLog sequence lengths used for resolution refinement.
        bias : float, default=-0.5
            Preferred FFTLog bias around which robustness is tested.
        bias_values : iterable of float, optional
            FFTLog bias values used to establish a local robust window.  An
            ``n``-stable but bias-sensitive result is not accepted.
        hstart : float, default=0.05
            Initial Ogata resolution parameter used by the coupled search.
        hdecrement : float, default=2.0
            Multiplicative refinement factor applied to the Ogata ``h`` search.
        maxiter : int, default=20
            Maximum number of Ogata ``h`` refinements per tested node count.
        ogata_n_values : iterable of int, optional
            Ordered Ogata node counts tested during coupled ``(N, h)``
            calibration.
        ogata_verification_levels : int, default=4
            Number of subsequent Ogata resolution levels required to remain
            stable after the first accepted candidate.
        q_boundary_check : bool, default=True
            If ``True`` and either input carries automatic q-sampling metadata,
            test the assembled interaction against a smooth taper near the
            outer represented q boundary.
        q_boundary_taper_fraction : float, default=0.1
            Fraction of the represented q interval over which the robustness
            probe applies its half-cosine taper.  Must satisfy ``0 < value < 1``.
        q_boundary_budget_fraction : float, default=0.5
            Fraction of the requested ``rtol``/``atol`` budget allocated to the
            q-boundary robustness probe.  Must satisfy ``0 < value <= 1``.
        verbose : bool, default=True
            Print one compact convergence report.  Set to ``False`` for batch
            calculations; the returned object retains the complete search.

        Returns
        -------
        InteractionConvergenceResult
            Reusable convergence result containing selected parameters and the
            full method-specific search history.  Use
            ``result.interaction(deltas, field1, field2, U_q)`` to construct a
            production interaction without rerunning calibration.

        Raises
        ------
        ValueError
            If method-specific controls or q-boundary settings are invalid.

        Notes
        -----
        ``result.converged`` describes the configured internal refinement and
        robustness criteria for the supplied represented inputs.  It does not
        assert pointwise or global agreement with an unknown exact solution.
        When an independent analytic or high-accuracy reference is available,
        evaluate that error separately.
        """
        from ._convergence import (
            absolute_linf_error,
            converge_fftlog_bias_resolution,
            converge_fixed_order_sequence,
            converge_ogata_coupled,
            relative_l2_error,
            relative_linf_error,
        )

        method = _validate_interaction_method(method, allow_experimental=cls._allow_experimental_methods)
        interpolator = validate_interpolator(interpolator)
        deltas = cls._validate_deltas(deltas)
        q_boundary_taper_fraction = float(q_boundary_taper_fraction)
        q_boundary_budget_fraction = float(q_boundary_budget_fraction)
        if not np.isfinite(q_boundary_taper_fraction) or not (0.0 < q_boundary_taper_fraction < 1.0):
            raise ValueError("q_boundary_taper_fraction must satisfy 0 < value < 1.")
        if not np.isfinite(q_boundary_budget_fraction) or not (0.0 < q_boundary_budget_fraction <= 1.0):
            raise ValueError("q_boundary_budget_fraction must satisfy 0 < value <= 1.")

        calibration = cls.__new__(cls)
        calibration._initialize(
            deltas,
            field1,
            field2,
            U_q,
            method=method,
            interpolator=interpolator,
            n=512,
            bias=bias,
            subdivisions=1,
            N=1024,
            h=None,
        )
        evaluated: dict[tuple, np.ndarray] = {}

        def evaluate(parameters):
            key = tuple(sorted((str(k), float(v)) for k, v in parameters.items()))
            if key in evaluated:
                return evaluated[key]

            calibration._fftlog_n = int(parameters.get("n", 512))
            calibration._fftlog_bias = float(parameters.get("bias", bias))
            calibration._subdivisions = int(parameters.get("subdivisions", 1))
            N = int(parameters.get("N", 1024))
            h = parameters.get("h", None)
            h = None if h is None else float(h)
            if N != calibration._N or h != calibration._h:
                calibration._ogata_cache.clear()
            calibration._N = N
            calibration._h = h

            H_mm = calibration._calculate_H_mm()
            values = np.asarray(
                (2.0 * np.pi * calibration.Phi_mm * H_mm).sum(axis=(0, 1)),
                dtype=np.complex128,
            )
            evaluated[key] = values
            return values

        if method in {"trapezoid", "simpson", "gl4", "gl8"}:
            expected_order = {
                "trapezoid": 2.0,
                "simpson": 4.0,
                "gl4": 8.0,
                "gl8": 16.0,
            }[method]
            # Keep a safety margin between a self-convergence estimate and
            # the user's requested tolerance.  This protects against mildly
            # non-smooth kernels whose observed finite-rule order can be
            # temporarily optimistic.
            search = converge_fixed_order_sequence(
                method,
                "subdivisions",
                subdivisions,
                evaluate,
                expected_order=expected_order,
                rtol=0.5 * float(rtol),
                atol=0.5 * float(atol),
            )
            search.metadata["requested_rtol"] = float(rtol)
            search.metadata["requested_atol"] = float(atol)
            search.metadata["internal_budget_fraction"] = 0.5
        elif method == "fftlog":
            # FFTLog has two independently tested numerical sensitivities:
            # transform length and bias.  Give each half of the requested
            # tolerance so their combination is not accepted merely because
            # both changes sit just below the final user target.
            search = converge_fftlog_bias_resolution(
                evaluate,
                n_values=n_values,
                bias_values=bias_values,
                preferred_bias=bias,
                rtol=0.5 * float(rtol),
                atol=0.5 * float(atol),
            )
            search.metadata["requested_rtol"] = float(rtol)
            search.metadata["requested_atol"] = float(atol)
            search.metadata["internal_budget_fraction"] = 0.5
        else:
            # Converge the quantity users actually consume: the assembled
            # interaction.  N is first converged at fixed h; neighboring h
            # plateaus then receive a second look-ahead check.  A
            # quarter-tolerance internal budget leaves room for both N and h
            # truncation effects without relying on a per-pair get_h seed.
            search = converge_ogata_coupled(
                evaluate,
                n_values=ogata_n_values,
                hstart=hstart,
                hdecrement=hdecrement,
                maxiter=maxiter,
                rtol=0.25 * float(rtol),
                atol=0.25 * float(atol),
            )
            search.metadata["requested_rtol"] = float(rtol)
            search.metadata["requested_atol"] = float(atol)
            search.metadata["internal_budget_fraction"] = 0.25
            # Retained for API compatibility with the first experimental
            # Ogata convergence implementation.  The coupled search now uses
            # its own two-level look-ahead and does not consume this value.
            search.metadata["legacy_ogata_verification_levels"] = int(
                ogata_verification_levels
            )

        # A self-converged interaction backend can still be wrong when the
        # observable is conditioned on tiny differences near the finite q
        # boundary of an automatically sampled HarmonicTransform.  Probe that
        # upstream support without an external reference by smoothly removing
        # only the final q-window and requiring the selected production result
        # to remain stable.  The check is intentionally limited to fields that
        # carry an automatic q-sampling convergence result; fixed user-supplied
        # fields continue to mean "integrate the represented input as given".
        sampled_inputs = [
            field
            for field in (field1, field2)
            if getattr(field, "sampling_convergence", None) is not None
        ]
        boundary_metadata = {
            "enabled": bool(q_boundary_check),
            "eligible_input_count": len(sampled_inputs),
            "tested": False,
            "taper_fraction": q_boundary_taper_fraction,
            "budget_fraction": q_boundary_budget_fraction,
            "requested_rtol": float(rtol),
            "requested_atol": float(atol),
        }
        if q_boundary_check and search.converged and sampled_inputs:
            tapered1 = (
                _BoundaryTaperedField(field1, q_boundary_taper_fraction)
                if getattr(field1, "sampling_convergence", None) is not None
                else field1
            )
            tapered2 = (
                _BoundaryTaperedField(field2, q_boundary_taper_fraction)
                if getattr(field2, "sampling_convergence", None) is not None
                else field2
            )
            selected = dict(search.selected_parameters)
            probe = cls(
                deltas,
                tapered1,
                tapered2,
                U_q,
                method=method,
                interpolator=interpolator,
                n=int(selected.get("n", 512)),
                bias=float(selected.get("bias", bias)),
                subdivisions=int(selected.get("subdivisions", 1)),
                N=int(selected.get("N", 1024)),
                h=selected.get("h", None),
            )
            baseline = np.asarray(search.selected_values, dtype=np.complex128)
            tapered_values = np.asarray(probe.V, dtype=np.complex128)
            rel_l2 = relative_l2_error(tapered_values, baseline)
            rel_linf = relative_linf_error(tapered_values, baseline)
            abs_linf = absolute_linf_error(tapered_values, baseline)
            peak = float(np.max(np.abs(baseline))) if baseline.size else 0.0
            boundary_rtol = q_boundary_budget_fraction * float(rtol)
            boundary_atol = q_boundary_budget_fraction * float(atol)
            passed = bool(
                np.isfinite(rel_l2)
                and rel_l2 <= boundary_rtol
                and np.isfinite(abs_linf)
                and abs_linf <= boundary_atol + boundary_rtol * peak
            )
            q_max_values = [
                float(np.asarray(field.q, dtype=float)[-1]) for field in sampled_inputs
            ]
            delta_max = float(np.max(np.linalg.norm(deltas, axis=1)))
            boundary_metadata.update(
                {
                    "tested": True,
                    "passed": passed,
                    "boundary_rtol": boundary_rtol,
                    "boundary_atol": boundary_atol,
                    "relative_l2_change": rel_l2,
                    "relative_linf_change": rel_linf,
                    "absolute_linf_change": abs_linf,
                    "reference_peak": peak,
                    "q_max_values": q_max_values,
                    "delta_max": delta_max,
                    "max_q_delta": max(q_max_values) * delta_max,
                    "status": "passed" if passed else "upstream_q_boundary_not_robust",
                }
            )
            if not passed:
                search.hyperparameter_robust = False
                search.metadata["pre_boundary_status"] = search.metadata.get("status")
                search.metadata["status"] = "upstream_q_boundary_not_robust"
        elif not q_boundary_check:
            boundary_metadata["status"] = "disabled"
        elif not sampled_inputs:
            boundary_metadata["status"] = "not_applicable_fixed_input"
        else:
            boundary_metadata["status"] = "not_tested_backend_unconverged"
        search.metadata["q_boundary_robustness"] = boundary_metadata

        result = InteractionConvergenceResult(
            search=search,
            method=method,
            interpolator=interpolator,
            rtol=float(rtol),
            atol=float(atol),
        )
        result._interaction_class = cls
        if verbose:
            print(result)
        return result

    @property
    def deltas(self) -> np.ndarray:
        """Cartesian displacement vectors."""
        return self._deltas.copy()

    @property
    def method(self) -> str:
        """Current interaction-transform method."""
        return self._method

    @property
    def interpolator(self) -> str:
        """Current momentum-space interpolator."""
        return self._interpolator

    @property
    def convergence(self):
        """Automatic convergence record attached to a calibrated production run."""
        return self._convergence_result

    def plot_convergence(self, title: str = "Interaction convergence"):
        """Plot the attached interaction-level convergence record.

        Parameters
        ----------
        title : str, default="Interaction convergence"
            Figure title. Pass an empty string to suppress it.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axes : ndarray of matplotlib.axes.Axes
            Method-specific convergence-diagnostic axes.

        Raises
        ------
        RuntimeError
            If the interaction was not constructed from
            ``Interaction.converge_parameters(...)``.

        Notes
        -----
        This method delegates to
        :meth:`InteractionConvergenceResult.plot_convergence`.  The plot shows
        internal self-convergence and robustness diagnostics, not an
        independent reference error.
        """
        if self._convergence_result is None:
            raise RuntimeError(
                "No convergence record is attached. Run "
                "Interaction.converge_parameters(...) and build the production "
                "interaction with result.interaction(...)."
            )
        return self._convergence_result.plot_convergence(title=title)

    def set_method(
        self,
        method: str,
        *,
        n: int | None = None,
        bias: float | None = None,
        subdivisions: int | None = None,
        N: int | None = None,
        h: float | None = None,
    ) -> Interaction:
        """Select the interaction quadrature and recompute ``V``.

        Parameters
        ----------
        method : {'fftlog', 'trapezoid', 'simpson', 'gl4', 'gl8', 'ogata'}
            Method used for the final radial interaction integrals.
        n : int or None, optional
            FFTLog sequence length. Used only by ``method='fftlog'``.
        bias : float or None, optional
            FFTLog power-law bias. Used only by ``method='fftlog'``.
        subdivisions : int or None, optional
            Finite-grid refinement used by ``trapezoid``, ``simpson``,
            ``gl4``, and ``gl8``.
        N : int or None, optional
            Ogata node count. Used only by ``method='ogata'``.
        h : float or None, optional
            Positive Ogata resolution parameter. Used only by
            ``method='ogata'``.

        Returns
        -------
        Interaction
            This instance after recomputing ``V`` and harmonic-pair
            contributions.

        Raises
        ------
        ValueError
            If ``method`` is unsupported or a supplied numerical parameter is
            invalid.
        ImportError
            If ``method='ogata'`` and the optional ``hankel`` dependency is
            unavailable.

        Notes
        -----
        Method-specific parameters that are not supplied retain their current
        values. Changing the method clears cached Ogata data and any attached
        interaction convergence record.
        """
        self._method = _validate_interaction_method(method, allow_experimental=type(self)._allow_experimental_methods)
        if n is not None:
            self._fftlog_n = HankelTransform._positive_int(n, "n")
        if bias is not None:
            bias = float(bias)
            if not np.isfinite(bias):
                raise ValueError("bias must be finite.")
            self._fftlog_bias = bias
        if subdivisions is not None:
            self._subdivisions = HankelTransform._positive_int(subdivisions, "subdivisions")
        if N is not None:
            self._N = HankelTransform._positive_int(N, "N")
        if h is not None:
            h = float(h)
            if not np.isfinite(h) or h <= 0.0:
                raise ValueError("h must be finite and positive.")
            self._h = h
        self._ogata_cache.clear()
        self._convergence_result = None
        self._compute()
        return self

    def set_interpolator(self, interpolator: str) -> Interaction:
        """Select the momentum-space interpolator and recompute ``V``.

        Parameters
        ----------
        interpolator : {'linear', 'cubic', 'pchip'}
            Interpolation used for the sampled momentum-space harmonics.

        Returns
        -------
        Interaction
            This instance after recomputing ``V`` and harmonic-pair
            contributions.

        Raises
        ------
        ValueError
            If ``interpolator`` is not one of the supported names.

        Notes
        -----
        Changing the interpolator clears the harmonic interpolation caches and
        any attached interaction convergence record.
        """
        self._interpolator = validate_interpolator(interpolator)
        self._mode_cache1.clear()
        self._mode_cache2.clear()
        self._convergence_result = None
        self._compute()
        return self

    def converge(
        self,
        *,
        rtol: float = 1.0e-3,
        atol: float = 0.0,
        subdivisions: Iterable[int] = (1, 2, 4, 8),
        n_values: Iterable[int] = (128, 256, 512, 1024),
        hstart: float = 0.05,
        hdecrement: float = 2.0,
        maxiter: int = 15,
        apply: bool = False,
    ):
        """Refine the active interaction method at fixed transformed fields.

        Parameters
        ----------
        rtol : float, default=1e-3
            Relative self-convergence threshold between successive interaction
            refinements.
        atol : float, default=0
            Absolute self-convergence threshold used together with ``rtol``.
        subdivisions : iterable of int, default=(1, 2, 4, 8)
            Candidate subdivision counts for the finite-grid methods.
        n_values : iterable of int, default=(128, 256, 512, 1024)
            Candidate FFTLog sequence lengths. The current ``bias`` is held
            fixed during this search.
        hstart : float, default=0.05
            Initial Ogata resolution parameter.
        hdecrement : float, default=2
            Factor controlling the Ogata resolution search.
        maxiter : int, default=15
            Maximum number of Ogata search iterations for each harmonic pair.
        apply : bool, default=False
            If ``True``, apply the selected parameters to this object and
            recompute the interaction.

        Returns
        -------
        ConvergenceResult
            Internal self-convergence record and selected parameters.

        Raises
        ------
        RuntimeError
            If ``apply=True`` but the search does not produce converged
            parameters.
        ImportError
            If the active method is ``'ogata'`` and the optional ``hankel``
            dependency is unavailable.

        Notes
        -----
        FFTLog refines ``n`` at fixed ``bias``. Finite-grid methods refine
        ``subdivisions``. Ogata searches ``N`` and ``h`` independently for the
        retained harmonic pairs. This method does not modify the q support of
        the input :class:`HarmonicTransform` objects. The reported error is an
        internal refinement diagnostic, not an independent reference error.
        """
        from ._convergence import (
            ConvergenceResult,
            ConvergenceStep,
            converge_ogata,
            converge_sequence,
        )

        saved = (self._N, self._h, self._fftlog_n, self._subdivisions)

        def evaluate(parameters):
            old = (self._N, self._h, self._fftlog_n, self._subdivisions)
            try:
                if "n" in parameters:
                    self._fftlog_n = int(parameters["n"])
                if "subdivisions" in parameters:
                    self._subdivisions = int(parameters["subdivisions"])
                self._ogata_cache.clear()
                H_mm = self._calculate_H_mm()
                return (2.0 * np.pi * self.Phi_mm * H_mm).sum(axis=(0, 1))
            finally:
                self._N, self._h, self._fftlog_n, self._subdivisions = old
                self._ogata_cache.clear()

        if self._method == "fftlog":
            result = converge_sequence(
                "fftlog",
                [{"n": int(value), "bias": float(self._fftlog_bias)} for value in n_values],
                evaluate,
                rtol=rtol,
                atol=atol,
            )
        elif self._method == "ogata":
            K = HankelTransform._representative_positive_values(self._delta_mags)
            if K is None:
                K = np.array([1.0])
            pair_results = []
            for m_raw in self._m_values1:
                m = int(m_raw)
                for mp_raw in self._m_values2:
                    mp = int(mp_raw)
                    order = abs(m - mp)
                    f1 = self._mode_callable(self._field1, m)
                    f2 = self._mode_callable(self._field2, mp)

                    def pair_function(q, f1=f1, f2=f2):
                        return f1(q) * np.conj(f2(q)) * self._U_q(q)

                    pair_results.append(
                        (
                            (m, mp),
                            converge_ogata(
                                pair_function,
                                order,
                                K=K,
                                rtol=rtol,
                                atol=atol,
                                hstart=hstart,
                                hdecrement=hdecrement,
                                maxiter=maxiter,
                            ),
                        )
                    )

            successful = [
                (pair, result)
                for pair, result in pair_results
                if result.resolution_converged
                and {"N", "h"}.issubset(result.selected_parameters)
            ]
            selected = (
                {
                    "N": max(int(result.selected_parameters["N"]) for _, result in successful),
                    "h": min(float(result.selected_parameters["h"]) for _, result in successful),
                }
                if successful
                else {}
            )
            all_converged = bool(pair_results) and len(successful) == len(pair_results)
            result = ConvergenceResult(
                method="ogata",
                rtol=rtol,
                atol=atol,
                selected_parameters=selected,
                selected_values=np.empty(0, dtype=np.complex128),
                steps=[
                    ConvergenceStep(
                        selected,
                        0.0,
                        converged=all_converged,
                        metadata={"source": "hankel.get_h"},
                    )
                ],
                resolution_converged=all_converged,
                metadata={
                    "kind": "hankel.get_h_across_harmonic_pairs",
                    "pairs": {
                        f"{m},{mp}": item.to_dict()
                        for (m, mp), item in pair_results
                    },
                    "n_converged_pairs": len(successful),
                    "n_pairs": len(pair_results),
                },
            )
        else:
            result = converge_sequence(
                self._method,
                [{"subdivisions": int(value)} for value in subdivisions],
                evaluate,
                rtol=rtol,
                atol=atol,
            )

        if apply:
            if not result.converged or not result.selected_parameters:
                raise RuntimeError("Cannot apply unconverged parameters.")
            if self._method == "fftlog":
                self._fftlog_n = int(result.selected_parameters["n"])
            elif self._method == "ogata":
                self._N = int(result.selected_parameters["N"])
                self._h = float(result.selected_parameters["h"])
            else:
                self._subdivisions = int(result.selected_parameters["subdivisions"])
            self._compute()
        else:
            self._N, self._h, self._fftlog_n, self._subdivisions = saved
        return result

    @staticmethod
    def _validate_deltas(deltas: np.ndarray) -> np.ndarray:
        values = np.asarray(deltas, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                "deltas must have shape (D, 2); "
                f"received {values.shape}."
            )
        if values.shape[0] == 0:
            raise ValueError("deltas must contain at least one displacement.")
        if not np.all(np.isfinite(values)):
            raise ValueError("deltas must contain only finite values.")
        return values.copy()

    @staticmethod
    def _field_m_values(field) -> np.ndarray:
        if not hasattr(field, "m_values"):
            raise TypeError("field must provide an m_values property.")
        values = np.asarray(field.m_values, dtype=int)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("field.m_values must be a non-empty one-dimensional array.")
        return values

    @staticmethod
    def _field_q_grid(field, m: int) -> np.ndarray:
        q = field.q
        if isinstance(q, dict):
            q = q[int(m)]
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or q.size < 2 or np.any(np.diff(q) <= 0.0):
            raise ValueError("field q grid must be one-dimensional and strictly increasing.")
        return q

    def _phase_factors(self) -> np.ndarray:
        m_difference = self._m_values1[:, None] - self._m_values2[None, :]
        # The exact reduced interaction contains J_{m'-m}.  Numerical
        # backends evaluate J_{|m-m'|}, so positive odd m-m' pairs require
        # the parity factor from J_{-(m-m')} = (-1)^(m-m') J_{m-m'}.
        bessel_parity = np.where(
            (m_difference > 0) & (m_difference % 2 != 0), -1.0, 1.0
        )
        return bessel_parity[:, :, None] * np.exp(
            1j * m_difference[:, :, None] * self._delta_angles[None, None, :]
        )

    def _mode_callable(self, field, m: int) -> Callable:
        if field is self._field1:
            cache = self._mode_cache1
        elif field is self._field2:
            cache = self._mode_cache2
        else:
            cache = None
        if cache is not None and int(m) in cache:
            return cache[int(m)]

        if hasattr(field, "F_q"):
            q = self._field_q_grid(field, m)
            result = Interpolator1D(
                q,
                field.F_q[int(m)],
                self._interpolator,
                cubic_bc_type="not-a-knot",
            )
        elif callable(field):
            result = lambda q: field(int(m), q)
        else:
            raise TypeError("field must provide F_q values or be callable as field(m, q).")

        if cache is not None:
            cache[int(m)] = result
        return result

    def _pair_grid(self, m: int, mp: int) -> np.ndarray:
        pair = (int(m), int(mp))
        if pair in self._pair_grid_cache:
            return self._pair_grid_cache[pair]
        q1 = self._field_q_grid(self._field1, m)
        q2 = self._field_q_grid(self._field2, mp)
        key = (q1.tobytes(), q2.tobytes())
        grid = self._common_grid_cache.get(key)
        if grid is None:
            grid = common_grid(q1, q2)
            self._common_grid_cache[key] = grid
        self._pair_grid_cache[pair] = grid
        return grid

    def _finite_geometry(self, m: int, mp: int, method: str):
        grid = self._pair_grid(m, mp)
        cache = self._evaluation_cache
        key = ("geometry", id(grid), int(self._subdivisions), method)
        if cache is not None and key in cache:
            return cache[key]

        panel_grid = subdivide_grid(grid, self._subdivisions)
        if method in ("gl4", "gl8"):
            gl_order = 4 if method == "gl4" else 8
            q, weights = composite_gauss_nodes_weights(panel_grid, gl_order)
        else:
            q, weights = panel_grid, None
        result = (q, weights)
        if cache is not None:
            cache[key] = result
        return result

    def _mode_values(self, field_index: int, m: int, q: np.ndarray) -> np.ndarray:
        cache = self._evaluation_cache
        key = ("mode", int(field_index), int(m), id(q))
        if cache is not None and key in cache:
            return cache[key]
        field = self._field1 if field_index == 1 else self._field2
        values = np.asarray(self._mode_callable(field, m)(q), dtype=np.complex128)
        if cache is not None:
            cache[key] = values
        return values

    def _kernel_values(self, q: np.ndarray) -> np.ndarray:
        cache = self._evaluation_cache
        key = ("kernel", id(q))
        if cache is not None and key in cache:
            return cache[key]
        values = np.asarray(self._U_q(q))
        if cache is not None:
            cache[key] = values
        return values

    def _positive_kernel_values(self, q: np.ndarray) -> np.ndarray:
        cache = self._evaluation_cache
        key = ("kernel_positive", id(q))
        if cache is not None and key in cache:
            return cache[key]
        values = np.asarray(self._U_q(q[q > 0.0]))
        if cache is not None:
            cache[key] = values
        return values

    def _bessel_matrix(self, order: int, q: np.ndarray) -> np.ndarray:
        cache = self._evaluation_cache
        key = ("bessel", int(order), id(q))
        if cache is not None and key in cache:
            return cache[key]
        values = jv(int(order), np.outer(self._delta_mags, q))
        if cache is not None:
            cache[key] = values
        return values

    def _get_ogata(self, order: int):
        hankel = _require_hankel()
        h = np.pi / self._N if self._h is None else self._h
        if order not in self._ogata_cache:
            self._ogata_cache[order] = hankel.HankelTransform(
                nu=order, N=self._N, h=h
            )
        return self._ogata_cache[order]

    def _finite_pair_integral(self, m: int, mp: int, order: int, method: str) -> np.ndarray:
        q, weights = self._finite_geometry(m, mp, method)
        f1 = self._mode_values(1, m, q)
        f2 = self._mode_values(2, mp, q)

        if method in ("gl4", "gl8"):
            base = weights * q * f1 * np.conj(f2) * self._kernel_values(q)
            return self._bessel_matrix(order, q) @ base

        base = np.zeros(q.size, dtype=np.complex128)
        positive = q > 0.0
        qp = q[positive]
        base[positive] = (
            qp
            * f1[positive]
            * np.conj(f2[positive])
            * self._positive_kernel_values(q)
        )
        if np.isclose(q[0], 0.0):
            if order == 0:
                cache = self._evaluation_cache
                qU0_key = ("qU0", float(qp[0]))
                if cache is not None and qU0_key in cache:
                    qU0 = cache[qU0_key]
                else:
                    qU0 = extrapolate_q_times_kernel_zero(self._U_q, qp[0])
                    if cache is not None:
                        cache[qU0_key] = qU0
                base[0] = f1[0] * np.conj(f2[0]) * qU0
            else:
                base[0] = 0.0

        matrix = self._bessel_matrix(order, q) * base[None, :]
        if method == "trapezoid":
            return trapezoid(matrix, x=q, axis=1)
        if method == "simpson":
            return simpson(matrix, x=q, axis=1)
        raise ValueError(method)

    def _fftlog_all_pairs(self) -> np.ndarray:
        shape = (len(self._m_values1), len(self._m_values2), len(self._delta_mags))
        H_mm = np.zeros(shape, dtype=np.complex128)
        positive = self._delta_mags > 0.0

        groups: dict[tuple[int, int, float, float], list[tuple[int, int, int, int]]] = {}
        if np.any(positive):
            for i, m_raw in enumerate(self._m_values1):
                m = int(m_raw)
                for j, mp_raw in enumerate(self._m_values2):
                    mp = int(mp_raw)
                    order = abs(m - mp)
                    grid = self._pair_grid(m, mp)
                    q_positive = grid[grid > 0.0]
                    if q_positive.size == 0:
                        raise ValueError("FFTLog requires positive q support.")
                    key = (order, id(grid), float(grid[-1]), float(q_positive[0]))
                    groups.setdefault(key, []).append((i, j, m, mp))

            for (order, _grid_id, native_q_max, native_q_step), pairs in groups.items():
                def sampled(q: np.ndarray, pairs=pairs) -> np.ndarray:
                    kernel = self._kernel_values(q)
                    rows = []
                    for _i, _j, m, mp in pairs:
                        left = self._mode_values(1, m, q)
                        right = self._mode_values(2, mp, q)
                        rows.append(left * np.conj(right) * kernel)
                    return np.stack(rows, axis=0)

                values, _ = _fftlog_hankel_transform_many(
                    sampled,
                    len(pairs),
                    self._delta_mags[positive],
                    order,
                    native_q_max=native_q_max,
                    native_q_step=native_q_step,
                    n=self._fftlog_n,
                    bias=self._fftlog_bias,
                    interpolator=self._interpolator,
                )
                for pair_index, (i, j, _m, _mp) in enumerate(pairs):
                    H_mm[i, j, positive] = values[pair_index]

        zero = ~positive
        if np.any(zero):
            for i, m_raw in enumerate(self._m_values1):
                m = int(m_raw)
                for j, mp_raw in enumerate(self._m_values2):
                    mp = int(mp_raw)
                    order = abs(m - mp)
                    if order == 0:
                        H_mm[i, j, zero] = self._finite_pair_integral(
                            m, mp, order, "gl4"
                        )[zero]
        return H_mm

    def _fftlog_pair_integral(self, m: int, mp: int, order: int) -> np.ndarray:
        grid = self._pair_grid(m, mp)
        f1 = self._mode_callable(self._field1, m)
        f2 = self._mode_callable(self._field2, mp)
        result = np.zeros(self._delta_mags.size, dtype=np.complex128)

        positive = self._delta_mags > 0.0
        if np.any(positive):
            q_positive = grid[grid > 0.0]
            if q_positive.size == 0:
                raise ValueError("FFTLog requires positive q support.")

            def pair_function(q):
                return f1(q) * np.conj(f2(q)) * self._U_q(q)

            values, _ = fftlog_hankel_transform(
                pair_function,
                self._delta_mags[positive],
                order,
                native_q_max=float(grid[-1]),
                native_q_step=float(q_positive[0]),
                n=self._fftlog_n,
                bias=self._fftlog_bias,
                interpolator=self._interpolator,
            )
            result[positive] = values

        zero = ~positive
        if np.any(zero) and order == 0:
            result[zero] = self._finite_pair_integral(m, mp, order, "gl4")[zero]
        return result

    def _ogata_pair_integral(self, m: int, mp: int, order: int) -> np.ndarray:
        f1 = self._mode_callable(self._field1, m)
        f2 = self._mode_callable(self._field2, mp)
        transform = self._get_ogata(order)

        def radial_integrand(q):
            return f1(q) * np.conj(f2(q)) * self._U_q(q)

        result = np.zeros(self._delta_mags.size, dtype=np.complex128)
        zero = np.isclose(self._delta_mags, 0.0)
        nonzero = ~zero
        if np.any(nonzero):
            result[nonzero] = transform.transform(
                radial_integrand,
                k=self._delta_mags[nonzero],
                ret_err=False,
            )
        if np.any(zero) and order == 0:
            result[zero] = self._finite_pair_integral(m, mp, order, "gl4")[zero]
        return result

    def _calculate_H_mm(self) -> np.ndarray:
        shape = (len(self._m_values1), len(self._m_values2), len(self._delta_mags))
        H_mm = np.zeros(shape, dtype=np.complex128)
        self._evaluation_cache = {}
        try:
            if self._method == "fftlog":
                return self._fftlog_all_pairs()
            for i, m_raw in enumerate(self._m_values1):
                m = int(m_raw)
                for j, mp_raw in enumerate(self._m_values2):
                    mp = int(mp_raw)
                    order = abs(m - mp)
                    if self._method == "ogata":
                        H_mm[i, j] = self._ogata_pair_integral(m, mp, order)
                    else:
                        H_mm[i, j] = self._finite_pair_integral(m, mp, order, self._method)
        finally:
            self._evaluation_cache = None
        return H_mm

    def _compute(self) -> None:
        self.H_mm = self._calculate_H_mm()
        self.V_mm = 2.0 * np.pi * self.Phi_mm * self.H_mm
        self.V = self.V_mm.sum(axis=(0, 1))
