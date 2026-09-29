"""Public result objects returned by QUARTIC2D convergence helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .interaction import HarmonicTransform, Interaction

from ._plot_style import (
    CRITERION_LINE_STYLE,
    DOCUMENTATION_RC_PARAMS,
    GUIDE_LINE_STYLE,
)
from ._plot_style import (
    clean_axes as _clean_axes,
)

__all__ = ["HarmonicConvergenceResult", "InteractionConvergenceResult"]


def _positive(values) -> np.ndarray:
    """Return positive finite values and NaN elsewhere for logarithmic plots."""
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr) & (arr > 0.0), arr, np.nan)


@dataclass
class HarmonicConvergenceResult:
    r"""Calibration result returned by :meth:`HarmonicTransform.converge_parameters`.

    The result separates an explicit convergence study from later production
    transforms.  It stores the selected numerical parameters together with the
    q-support, q-grid, and radial-quadrature diagnostics used to select them.

    The requested ``rtol`` is a self-convergence criterion, not an independent
    reference-error guarantee.  On each refinement comparison, acceptance
    requires both

    .. math::

        \frac{\|f^{(n)}-f^{(n-1)}\|_2}
        {\|f^{(n-1)}\|_2}\leq \mathrm{rtol}

    and

    .. math::

        \|f^{(n)}-f^{(n-1)}\|_\infty
        \leq \mathrm{atol}
        +\mathrm{rtol}\,\|f^{(n-1)}\|_\infty.

    The relative :math:`L^\infty` change is recorded as a diagnostic but is not
    an additional acceptance condition.  ``q_tail_rtol`` is separate and limits
    the estimated relative :math:`L^2` norm omitted beyond the selected
    ``q_max``.

    Parameters
    ----------
    sampling : object
        Structured q-support and q-grid convergence record.
    quadrature : object
        Structured radial-quadrature convergence record.
    method : str
        Selected radial Hankel quadrature method.
    interpolator : str
        Radial interpolator used during calibration.
    rtol, atol : float
        Relative and absolute self-convergence thresholds.
    q_tail_rtol : float
        Relative L2 tail budget used to select momentum support.

    Attributes
    ----------
    converged : bool
        ``True`` only when q support, q-grid interpolation, and radial
        quadrature satisfy their configured criteria.
    parameters : dict
        Constructor keyword arguments selected for a production
        :class:`HarmonicTransform`.

    Notes
    -----
    Use :meth:`transform` to build a production transform without rerunning the
    convergence study.  Use :meth:`to_dict` when the calibration record should
    be stored with numerical results or provenance metadata.
    """

    sampling: Any
    quadrature: Any
    method: str
    interpolator: str
    rtol: float
    atol: float
    q_tail_rtol: float

    @property
    def converged(self) -> bool:
        """Whether all configured harmonic-transform criteria were satisfied."""
        return bool(self.sampling.converged and self.quadrature.converged)

    @property
    def parameters(self) -> dict:
        """Selected keyword arguments for a production ``HarmonicTransform``."""
        params = {
            "q_max": float(self.sampling.q_max),
            "n_q": int(self.sampling.n_q),
            "method": self.method,
            "interpolator": self.interpolator,
        }
        if self.sampling.q_grid is not None:
            params["q_grid"] = [float(x) for x in np.asarray(self.sampling.q_grid)]
        params.update(self.quadrature.selected_parameters)
        return params

    def transform(self, decomposition, *, check: bool = False) -> HarmonicTransform:
        """Build a production transform from the selected parameters.

        Parameters
        ----------
        decomposition : petal2d.PolarDecomposition
            PETAL2D decomposition to transform.  In the usual workflow this is
            the same decomposition used for calibration.
        check : bool, default=False
            Run the inexpensive post-transform fault detectors in addition to
            reusing the stored calibration record.

        Returns
        -------
        HarmonicTransform
            Production transform carrying the stored sampling and quadrature
            convergence records.

        Notes
        -----
        This method does not repeat the convergence search.  Reusing selected
        parameters for a materially different field should be justified by a
        separate calibration or validation study.
        """
        from .interaction import HarmonicTransform

        field = HarmonicTransform(decomposition, check=check, **self.parameters)
        field._sampling_convergence = self.sampling
        field._quadrature_convergence = self.quadrature
        return field

    def plot_convergence(self, title: str = "Harmonic-transform convergence"):
        r"""Plot the convergence record used to select production parameters.

        The three panels show momentum-support selection, off-grid q-sampling
        refinement, and radial-quadrature refinement.  The plotted thresholds
        are internal self-convergence criteria.  They are not independent
        reference-error bounds.

        Parameters
        ----------
        title : str, default="Harmonic-transform convergence"
            Figure title.  Pass an empty string to suppress it.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axes : ndarray of matplotlib.axes.Axes, shape (3,)
            Momentum-support, q-grid, and radial-quadrature axes.

        Notes
        -----
        The q-support panel uses the separate ``q_tail_rtol`` criterion.  The
        other two panels show relative :math:`L^2` refinement changes.  Final
        acceptance also requires the peak-scaled absolute maximum-change
        condition controlled by ``atol``; a single horizontal line cannot
        represent that scale-dependent condition.
        """
        from matplotlib import pyplot as plt

        sampling = self.sampling
        quadrature = self.quadrature
        modes = sorted(int(m) for m in sampling.tail_modes)
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        colors = {
            m: cycle[i % len(cycle)] if cycle else None
            for i, m in enumerate(modes)
        }

        with plt.rc_context(DOCUMENTATION_RC_PARAMS):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(11.4, 3.45),
                layout="constrained",
            )

            # Momentum support: one required q per retained harmonic.
            required = [float(sampling.tail_modes[m].q_required) for m in modes]
            for m, value in zip(modes, required):
                axes[0].plot(
                    [m],
                    [value],
                    marker="o",
                    linestyle="none",
                    color=colors[m],
                )
            axes[0].axhline(float(sampling.q_max), **CRITERION_LINE_STYLE)
            axes[0].annotate(
                rf"selected $q_{{\max}}={sampling.q_max:.4g}$",
                xy=(0.98, float(sampling.q_max)),
                xycoords=("axes fraction", "data"),
                xytext=(-4, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=8,
            )
            ceiling = float(sampling.q_ceiling)
            if np.isfinite(ceiling) and ceiling > float(sampling.q_max):
                axes[0].axhline(ceiling, **GUIDE_LINE_STYLE)
                axes[0].annotate(
                    "radial-sampling ceiling",
                    xy=(0.98, ceiling),
                    xycoords=("axes fraction", "data"),
                    xytext=(-4, 4),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="0.4",
                )
            axes[0].set_xlabel(r"harmonic $m$")
            axes[0].set_ylabel(r"required $q$")
            if modes:
                axes[0].set_xticks(modes)
            axes[0].set_ylim(bottom=0.0)
            _clean_axes(axes[0])

            # Off-grid q interpolation refinement.
            interpolation_steps = list(sampling.interpolation_steps)
            for m in modes:
                x = [int(step.n_q) for step in interpolation_steps]
                y = _positive(
                    [step.harmonic_errors[m]["relative_l2"] for step in interpolation_steps]
                )
                axes[1].plot(
                    x,
                    y,
                    marker="o",
                    color=colors[m],
                )
                finite = np.flatnonzero(np.isfinite(y))
                if finite.size:
                    j = int(finite[-1])
                    axes[1].annotate(
                        rf"$m={m}$",
                        (x[j], y[j]),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=7.5,
                        color=colors[m],
                    )
            axes[1].axhline(float(sampling.interpolation_rtol), **CRITERION_LINE_STYLE)
            axes[1].set_xlabel(r"$N_q$")
            axes[1].set_ylabel(r"$L^2$ error")
            _clean_axes(axes[1], log_y=True)

            # Radial quadrature refinement.
            qmeta = quadrature.metadata.get("harmonics", {})
            for m in modes:
                item = qmeta.get(str(m), {})
                steps = item.get("steps", [])
                pairs = [
                    (
                        step["parameters"].get("subdivisions"),
                        step.get("relative_l2_change"),
                    )
                    for step in steps
                ]
                pairs = [
                    (int(x), float(y))
                    for x, y in pairs
                    if x is not None and y is not None and np.isfinite(y) and y > 0.0
                ]
                if pairs:
                    px = [x for x, _ in pairs]
                    py = [y for _, y in pairs]
                    axes[2].plot(
                        px,
                        py,
                        marker="o",
                        color=colors[m],
                    )
                    axes[2].annotate(
                        rf"$m={m}$",
                        (px[-1], py[-1]),
                        xytext=(4, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=7.5,
                        color=colors[m],
                    )
            axes[2].axhline(float(quadrature.rtol), **CRITERION_LINE_STYLE)
            axes[2].set_xlabel("radial subdivisions")
            axes[2].set_ylabel(r"$L^2$ error")
            _clean_axes(axes[2], log_y=True)

            if title:
                fig.suptitle(title)
            return fig, axes

    def to_dict(self) -> dict:
        """Return a JSON-serializable calibration record.

        Returns
        -------
        dict
            Selected parameters, configured tolerances, and nested sampling and
            quadrature diagnostics.
        """
        return {
            "converged": self.converged,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "q_tail_rtol": float(self.q_tail_rtol),
            "parameters": self.parameters,
            "sampling": self.sampling.to_dict(),
            "quadrature": self.quadrature.to_dict(),
        }

    def __str__(self) -> str:
        p = self.parameters
        lines = [
            "HarmonicTransform convergence",
            "-----------------------------",
            f"numerical target (rtol) : {self.rtol:.1e}",
            f"absolute target (atol)  : {self.atol:.1e}",
            f"q-tail target           : {self.q_tail_rtol:.1e}  (relative L2 norm beyond q_max)",
            f"q_max                   : {p['q_max']:.6g}",
            f"n_q                     : {p['n_q']}",
            f"method                  : {p['method']}",
        ]
        if "subdivisions" in p:
            lines.append(f"radial subdivisions     : {p['subdivisions']}")
        if "N" in p:
            lines.append(f"Ogata N                 : {p['N']}")
        if "h" in p:
            lines.append(f"Ogata h                 : {p['h']:.6g}")
        lines.extend(
            [
                "",
                f"q support               : {'PASS' if self.sampling.tail_converged else 'FAIL'}",
                f"q-grid interpolation    : {'PASS' if self.sampling.interpolation_converged else 'FAIL'}",
                f"radial quadrature       : {'PASS' if self.quadrature.converged else 'FAIL'}",
            ]
        )
        return "\n".join(lines)


@dataclass
class InteractionConvergenceResult:
    r"""Calibration result returned by :meth:`Interaction.converge_parameters`.

    The result stores the complete interaction-level refinement search and the
    parameters selected for production use.  Convergence is measured on the
    assembled interaction over the supplied displacement set.  It is an
    internal self-convergence statement for those represented inputs, not an
    independent reference-error guarantee.

    Finite-grid methods may additionally use known-order and Richardson checks.
    FFTLog requires stability in both resolution and the configured local bias
    window.  Ogata is calibrated in its coupled ``(N, h)`` parameter space.
    When automatically sampled :class:`HarmonicTransform` inputs are supplied,
    the calibration can also probe sensitivity to the finite upstream q
    boundary.

    Parameters
    ----------
    search : object
        Structured method-specific interaction convergence record.
    method : str
        Interaction integration method that was calibrated.
    interpolator : str
        Momentum-space interpolator used during calibration.
    rtol, atol : float
        Relative and absolute self-convergence thresholds.

    Attributes
    ----------
    converged : bool
        Whether the configured interaction-level criteria were satisfied.
    parameters : dict
        Constructor keyword arguments selected for a production
        :class:`Interaction`.

    Notes
    -----
    Use :meth:`interaction` to construct a production interaction without
    rerunning calibration.  Use :meth:`to_dict` to retain the search record with
    numerical results or provenance metadata.
    """

    search: Any
    method: str
    interpolator: str
    rtol: float
    atol: float
    _interaction_class: type | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def converged(self) -> bool:
        """Whether the configured interaction-level criteria were satisfied."""
        return bool(self.search.converged)

    @property
    def parameters(self) -> dict[str, Any]:
        """Selected keyword arguments for a production ``Interaction``."""
        params = {
            "method": str(self.method),
            "interpolator": str(self.interpolator),
        }
        params.update(self.search.selected_parameters)
        return params

    def interaction(self, deltas, field1, field2, U_q) -> Interaction:
        """Build a production interaction from the selected parameters.

        Parameters
        ----------
        deltas : array_like, shape (D, 2)
            Cartesian displacement vectors at which to evaluate the interaction.
        field1, field2 : HarmonicTransform
            Momentum-space transition fields.  In the usual workflow these are
            the same represented inputs used during calibration.
        U_q : callable
            Radial momentum-space interaction kernel.

        Returns
        -------
        Interaction
            Production interaction carrying this convergence result through its
            ``convergence`` attribute.

        Raises
        ------
        RuntimeError
            If the calibration did not satisfy its configured criteria.

        Notes
        -----
        This method does not repeat the convergence search.  Reuse for a wider
        displacement domain, different kernel, or materially different fields
        requires a separate numerical justification.
        """
        if not self.converged:
            raise RuntimeError("Cannot build an interaction from unconverged parameters.")
        if self._interaction_class is None:
            from .interaction import Interaction

            interaction_class = Interaction
        else:
            interaction_class = self._interaction_class

        interaction = interaction_class(deltas, field1, field2, U_q, **self.parameters)
        interaction._convergence_result = self
        return interaction

    def plot_convergence(self, title: str = "Interaction convergence"):
        r"""Plot method-specific interaction self-convergence diagnostics.

        The plot adapts to the calibrated integration method.  Finite-grid
        rules show the L2 refinement error.  FFTLog shows resolution convergence
        across tested biases together with the bias/resolution search.  Ogata
        shows coupled ``h`` and ``N`` refinement.

        Parameters
        ----------
        title : str, default="Interaction convergence"
            Figure title.  Pass an empty string to suppress it.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Created figure.
        axes : ndarray of matplotlib.axes.Axes
            Method-specific convergence axes. Finite-grid methods return one
            axis; FFTLog, Ogata, and fallback diagnostics return two.

        Notes
        -----
        These panels visualize the internal refinement and robustness tests
        recorded by :meth:`Interaction.converge_parameters`.  They do not show
        an independent reference error.  The relative :math:`L^2` curves are
        only one part of acceptance; the peak-scaled absolute maximum-change
        criterion controlled by ``atol`` is also enforced by the search.
        """
        from matplotlib import pyplot as plt

        search = self.search
        kind = str(search.metadata.get("kind", ""))
        selected = dict(search.selected_parameters)

        with plt.rc_context(DOCUMENTATION_RC_PARAMS):
            n_axes = 1 if kind == "fixed_order_richardson" else 2
            fig, axes = plt.subplots(
                1,
                n_axes,
                figsize=(4.6 if n_axes == 1 else 7.8, 3.45),
                layout="constrained",
                squeeze=False,
            )
            axes = axes.ravel()

            if kind == "fixed_order_richardson":
                self._plot_finite_convergence(axes[0], search, selected)
            elif kind == "fftlog_resolution_and_bias":
                self._plot_fftlog_convergence(axes, search, selected)
            elif kind == "ogata_coupled_N_h":
                self._plot_ogata_convergence(axes, search, selected)
            else:
                self._plot_generic_convergence(axes, search, selected)

            if title:
                fig.suptitle(title)

            boundary = search.metadata.get("q_boundary_robustness", {})
            status = boundary.get("status") if isinstance(boundary, dict) else None
            if status and status not in {"disabled", "not_applicable_fixed_input"}:
                target_axis = axes[-1]
                target_axis.set_title(
                    f"q-boundary: {str(status).replace('_', ' ')}",
                    loc="right",
                    fontsize=8,
                    color="0.35",
                    pad=5,
                )
            return fig, axes

    def _plot_finite_convergence(self, axis, search, selected) -> None:
        """Populate finite-rule L2 convergence diagnostics."""
        parameter_name = str(search.metadata.get("parameter_name", "resolution"))
        points = [
            (
                step.parameters.get(parameter_name),
                step.relative_l2_change,
            )
            for step in search.steps
        ]
        points = [
            (float(x), float(y))
            for x, y in points
            if x is not None and y is not None and np.isfinite(y) and y > 0.0
        ]
        if points:
            axis.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                label=r"$L^2$ error",
            )
        axis.axhline(
            float(search.rtol),
            label="target",
            **CRITERION_LINE_STYLE,
        )
        selected_x = selected.get(parameter_name)
        if selected_x is not None:
            axis.axvline(float(selected_x), **GUIDE_LINE_STYLE)
        axis.set_xlabel(parameter_name.replace("_", " "))
        axis.set_ylabel(r"$L^2$ error")
        _clean_axes(axis, log_y=True)
        axis.legend(frameon=True, framealpha=1.0, loc="best")

    def _plot_fftlog_convergence(self, axes, search, selected) -> None:
        """Populate FFTLog resolution and bias-robustness panels."""
        from matplotlib import pyplot as plt

        selected_bias = selected.get("bias")
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        plotted = 0
        bias_summary = []
        for i, step in enumerate(search.steps):
            bias = float(step.parameters.get("bias"))
            nested = (step.metadata or {}).get("resolution_search", {})
            nested_steps = nested.get("steps", []) if isinstance(nested, dict) else []
            points = []
            for item in nested_steps:
                params = item.get("parameters", {})
                n = params.get("n")
                err = item.get("relative_l2_change")
                if n is not None and err is not None and np.isfinite(err) and err > 0.0:
                    points.append((int(n), float(err)))
            color = cycle[i % len(cycle)] if cycle else None
            is_selected = selected_bias is not None and np.isclose(bias, float(selected_bias))
            if points:
                axes[0].plot(
                    [x for x, _ in points],
                    [y for _, y in points],
                    marker="o",
                    color=color,
                    alpha=1.0 if is_selected else 0.55,
                    linewidth=2.0 if is_selected else 1.1,
                    label=rf"bias={bias:g}" + (" (selected)" if is_selected else ""),
                )
                plotted += 1
            converged = bool((step.metadata or {}).get("resolution_converged", False))
            n_selected = step.parameters.get("n")
            if n_selected is not None:
                bias_summary.append((bias, int(n_selected), converged, is_selected, color))

        axes[0].axhline(float(search.rtol), **CRITERION_LINE_STYLE)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("FFTLog length N")
        axes[0].set_ylabel(r"$L^2$ error")
        _clean_axes(axes[0], log_y=True)
        if plotted:
            axes[0].legend(frameon=False, loc="best")

        for bias, n_value, converged, is_selected, color in bias_summary:
            axes[1].plot(
                [bias],
                [n_value],
                marker="*" if is_selected else ("o" if converged else "x"),
                markersize=9 if is_selected else 5.5,
                linestyle="none",
                color=color,
            )

        robust_windows = [
            item
            for item in search.metadata.get("bias_windows", [])
            if isinstance(item, dict) and item.get("robust")
        ]
        if robust_windows:
            biases = [float(x) for x in robust_windows[0].get("biases", [])]
            if biases:
                axes[1].axvspan(
                    min(biases),
                    max(biases),
                    color="0.5",
                    alpha=0.10,
                    linewidth=0.0,
                )
        axes[1].set_xlabel("FFTLog bias")
        axes[1].set_ylabel("selected resolution N")
        if bias_summary:
            axes[1].set_yscale("log", base=2)
        _clean_axes(axes[1])
        axes[1].text(
            0.98,
            0.03,
            "star: selected\ncircle: resolution stable\nx: unresolved",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
        )

    def _plot_ogata_convergence(self, axes, search, selected) -> None:
        """Populate coupled Ogata h/N refinement panels."""
        rows = []
        for step in search.steps:
            h = step.parameters.get("h")
            n_value = step.parameters.get("N")
            err = step.relative_l2_change
            if h is not None and n_value is not None:
                rows.append(
                    (
                        float(h),
                        int(n_value),
                        None if err is None else float(err),
                    )
                )
        err_rows = [row for row in rows if row[2] is not None and row[2] > 0.0]
        if err_rows:
            axes[0].plot(
                [row[0] for row in err_rows],
                [row[2] for row in err_rows],
                marker="o",
            )
        axes[0].axhline(float(search.rtol), **CRITERION_LINE_STYLE)
        selected_h = selected.get("h")
        if selected_h is not None:
            axes[0].axvline(float(selected_h), **GUIDE_LINE_STYLE)
        axes[0].set_xscale("log")
        axes[0].invert_xaxis()
        axes[0].set_xlabel("Ogata h (finer to the right)")
        axes[0].set_ylabel(r"$L^2$ error")
        _clean_axes(axes[0], log_y=True)

        if rows:
            axes[1].plot(
                [row[0] for row in rows],
                [row[1] for row in rows],
                marker="o",
            )
            axes[1].set_xscale("log")
            axes[1].set_yscale("log", base=2)
            axes[1].invert_xaxis()
        axes[1].set_xlabel("Ogata h (finer to the right)")
        axes[1].set_ylabel("selected Ogata N")
        _clean_axes(axes[1])

    def _plot_generic_convergence(self, axes, search, selected) -> None:
        """Populate a conservative fallback view for an unknown search kind."""
        x = np.arange(len(search.steps), dtype=int)
        errors = _positive(
            [
                np.nan if step.relative_l2_change is None else step.relative_l2_change
                for step in search.steps
            ]
        )
        axes[0].plot(x, errors, marker="o")
        axes[0].axhline(float(search.rtol), **CRITERION_LINE_STYLE)
        axes[0].set_xlabel("refinement step")
        axes[0].set_ylabel(r"$L^2$ error")
        _clean_axes(axes[0], log_y=True)

        runtimes = [float(step.runtime_seconds) for step in search.steps]
        axes[1].plot(x, runtimes, marker="o")
        axes[1].set_xlabel("refinement step")
        axes[1].set_ylabel("evaluation time [s]")
        _clean_axes(axes[1])

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable calibration record.

        Parameters
        ----------
        include_values : bool, default=False
            Include stored interaction arrays from each refinement step.  Leave
            ``False`` for compact provenance records.

        Returns
        -------
        dict
            Selected parameters, configured tolerances, and the method-specific
            convergence search history.
        """
        return {
            "converged": self.converged,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "parameters": self.parameters,
            "search": self.search.to_dict(include_values=include_values),
        }

    def __str__(self) -> str:
        lines = [
            "Interaction convergence",
            "-----------------------",
            f"numerical target (rtol) : {self.rtol:.1e}",
            f"absolute target (atol)  : {self.atol:.1e}",
            f"method                  : {self.method}",
            f"status                  : {'PASS' if self.converged else 'FAIL'}",
        ]
        for key, value in self.search.selected_parameters.items():
            if isinstance(value, float):
                lines.append(f"{key:24s}: {value:.6g}")
            else:
                lines.append(f"{key:24s}: {value}")
        return "\n".join(lines)
