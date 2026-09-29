"""Internal plotting semantics shared by QUARTIC2D figures and helper plots.

This module centralizes semantic encodings, not user-global Matplotlib state.
Callers should apply ``DOCUMENTATION_RC_PARAMS`` or their own local rc context.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

METHOD_ORDER = ("trapezoid", "simpson", "gl4", "gl8", "fftlog", "ogata")

METHOD_LABELS = {
    "trapezoid": "Trapezoid",
    "simpson": "Simpson",
    "gl4": "GL4",
    "gl8": "GL8",
    "fftlog": "FFTLog",
    "ogata": "Ogata",
}

# Okabe-Ito-derived palette plus redundant marker/line encodings.  Color is
# never the only distinction between numerical methods.
METHOD_STYLES: Mapping[str, Mapping[str, Any]] = {
    "trapezoid": {"color": "#0072B2", "marker": "v", "linestyle": ":"},
    "simpson": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "gl4": {"color": "#009E73", "marker": "s", "linestyle": "--"},
    "gl8": {"color": "#CC79A7", "marker": "P", "linestyle": "-."},
    "fftlog": {"color": "#D55E00", "marker": "^", "linestyle": "-"},
    "ogata": {"color": "#56B4E9", "marker": "D", "linestyle": "--"},
}


STATUS_LABELS = {
    "accepted_reference_pass": "automatic acceptance + reference pass",
    "qualified_after_refusal": "reference-qualified point after automatic refusal",
    "conservative_refusal": "automatic refusal; no qualified point in selector path",
    "tested_box_not_capable": "no capable point found in tested box",
    "accepted_reference_fail": "automatic acceptance + reference fail",
}

# Selector outcomes use their own semantic channel.  These colors/markers are
# intentionally independent of METHOD_STYLES so a status symbol cannot be
# mistaken for a numerical-method identity.
STATUS_STYLES: Mapping[str, Mapping[str, Any]] = {
    "accepted_reference_pass": {"color": "#009E73", "marker": "o", "fillstyle": "full"},
    "qualified_after_refusal": {"color": "#56B4E9", "marker": "D", "fillstyle": "none"},
    "conservative_refusal": {"color": "#E69F00", "marker": "^", "fillstyle": "none"},
    "tested_box_not_capable": {"color": "0.40", "marker": "x", "fillstyle": "none"},
    "accepted_reference_fail": {"color": "#D55E00", "marker": "s", "fillstyle": "none"},
}

DOMAIN_STYLES: Mapping[str, Mapping[str, Any]] = {
    "standard": {"color": "#0072B2", "marker": "o", "hatch": ""},
    "large": {"color": "#D55E00", "marker": "s", "hatch": "//"},
}

REFERENCE_LINE_STYLE = {"color": "0.15", "linestyle": "-", "linewidth": 1.45}
CRITERION_LINE_STYLE = {"color": "0.35", "linestyle": "--", "linewidth": 1.0}
GUIDE_LINE_STYLE = {"color": "0.50", "linestyle": ":", "linewidth": 0.9}
ZERO_LINE_STYLE = {"color": "0.60", "linestyle": "-", "linewidth": 0.8}

DOCUMENTATION_RC_PARAMS = {
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 8.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 5.0,
    "savefig.bbox": "tight",
}


def method_label(method: str) -> str:
    """Return the canonical display label for a numerical method."""
    return METHOD_LABELS.get(method, method)


def method_style(method: str, *, label: bool = False) -> dict[str, Any]:
    """Return a copy of the canonical style for ``method``."""
    style = dict(METHOD_STYLES[method])
    if label:
        style["label"] = method_label(method)
    return style


def domain_style(domain: str, *, label: bool = False) -> dict[str, Any]:
    """Return a copy of the canonical standard/large-displacement style."""
    style = dict(DOMAIN_STYLES[domain])
    if label:
        style["label"] = "standard" if domain == "standard" else "large displacement"
    return style


def status_style(status: str, *, label: bool = False) -> dict[str, Any]:
    """Return a copy of the canonical selector-status style."""
    style = dict(STATUS_STYLES[status])
    if label:
        style["label"] = STATUS_LABELS[status]
    return style


def clean_axes(ax, *, grid: bool = True, log_y: bool = False) -> None:
    """Apply restrained documentation/helper axis formatting in-place."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, which="major", linewidth=0.6, alpha=0.20)
        ax.set_axisbelow(True)
    if log_y:
        ax.set_yscale("log")



def zero_line(ax, **kwargs):
    """Draw the canonical neutral zero baseline."""
    style = dict(ZERO_LINE_STYLE)
    style.update(kwargs)
    return ax.axhline(0.0, **style)


def criterion_line(ax, value: float, **kwargs):
    """Draw a neutral line for an internal or comparison criterion."""
    style = dict(CRITERION_LINE_STYLE)
    style.update(kwargs)
    return ax.axhline(value, **style)


def guide_line(ax, value: float, *, orientation: str = "horizontal", **kwargs):
    """Draw a neutral contextual guide line."""
    style = dict(GUIDE_LINE_STYLE)
    style.update(kwargs)
    if orientation == "horizontal":
        return ax.axhline(value, **style)
    if orientation == "vertical":
        return ax.axvline(value, **style)
    raise ValueError("orientation must be 'horizontal' or 'vertical'")
