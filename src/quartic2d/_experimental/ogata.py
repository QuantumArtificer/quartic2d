"""Backward-compatible aliases for the now-public Ogata backend.

New code should import :class:`quartic2d.HankelTransform`,
:class:`quartic2d.HarmonicTransform`, and :class:`quartic2d.Interaction` directly
and select ``method="ogata"``.
"""

from __future__ import annotations

from .._numerics import hankel_transform_sampled as _hankel_transform_sampled
from ..interaction import (
    HankelTransform as _HankelTransform,
)
from ..interaction import (
    HarmonicTransform as _HarmonicTransform,
)
from ..interaction import (
    Interaction as _Interaction,
)


class HankelTransform(_HankelTransform):
    _allow_experimental_methods = True


class HarmonicTransform(_HarmonicTransform):
    _allow_experimental_methods = True
    _hankel_transform_type = HankelTransform


class Interaction(_Interaction):
    _allow_experimental_methods = True


def hankel_transform_sampled(*args, **kwargs):
    kwargs["_allow_experimental"] = True
    return _hankel_transform_sampled(*args, **kwargs)
