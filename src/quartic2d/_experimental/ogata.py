"""Experimental Ogata Hankel-transform backend."""

from __future__ import annotations

from .._numerics import hankel_transform_sampled as _hankel_transform_sampled
from ..interaction import (
    HankelTransform as _HankelTransform,
    HarmonicTransform as _HarmonicTransform,
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
