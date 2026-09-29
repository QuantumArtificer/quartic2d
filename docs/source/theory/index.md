# Theory

QUARTIC2D starts from the physical four-center matrix element and reduces it to reusable angular-harmonic form factors and one-dimensional radial momentum integrals. The reduction separates three ingredients that should remain conceptually distinct:

- the orbital or transition-field structure, represented by $F_m(q)$;
- the scalar radial interaction, represented by $U(q)$;
- the relative geometry, represented by translation-Bessel factors and the displacement angle.

This separation is the reason a transition-field transform can be reused across many displacement vectors and, when its represented momentum interval is sufficient, across different radial kernels.

```{toctree}
:maxdepth: 2

four_center_reduction
conventions
```

The derivation is written for two general complex transition fields. Density-density terms are a special case in which the relevant orbital products happen to be densities. The assumptions that delimit the reduction are collected separately in {doc}`../limitations`.
