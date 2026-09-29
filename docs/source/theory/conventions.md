# Fourier and coordinate conventions

## Real-space angular expansion

PETAL2D uses

$$
\rho(r,\theta)=\sum_{m\in\mathbb Z}\rho_m(r)e^{im\theta},
$$

with

$$
\rho_m(r)=\frac{1}{2\pi}\int_0^{2\pi}
\rho(r,\theta)e^{-im\theta}\,d\theta.
$$

The integer $m$ is the signed angular-harmonic index.

## Transition-field Fourier transform

QUARTIC2D uses

$$
\rho(\mathbf r)=
\int\frac{d^2\mathbf q}{2\pi}
\rho(\mathbf q)e^{i\mathbf q\cdot\mathbf r},
$$

and

$$
\rho(\mathbf q)=
\frac{1}{2\pi}\int d^2\mathbf r\,
\rho(\mathbf r)e^{-i\mathbf q\cdot\mathbf r}.
$$

With the Hankel definition

$$
F_m(q)=\int_0^\infty r\,dr\,\rho_m(r)J_m(qr),
$$

the transformed field is

$$
\rho(q,\phi_q)
=\sum_m(-i)^m e^{im\phi_q}F_m(q).
$$

## Kernel Fourier transform

The radial interaction kernel is defined by

$$
U(\mathbf R)=
\int\frac{d^2\mathbf q}{(2\pi)^2}
U(q)e^{i\mathbf q\cdot\mathbf R}.
$$

The callable supplied to `Interaction` is this $U(q)$.

## Displacement sign

For local transition-field expansion centers $\mathbf R_1$ and $\mathbf R_2$, define

$$
\boldsymbol\delta=\mathbf R_1-\mathbf R_2.
$$

With the local-coordinate convention derived in {doc}`four_center_reduction`, QUARTIC2D uses

$$
e^{-i\mathbf q\cdot\boldsymbol\delta}
$$

in the momentum-space interaction. The input `deltas` array therefore fixes both the displacement magnitude and its polar angle.

## Negative Hankel order

For integer $k\ge0$,

$$
J_{-k}(x)=(-1)^kJ_k(x).
$$

`HankelTransform` preserves the sign convention for negative harmonic indices. `Interaction` uses nonnegative translation Bessel orders internally and includes the corresponding parity in `Phi_mm`.

## Units

QUARTIC2D does not define a unit system. If the real-space coordinates have dimensions of length, q has dimensions of inverse length. The dimensions of the final matrix element follow from the normalization of the transition fields and the supplied $U(q)$.

For reproducible calculations, record the coordinate unit, orbital normalization, kernel convention, and any dielectric or screening parameters together with the numerical tolerances.
