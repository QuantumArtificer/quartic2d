# Theory and conventions

## Angular-harmonic expansion

PETAL2D represents a localized scalar field as

$$
\rho(r,\theta)=\sum_m \rho_m(r)e^{im\theta}.
$$

QUARTIC2D applies the radial Hankel transform

$$
F_m(q)=\int_0^\infty r\,\rho_m(r)J_m(qr)\,dr.
$$

For negative integer order,

$$
J_{-m}(x)=(-1)^mJ_m(x),
$$

which fixes the sign convention used for negative harmonic indices.

## Radial interaction kernel

Let $U(q)$ be a radially symmetric momentum-space interaction kernel and let

$$
\boldsymbol\Delta=\Delta(\cos\phi_\Delta,\sin\phi_\Delta).
$$

For two transformed harmonic sets, QUARTIC2D evaluates

$$
H_{mm'}(\Delta)=
\int_0^\infty q\,dq\,
J_{|m-m'|}(q\Delta)
F_m(q)F_{m'}^*(q)U(q),
$$

with angular prefactor

$$
\Phi_{mm'}(\phi_\Delta)=
\eta_{mm'}\exp\left[i(m-m')\phi_\Delta\right],
$$

where

$$
\eta_{mm'}=
\begin{cases}
(-1)^{m-m'}, & m>m',\\
1, & m\le m'.
\end{cases}
$$

The factor $\eta_{mm'}$ is required because the numerical backends use the
nonnegative Bessel order $|m-m'|$. Equivalently, the exact angular reduction
may be written directly with the signed order $J_{m'-m}(q\Delta)$ and no
separate parity factor.

The harmonic-pair contribution is

$$
V_{mm'}(\boldsymbol\Delta)=
2\pi\,\Phi_{mm'}H_{mm'},
$$

and

$$
V(\boldsymbol\Delta)=\sum_{m,m'}V_{mm'}(\boldsymbol\Delta).
$$

## Zero displacement

At $\Delta=0$, only $m=m'$ contributes because $J_n(0)=0$ for nonzero integer $n$.
