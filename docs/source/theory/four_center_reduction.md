# Four-center reduction

## From four orbitals to two transition fields

The four-center matrix element is

$$
U_{1234}=
\iint d^2\mathbf r\,d^2\mathbf r'\,
\phi_1^*(\mathbf r)\phi_2^*(\mathbf r')
U(|\mathbf r-\mathbf r'|)
\phi_3(\mathbf r)\phi_4(\mathbf r').
$$

Define the two transition fields

$$
\rho_{13}(\mathbf r)=\phi_1^*(\mathbf r)\phi_3(\mathbf r),
\qquad
\rho_{42}(\mathbf r')=\phi_4^*(\mathbf r')\phi_2(\mathbf r').
$$

Then

$$
U_{1234}=
\iint d^2\mathbf r\,d^2\mathbf r'\,
\rho_{13}(\mathbf r)
U(|\mathbf r-\mathbf r'|)
\rho_{42}^*(\mathbf r').
$$

No reality or positivity assumption has been made. The same reduction therefore
covers ordinary densities and complex off-diagonal transition fields.

For translated localized fields, choose expansion centers $\mathbf R_1$ and
$\mathbf R_2$ and local coordinates

$$
\mathbf s=\mathbf r-\mathbf R_1,
\qquad
\mathbf t=\mathbf r'-\mathbf R_2.
$$

Write the locally represented fields as

$$
\widetilde\rho_{13}(\mathbf s)
=\rho_{13}(\mathbf R_1+\mathbf s),
\qquad
\widetilde\rho_{42}(\mathbf t)
=\rho_{42}(\mathbf R_2+\mathbf t),
$$

and define

$$
\boldsymbol\delta=\mathbf R_1-\mathbf R_2
=\delta(\cos\phi_\delta,\sin\phi_\delta).
$$

The interaction then becomes

$$
U_{1234}(\boldsymbol\delta)=
\iint d^2\mathbf s\,d^2\mathbf t\,
\widetilde\rho_{13}(\mathbf s)
U(|\mathbf s-\mathbf t+\boldsymbol\delta|)
\widetilde\rho_{42}^*(\mathbf t).
$$

This equation fixes the displacement convention used throughout QUARTIC2D.

## Momentum-space form

Using the Fourier conventions in {doc}`conventions`, the centered transition
fields satisfy

$$
U_{1234}(\boldsymbol\delta)
=
\int d^2\mathbf q\,
\widetilde\rho_{13}(\mathbf q)
U(q)
\widetilde\rho_{42}^*(\mathbf q)
 e^{-i\mathbf q\cdot\boldsymbol\delta},
$$

where $q=|\mathbf q|$ and $U(q)$ is the radial momentum-space kernel supplied
to `Interaction`.

The three ingredients are now separated: the two transformed transition fields
contain the orbital structure, $U(q)$ contains the radial interaction, and the
phase factor contains the relative displacement.

## Angular harmonics

For either centered transition field,

$$
\widetilde\rho(r,\theta)=\sum_m\rho_m(r)e^{im\theta},
$$

with

$$
\rho_m(r)=\frac{1}{2\pi}\int_0^{2\pi}
\widetilde\rho(r,\theta)e^{-im\theta}\,d\theta.
$$

Define the radial Hankel transform

$$
F_m(q)=\int_0^\infty r\,dr\,\rho_m(r)J_m(qr).
$$

The momentum-space field is

$$
\widetilde\rho(q,\phi_q)
=\sum_m(-i)^m e^{im\phi_q}F_m(q).
$$

For the two transition fields,

$$
\widetilde\rho_{13}(q,\phi_q)
=\sum_m(-i)^m e^{im\phi_q}F_{13,m}(q),
$$

and

$$
\widetilde\rho_{42}^*(q,\phi_q)
=\sum_{m'}i^{m'}e^{-im'\phi_q}F_{42,m'}^*(q).
$$

The product of their angular phases contains

$$
(-i)^{m-m'}e^{i(m-m')\phi_q}.
$$

## Displacement integral

The remaining angular integral is

$$
\int_0^{2\pi}d\phi_q\,
e^{in\phi_q}
e^{-iq\delta\cos(\phi_q-\phi_\delta)}
=2\pi(-i)^n e^{in\phi_\delta}J_n(q\delta),
$$

for integer $n$. Taking $n=m-m'$ and combining the phase factors gives

$$
\boxed{
U_{1234}(\boldsymbol\delta)
=2\pi\sum_{m,m'}e^{i(m-m')\phi_\delta}
\int_0^\infty q\,dq\,
U(q)F_{13,m}(q)F_{42,m'}^*(q)
J_{m'-m}(q\delta)
}.
$$

This is the central reduced formula used by QUARTIC2D. The `Interaction.V`
array stores its value on the requested displacement vectors.

## Computational factorization and reuse

The reduced expression separates the problem into quantities with different reuse patterns:

$$
\underbrace{F_{13,m}(q)F_{42,m'}^*(q)}_{\text{transition-field structure}}
\;\underbrace{U(q)}_{\text{radial interaction}}
\;\underbrace{J_{m'-m}(q\delta)e^{i(m-m')\phi_\delta}}_{\text{relative geometry}}.
$$

The transition-field transforms are computed once for a chosen represented momentum interval. Changing the displacement vectors then changes only the translation-Bessel factor, so the real-space angular decomposition and radial Hankel transforms do not have to be repeated. Changing the radial kernel requires a new interaction integral, but not new transition-field transforms, provided the existing q interval resolves the new kernel and the transformed fields over the required support.

This distinction is important for parameter sweeps. Field preparation and harmonic transformation can be amortized across many separations and compatible interaction kernels, while each new kernel/displacement family can still be calibrated at the assembled-interaction level.

## Equivalent nonnegative-order form

The numerical radial routines evaluate nonnegative translation-Bessel orders.
For an integer $k\ge0$,

$$
J_{-k}(x)=(-1)^kJ_k(x).
$$

Define

$$
\eta_{mm'}=
\begin{cases}
(-1)^{m-m'}, & m>m',\\
1, & m\le m'.
\end{cases}
$$

The same interaction can then be written as

$$
H_{mm'}(\delta)=
\int_0^\infty q\,dq\,
U(q)F_{13,m}(q)F_{42,m'}^*(q)
J_{|m-m'|}(q\delta),
$$

$$
\Phi_{mm'}(\phi_\delta)
=\eta_{mm'}e^{i(m-m')\phi_\delta},
$$

and

$$
U_{1234}^{(mm')}(\boldsymbol\delta)
=2\pi\Phi_{mm'}H_{mm'},
\qquad
U_{1234}(\boldsymbol\delta)=\sum_{m,m'}U_{1234}^{(mm')}(\boldsymbol\delta).
$$

### Correspondence to the API

After the mathematical reduction is defined, the stored arrays map directly onto it: `Interaction.H_mm` contains the radial pair integrals, `Interaction.Phi_mm` contains the angular/displacement phases, `Interaction.V_mm` contains the resolved pair contributions, and `Interaction.V` is their sum.

## Zero displacement

At $\delta=0$,

$$
J_n(0)=0
$$

for every nonzero integer $n$. Only $m=m'$ terms survive. Zero-displacement
calculations are therefore useful checks of radial normalization, but they
cannot by themselves test the signed-order phase carried by odd $m-m'$ pairs
at finite displacement.
