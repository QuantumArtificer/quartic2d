# Performance and reuse

QUARTIC2D is most efficient when numerical calibration is separated from repeated
production evaluation. The package therefore exposes transformed fields,
calibration results, and fixed production parameters instead of hiding every
search inside a constructor call.

## Reuse transformed fields

A transition field only needs to be transformed once for a fixed PETAL2D
representation and transform parameter set. If the same orbital product appears
in several direct, exchange, pair-hopping, or correlated-hopping matrix
elements, reuse its `HarmonicTransform`.

If the transformed fields contain $N_m^{(1)}$ and $N_m^{(2)}$ retained
harmonics, the interaction contains

$$
N_p=N_m^{(1)}N_m^{(2)}
$$

harmonic pairs. Avoiding numerically irrelevant angular channels upstream
therefore reduces both transform storage and downstream interaction work.

## Separate calibration from production

The intended quantitative workflow is

```python
calibration = Interaction.converge_parameters(
    deltas,
    field_13,
    field_42,
    U_q,
    rtol=1e-4,
)

fig, axes = calibration.plot_convergence()

interaction = calibration.interaction(
    deltas,
    field_13,
    field_42,
    U_q,
)
```

`calibration.interaction(...)` reuses the selected numerical parameters and
attaches the convergence record to the production object. It does not repeat the
search.

The same pattern applies to the upstream harmonic transform:

```python
hcal = HarmonicTransform.converge_parameters(
    decomposition, rtol=1e-4, q_tail_rtol=1e-3
)
field = hcal.transform(decomposition)
```

For repeated calculations, store the compact provenance records:

```python
harmonic_record = hcal.to_dict()
interaction_record = calibration.to_dict()
```

Do not copy a parameter dictionary from an unrelated benchmark or example and
treat it as calibrated for a new problem.

## Reuse within a numerical family

If many nearby parameter points share the same numerical character, calibrate
representative difficult members and form a conservative common production
envelope. The family should be defined by numerical structure rather than by a
physical label alone. Relevant features include field extent, radial tails,
nodes, angular content, kernel nonanalyticities, and maximum displacement.

The procedure is developed in {ref}`family-calibration`. Spot-check both
boundaries and representative interior points. If one member requires
qualitatively different resolution, split the family instead of over-resolving
every point.

## Transform scaling

For $N_m$ retained harmonics, $N_q$ stored momentum points, and $N_s$ effective
radial quadrature nodes, the leading sampled transform work is

$$
T_H=\Theta(N_mN_qN_s).
$$

The implementation batches q points, so the full $N_qN_s$ temporary product is
not necessarily retained in memory at once. Measured scaling and peak-memory
evidence are reported in {doc}`../validation/performance`.

## Finite-rule interaction scaling

For $N_D$ requested displacement vectors and $N_{q,s}$ effective q quadrature
nodes,

$$
T_{\mathrm{finite}}=\Theta(N_pN_DN_{q,s}).
$$

At fixed field and numerical resolution, finite-rule production therefore grows
approximately linearly with the number of requested displacements.

## FFTLog scaling

FFTLog {cite:p}`Hamilton2000` uses logarithmically sampled fast Hankel transforms. With transform length $N_F$,

$$
T_{\mathrm{FFTLog}}
=\Theta\!\left[N_p\left(N_F\log N_F+N_D\right)\right].
$$

FFTLog pays for a logarithmic transform for each harmonic pair and then evaluates
many displacements from that representation. This can be advantageous for some
large displacement batches, but the performance advantage is useful only when
the transform-length, bias, and boundary checks converge for the target
field/kernel family.

## Memory and output size

The retained interaction decomposition contains arrays with shape

$$
(N_m^{(1)},N_m^{(2)},N_D).
$$

Keeping `Phi_mm`, `H_mm`, and `V_mm` is useful for physical interpretation and
diagnostics, but their storage scales with the harmonic-pair count and number of
displacements. A production study with many harmonics and a dense displacement
mesh should include these arrays in its memory planning.

Peak-RSS measurements and finite-range empirical scaling belong in
{doc}`../validation/performance`. They are evidence for specific tested
configurations rather than universal performance guarantees.
