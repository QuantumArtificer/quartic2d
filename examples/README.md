# Examples

This directory contains only the polished executable examples used by the
guided documentation. The analytic fields are chosen for clarity; publication
stress workloads live under `benchmarks/`, and developer diagnostics live under
`tools/diagnostics/`.

- `isotropic_interaction.py`: normalized Gaussian, analytic transform reference,
  automatic transform calibration, and Coulomb/Rytova–Keldysh interactions.
- `four_center_interaction.py`: direct and exchange four-center terms from
  normalized $s$ and $p_x$ orbitals.
- `anisotropic_interaction.py`: orientation dependence and harmonic-pair
  contributions for an $m=0,\pm2$ field.
- `sampled_data.py`: complex Cartesian array input carried through PETAL2D,
  harmonic transforms, and an off-diagonal interaction.
- `complex_transition_field.py`: complex transition fields with exchange,
  pair-hopping, and correlated-hopping channels.
- `screening_family_sweep.py`: calibrated reuse across a bounded
  Rytova–Keldysh screening family.
- `large_separation.py`: long-distance Coulomb interactions with an analytic
  Gaussian reference and specialist oscillatory methods.

Every script is import-safe: importing it defines the reusable fields and helper
functions without executing the calculation. Run a complete example with

```bash
python examples/isotropic_interaction.py
```

The corresponding equations, selected output, plots, and interpretation are in
`docs/source/examples/`.
