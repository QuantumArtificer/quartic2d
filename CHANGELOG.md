# Changelog

All notable changes to QUARTIC2D will be documented in this file.

## [0.1.0] - Unreleased

### Added
- PETAL2D integration for transforming retained angular harmonics and evaluating radial-kernel interactions.
- Sampled radial quadratures using trapezoidal, Simpson, composite 4-point Gauss--Legendre, and composite 8-point Gauss--Legendre rules.
- Ogata Hankel quadrature through the `hankel` package.
- FFTLog interaction transforms through `scipy.fft.fht`.
- `rtol`/`atol` convergence interfaces for sampled quadratures, FFTLog resolution, and Ogata parameter selection.
- Harmonic-resolved public interaction outputs `Phi_mm`, `H_mm`, `V_mm`, and total `V`.
- Unit tests, analytic validation, executable examples, documentation, and numerical-method benchmarks.
- GitHub Actions workflows for tests, documentation, distribution checks, and PyPI Trusted Publishing.
- Citation metadata and release tooling for archival releases.
- Fast scale-aware Step-2 defaults with inexpensive post-transform fault detection.
- Opt-in Step-2 convergence of q support, q-grid interpolation, and radial quadrature with reusable calibrated parameters.
- Step-2 plotting for transformed harmonics and optional convergence diagnostics.

### Changed
- Simpson is the default sampled radial quadrature; GL4 and GL8 remain available as higher-order finite-quadrature backends.
- FFTLog is the default interaction transform with `n=512` and `bias=-0.5`.
- Radial support is read from the public PETAL2D `cutoff_radius` mapping.
- Numerical backend parameters use established names including `rtol`, `atol`, `subdivisions`, `N`, `h`, `nu`, `n`, `bias`, and `offset`.
- Cubic interpolation in q space uses not-a-knot boundary conditions; radial-profile cubic interpolation remains natural.
- Finite-grid harmonic convergence is certified independently for every retained harmonic before a common resolution is selected.

### Removed
- Radial tapering and zero-padding controls, which did not improve the validated sampled-data workflow.
- QDHT support.
- Superseded exploratory benchmark and diagnostic scripts from the release surface.

### Fixed
- Negative integer Hankel orders use the correct Bessel-function sign convention.
- Zero-momentum and zero-displacement special cases use the finite represented support consistently.
- Singular kernels are not required to be evaluated at `q=0` by Gauss--Legendre interaction quadratures.
- FFTLog accepts displacement arrays in arbitrary input order.
- Benchmark resume/schema handling and calibrated-parameter reporting are consistent across numerical backends.
