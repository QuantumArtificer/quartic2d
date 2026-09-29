# Changelog

All notable changes to QUARTIC2D will be documented in this file.

## [0.1.0] - Unreleased

### Added
- PETAL2D integration for transforming retained angular harmonics and evaluating radial-kernel interactions.
- Sampled radial quadratures using trapezoidal, Simpson, composite 4-point Gauss--Legendre, and composite 8-point Gauss--Legendre rules.
- Public Ogata Hankel quadrature through the optional `hankel` package (`quartic2d[ogata]`).
- FFTLog interaction transforms through `scipy.fft.fht`.
- `rtol`/`atol` convergence interfaces for sampled quadratures, FFTLog resolution, and Ogata parameter selection.
- Harmonic-resolved public interaction outputs `Phi_mm`, `H_mm`, `V_mm`, and total `V`.
- Unit tests, analytic validation, guided examples, user documentation, and publication-grade numerical benchmarks.
- GitHub Actions workflows for tests, documentation, distribution checks, PyPI Trusted Publishing, and tag-driven GitHub releases.
- Citation metadata, GitHub Pages documentation, PyPI Trusted Publishing, and Zenodo-oriented release guidance.
- Fast scale-aware `HarmonicTransform` defaults with inexpensive post-transform fault detection.
- Opt-in `HarmonicTransform` convergence of q support, q-grid interpolation, and radial quadrature with reusable calibrated parameters.
- `HarmonicTransform` plotting for transformed harmonics and optional convergence diagnostics.
- `HarmonicConvergenceResult` and `InteractionConvergenceResult` as public top-level calibration result types with reusable production constructors and serialization helpers.
- Object-oriented API documentation with separate pages for public classes and callable methods.

### Changed
- Simpson is the default sampled radial quadrature; GL4 and GL8 remain available as higher-order finite-quadrature backends.
- GL4 is the default interaction transform; FFTLog and Ogata remain explicit workload-dependent alternatives.
- Radial support is read from the public PETAL2D `cutoff_radius` mapping.
- Numerical backend parameters use established names including `rtol`, `atol`, `subdivisions`, `N`, `h`, `nu`, `n`, `bias`, and `offset`.
- Cubic interpolation in q space uses not-a-knot boundary conditions; radial-profile cubic interpolation remains natural.
- Finite-grid harmonic convergence is checked independently for every retained harmonic before a common resolution is selected.

### Removed
- Radial tapering and zero-padding controls, which did not improve the validated sampled-data calculation.
- QDHT support.
- Superseded exploratory benchmark and diagnostic scripts from the release surface.
- The tracked publication-run log; benchmark logs are local artifacts and are ignored by Git.

### Fixed
- Negative integer Hankel orders use the correct Bessel-function sign convention.
- Mixed-parity interaction pairs retain the signed-order Bessel phase when numerical backends evaluate nonnegative order `|m-m'|`.
- Zero-momentum and zero-displacement special cases use the finite represented support consistently.
- Singular kernels are not required to be evaluated at `q=0` by Gauss--Legendre interaction quadratures.
- FFTLog accepts displacement arrays in arbitrary input order.
- Benchmark resume/schema handling and calibrated-parameter reporting are consistent across numerical backends.
