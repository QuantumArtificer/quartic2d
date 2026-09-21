# Getting started

## Installation

Install QUARTIC2D with

```bash
python -m pip install quartic2d
```

For development:

```bash
git clone https://github.com/QuantumArtificer/quartic2d.git
cd quartic2d
python -m pip install -e ".[test,docs,release]"
```

## Workflow

QUARTIC2D starts from a PETAL2D decomposition.

### Step 1: decompose the field with PETAL2D

```python
import numpy as np
from petal2d import PolarDecomposition

x = np.linspace(-6.0, 6.0, 161)
y = np.linspace(-6.0, 6.0, 161)


def density(x, y):
    return np.exp(-(x**2 + y**2)) / np.pi


dec = PolarDecomposition(density, x, y, Nr=161, Ntheta=256)
```

PETAL2D decides which angular harmonics are needed and provides their radial profiles and support radii.

### Step 2: transform the retained radial profiles

```python
from quartic2d import HarmonicTransform

transformed = HarmonicTransform(dec)
```

That is the normal production path. QUARTIC2D chooses a scale-aware q range and q-grid density, uses Simpson quadrature with the benchmarked default radial refinement, and computes the transform once.

The object also runs very cheap sanity checks. They inspect the transform that was already calculated; they do not launch additional Hankel transforms. If no warning appears, continue normally.

You can evaluate a transformed harmonic at any momentum inside the sampled range:

```python
transformed(0, 1.0)
transformed(0, np.linspace(0.0, 4.0, 100))
```

### What the main Step-2 parameters mean

`q_max` answers **how far in momentum space the form factor is represented**. A value that is too small cuts off a transform that has not yet decayed. A value that is too large can ask the radial data to represent momentum structure it does not contain.

`n_q` answers **how finely that momentum interval is tabulated**. Too few points can make interpolation miss structure between neighboring q samples. Increasing `n_q` mainly costs memory and evaluations of the radial integral.

`subdivisions` answers **how finely each PETAL2D radial interval is integrated**. With the default `subdivisions=2`, every original radial interval is split once before composite Simpson integration. This affects the accuracy and cost of the Hankel quadrature itself.

Most users should leave all three at their defaults unless a diagnostic warns or a production workload needs deliberate tuning.

### When to run convergence

Convergence is opt-in because it repeats calculations at several numerical resolutions.

```python
calibration = HarmonicTransform.converge_parameters(
    dec,
    rtol=1e-4,
    atol=1e-12,
    q_tail_rtol=1e-3,
)
```

The report distinguishes two error questions:

- `rtol` / `atol`: are the q-grid interpolation and radial quadrature numerically stable on the interval we represent?
- `q_tail_rtol`: has the form factor decayed far enough that the omitted q-space tail contains at most the requested relative L2 norm?

The returned parameters can be reused without rerunning convergence:

```python
transformed = calibration.transform(dec)
```

### Step 3: evaluate an interaction

```python
from quartic2d import Interaction


def U_q(q):
    kappa = 0.25
    return 2.0 * np.pi / np.sqrt(q**2 + kappa**2)


deltas = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
result = Interaction(deltas, transformed, transformed, U_q)
result.V
```

See {doc}`user_guide` for numerical diagnostics and backend selection, {doc}`examples` for executable examples, and {doc}`reference/index` for the complete API.
