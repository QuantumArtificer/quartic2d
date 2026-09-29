# Figure design system

QUARTIC2D uses one semantic visual language across public plotting helpers,
documentation figures, validation plots, and manuscript artifacts. Plotting code
should import semantic styles from `quartic2d._plot_style` rather than assign a
new color or marker to a numerical method locally.

## Numerical methods

Each numerical method has a fixed color, marker, and line style. The marker and
line style are intentional redundant encodings so method identity does not rely
on color alone. The canonical labels are `Trapezoid`, `Simpson`, `GL4`, `GL8`,
`FFTLog`, and `Ogata`.

A figure may omit one or more encodings when they do not apply, but it should
not reassign them. In particular, a method must not change color or marker from
one figure to another.

## Reference and guide lines

Independent references use a dark neutral line. Zero baselines, numerical
criteria, asymptotic guides, and selected-parameter guides use neutral gray
styles. They must not borrow a method color because that visually associates a
reference or threshold with one backend.

A line representing a self-convergence threshold must be described as such. A
line on an independently measured reference-error plot is a reference-error
comparison level, not an `rtol` error bound.

## Legends

Legends must not obscure data. Repeated panel semantics should normally use one
figure-level legend or direct labels rather than duplicate legends on every
axis. Equations and explanatory sentences belong in the caption or prose, not
inside legend entries.

When a figure contains only a few isolated points or curves, direct labels are
preferred when they remain unambiguous. Missing results and exact zero values
must be distinguishable from data that were not run or not qualified.

## Multi-panel figures and colorbars

Multi-panel figures use `(a)`, `(b)`, ... labels when the caption discusses
panels separately. Panels that encode the same physical quantity should share
axis limits and color normalization when comparison by position or color is
intended. Shared normalization should use a shared colorbar. If separate color
scales are scientifically necessary, the distinction must be explicit.

Dense raster-like content may be rasterized inside an SVG to avoid visible cell
edges while retaining vector text and axes.

## Documentation display sizes

Sphinx figures use one of three display classes instead of ad hoc percentage
widths:

- `q2d-figure-compact` for simple single-panel figures;
- `q2d-figure-standard` for ordinary figures;
- `q2d-figure-wide` for genuinely wide or dense multi-panel figures.

All three collapse to the available content width on narrow screens. Figure
redesign should favor splitting an overloaded panel composition over assigning
it an ever-larger display class.
