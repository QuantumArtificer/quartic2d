"""Extended PETAL2D/QUARTIC2D diagnostic report for development use."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from petal2d import PolarDecomposition

from quartic2d import HarmonicTransform


def print_table(title, headers, rows):
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in text_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    print(f"\n{title}")
    print("  ".join(str(header).ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


x = np.linspace(-6.0, 6.0, 161)
y = np.linspace(-6.0, 6.0, 161)


def density(x, y):
    return np.exp(-(x**2 + y**2)) * (1.0 + 0.32 * (x**2 - y**2))


dec = PolarDecomposition(
    density,
    x,
    y,
    Nr=256,
    Ntheta=256,
    recon_err_tol=1.0e-4,
)

print_table(
    "PETAL2D retained harmonics",
    ("m", "power fraction", "cutoff radius"),
    [
        (
            int(m),
            f"{dec.power_fracs[int(m)]:.6e}",
            f"{dec.cutoff_radius[int(m)]:.6g}",
        )
        for m in dec.m_sorted
    ],
)

dec.plot_harmonics_with_hist("PETAL2D radial harmonics")
dec.plot_original_vs_reconstructions("PETAL2D reconstruction")

start = perf_counter()
transformed = HarmonicTransform(dec)
default_runtime = perf_counter() - start

diag = transformed.diagnostics
print_table(
    "Default harmonic transform",
    ("quantity", "value"),
    [
        ("harmonics", ", ".join(str(int(m)) for m in transformed.m_values)),
        ("q_max", f"{transformed.q[-1]:.8g}"),
        ("n_q", transformed.q.size),
        ("delta q", f"{diag.dq:.6g}"),
        ("radial q ceiling", f"{diag.q_ceiling:.8g}"),
        ("q sampling factor", f"{diag.q_sampling_factor:.4f}"),
        ("method", transformed.method),
        ("interpolator", transformed.interpolator),
        ("diagnostics", "PASS" if diag.healthy else "CHECK"),
        ("runtime [s]", f"{default_runtime:.6f}"),
    ],
)

print_table(
    "Fast diagnostics by harmonic",
    ("m", "boundary amplitude ratio", "boundary power fraction"),
    [
        (
            m,
            f"{values['boundary_amplitude_ratio']:.6e}",
            f"{values['boundary_power_fraction']:.6e}",
        )
        for m, values in sorted(diag.per_harmonic.items())
    ],
)

if diag.issues:
    print_table(
        "Diagnostic issues",
        ("code", "harmonics", "message"),
        [
            (
                issue.code,
                ", ".join(str(m) for m in issue.harmonics) or "-",
                issue.message,
            )
            for issue in diag.issues
        ],
    )

transformed.plot_harmonics("Quartic2D Hankel transforms")

start = perf_counter()
calibration = HarmonicTransform.converge_parameters(
    dec,
    rtol=1.0e-4,
    atol=1.0e-12,
    q_tail_rtol=1.0e-3,
    verbose=False,
)
convergence_runtime = perf_counter() - start

print_table(
    "Converged parameters",
    ("parameter", "value"),
    [
        ("q_max", f"{calibration.parameters['q_max']:.8g}"),
        ("n_q", calibration.parameters["n_q"]),
        ("method", calibration.parameters["method"]),
        ("interpolator", calibration.parameters["interpolator"]),
        (
            "subdivisions",
            calibration.parameters.get("subdivisions", "-"),
        ),
        ("rtol", f"{calibration.rtol:.2e}"),
        ("atol", f"{calibration.atol:.2e}"),
        ("q_tail_rtol", f"{calibration.q_tail_rtol:.2e}"),
        ("convergence runtime [s]", f"{convergence_runtime:.6f}"),
    ],
)

print_table(
    "q-support convergence by harmonic",
    ("m", "required q", "tail error at q_max", "Parseval error", "resolved"),
    [
        (
            m,
            "-" if result.q_required is None else f"{result.q_required:.8g}",
            f"{result.tail_error_at_q_max:.6e}",
            f"{result.parseval_relative_error:.6e}",
            "yes" if result.resolved else "no",
        )
        for m, result in sorted(calibration.sampling.tail_modes.items())
    ],
)

print_table(
    "q-grid interpolation convergence",
    ("oversampling", "n_q", "worst rel. L2", "worst rel. Linf", "converged"),
    [
        (
            f"{step.oversampling:.4g}",
            step.n_q,
            f"{max(errors['relative_l2'] for errors in step.harmonic_errors.values()):.6e}",
            f"{max(errors['relative_linf'] for errors in step.harmonic_errors.values()):.6e}",
            "yes" if step.converged else "no",
        )
        for step in calibration.sampling.interpolation_steps
    ],
)

quadrature_rows = []
for m, result in sorted(calibration.quadrature.metadata["harmonics"].items(), key=lambda item: int(item[0])):
    measured_steps = [
        step for step in result["steps"] if step["relative_l2_change"] is not None
    ]
    final_step = measured_steps[-1]
    quadrature_rows.append(
        (
            m,
            result["selected_parameters"].get("subdivisions", "-"),
            f"{final_step['relative_l2_change']:.6e}",
            f"{final_step['relative_linf_change']:.6e}",
            f"{final_step['absolute_linf_change']:.6e}",
            "yes" if result["converged"] else "no",
        )
    )

print_table(
    "Radial quadrature convergence by harmonic",
    ("m", "subdivisions", "rel. L2 change", "rel. Linf change", "abs. Linf change", "converged"),
    quadrature_rows,
)

start = perf_counter()
calibrated = calibration.transform(dec)
calibrated_runtime = perf_counter() - start

print_table(
    "Production transform from converged parameters",
    ("quantity", "value"),
    [
        ("q_max", f"{calibrated.q[-1]:.8g}"),
        ("n_q", calibrated.q.size),
        ("runtime [s]", f"{calibrated_runtime:.6f}"),
    ],
)

calibrated.plot_harmonics("Converged Hankel transforms")
calibrated.plot_convergence("Harmonic-transform convergence")

plt.show()
