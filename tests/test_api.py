# ---- test_public_api.py ----
import importlib.util
from pathlib import Path

import quartic2d


def test_public_api_is_minimal_and_explicit():
    assert quartic2d.__all__ == [
        "HankelTransform",
        "HarmonicConvergenceResult",
        "HarmonicTransform",
        "Interaction",
        "InteractionConvergenceResult",
        "__version__",
    ]


def test_convergence_results_are_public_types():
    assert quartic2d.HarmonicConvergenceResult.__module__ == "quartic2d.convergence"
    assert quartic2d.InteractionConvergenceResult.__module__ == "quartic2d.convergence"


def test_version_is_public():
    assert quartic2d.__version__ == "0.1.0"


def test_legacy_top_level_names_are_not_exposed():
    for name in (
        "HankelofHarmonics",
        "Interact",
        "available_quadratures",
        "available_transforms",
        "available_interpolators",
        "converge_sequence",
        "relative_l2_error",
    ):
        assert not hasattr(quartic2d, name)

# ---- test_anisotropic_example.py ----
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"
PUBLIC_EXAMPLES = (
    "isotropic_interaction.py",
    "four_center_interaction.py",
    "anisotropic_interaction.py",
    "sampled_data.py",
    "complex_transition_field.py",
    "screening_family_sweep.py",
    "large_separation.py",
)


def load_example_module(filename):
    path = EXAMPLE_DIR / filename
    module_name = f"quartic2d_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_examples_are_import_safe(capsys):
    for filename in PUBLIC_EXAMPLES:
        load_example_module(filename)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_anisotropic_example_keeps_only_physical_harmonics():
    example = load_example_module("anisotropic_interaction.py")
    dec = example.make_decomposition()

    assert dec.m_sorted == [0, 2, -2]

# ---- convergence plotting helpers ----
def test_harmonic_convergence_result_plotter_returns_three_axes():
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg", force=True)

    sampling = SimpleNamespace(
        tail_modes={0: SimpleNamespace(q_required=4.0), 2: SimpleNamespace(q_required=5.0)},
        q_max=5.2,
        q_ceiling=12.0,
        q_tail_rtol=1.0e-3,
        interpolation_rtol=1.0e-4,
        q_grid=None,
        converged=True,
        tail_converged=True,
        interpolation_converged=True,
        interpolation_steps=[
            SimpleNamespace(
                n_q=25,
                harmonic_errors={
                    0: {"relative_l2": 3.0e-4},
                    2: {"relative_l2": 5.0e-4},
                },
            ),
            SimpleNamespace(
                n_q=41,
                harmonic_errors={
                    0: {"relative_l2": 4.0e-5},
                    2: {"relative_l2": 7.0e-5},
                },
            ),
        ],
        to_dict=lambda: {},
    )
    quadrature = SimpleNamespace(
        rtol=1.0e-4,
        atol=1.0e-12,
        converged=True,
        selected_parameters={"subdivisions": 4},
        metadata={
            "harmonics": {
                "0": {
                    "steps": [
                        {"parameters": {"subdivisions": 1}, "relative_l2_change": None},
                        {"parameters": {"subdivisions": 2}, "relative_l2_change": 2.0e-4},
                        {"parameters": {"subdivisions": 4}, "relative_l2_change": 2.0e-5},
                    ]
                },
                "2": {
                    "steps": [
                        {"parameters": {"subdivisions": 1}, "relative_l2_change": None},
                        {"parameters": {"subdivisions": 2}, "relative_l2_change": 4.0e-4},
                        {"parameters": {"subdivisions": 4}, "relative_l2_change": 5.0e-5},
                    ]
                },
            }
        },
        to_dict=lambda: {},
    )
    result = quartic2d.HarmonicConvergenceResult(
        sampling=sampling,
        quadrature=quadrature,
        method="simpson",
        interpolator="cubic",
        rtol=1.0e-4,
        atol=1.0e-12,
        q_tail_rtol=1.0e-3,
    )

    fig, axes = result.plot_convergence(title="")
    assert len(axes) == 3
    fig.clf()


def test_interaction_convergence_result_plotter_handles_finite_search():
    from types import SimpleNamespace

    import matplotlib

    matplotlib.use("Agg", force=True)

    steps = [
        SimpleNamespace(
            parameters={"subdivisions": 1},
            relative_l2_change=None,
            runtime_seconds=0.01,
            metadata={},
        ),
        SimpleNamespace(
            parameters={"subdivisions": 2},
            relative_l2_change=2.0e-5,
            runtime_seconds=0.02,
            metadata={},
        ),
        SimpleNamespace(
            parameters={"subdivisions": 4},
            relative_l2_change=1.0e-7,
            runtime_seconds=0.04,
            metadata={"observed_order_l2": 7.8},
        ),
    ]
    search = SimpleNamespace(
        metadata={
            "kind": "fixed_order_richardson",
            "parameter_name": "subdivisions",
            "expected_order": 8.0,
        },
        selected_parameters={"subdivisions": 2},
        rtol=5.0e-5,
        atol=5.0e-13,
        steps=steps,
        converged=True,
        to_dict=lambda **kwargs: {},
    )
    result = quartic2d.InteractionConvergenceResult(
        search=search,
        method="gl4",
        interpolator="cubic",
        rtol=1.0e-4,
        atol=1.0e-12,
    )

    fig, axes = result.plot_convergence(title="")
    assert len(axes) == 1
    assert axes[0].get_ylabel() == r"$L^2$ error"
    fig.clf()


def test_interaction_plot_convergence_delegates_to_attached_result():
    class FakeResult:
        def plot_convergence(self, title):
            return "figure", title

    interaction = quartic2d.Interaction.__new__(quartic2d.Interaction)
    interaction._convergence_result = FakeResult()

    assert interaction.plot_convergence("custom") == ("figure", "custom")
