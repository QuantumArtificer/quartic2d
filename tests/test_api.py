# ---- test_public_api.py ----
import quartic2d


def test_public_api_is_minimal_and_explicit():
    assert quartic2d.__all__ == [
        "HankelTransform",
        "HarmonicTransform",
        "Interaction",
        "__version__",
    ]


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
import importlib.util
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "anisotropic_interaction.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("quartic2d_anisotropic_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_anisotropic_example_keeps_only_physical_harmonics():
    example = load_example_module()
    dec = example.make_decomposition()

    assert dec.m_sorted == [0, 2, -2]
