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
