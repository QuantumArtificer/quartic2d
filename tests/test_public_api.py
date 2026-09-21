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
