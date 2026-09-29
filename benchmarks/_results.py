"""Publication-result consolidation for the QUARTIC2D benchmark suite.

Numerical runners write their native JSON into a temporary staging directory.
A completed suite is promoted into a small, human-readable set of evidence files
organized by the scientific question each benchmark answers.  Native result
objects are embedded without numerical transformation so downstream manuscript
figures consume exactly the same benchmark values.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmarks._common import write_json

SCHEMA_VERSION = 1

# output filename -> (benchmark class, description, section key -> native filename)
PUBLICATION_RESULT_SETS: dict[str, tuple[str, str, dict[str, str]]] = {
    "harmonic.json": (
        "HarmonicTransform validation",
        "Accuracy, automatic convergence, and finite-quadrature validation for the harmonic transform stage.",
        {
            "reference_accuracy_and_tolerance_response": "harmonic_transform.json",
            "automatic_convergence_validation": "harmonic_convergence_1e-4.json",
            "finite_quadrature_method_matrix": "harmonic_method_matrix_1e-4.json",
        },
    ),
    "interaction_accuracy.json": (
        "Interaction accuracy",
        "Fixed-field reference accuracy, production timing, and the primary-tolerance public-method comparison.",
        {
            "broad_accuracy_and_timing": "interaction_accuracy.json",
            "primary_tolerance_method_matrix": "interaction_method_matrix_1e-4.json",
        },
    ),
    "interaction_convergence.json": (
        "Interaction automatic convergence",
        "Automatic-certification reliability and bounded capability at the primary and practical large-δ tolerances.",
        {
            "primary_tolerance": "interaction_convergence.json",
            "large_delta_practical_tolerance": "interaction_convergence_large_1e-3.json",
        },
    ),
    "pipeline.json": (
        "Pipeline validation",
        "Cross-stage and true end-to-end validation from harmonic decomposition through final interactions.",
        {
            "cross_stage_standard_delta": "cross_stage_standard.json",
            "cross_stage_large_delta": "cross_stage_large.json",
            "end_to_end_standard_delta": "end_to_end_standard.json",
            "end_to_end_large_delta": "end_to_end_large.json",
        },
    ),
    "performance.json": (
        "Performance and reuse",
        "Automatic-calibration overhead and qualified fixed-configuration production timing.",
        {
            "automatic_convergence_cost_primary_tolerance": "autoconvergence_performance.json",
            "automatic_convergence_cost_large_delta_practical_tolerance": "autoconvergence_performance_large_1e-3.json",
            "qualified_fixed_configuration_large_delta_practical_tolerance": "fixed_configuration_performance_large_1e-3.json",
        },
    ),
    "scaling.json": (
        "Computational scaling",
        "Measured runtime scaling for HarmonicTransform and Interaction against source-derived work measures.",
        {
            "harmonic_transform": "harmonic_scaling.json",
            "interaction": "interaction_scaling.json",
        },
    ),
    "memory.json": (
        "Peak-memory scaling",
        "Fresh-process incremental peak-RSS scaling for HarmonicTransform and Interaction.",
        {
            "harmonic_transform": "harmonic_memory.json",
            "interaction": "interaction_memory.json",
        },
    ),
}


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def consolidate_publication_results(
    staging_dir: Path,
    output_dir: Path,
    *,
    require_complete: bool = True,
) -> list[Path]:
    """Promote native runner JSON into the canonical publication result set.

    Native result objects are nested unchanged under descriptive section names.
    This is deliberately lossless: no benchmark value is rounded, renamed, or
    recomputed during consolidation.
    """
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    missing: list[Path] = []
    for filename, (benchmark_class, description, sections) in PUBLICATION_RESULT_SETS.items():
        consolidated_sections: dict[str, dict] = {}
        for section_name, native_filename in sections.items():
            source = staging_dir / native_filename
            if not source.exists():
                missing.append(source)
                continue
            consolidated_sections[section_name] = _load_json(source)

        if consolidated_sections:
            result = {
                "schema_version": SCHEMA_VERSION,
                "result_type": "publication_benchmark_collection",
                "benchmark_class": benchmark_class,
                "description": description,
                "sections": consolidated_sections,
            }
            target = output_dir / filename
            write_json(target, result)
            written.append(target)

    if require_complete and missing:
        rel = [str(path.relative_to(staging_dir)) for path in missing]
        raise RuntimeError(
            "publication result consolidation is incomplete; missing native result files: "
            + ", ".join(rel)
        )
    return written


def remove_staging_directory(path: Path) -> None:
    """Remove the temporary native-result directory after successful promotion."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def load_publication_results(results_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Load canonical evidence into the legacy logical keys used by figures.

    Returns
    -------
    data
        Mapping consumed by ``manuscript_artifacts.py``.
    provenance
        Mapping from each logical key to its consolidated file/section location.
    """
    results_dir = Path(results_dir)
    bundles = {
        filename: _load_json(results_dir / filename)
        for filename in PUBLICATION_RESULT_SETS
    }

    mapping = {
        "harmonic_transform": ("harmonic.json", "reference_accuracy_and_tolerance_response"),
        "harmonic_convergence_1e-4": ("harmonic.json", "automatic_convergence_validation"),
        "harmonic_method_matrix_1e-4": ("harmonic.json", "finite_quadrature_method_matrix"),
        "interaction_accuracy": ("interaction_accuracy.json", "broad_accuracy_and_timing"),
        "interaction_method_matrix_1e-4": ("interaction_accuracy.json", "primary_tolerance_method_matrix"),
        "interaction_convergence": ("interaction_convergence.json", "primary_tolerance"),
        "interaction_convergence_large_1e-3": ("interaction_convergence.json", "large_delta_practical_tolerance"),
        "cross_stage_standard": ("pipeline.json", "cross_stage_standard_delta"),
        "cross_stage_large": ("pipeline.json", "cross_stage_large_delta"),
        "end_to_end_standard": ("pipeline.json", "end_to_end_standard_delta"),
        "end_to_end_large": ("pipeline.json", "end_to_end_large_delta"),
        "autoconvergence_performance": ("performance.json", "automatic_convergence_cost_primary_tolerance"),
        "autoconvergence_performance_large_1e-3": ("performance.json", "automatic_convergence_cost_large_delta_practical_tolerance"),
        "fixed_performance_large_1e-3": ("performance.json", "qualified_fixed_configuration_large_delta_practical_tolerance"),
        "harmonic_scaling": ("scaling.json", "harmonic_transform"),
        "interaction_scaling": ("scaling.json", "interaction"),
        "harmonic_memory": ("memory.json", "harmonic_transform"),
        "interaction_memory": ("memory.json", "interaction"),
    }

    data: dict[str, dict] = {}
    provenance: dict[str, str] = {}
    for logical_key, (filename, section) in mapping.items():
        data[logical_key] = bundles[filename]["sections"][section]
        provenance[logical_key] = f"{filename} :: sections.{section}"
    return data, provenance
