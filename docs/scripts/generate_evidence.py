"""Generate tracked validation prose/tables from canonical publication JSON.

The benchmark JSON bundle is the numerical source of truth. This script writes
small Markdown fragments used by the Sphinx validation pages plus a metadata
file containing input hashes. The generated fragments are tracked so a fresh
source checkout can build the documentation without carrying the large,
ignored publication result bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "benchmarks" / "results"
DEFAULT_OUTPUT = ROOT / "docs" / "source" / "_generated" / "validation"

REQUIRED_RESULTS = (
    "manifest.json",
    "harmonic.json",
    "interaction_accuracy.json",
    "interaction_convergence.json",
    "pipeline.json",
    "performance.json",
    "scaling.json",
    "memory.json",
)

METHOD_LABELS = {
    "trapezoid": "Trapezoid",
    "simpson": "Simpson",
    "gl4": "GL4",
    "gl8": "GL8",
    "fftlog": "FFTLog",
    "ogata": "Ogata",
}
CASE_LABELS = {
    "isotropic_coulomb": "isotropic Coulomb",
    "anisotropic_rk_strong": "anisotropic RK",
    "complex_gate": "complex dual gate",
    "nodal_rpa": "nodal RPA",
}
SELECTOR_LABELS = {
    "converged_minimal": "accepted",
    "converged_nonminimal": "accepted",
    "converged_reference_pass": "accepted",
    "tested_box_not_capable": "refused",
    "selector_miss_backend_capable": "refused",
    "conservative_rejection_terminal_pass": "refused",
    "conservative_rejection": "refused",
    "missed_convergence": "refused",
}
QUALIFICATION_LABELS = {
    "converged_minimal": "reference pass",
    "converged_nonminimal": "reference pass",
    "converged_reference_pass": "reference pass",
    "tested_box_not_capable": "not established in tested box",
    "selector_miss_backend_capable": "qualified point found",
    "conservative_rejection_terminal_pass": "terminal point passes reference",
    "conservative_rejection": "not established",
    "missed_convergence": "not established",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_fingerprint(results_dir: Path) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    digest = hashlib.sha256()
    for name in REQUIRED_RESULTS:
        path = results_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical publication result: {path}")
        file_hash = _sha256(path)
        hashes[name] = file_hash
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), hashes


def _load(results_dir: Path, name: str):
    return json.loads((results_dir / name).read_text())


def _sci(value: float, digits: int = 3) -> str:
    if value == 0:
        return "$0$"
    exponent = int(f"{abs(value):e}".split("e")[1])
    mantissa = value / (10.0**exponent)
    mantissa_text = f"{mantissa:.{digits - 1}f}".rstrip("0").rstrip(".")
    return rf"${mantissa_text}\times10^{{{exponent}}}$"


def _seconds(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3g} s"
    if value >= 0.01:
        return f"{value:.3f} s"
    return f"{value:.3g} s"


def _generated_header(fingerprint: str) -> str:
    return (
        "<!-- GENERATED FILE: do not edit by hand.\n"
        "     Source: canonical publication JSON via docs/scripts/generate_evidence.py\n"
        f"     Publication bundle SHA-256: {fingerprint}\n"
        "-->\n\n"
    )


def render_harmonic_primary(harmonic: dict) -> str:
    section = harmonic["sections"]["finite_quadrature_method_matrix"]
    methods = section["settings"]["methods"]
    accuracy_lines = [
        "| Method | Workloads complete | Worst in-domain $L^2$ error |",
        "| --- | ---: | ---: |",
    ]
    production_lines = [
        "| Method | Median production time | Selected subdivisions (workload count) |",
        "| --- | ---: | --- |",
    ]
    for method in methods:
        completed = []
        times = []
        errors = []
        subdivisions = []
        for row in section["rows"]:
            result = row["methods"][method]
            if result["status"] != "complete":
                continue
            completed.append(row)
            times.append(result["timing"]["median_seconds"])
            errors.append(result["worst_in_domain_relative_l2_total_norm"])
            subdivisions.append(int(result["subdivisions_selected"]))
        counts = Counter(subdivisions)
        subdivision_text = ", ".join(
            f"{value}: {count}/{len(completed)}" for value, count in sorted(counts.items())
        )
        accuracy_lines.append(
            f"| {METHOD_LABELS[method]} | {len(completed)}/{len(section['rows'])} | "
            f"{_sci(max(errors), 2)} |"
        )
        production_lines.append(
            f"| {METHOD_LABELS[method]} | {_seconds(statistics.median(times))} | "
            f"{subdivision_text} |"
        )
    return "\n".join(accuracy_lines) + "\n\n" + "\n".join(production_lines) + "\n"


def render_interaction_primary(interaction_accuracy: dict) -> str:
    section = interaction_accuracy["sections"]["primary_tolerance_method_matrix"]
    status_lines = [
        "| Method | Automatic acceptance | Reference pass among accepted | "
        "Conservative refusals with a passing tested point | Accepted / reference fail |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    accuracy_lines = [
        "| Method | Worst relative $L^2$ among accepted results | "
        "Worst peak-normalized max. error among accepted results |",
        "| --- | ---: | ---: |",
    ]
    for row in section["summary"]:
        accepted = int(row["n_automatic_converged"])
        ref_pass = int(row["n_reference_pass"])
        status_lines.append(
            f"| {METHOD_LABELS[row['method']]} | {accepted} / {row['n_cases']} | "
            f"{ref_pass} / {accepted} | {row['n_conservative_rejection_oracle_pass']} | "
            f"{row['n_false_positive']} |"
        )
        accuracy_lines.append(
            f"| {METHOD_LABELS[row['method']]} | {_sci(row['worst_relative_l2'])} | "
            f"{_sci(row['worst_relative_peak'])} |"
        )
    return "\n".join(status_lines) + "\n\n" + "\n".join(accuracy_lines) + "\n"



def render_interaction_tolerance_response(interaction_accuracy: dict) -> str:
    section = interaction_accuracy["sections"]["broad_accuracy_and_timing"]
    target = 1.0e-5
    rows = {
        row["method"]: row
        for row in section["summary"]
        if float(row["requested_tolerance"]) == target
    }
    parts = []
    for method in ("simpson", "gl4", "fftlog"):
        row = rows[method]
        parts.append(
            f"{METHOD_LABELS[method]} {row['n_automatic_converged']}/{row['n_cases']} automatic acceptances "
            f"and {row['n_reference_pass']}/{row['n_cases']} independent-reference passes"
        )
    return (
        "At `Interaction.rtol = 1e-5`, the same broad matrix gives "
        + "; ".join(parts)
        + ". This records the boundary of the declared search boxes; it does not imply "
          "mathematical impossibility outside those boxes.\n"
    )

def render_fftlog_coverage(interaction_accuracy: dict) -> str:
    section = interaction_accuracy["sections"]["primary_tolerance_method_matrix"]
    rows = [row for row in section["rows"] if row["method"] == "fftlog"]
    workload_order = []
    workload_counts: OrderedDict[str, list[int]] = OrderedDict()
    kernel_counts: OrderedDict[str, list[int]] = OrderedDict()
    for row in rows:
        workload = row["workload"]
        if workload not in workload_counts:
            workload_counts[workload] = [0, 0]
            workload_order.append(workload)
        workload_counts[workload][1] += 1
        workload_counts[workload][0] += int(bool(row["automatic_converged"]))
        family = row["kernel_family"]
        if family not in kernel_counts:
            kernel_counts[family] = [0, 0]
        kernel_counts[family][1] += 1
        kernel_counts[family][0] += int(bool(row["automatic_converged"]))

    workload_labels = {
        "gaussian_isotropic": "isotropic Gaussian",
        "gaussian_odd_pair": "odd-pair Gaussian",
        "gaussian_anisotropic_m2": "anisotropic Gaussian",
        "gaussian_high_order_m4": "high-order $m=4$ continuity",
        "nodal_mixed": "nodal mixed",
        "complex_mixed": "complex mixed-parity",
        "exponential_cusp": "exponential cusp",
        "oscillatory_exponential": "oscillatory exponential",
        "algebraic_mixed": "algebraic",
    }
    kernel_labels = {
        "rytova_keldysh": "Rytova–Keldysh",
        "coulomb": "bare Coulomb",
        "thomas_fermi": "Thomas–Fermi",
        "yukawa": "Yukawa",
        "helmholtz_yukawa_2d": "2D Helmholtz/Yukawa",
        "single_gate": "single gate",
        "dual_gate": "dual gate",
        "interlayer_coulomb": "interlayer Coulomb",
        "gated_interlayer": "gated interlayer",
        "static_2deg_rpa": "static 2DEG RPA",
        "numerical_control": "smooth numerical control",
    }
    lines = ["**Field-workload coverage**", ""]
    for key in workload_order:
        accepted, total = workload_counts[key]
        lines.append(f"- {workload_labels.get(key, key)}: {accepted}/{total} accepted.")
    lines += ["", "**Kernel-family coverage**", ""]
    for key, (accepted, total) in kernel_counts.items():
        lines.append(f"- {kernel_labels.get(key, key)}: {accepted}/{total} accepted.")
    return "\n".join(lines) + "\n"


def render_canonical_outcomes(interaction_convergence: dict) -> str:
    rows = interaction_convergence["sections"]["primary_tolerance"]["rows"]
    methods = ("simpson", "gl4", "fftlog", "ogata")
    cases = ("isotropic_coulomb", "anisotropic_rk_strong", "complex_gate", "nodal_rpa")
    lookup = {(r["delta_domain"], r["case"], r["method"]): r for r in rows}

    selector_lines = [
        "**Automatic selector outcome**",
        "",
        "| Domain | Case | Simpson | GL4 | FFTLog | Ogata |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    qualification_lines = [
        "**Independent qualification outcome**",
        "",
        "| Domain | Case | Simpson | GL4 | FFTLog | Ogata |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for domain in ("standard", "large"):
        for case in cases:
            selector_cells = []
            qualification_cells = []
            for method in methods:
                cls = lookup[(domain, case, method)]["classification"]
                selector_cells.append(SELECTOR_LABELS[cls])
                qualification_cells.append(QUALIFICATION_LABELS[cls])
            selector_lines.append(
                f"| {domain} | {CASE_LABELS[case]} | " + " | ".join(selector_cells) + " |"
            )
            qualification_lines.append(
                f"| {domain} | {CASE_LABELS[case]} | "
                + " | ".join(qualification_cells)
                + " |"
            )
    return "\n".join(selector_lines) + "\n\n" + "\n".join(qualification_lines) + "\n"


def render_reference_stability(interaction_convergence: dict) -> str:
    refs = interaction_convergence["sections"]["primary_tolerance"]["reference_stability"]
    complex_row = refs["large::complex_gate"]
    nodal_row = refs["large::nodal_rpa"]
    return (
        "For the complex mixed-parity dual-gate case, the final two phase-resolved "
        f"reference levels differ by {_sci(complex_row['relative_l2_change'])} in relative "
        f"$L^2$ and {_sci(complex_row['relative_peak_change'])} in peak-normalized maximum "
        f"error. The declared stability threshold is {_sci(complex_row['required_max_change'])}.\n\n"
        "The harder nodal-RPA reference requires additional phase refinement and reaches a "
        f"final coarse/fine relative-$L^2$ change of {_sci(nodal_row['relative_l2_change'])}.\n"
    )


def render_cross_stage(pipeline: dict) -> str:
    section = pipeline["sections"]["cross_stage_standard_delta"]
    rows = {row["method"]: row for row in section["summary"]}
    simpson = rows["simpson"]
    gl4 = rows["gl4"]
    return (
        "The standard-domain cross-stage suite evaluates "
        f"{simpson['n_cases']} cases after adaptive `HarmonicTransform` sampling. "
        f"Simpson produces {simpson['n_reference_pass']}/{simpson['n_cases']} independent-reference "
        f"passes and GL4 produces {gl4['n_reference_pass']}/{gl4['n_cases']}. Automatic selector "
        "outcomes remain recorded separately from those reference comparisons.\n"
    )


def render_end_to_end(pipeline: dict) -> str:
    case_labels = {
        "gaussian_coulomb": "isotropic Coulomb",
        "anisotropic_rk_strong": "anisotropic RK",
        "nodal_rpa": "nodal RPA",
    }
    lines = [
        "| Domain | Case | PETAL2D relative $L^2$ | `HarmonicTransform` relative $L^2$ | "
        "`Interaction` relative $L^2$ |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for domain, section_name in (
        ("standard", "end_to_end_standard_delta"),
        ("large", "end_to_end_large_delta"),
    ):
        for case in pipeline["sections"][section_name]["cases"]:
            interaction_errors = [float(row["relative_l2"]) for row in case["interactions"]]
            lines.append(
                f"| {domain} | {case_labels.get(case['name'], case['name'])} | "
                f"{_sci(case['petal2d']['relative_l2'])} | "
                f"{_sci(case['harmonic_transform']['analytic_error']['relative_l2'])} | "
                f"{_sci(max(interaction_errors))} |"
            )
    return "\n".join(lines) + "\n"


def render_standard_calibration_cost(performance: dict) -> str:
    summary = performance["sections"]["automatic_convergence_cost_primary_tolerance"]["summary"]
    rows = {
        row["method"]: row
        for row in summary
        if row["stage"] == "Interaction" and row["delta_domain"] == "standard"
    }
    lines = [
        "| Method | Median calibration | Median selected production | "
        "Median per-case calibration / production ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("simpson", "gl4", "fftlog"):
        row = rows[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_seconds(row['median_calibration_seconds'])} | "
            f"{_seconds(row['median_selected_production_seconds'])} | "
            f"{row['median_calibration_to_selected_production']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def render_large_delta_timing(performance: dict) -> str:
    rows = performance["sections"]["qualified_fixed_configuration_large_delta_practical_tolerance"]["rows"]
    lookup = {(row["case"], row["method"]): row for row in rows}
    methods = ("simpson", "gl4", "fftlog", "ogata")
    cases = ("isotropic_coulomb", "anisotropic_rk_strong", "complex_gate", "nodal_rpa")
    qualification_labels = {
        "automatic_certificate": "automatic",
        "independent_oracle_qualification": "independent",
        "independent_terminal_qualification": "independent (terminal)",
    }

    timing_lines = [
        "**Median fixed-production time**",
        "",
        "| Case | Simpson | GL4 | FFTLog | Ogata |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    qualification_lines = [
        "**Qualification source**",
        "",
        "| Case | Simpson | GL4 | FFTLog | Ogata |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        timing_values = []
        qualification_values = []
        for method in methods:
            row = lookup.get((case, method))
            if row is None:
                timing_values.append("not qualified")
                qualification_values.append("not qualified")
                continue
            timing_values.append(_seconds(row["production_timing"]["median_seconds"]))
            qualification_values.append(qualification_labels[row["qualification_kind"]])
        timing_lines.append(f"| {CASE_LABELS[case]} | " + " | ".join(timing_values) + " |")
        qualification_lines.append(
            f"| {CASE_LABELS[case]} | " + " | ".join(qualification_values) + " |"
        )
    return "\n".join(timing_lines) + "\n\n" + "\n".join(qualification_lines) + "\n"


RENDERERS = {
    "harmonic_primary.md": lambda data: render_harmonic_primary(data["harmonic.json"]),
    "interaction_primary.md": lambda data: render_interaction_primary(data["interaction_accuracy.json"]),
    "interaction_tolerance_response.md": lambda data: render_interaction_tolerance_response(data["interaction_accuracy.json"]),
    "fftlog_coverage.md": lambda data: render_fftlog_coverage(data["interaction_accuracy.json"]),
    "canonical_outcomes.md": lambda data: render_canonical_outcomes(data["interaction_convergence.json"]),
    "reference_stability.md": lambda data: render_reference_stability(data["interaction_convergence.json"]),
    "cross_stage.md": lambda data: render_cross_stage(data["pipeline.json"]),
    "end_to_end.md": lambda data: render_end_to_end(data["pipeline.json"]),
    "standard_calibration_cost.md": lambda data: render_standard_calibration_cost(data["performance.json"]),
    "large_delta_timing.md": lambda data: render_large_delta_timing(data["performance.json"]),
}


def generate(results_dir: Path, output_dir: Path) -> None:
    fingerprint, input_hashes = publication_fingerprint(results_dir)
    data = {name: _load(results_dir, name) for name in REQUIRED_RESULTS}
    manifest = data["manifest.json"]
    if manifest.get("status") != "passed":
        raise RuntimeError("Canonical publication manifest does not report status='passed'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes: dict[str, str] = {}
    for name, renderer in RENDERERS.items():
        path = output_dir / name
        path.write_text(_generated_header(fingerprint) + renderer(data))
        output_hashes[name] = _sha256(path)
        print(path.relative_to(ROOT))

    metadata = {
        "schema_version": 1,
        "publication_bundle_sha256": fingerprint,
        "input_sha256": input_hashes,
        "source_manifest": {
            "status": manifest.get("status"),
            "profile": manifest.get("profile"),
            "repository": manifest.get("repository"),
            "provenance_assessment": manifest.get("provenance_assessment"),
        },
        "generated_outputs_sha256": output_hashes,
    }
    metadata_path = output_dir / "snapshot.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(metadata_path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
