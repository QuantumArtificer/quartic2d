#!/usr/bin/env python3
"""Canonical QUARTIC2D benchmark launcher.

The launcher separates manuscript-grade evidence from the lightweight documentation
validation. Publication commands use fixed arguments so the final dataset is
reproducible and auditable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmarks._common import git_metadata, write_json
from benchmarks._results import (
    consolidate_publication_results,
    remove_staging_directory,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


@dataclass(frozen=True)
class Job:
    name: str
    tier: str
    module: str
    args: tuple[str, ...]
    purpose: str


def _publication_jobs(result_dir: Path = RESULTS) -> tuple[Job, ...]:
    pub = Path(result_dir)
    return (
        Job(
            "harmonic-transform",
            "publication",
            "benchmarks.harmonic_transform",
            (
                "--output", str(pub / "harmonic_transform.json"),
                "--tolerances", "1e-3,1e-4,1e-5",
                "--required-tolerances", "1e-3,1e-4",
                "--methods", "simpson,gl4",
                "--warmups", "2",
                "--repeats", "7",
                "--require-complete",
            ),
            "Reference accuracy, requested-tolerance response, and input qualification for the principal HarmonicTransform methods.",
        ),
        Job(
            "harmonic-autoconvergence",
            "publication",
            "benchmarks.harmonic_convergence",
            (
                "--output", str(pub / "harmonic_convergence_1e-4.json"),
                "--methods", "simpson,gl4",
                "--rtol", "1e-4",
                "--q-tail-rtol", "1e-3",
                "--n-check", "4001",
                "--include-stress",
            ),
            "Automatic HarmonicTransform certification against analytic generality workloads, including explicit stress refusals.",
        ),
        Job(
            "harmonic-method-matrix",
            "publication",
            "benchmarks.harmonic_transform",
            (
                "--output", str(pub / "harmonic_method_matrix_1e-4.json"),
                "--tolerances", "1e-4",
                "--methods", "trapezoid,simpson,gl4,gl8",
                "--warmups", "2",
                "--repeats", "5",
            ),
            "Supplemental finite-quadrature comparison at the primary publication tolerance.",
        ),
        Job(
            "interaction-accuracy",
            "publication",
            "benchmarks.interaction_accuracy",
            (
                "--output", str(pub / "interaction_accuracy.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--methods", "simpson,gl4,fftlog",
                "--tolerances", "1e-3,1e-4,1e-5",
                "--groups", "continuity,kernels,hard,control",
                "--n-delta", "32",
                "--reference-order", "24",
                "--warmups", "2",
                "--repeats", "5",
            ),
            "Broad fixed-field Interaction accuracy and production timing against independent references.",
        ),
        Job(
            "interaction-method-matrix",
            "publication",
            "benchmarks.interaction_accuracy",
            (
                "--output", str(pub / "interaction_method_matrix_1e-4.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--methods", "trapezoid,simpson,gl4,gl8,fftlog",
                "--tolerances", "1e-4",
                "--groups", "continuity,kernels,hard,control",
                "--n-delta", "32",
                "--reference-order", "24",
                "--warmups", "2",
                "--repeats", "5",
            ),
            "Supplemental public-backend accuracy/runtime matrix at the primary publication tolerance.",
        ),
        Job(
            "interaction-convergence",
            "publication",
            "benchmarks.interaction_convergence",
            (
                "--output", str(pub / "interaction_convergence.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--methods", "simpson,gl4,fftlog,ogata",
                "--tolerances", "1e-4",
                "--cases", "default",
                "--delta-domains", "standard,large",
                "--n-delta-standard", "32",
                "--n-delta-large", "12",
                "--reference-order", "24",
            ),
            "Automatic-certification reliability, large-delta oracle stability, and method coverage.",
        ),
        Job(
            "interaction-large-practical-convergence",
            "publication",
            "benchmarks.interaction_convergence",
            (
                "--output", str(pub / "interaction_convergence_large_1e-3.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--methods", "simpson,gl4,fftlog,ogata",
                "--tolerances", "1e-3",
                "--cases", "default",
                "--delta-domains", "large",
                "--n-delta-large", "12",
                "--reference-order", "24",
            ),
            "Practical-tier large-displacement capability/certification comparison at 1e-3 on the canonical four-case matrix.",
        ),
        Job(
            "interaction-large-practical-performance",
            "publication",
            "benchmarks.autoconvergence_performance",
            (
                "--output", str(pub / "autoconvergence_performance_large_1e-3.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--stages", "interaction",
                "--interaction-cases", "default",
                "--methods", "simpson,gl4,fftlog,ogata",
                "--tolerances", "1e-3",
                "--delta-domains", "large",
                "--n-delta", "12",
                "--calibration-repeats", "5",
                "--production-warmups", "2",
                "--production-repeats", "9",
            ),
            "Practical-tier large-displacement calibration and fixed-parameter production timing at 1e-3 on the same canonical four-case matrix.",
        ),
        Job(
            "interaction-large-qualified-fixed-performance",
            "publication",
            "benchmarks.fixed_configuration_performance",
            (
                "--output", str(pub / "fixed_configuration_performance_large_1e-3.json"),
                "--qualification-results", str(pub / "interaction_convergence_large_1e-3.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--cases", "default",
                "--methods", "simpson,gl4,fftlog,ogata",
                "--delta-domain", "large",
                "--tolerance", "1e-3",
                "--warmups", "2",
                "--repeats", "9",
            ),
            "Fixed-parameter production timing for every independently qualified practical-tier large-displacement configuration, using minimum oracle-proven finite-rule parameters after conservative refusals and reference-passing terminal parameters for other backends.",
        ),
        Job(
            "cross-stage-standard",
            "publication",
            "benchmarks.cross_stage",
            (
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--harmonic-rtol", "1e-4",
                "--q-tail-rtol", "1e-3",
                "--interaction-rtol", "1e-4",
                "--methods", "simpson,gl4",
                "--groups", "continuity,kernels,hard,control",
                "--delta-domain", "standard",
                "--n-delta", "32",
                "--reference-order", "24",
                "--output", str(pub / "cross_stage_standard.json"),
            ),
            "Broad adaptive HarmonicTransform -> Interaction composition validation with analytically supplied radial harmonics.",
        ),
        Job(
            "cross-stage-large",
            "publication",
            "benchmarks.cross_stage",
            (
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--harmonic-rtol", "1e-4",
                "--q-tail-rtol", "1e-3",
                "--interaction-rtol", "1e-4",
                "--methods", "simpson,gl4,fftlog,ogata",
                "--groups", "kernels",
                "--cases",
                "kernels__gaussian_isotropic__coulomb,kernels__gaussian_anisotropic_m2__rk_r0_10,kernels__nodal_mixed__rpa_2deg_kf1_qtf1,kernels__complex_mixed__dual_gate_d1",
                "--delta-domain", "large",
                "--n-delta", "12",
                "--reference-order", "24",
                "--output", str(pub / "cross_stage_large.json"),
            ),
            "Representative large-displacement cross-stage validation across finite-grid, FFTLog, and public Ogata backends; safe upstream-boundary refusals are recorded explicitly.",
        ),
        Job(
            "end-to-end-standard",
            "publication",
            "benchmarks.end_to_end",
            (
                "--cases", "gaussian_coulomb,anisotropic_rk_strong,nodal_rpa",
                "--delta-domain", "standard",
                "--methods", "gl4",
                "--n-delta", "32",
                "--petal-recon-rtol", "1e-5",
                "--petal-profile-rtol", "1e-4",
                "--harmonic-rtol", "1e-4",
                "--q-tail-rtol", "1e-3",
                "--interaction-rtol", "1e-4",
                "--reference-order", "24",
                "--output", str(pub / "end_to_end_standard.json"),
            ),
            "True PETAL2D -> HarmonicTransform -> Interaction validation against analytic-form-factor references.",
        ),
        Job(
            "end-to-end-large",
            "publication",
            "benchmarks.end_to_end",
            (
                "--cases", "gaussian_coulomb,anisotropic_rk_strong,nodal_rpa",
                "--delta-domain", "large",
                "--methods", "gl4",
                "--n-delta", "4",
                "--petal-recon-rtol", "1e-5",
                "--petal-profile-rtol", "1e-4",
                "--harmonic-rtol", "1e-4",
                "--q-tail-rtol", "1e-3",
                "--interaction-rtol", "1e-4",
                "--reference-order", "24",
                "--output", str(pub / "end_to_end_large.json"),
            ),
            "True PETAL2D -> HarmonicTransform -> Interaction validation through delta=1e4.",
        ),
        Job(
            "autoconvergence-performance",
            "publication",
            "benchmarks.autoconvergence_performance",
            (
                "--output", str(pub / "autoconvergence_performance.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--stages", "harmonic,interaction",
                "--methods", "simpson,gl4,fftlog",
                "--tolerances", "1e-4",
                "--delta-domains", "standard,large",
                "--n-delta", "32",
                "--calibration-repeats", "5",
                "--production-warmups", "2",
                "--production-repeats", "9",
            ),
            "Repeated automatic-convergence overhead and selected production timing.",
        ),
        Job(
            "harmonic-scaling",
            "publication",
            "benchmarks.harmonic_scaling",
            (
                "--output", str(pub / "harmonic_scaling.json"),
                "--warmups", "2",
                "--repeats", "7",
            ),
            "HarmonicTransform complexity and work-collapse evidence.",
        ),
        Job(
            "interaction-scaling",
            "publication",
            "benchmarks.interaction_scaling",
            (
                "--output", str(pub / "interaction_scaling.json"),
                "--warmups", "2",
                "--repeats", "7",
            ),
            "Interaction finite-grid/FFTLog scaling evidence.",
        ),
        Job(
            "harmonic-memory",
            "publication",
            "benchmarks.harmonic_memory",
            ("--output", str(pub / "harmonic_memory.json"), "--repeats", "3"),
            "Peak-memory scaling for HarmonicTransform.",
        ),
        Job(
            "interaction-memory",
            "publication",
            "benchmarks.interaction_memory",
            ("--output", str(pub / "interaction_memory.json"), "--repeats", "3"),
            "Peak-memory scaling for Interaction.",
        ),
    )


def _documentation_jobs(result_dir: Path = RESULTS) -> tuple[Job, ...]:
    docs = Path(result_dir)
    return (
        Job(
            "gaussian-workflow",
            "documentation",
            "benchmarks.gaussian_validation",
            ("--quick", "--output", str(docs / "gaussian_validation.json")),
            "Cheap analytic workflow check used by documentation/examples; not paper evidence.",
        ),
    )



JOB_GROUPS = {
    "harmonic-transform": "Harmonic transform",
    "harmonic-autoconvergence": "Harmonic transform",
    "harmonic-method-matrix": "Harmonic transform",
    "interaction-accuracy": "Interaction",
    "interaction-method-matrix": "Interaction",
    "interaction-convergence": "Interaction",
    "interaction-large-practical-convergence": "Interaction",
    "interaction-large-practical-performance": "Performance and reuse",
    "interaction-large-qualified-fixed-performance": "Performance and reuse",
    "autoconvergence-performance": "Performance and reuse",
    "cross-stage-standard": "Pipeline validation",
    "cross-stage-large": "Pipeline validation",
    "end-to-end-standard": "Pipeline validation",
    "end-to-end-large": "Pipeline validation",
    "harmonic-scaling": "Computational scaling",
    "interaction-scaling": "Computational scaling",
    "harmonic-memory": "Peak-memory scaling",
    "interaction-memory": "Peak-memory scaling",
    "gaussian-workflow": "Documentation validation",
}

JOB_TITLES = {
    "harmonic-transform": "HarmonicTransform reference accuracy",
    "harmonic-autoconvergence": "HarmonicTransform automatic convergence",
    "harmonic-method-matrix": "HarmonicTransform finite-quadrature matrix",
    "interaction-accuracy": "Interaction reference accuracy",
    "interaction-method-matrix": "Interaction method matrix",
    "interaction-convergence": "Interaction automatic convergence",
    "interaction-large-practical-convergence": "Large-δ practical convergence",
    "interaction-large-practical-performance": "Large-δ automatic-convergence cost",
    "interaction-large-qualified-fixed-performance": "Large-δ qualified production timing",
    "autoconvergence-performance": "Automatic-convergence performance",
    "cross-stage-standard": "Cross-stage validation: standard δ",
    "cross-stage-large": "Cross-stage validation: large δ",
    "end-to-end-standard": "End-to-end validation: standard δ",
    "end-to-end-large": "End-to-end validation: large δ",
    "harmonic-scaling": "HarmonicTransform runtime scaling",
    "interaction-scaling": "Interaction runtime scaling",
    "harmonic-memory": "HarmonicTransform peak-memory scaling",
    "interaction-memory": "Interaction peak-memory scaling",
    "gaussian-workflow": "Gaussian workflow validation",
}

GROUP_ORDER = {
    "Harmonic transform": 0,
    "Interaction": 1,
    "Performance and reuse": 2,
    "Pipeline validation": 3,
    "Computational scaling": 4,
    "Peak-memory scaling": 5,
    "Documentation validation": 6,
}


def _jobs_for_profile(profile: str, result_dir: Path = RESULTS) -> list[Job]:
    jobs = list(_publication_jobs(result_dir) + _documentation_jobs(result_dir))
    if profile != "all":
        jobs = [job for job in jobs if job.tier == profile]
    original_order = {job.name: index for index, job in enumerate(jobs)}
    jobs.sort(key=lambda job: (GROUP_ORDER.get(JOB_GROUPS.get(job.name, ""), 99), original_order[job.name]))
    return jobs


def _job_group(job: Job) -> str:
    return JOB_GROUPS.get(job.name, job.tier.replace("_", " ").title())


def _human_name(job: Job) -> str:
    return JOB_TITLES.get(job.name, job.name.replace("-", " ").title())


def _single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _print_jobs(jobs: Iterable[Job]) -> None:
    current_group = None
    for job in jobs:
        group = _job_group(job)
        if group != current_group:
            if current_group is not None:
                print()
            print(group)
            print("-" * len(group))
            current_group = group
        print(f"  {_human_name(job)}")
        print(textwrap.fill(job.purpose, width=92, initial_indent="    ", subsequent_indent="    "))


def _structured_arguments(args: tuple[str, ...]) -> dict[str, object]:
    """Represent argparse-style arguments as readable JSON fields."""
    out: dict[str, object] = {}
    i = 0
    while i < len(args):
        item = args[i]
        if not item.startswith("--"):
            out.setdefault("positional", []).append(item)
            i += 1
            continue
        key = item[2:].replace("-", "_")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out[key] = args[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:04.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes:02d} min"


def _stream_job(cmd: list[str], *, env: dict[str, str], width: int = 100) -> int:
    """Run one benchmark while wrapping unusually long terminal lines."""
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if not line:
            print()
            continue
        if len(line) <= width:
            print(f"    {line}", flush=True)
        else:
            print(
                textwrap.fill(
                    line,
                    width=width,
                    initial_indent="    ",
                    subsequent_indent="      ",
                    break_long_words=False,
                    break_on_hyphens=False,
                ),
                flush=True,
            )
    return int(proc.wait())


def _clean_previous_publication_outputs() -> None:
    """Start a full publication rerun from an empty generated-results tree."""
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical QUARTIC2D benchmark suite and promote a consolidated result set."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        choices=("publication", "documentation", "all"),
        default="publication",
    )
    parser.add_argument("--list", action="store_true", help="list jobs without executing them")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated job names within the selected profile",
    )
    # Backward-compatible no-op.  Dirty trees are always allowed; provenance is
    # detected and recorded in the manifest instead of blocking execution.
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="print the complete Python command for each runner",
    )
    args = parser.parse_args()

    # Use a placeholder result location for listing/filtering. Execution jobs are
    # rebuilt below with their actual staging directory.
    listed_jobs = _jobs_for_profile(args.profile, RESULTS / ".work" / "list")
    wanted: set[str] = set()
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        available = {job.name for job in listed_jobs}
        unknown = sorted(wanted - available)
        if unknown:
            raise SystemExit(f"unknown jobs for profile {args.profile!r}: {unknown}")
        listed_jobs = [job for job in listed_jobs if job.name in wanted]

    if args.list:
        _print_jobs(listed_jobs)
        return
    if not listed_jobs:
        raise SystemExit("no benchmark jobs selected")

    git = git_metadata()

    print("\n" + "=" * 100)
    print("REPOSITORY PROVENANCE")
    print("=" * 100)
    source_state = git.get("source_state")
    if source_state == "exact_commit":
        print("Source state : CLEAN COMMIT")
        print("Assessment   : exact commit provenance; rerun from the recorded commit")
    elif source_state == "working_tree_modified":
        print("Source state : DIRTY WORKING TREE")
        print("Assessment   : benchmark will run normally; provenance includes the working-tree fingerprint")
    else:
        print("Source state : GIT STATUS UNAVAILABLE")
        print("Assessment   : benchmark will run normally; Git provenance cannot be fully assessed")
    print(f"Branch       : {git.get('branch') or 'unknown'}")
    print(f"Commit       : {git.get('commit') or 'unknown'}")
    if git.get("dirty"):
        changed = git.get("changed_paths", [])
        print(f"Changed paths: {len(changed)}")
        shown = changed[:12]
        for item in shown:
            print(f"    {item.get('status', '??'):>2s}  {item.get('path', '')}")
        if len(changed) > len(shown):
            print(f"    ... {len(changed) - len(shown)} more path(s); see manifest.json for the full list")
        if git.get("diff_sha256"):
            print(f"Fingerprint  : sha256:{git['diff_sha256'][:16]}… (full value in manifest.json)")

    subset_name = "__".join(job.name for job in listed_jobs) if args.only else ""
    full_publication = args.profile == "publication" and not args.only
    if full_publication:
        final_dir = RESULTS
        staging_dir = RESULTS / ".work" / "publication"
        manifest_path = RESULTS / "manifest.json"
        _clean_previous_publication_outputs()
    elif args.profile == "documentation" and not args.only:
        final_dir = RESULTS / "documentation"
        staging_dir = final_dir / ".work"
        manifest_path = final_dir / "manifest.json"
    else:
        label = subset_name or args.profile
        final_dir = RESULTS / "subsets" / label
        staging_dir = final_dir / ".work"
        manifest_path = final_dir / "manifest.json"

    if staging_dir.exists():
        remove_staging_directory(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    jobs = _jobs_for_profile(args.profile, staging_dir)
    if wanted:
        jobs = [job for job in jobs if job.name in wanted]

    env = _single_thread_env()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "result_type": "benchmark_run_manifest",
        "profile": args.profile,
        "selection": "full" if not args.only else "subset",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "repository": git,
        "provenance_assessment": {
            "source_state": git.get("source_state"),
            "reproducibility": git.get("reproducibility"),
            "benchmark_execution_allowed": True,
        },
        "execution_policy": {
            "thread_count": 1,
            "thread_environment": {
                key: env.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
        "benchmark_groups": [],
        "consolidated_outputs": [],
        "status": "running",
    }
    write_json(manifest_path, manifest)

    current_group = None
    group_record = None
    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        group = _job_group(job)
        if group != current_group:
            current_group = group
            group_record = {"name": group, "benchmarks": []}
            manifest["benchmark_groups"].append(group_record)
            print("\n" + "=" * 100)
            print(group.upper())
            print("=" * 100)

        print(f"\n[{index:02d}/{total:02d}] {_human_name(job)}")
        print("-" * min(100, max(36, len(_human_name(job)) + 4)))
        print(textwrap.fill(job.purpose, width=100, initial_indent="Purpose: ", subsequent_indent="         "))
        if args.show_commands:
            command_text = " ".join([sys.executable, "-m", job.module, *job.args])
            print(textwrap.fill(command_text, width=100, initial_indent="Command: ", subsequent_indent="         ", break_long_words=False))
        print("Status : RUNNING", flush=True)

        cmd = [sys.executable, "-m", job.module, *job.args]
        record = {
            "name": job.name,
            "title": _human_name(job),
            "module": job.module,
            "purpose": job.purpose,
            "arguments": _structured_arguments(job.args),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_at_utc": None,
            "elapsed_seconds": None,
            "return_code": None,
            "status": "running",
        }
        assert group_record is not None
        group_record["benchmarks"].append(record)
        write_json(manifest_path, manifest)

        t0 = time.perf_counter()
        try:
            return_code = _stream_job(cmd, env=env)
            record["return_code"] = return_code
            record["status"] = "passed" if return_code == 0 else "failed"
        finally:
            elapsed = time.perf_counter() - t0
            record["elapsed_seconds"] = float(elapsed)
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, manifest)

        label = "PASS" if record["status"] == "passed" else "FAIL"
        print(f"Status : {label} ({_format_elapsed(elapsed)})", flush=True)
        if record["return_code"] != 0:
            print(f"Native result staging was preserved at: {staging_dir}")
            raise SystemExit(record["return_code"])

    if full_publication:
        outputs = consolidate_publication_results(staging_dir, final_dir, require_complete=True)
        manifest["consolidated_outputs"] = [path.name for path in outputs]
        remove_staging_directory(staging_dir)
    elif args.profile == "documentation" and not args.only:
        # The documentation profile has one intentionally lightweight result.
        source = staging_dir / "gaussian_validation.json"
        target = final_dir / "gaussian_validation.json"
        if source.exists():
            target.write_bytes(source.read_bytes())
            manifest["consolidated_outputs"] = [target.name]
        remove_staging_directory(staging_dir)
    else:
        # Subsets are diagnostic runs. Keep their native JSON together with the
        # manifest rather than overwriting the canonical full-suite evidence.
        native_dir = final_dir / "native_results"
        if native_dir.exists():
            remove_staging_directory(native_dir)
        staging_dir.rename(native_dir)
        manifest["consolidated_outputs"] = [str(path.relative_to(final_dir)) for path in sorted(native_dir.glob("*.json"))]

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "passed"
    write_json(manifest_path, manifest)

    print("\n" + "=" * 100)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 100)
    print("Status  : PASS")
    print(f"Results : {final_dir}")
    if manifest["consolidated_outputs"]:
        print("Files   :")
        for filename in manifest["consolidated_outputs"]:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
