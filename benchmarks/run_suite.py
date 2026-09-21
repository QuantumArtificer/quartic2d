#!/usr/bin/env python3
"""Canonical QUARTIC2D benchmark launcher.

The launcher separates manuscript-grade evidence from documentation and
engineering/development diagnostics. Publication commands use fixed arguments so
that the final dataset is reproducible and auditable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmarks._common import git_metadata

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


@dataclass(frozen=True)
class Job:
    name: str
    tier: str
    module: str
    args: tuple[str, ...]
    purpose: str


def _publication_jobs() -> tuple[Job, ...]:
    pub = RESULTS / "publication"
    return (
        Job(
            "harmonic-transform",
            "publication",
            "benchmarks.publication.harmonic_transform",
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
            "benchmarks.publication.harmonic_convergence",
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
            "benchmarks.publication.harmonic_transform",
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
            "benchmarks.publication.interaction_accuracy",
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
            "benchmarks.publication.interaction_accuracy",
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
            "benchmarks.publication.interaction_convergence",
            (
                "--output", str(pub / "interaction_convergence.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--methods", "simpson,gl4,fftlog,ogata",
                "--tolerances", "1e-4",
                "--cases", "default",
                "--delta-domains", "standard,large",
                "--n-delta", "32",
                "--reference-order", "24",
            ),
            "Automatic-certification reliability, large-delta oracle stability, and method coverage.",
        ),
        Job(
            "cross-stage-standard",
            "publication",
            "benchmarks.publication.cross_stage",
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
            "benchmarks.publication.cross_stage",
            (
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--harmonic-rtol", "1e-4",
                "--q-tail-rtol", "1e-3",
                "--interaction-rtol", "1e-4",
                "--methods", "gl4",
                "--groups", "kernels",
                "--cases",
                "kernels__gaussian_isotropic__coulomb,kernels__gaussian_anisotropic_m2__rk_r0_10,kernels__nodal_mixed__rpa_2deg_kf1_qtf1,kernels__complex_mixed__dual_gate_d1",
                "--delta-domain", "large",
                "--n-delta", "32",
                "--reference-order", "24",
                "--output", str(pub / "cross_stage_large.json"),
            ),
            "Representative large-displacement cross-stage validation with fixed analytic radial inputs.",
        ),
        Job(
            "end-to-end-standard",
            "publication",
            "benchmarks.publication.end_to_end",
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
            "benchmarks.publication.end_to_end",
            (
                "--cases", "gaussian_coulomb,anisotropic_rk_strong,nodal_rpa",
                "--delta-domain", "large",
                "--methods", "gl4",
                "--n-delta", "32",
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
            "benchmarks.publication.autoconvergence_performance",
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
            "benchmarks.publication.harmonic_scaling",
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
            "benchmarks.publication.interaction_scaling",
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
            "benchmarks.publication.harmonic_memory",
            ("--output", str(pub / "harmonic_memory.json"), "--repeats", "3"),
            "Peak-memory scaling for HarmonicTransform.",
        ),
        Job(
            "interaction-memory",
            "publication",
            "benchmarks.publication.interaction_memory",
            ("--output", str(pub / "interaction_memory.json"), "--repeats", "3"),
            "Peak-memory scaling for Interaction.",
        ),
    )


def _documentation_jobs() -> tuple[Job, ...]:
    docs = RESULTS / "documentation"
    return (
        Job(
            "gaussian-workflow",
            "documentation",
            "benchmarks.documentation.gaussian_validation",
            ("--quick", "--output", str(docs / "gaussian.json")),
            "Cheap analytic workflow check used by documentation/examples; not paper evidence.",
        ),
    )


def _development_jobs() -> tuple[Job, ...]:
    dev = RESULTS / "development"
    pub = RESULTS / "publication"
    return (
        Job(
            "legacy-backend-matrix",
            "development",
            "benchmarks.development.backend_matrix",
            ("--quick", "--output-dir", str(dev / "legacy_backend_matrix")),
            "Historical engineering matrix retained for regression/context; not current manuscript evidence.",
        ),
        Job(
            "q-sampling-diagnostics",
            "development",
            "benchmarks.development.q_sampling",
            ("--output", str(dev / "q_sampling.json")),
            "Sampling-theory diagnostic retained for engineering regression and investigation.",
        ),
        Job(
            "autoconvergence-screen",
            "development",
            "benchmarks.development.autoconvergence_screen",
            (
                "--output", str(dev / "autoconvergence_screen.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
            ),
            "One-shot method screen; never publication timing evidence.",
        ),
        Job(
            "autoconvergence-profile",
            "development",
            "benchmarks.development.autoconvergence_profile",
            (
                "--output", str(dev / "autoconvergence_profile.json"),
                "--harmonic-results", str(pub / "harmonic_transform.json"),
                "--profile-dir", str(dev / "profiles" / "autoconvergence"),
            ),
            "cProfile hotspot diagnostics; never publication timing evidence.",
        ),
    )


JOBS = _publication_jobs() + _documentation_jobs() + _development_jobs()


def _jobs_for_profile(profile: str) -> list[Job]:
    if profile == "all":
        return list(JOBS)
    return [job for job in JOBS if job.tier == profile]


def _single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _print_jobs(jobs: Iterable[Job]) -> None:
    for job in jobs:
        print(f"{job.tier:13s} {job.name:30s} {job.purpose}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile",
        nargs="?",
        choices=("publication", "documentation", "development", "all"),
        default="publication",
    )
    parser.add_argument("--list", action="store_true", help="list jobs without executing them")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated job names within the selected profile",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow publication benchmarks from a dirty Git tree (not recommended for manuscript evidence)",
    )
    args = parser.parse_args()

    jobs = _jobs_for_profile(args.profile)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        available = {job.name for job in jobs}
        unknown = sorted(wanted - available)
        if unknown:
            raise SystemExit(f"unknown jobs for profile {args.profile!r}: {unknown}")
        jobs = [job for job in jobs if job.name in wanted]

    if args.list:
        _print_jobs(jobs)
        return

    if not jobs:
        raise SystemExit("no benchmark jobs selected")

    git = git_metadata()
    if args.profile == "publication" and not args.allow_dirty and git.get("dirty"):
        raise SystemExit(
            "publication benchmarks require a clean Git working tree. "
            "Commit/stash the intended source state first, or pass --allow-dirty explicitly."
        )

    env = _single_thread_env()
    manifest_path = RESULTS / args.profile / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "profile": args.profile,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "git": git,
        "thread_environment": {
            key: env.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "jobs": [],
        "complete": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for job in jobs:
        print(f"\n=== {job.tier}: {job.name} ===", flush=True)
        cmd = [sys.executable, "-m", job.module, *job.args]
        print("$ " + " ".join(cmd), flush=True)
        record = {
            "name": job.name,
            "tier": job.tier,
            "module": job.module,
            "args": list(job.args),
            "purpose": job.purpose,
            "command": cmd,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_at_utc": None,
            "returncode": None,
            "status": "running",
        }
        manifest["jobs"].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
            record["returncode"] = int(completed.returncode)
            record["status"] = "passed" if completed.returncode == 0 else "failed"
        finally:
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if record["returncode"] != 0:
            raise SystemExit(record["returncode"])

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
