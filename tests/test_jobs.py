from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from moeforge.cli import main
from moeforge.jobs import JobLaunchError, JobLaunchOptions, ModalCollectOptions, collect_modal_artifact, launch_background_job


def test_launch_background_job_dry_run_records_command(tmp_path: Path) -> None:
    manifest = launch_background_job(
        JobLaunchOptions(
            name="modal smoke",
            command=["--", "modal", "run", "--detach", "example.py", "--run-name", "smoke", "--spawn"],
            output_dir=tmp_path,
            dry_run=True,
        )
    )

    saved = json.loads((tmp_path / "modal-smoke" / "job.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert manifest["pid"] is None
    assert saved["env"]["PYTHONIOENCODING"] == "utf-8"
    assert saved["command"] == ["modal", "run", "--detach", "example.py", "--run-name", "smoke", "--spawn"]
    assert Path(saved["command_path"]).exists()


def test_job_launch_cli_dry_run(tmp_path: Path) -> None:
    status = main(
        [
            "job-launch",
            "--name",
            "bench",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--",
            "modal",
            "run",
            "--detach",
            "benchmark.py",
            "--spawn",
        ]
    )

    payload = json.loads((tmp_path / "bench" / "job.json").read_text(encoding="utf-8"))
    assert status == 0
    assert payload["dry_run"] is True
    assert payload["command"][:4] == ["modal", "run", "--detach", "benchmark.py"]
    assert payload["command"][-1] == "--spawn"


def test_launch_background_job_requires_command(tmp_path: Path) -> None:
    with pytest.raises(JobLaunchError, match="command"):
        launch_background_job(JobLaunchOptions(name="empty", command=[], output_dir=tmp_path))


def test_collect_modal_artifact_dry_run_parses_spawn_manifest(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    stdout = job_dir / "stdout.log"
    stdout.write_text(
        "logs first\n"
        + json.dumps(
            {
                "format": "moeforge_modal_recovery_spawn",
                "run_name": "smoke",
                "expected_report": "\\vol\\recovery-runs\\smoke\\modal-recovery-manifest.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    job_manifest = job_dir / "job.json"
    job_manifest.write_text(
        json.dumps(
            {
                "format": "moeforge_background_job",
                "output_dir": str(job_dir),
                "stdout": str(stdout),
            }
        ),
        encoding="utf-8",
    )

    report = collect_modal_artifact(ModalCollectOptions(job_manifest=job_manifest, dry_run=True))

    assert report["status"] == "planned"
    assert report["remote_path"] == "/vol/recovery-runs/smoke/modal-recovery-manifest.json"
    assert report["local_path"].endswith("modal-recovery-manifest.json")
    assert (job_dir / "modal-collect.json").exists()


def test_collect_modal_artifact_runs_modal_volume_get(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job_manifest = job_dir / "job.json"
    job_manifest.write_text(
        json.dumps(
            {
                "format": "moeforge_modal_recovery_spawn",
                "output_dir": str(job_dir),
                "expected_report": "/vol/runs/demo/modal-benchmark-manifest.json",
            }
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        Path(command[-1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="copied", stderr="")

    report = collect_modal_artifact(
        ModalCollectOptions(job_manifest=job_manifest, volume="bench-volume"),
        runner=fake_runner,
    )

    assert report["status"] == "collected"
    assert calls == [
        [
            "modal",
            "volume",
            "get",
            "--force",
            "bench-volume",
            "/vol/runs/demo/modal-benchmark-manifest.json",
            str(job_dir / "modal-benchmark-manifest.json"),
        ]
    ]


def test_job_collect_cli_dry_run(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    job_manifest = job_dir / "job.json"
    job_manifest.write_text(
        json.dumps(
            {
                "format": "moeforge_modal_recovery_spawn",
                "output_dir": str(job_dir),
                "expected_report": "/vol/recovery-runs/demo/modal-recovery-manifest.json",
            }
        ),
        encoding="utf-8",
    )

    status = main(["job-collect", "--job", str(job_manifest), "--dry-run"])

    report = json.loads((job_dir / "modal-collect.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "planned"
