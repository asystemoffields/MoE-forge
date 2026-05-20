from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from moeforge.cli import main
from moeforge.jobs import (
    JobLaunchError,
    JobLaunchOptions,
    ModalCollectOptions,
    RunStatusOptions,
    collect_modal_artifact,
    launch_background_job,
    run_status,
)


def _write_status_job(
    tmp_path: Path,
    *,
    run_name: str = "demo",
    app_id: str = "ap-TEST123",
) -> Path:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    stdout = job_dir / "stdout.log"
    stdout.write_text(
        f"View run at https://modal.com/apps/ws/main/{app_id}\n"
        + json.dumps(
            {
                "format": "moeforge_modal_recovery_spawn",
                "run_name": run_name,
                "run_dir": f"/vol/recovery-runs/{run_name}",
                "dashboard_url": "https://modal.com/id/fc-XYZ",
                "expected_report": f"/vol/recovery-runs/{run_name}/modal-recovery-manifest.json",
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
                "name": run_name,
                "output_dir": str(job_dir),
                "stdout": str(stdout),
                "stderr": str(job_dir / "stderr.log"),
            }
        ),
        encoding="utf-8",
    )
    return job_manifest


def test_run_status_collected_when_local_artifact_present(tmp_path: Path) -> None:
    job_manifest = _write_status_job(tmp_path)
    (job_manifest.parent / "modal-recovery-manifest.json").write_text("{}", encoding="utf-8")

    report = run_status(RunStatusOptions(job_manifest=job_manifest, query_remote=False))

    assert report["state"] == "collected"
    assert report["done"] is True
    assert report["next_commands"][0].startswith("moe-forge summarize")


def test_run_status_completed_when_remote_artifact_present(tmp_path: Path) -> None:
    job_manifest = _write_status_job(tmp_path)

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["modal", "volume", "ls"]
        assert command[-1] == "recovery-runs/demo"  # /vol prefix stripped
        return subprocess.CompletedProcess(command, 0, stdout="modal-recovery-manifest.json\n", stderr="")

    report = run_status(RunStatusOptions(job_manifest=job_manifest), runner=fake_runner)

    assert report["state"] == "completed"
    assert report["remote_artifact"]["present"] is True
    assert any("job-collect" in cmd for cmd in report["next_commands"])


def test_run_status_running_when_app_active_and_no_artifact(tmp_path: Path) -> None:
    job_manifest = _write_status_job(tmp_path, app_id="ap-RUN9")

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["modal", "volume", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="train.txt\nbefore\n", stderr="")
        if command[:3] == ["modal", "app", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout="| ap-RUN9 | desc | ephemeral | 1 |\n", stderr="")
        raise AssertionError(command)

    report = run_status(RunStatusOptions(job_manifest=job_manifest), runner=fake_runner)

    assert report["state"] == "running"
    assert report["running"] is True


def test_run_status_failed_when_app_stopped_without_artifact(tmp_path: Path) -> None:
    job_manifest = _write_status_job(tmp_path, app_id="ap-DEAD1")

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["modal", "volume", "ls"]:
            return subprocess.CompletedProcess(command, 0, stdout="(no manifest here)\n", stderr="")
        if command[:3] == ["modal", "app", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout="| ap-DEAD1 | desc | stopped | 0 |\n", stderr="")
        raise AssertionError(command)

    report = run_status(RunStatusOptions(job_manifest=job_manifest), runner=fake_runner)

    assert report["state"] == "failed"
    status = main(["status", "--job", str(job_manifest), "--no-remote"])
    # CLI returns 0 for an offline (pending/unknown) check.
    assert status == 0


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
    runner_kwargs: list[dict[str, object]] = []

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        runner_kwargs.append(kwargs)
        Path(command[-1]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="copied", stderr="")

    report = collect_modal_artifact(
        ModalCollectOptions(job_manifest=job_manifest, volume="bench-volume"),
        runner=fake_runner,
    )

    assert report["status"] == "collected"
    # The /vol mount prefix must be stripped so `modal volume get` resolves the path.
    assert report["volume_path"] == "runs/demo/modal-benchmark-manifest.json"
    assert calls == [
        [
            "modal",
            "volume",
            "get",
            "--force",
            "bench-volume",
            "runs/demo/modal-benchmark-manifest.json",
            str(job_dir / "modal-benchmark-manifest.json"),
        ]
    ]
    assert runner_kwargs[0]["encoding"] == "utf-8"
    assert runner_kwargs[0]["errors"] == "replace"


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
