from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.cli import main
from moeforge.jobs import JobLaunchError, JobLaunchOptions, launch_background_job


def test_launch_background_job_dry_run_records_command(tmp_path: Path) -> None:
    manifest = launch_background_job(
        JobLaunchOptions(
            name="modal smoke",
            command=["--", "modal", "run", "example.py", "--run-name", "smoke", "--spawn"],
            output_dir=tmp_path,
            dry_run=True,
        )
    )

    saved = json.loads((tmp_path / "modal-smoke" / "job.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert manifest["pid"] is None
    assert saved["env"]["PYTHONIOENCODING"] == "utf-8"
    assert saved["command"] == ["modal", "run", "example.py", "--run-name", "smoke", "--spawn"]
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
            "benchmark.py",
            "--spawn",
        ]
    )

    payload = json.loads((tmp_path / "bench" / "job.json").read_text(encoding="utf-8"))
    assert status == 0
    assert payload["dry_run"] is True
    assert payload["command"][:3] == ["modal", "run", "benchmark.py"]
    assert payload["command"][-1] == "--spawn"


def test_launch_background_job_requires_command(tmp_path: Path) -> None:
    with pytest.raises(JobLaunchError, match="command"):
        launch_background_job(JobLaunchOptions(name="empty", command=[], output_dir=tmp_path))
