from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any


class JobLaunchError(RuntimeError):
    """Raised when a background job cannot be launched."""


@dataclass(slots=True)
class JobLaunchOptions:
    name: str
    command: list[str]
    output_dir: Path = Path("outputs/jobs")
    dry_run: bool = False


def launch_background_job(options: JobLaunchOptions) -> dict[str, Any]:
    if not options.name.strip():
        raise JobLaunchError("job name is required")
    command = _normalized_command(options.command)
    if not command:
        raise JobLaunchError("job command is required")

    job_dir = options.output_dir / _safe_name(options.name)
    job_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    command_path = job_dir / ("command.ps1" if os.name == "nt" else "command.sh")
    manifest_path = job_dir / "job.json"
    command_text = _command_text(command)
    command_path.write_text(command_text, encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    manifest: dict[str, Any] = {
        "format": "moeforge_background_job",
        "name": options.name,
        "command": command,
        "command_text": command_text,
        "command_path": str(command_path),
        "output_dir": str(job_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "dry_run": options.dry_run,
        "pid": None,
        "env": {"PYTHONIOENCODING": env["PYTHONIOENCODING"]},
        "status": "planned" if options.dry_run else "launched",
        "notes": [
            "Use this manifest to reconnect command output with remote Modal artifacts.",
            "For Modal jobs, prefer modal run --detach with local entrypoints that call Modal .spawn() and print the remote artifact path.",
        ],
    }
    if not options.dry_run:
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                cwd=Path.cwd(),
                env=env,
                close_fds=True,
                start_new_session=os.name != "nt",
                creationflags=_windows_creation_flags(),
            )
        except OSError as exc:
            stdout.close()
            stderr.close()
            raise JobLaunchError(f"could not launch job: {exc}") from exc
        manifest["pid"] = process.pid
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _normalized_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def _command_text(command: list[str]) -> str:
    if os.name == "nt":
        return " ".join(_quote_powershell(arg) for arg in command) + "\n"
    return shlex.join(command) + "\n"


def _quote_powershell(value: str) -> str:
    if value == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:/\\=")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "''") + "'"


def _windows_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "-" for char in value.strip())
    return safe.strip(".-") or "job"
