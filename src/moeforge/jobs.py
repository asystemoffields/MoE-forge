from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import subprocess
from typing import Any, Callable


class JobLaunchError(RuntimeError):
    """Raised when a background job cannot be launched."""


@dataclass(slots=True)
class JobLaunchOptions:
    name: str
    command: list[str]
    output_dir: Path = Path("outputs/jobs")
    dry_run: bool = False


@dataclass(slots=True)
class ModalCollectOptions:
    job_manifest: Path
    output_dir: Path | None = None
    volume: str = "moeforge-benchmarks"
    remote_path: str | None = None
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


def collect_modal_artifact(
    options: ModalCollectOptions,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    job = _load_json_object(options.job_manifest, label="job manifest")
    job_dir = Path(str(job.get("output_dir") or options.job_manifest.parent))
    output_dir = options.output_dir or job_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spawn_manifest = _spawn_manifest_from_job(job=job, job_manifest=options.job_manifest)
    remote_path = _normalize_modal_path(options.remote_path or str(spawn_manifest.get("expected_report") or ""))
    if not remote_path:
        raise JobLaunchError("could not find Modal expected_report; pass --remote-path explicitly")
    local_path = output_dir / PurePosixPath(remote_path).name
    command = ["modal", "volume", "get", "--force", options.volume, remote_path, str(local_path)]
    report_path = output_dir / "modal-collect.json"
    report: dict[str, Any] = {
        "format": "moeforge_modal_collect",
        "status": "planned" if options.dry_run else "pending",
        "job_manifest": str(options.job_manifest),
        "spawn_manifest": spawn_manifest,
        "volume": options.volume,
        "remote_path": remote_path,
        "local_path": str(local_path),
        "command": command,
        "dry_run": options.dry_run,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "report_path": str(report_path),
    }
    if not options.dry_run:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        completed = runner(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )
        report["returncode"] = completed.returncode
        report["stdout"] = completed.stdout
        report["stderr"] = completed.stderr
        report["status"] = "collected" if completed.returncode == 0 and local_path.exists() else "unavailable"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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


def _spawn_manifest_from_job(*, job: dict[str, Any], job_manifest: Path) -> dict[str, Any]:
    if job.get("expected_report"):
        return job
    stdout_path = _recorded_path(job.get("stdout"), base_dir=job_manifest.parent)
    if stdout_path is None or not stdout_path.exists():
        return {}
    objects = _json_objects_from_text(stdout_path.read_text(encoding="utf-8", errors="replace"))
    for payload in reversed(objects):
        if payload.get("expected_report") and str(payload.get("format", "")).startswith("moeforge_modal_"):
            return payload
    return {}


def _json_objects_from_text(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _normalize_modal_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    if path.startswith("//"):
        path = "/" + path.lstrip("/")
    return path


def _recorded_path(value: Any, *, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    base_candidate = base_dir / path.name
    if base_candidate.exists():
        return base_candidate
    return path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JobLaunchError(f"could not read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JobLaunchError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise JobLaunchError(f"{label} must be a JSON object: {path}")
    return payload
