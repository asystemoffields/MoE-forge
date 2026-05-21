from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
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


@dataclass(slots=True)
class RunStatusOptions:
    job_manifest: Path | None = None
    name: str | None = None
    jobs_dir: Path = Path("outputs/modal-jobs")
    volume: str = "moeforge-benchmarks"
    query_remote: bool = True


@dataclass(slots=True)
class RunsListOptions:
    jobs_dir: Path = Path("outputs/modal-jobs")
    volume: str = "moeforge-benchmarks"
    query_remote: bool = False


# Modal mounts the benchmarks volume at this path inside the container, but the `modal volume`
# CLI addresses files relative to the volume root, so the mount prefix must be stripped.
MODAL_VOLUME_MOUNT = "/vol"


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
    volume_path = _volume_relative_path(remote_path)
    local_path = output_dir / PurePosixPath(remote_path).name
    command = ["modal", "volume", "get", "--force", options.volume, volume_path, str(local_path)]
    report_path = output_dir / "modal-collect.json"
    report: dict[str, Any] = {
        "format": "moeforge_modal_collect",
        "status": "planned" if options.dry_run else "pending",
        "job_manifest": str(options.job_manifest),
        "spawn_manifest": spawn_manifest,
        "volume": options.volume,
        "remote_path": remote_path,
        "volume_path": volume_path,
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


def run_status(
    options: RunStatusOptions,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Reconcile a background job's local + Modal state into one decision-ready status report."""
    job_manifest = _resolve_status_manifest(options)
    job = _load_json_object(job_manifest, label="job manifest")
    job_dir = Path(str(job.get("output_dir") or job_manifest.parent))
    spawn_manifest = _spawn_manifest_from_job(job=job, job_manifest=job_manifest)

    expected_report = _normalize_modal_path(str(spawn_manifest.get("expected_report") or ""))
    run_dir_remote = _normalize_modal_path(str(spawn_manifest.get("run_dir") or ""))
    run_name = spawn_manifest.get("run_name") or job.get("name")
    dashboard_url = spawn_manifest.get("dashboard_url")
    artifact_name = PurePosixPath(expected_report).name if expected_report else "modal-recovery-manifest.json"

    local_artifact = job_dir / artifact_name
    local_present = local_artifact.exists()

    volume_dir = ""
    if run_dir_remote:
        volume_dir = _volume_relative_path(run_dir_remote)
    elif expected_report:
        volume_dir = str(PurePosixPath(_volume_relative_path(expected_report)).parent)

    remote_present: bool | None = None
    if options.query_remote and volume_dir and not local_present:
        listing = _modal_text_run(runner, ["modal", "volume", "ls", options.volume, volume_dir])
        if listing is not None and listing.returncode == 0:
            remote_present = artifact_name in (listing.stdout or "")
        elif listing is not None:
            remote_present = False

    app_id = _app_id_from_job(job=job, job_manifest=job_manifest)
    app_state: str | None = None
    if options.query_remote and app_id and not local_present and not remote_present:
        apps = _modal_text_run(runner, ["modal", "app", "list"])
        if apps is not None and apps.returncode == 0:
            app_state = _app_state_from_listing(apps.stdout or "", app_id=app_id)

    state, running, done = _derive_run_state(
        local_present=local_present, remote_present=remote_present, app_state=app_state
    )
    report = {
        "format": "moeforge_run_status",
        "run_name": run_name,
        "state": state,
        "running": running,
        "done": done,
        "job_manifest": str(job_manifest),
        "local_artifact": {"path": str(local_artifact), "present": local_present},
        "remote_artifact": {
            "volume": options.volume,
            "dir": volume_dir,
            "name": artifact_name,
            "present": remote_present,
        },
        "app": {"id": app_id, "state": app_state},
        "dashboard_url": dashboard_url,
        "logs": {"stdout": job.get("stdout"), "stderr": job.get("stderr")},
        "next_commands": _status_next_commands(
            state=state,
            job_manifest=job_manifest,
            local_artifact=local_artifact,
            dashboard_url=dashboard_url,
            stderr=job.get("stderr"),
        ),
    }
    return report


def list_runs(
    options: RunsListOptions,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Index every background job under jobs_dir into one ledger row per run (state + headline metric)."""
    from .summary import RunSummaryError, summarize_run

    rows: list[dict[str, Any]] = []
    for job_json in sorted(options.jobs_dir.glob("*/job.json")):
        try:
            status = run_status(
                RunStatusOptions(
                    job_manifest=job_json,
                    volume=options.volume,
                    query_remote=options.query_remote,
                ),
                runner=runner,
            )
        except JobLaunchError:
            continue
        row: dict[str, Any] = {
            "run_name": status.get("run_name"),
            "state": status.get("state"),
            "job_manifest": status.get("job_manifest"),
        }
        local_artifact = Path(str(_dict(status.get("local_artifact")).get("path") or ""))
        if local_artifact.exists():
            try:
                summary = summarize_run(report_path=local_artifact)
            except RunSummaryError:
                summary = None
            if summary is not None:
                row["routing_gap"] = _dict(summary.get("verdicts")).get("routing_gap")
                row["learned_router_kl"] = _dict(_dict(summary.get("metrics")).get("learned_router")).get("kl_after")
                row["headline"] = summary.get("headline")
        rows.append(row)
    return {
        "format": "moeforge_runs_index",
        "jobs_dir": str(options.jobs_dir),
        "run_count": len(rows),
        "runs": rows,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_status_manifest(options: RunStatusOptions) -> Path:
    if options.job_manifest is not None:
        manifest = options.job_manifest
        if manifest.is_dir():
            manifest = manifest / "job.json"
        return manifest
    if options.name:
        return options.jobs_dir / _safe_name(options.name) / "job.json"
    raise JobLaunchError("run status requires --job or --name")


def _modal_text_run(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    command: list[str],
) -> subprocess.CompletedProcess[str] | None:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        return runner(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
        )
    except OSError:
        return None


def _app_id_from_job(*, job: dict[str, Any], job_manifest: Path) -> str | None:
    stdout_path = _recorded_path(job.get("stdout"), base_dir=job_manifest.parent)
    if stdout_path is None or not stdout_path.exists():
        return None
    matches = re.findall(r"ap-[A-Za-z0-9]+", stdout_path.read_text(encoding="utf-8", errors="replace"))
    return matches[-1] if matches else None


def _app_state_from_listing(text: str, *, app_id: str) -> str | None:
    for line in text.splitlines():
        if app_id not in line:
            continue
        lowered = line.lower()
        if "stopped" in lowered:
            return "stopped"
        if any(token in lowered for token in ("running", "ephemeral", "deployed", "deploying")):
            return "running"
        return "unknown"
    return None


def _derive_run_state(
    *,
    local_present: bool,
    remote_present: bool | None,
    app_state: str | None,
) -> tuple[str, bool | None, bool]:
    if local_present:
        return "collected", False, True
    if remote_present:
        return "completed", False, True
    if remote_present is False:
        if app_state == "running":
            return "running", True, False
        if app_state == "stopped":
            return "failed", False, False
        return "pending", None, False
    return "unknown", None, False


def _status_next_commands(
    *,
    state: str,
    job_manifest: Path,
    local_artifact: Path,
    dashboard_url: Any,
    stderr: Any,
) -> list[str]:
    if state == "collected":
        return [f"moe-forge summarize {local_artifact}"]
    if state == "completed":
        return [
            f"moe-forge job-collect --job {job_manifest}",
            f"moe-forge summarize {local_artifact}",
        ]
    if state == "running":
        commands = [f"moe-forge status --job {job_manifest}  # re-check later"]
        if dashboard_url:
            commands.append(f"open {dashboard_url}")
        return commands
    if state == "failed":
        commands = []
        if stderr:
            commands.append(f"inspect logs: {stderr}")
        if dashboard_url:
            commands.append(f"open {dashboard_url}")
        commands.append(f"moe-forge status --job {job_manifest}  # confirm after checking logs")
        return commands
    return [f"moe-forge status --job {job_manifest}  # state not yet determinable"]


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


def _volume_relative_path(value: str, *, mount: str = MODAL_VOLUME_MOUNT) -> str:
    """Strip the in-container mount prefix so `modal volume` sees a volume-root-relative path."""
    path = _normalize_modal_path(value)
    mount = mount.rstrip("/")
    if path == mount:
        return ""
    if path.startswith(mount + "/"):
        return path[len(mount) + 1 :]
    return path.lstrip("/")


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
