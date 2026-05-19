from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inspectors import inspect_model
from .wrapper import load_wrapper_config


class PreflightError(RuntimeError):
    """Raised when a preflight report cannot be written."""


def run_preflight(
    *,
    model: str | None = None,
    recipe: Path | None = None,
    profile: Path | None = None,
    manifest: Path | None = None,
    artifact: Path | None = None,
    wrapper: Path | None = None,
    recovery_config: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    if model:
        artifacts["model"] = _inspect_model(model, checks)
    else:
        _check(checks, "model.provided", "warn", "no dense model was provided", "Pass --model to inspect conversion readiness.")

    if recipe is not None:
        artifacts["recipe"] = _check_recipe(recipe, checks)
    if profile is not None:
        artifacts["profile"] = _check_json_file(profile, label="profile", checks=checks)
    if manifest is not None:
        artifacts["manifest"] = _check_manifest(manifest, checks)
    if artifact is not None:
        artifacts["artifact"] = _check_artifact(artifact, checks)
    if wrapper is not None:
        artifacts["wrapper"] = _check_wrapper(wrapper, checks)
    if recovery_config is not None:
        artifacts["recovery_config"] = _check_recovery_config(recovery_config, checks)

    next_commands = _next_commands(
        model=model,
        recipe=recipe,
        profile=profile,
        manifest=manifest,
        artifact=artifact,
        wrapper=wrapper,
        recovery_config=recovery_config,
    )
    report = {
        "format": "moeforge_preflight",
        "status": "ready" if not any(check["status"] == "fail" for check in checks) else "blocked",
        "passed": not any(check["status"] == "fail" for check in checks),
        "check_count": len(checks),
        "failed_check_count": sum(1 for check in checks if check["status"] == "fail"),
        "warning_count": sum(1 for check in checks if check["status"] == "warn"),
        "checks": checks,
        "artifacts": artifacts,
        "next_commands": next_commands,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _inspect_model(model: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        info = inspect_model(model)
    except Exception as exc:
        _check(checks, "model.inspect", "fail", f"could not inspect model: {exc}", "Check the model path or HF id.")
        return {"input": model}
    payload = info.to_dict()
    _check(checks, "model.inspect", "pass", f"inspected {payload.get('source_format')} model", None)
    if payload.get("dense") is False:
        _check(checks, "model.dense", "warn", "source model already appears to be MoE", "Use a dense checkpoint for dense-to-MoE conversion.")
    else:
        _check(checks, "model.dense", "pass", "source model appears dense", None)
    if not payload.get("adapter_family"):
        _check(checks, "model.adapter", "warn", "no supported architecture adapter was detected", "Add or harden an adapter before carving.")
    else:
        _check(checks, "model.adapter", "pass", f"adapter family: {payload['adapter_family']}", None)
    return payload


def _check_recipe(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _check_json_file(path, label="recipe", checks=checks)
    if not payload:
        return payload
    experts = _int(payload.get("experts"))
    if experts <= 0:
        _check(checks, "recipe.experts", "fail", "recipe must specify a positive experts value", "Run `moe-forge plan` or edit experts.")
    else:
        _check(checks, "recipe.experts", "pass", f"experts: {experts}", None)
    if payload.get("moe_layers") in (None, [], ""):
        _check(checks, "recipe.layers", "fail", "recipe is missing moe_layers", "Set moe_layers to a list, range, or `all`.")
    else:
        _check(checks, "recipe.layers", "pass", f"moe_layers: {payload.get('moe_layers')}", None)
    shared_ratio = payload.get("shared_ratio")
    if shared_ratio is not None and not (0.0 <= _float(shared_ratio) < 1.0):
        _check(checks, "recipe.shared_ratio", "fail", "shared_ratio must be >= 0 and < 1", "Choose a valid shared_ratio.")
    return payload


def _check_manifest(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _check_json_file(path, label="manifest", checks=checks)
    if not payload:
        return payload
    layers = payload.get("layers") if isinstance(payload.get("layers"), list) else []
    _check(
        checks,
        "manifest.layers",
        "pass" if layers else "fail",
        f"manifest layers: {len(layers)}",
        None if layers else "Run `moe-forge carve-manifest` with a valid recipe.",
    )
    if _int(payload.get("experts")) <= 0:
        _check(checks, "manifest.experts", "fail", "manifest has no positive expert count", "Regenerate the manifest.")
    return payload


def _check_artifact(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    exists = path.exists()
    _check(
        checks,
        "artifact.exists",
        "pass" if exists else "fail",
        f"artifact {'exists' if exists else 'is missing'}: {path}",
        None if exists else "Run `moe-forge carve-apply`.",
    )
    if exists and path.suffix != ".safetensors":
        _check(checks, "artifact.type", "warn", "artifact is not a .safetensors file", "Use the carved safetensors artifact for wrapper export.")
    return {"path": str(path), "exists": exists, "bytes": path.stat().st_size if exists else None}


def _check_wrapper(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    config_path = path / "moeforge_config.json"
    if not config_path.exists():
        _check(checks, "wrapper.config", "fail", f"wrapper config is missing: {config_path}", "Run `moe-forge wrapper-export`.")
        return {"path": str(path), "exists": path.exists()}
    try:
        config = load_wrapper_config(config_path)
    except Exception as exc:
        _check(checks, "wrapper.config", "fail", f"could not load wrapper config: {exc}", "Regenerate the wrapper package.")
        return {"path": str(path), "exists": path.exists()}
    _check(checks, "wrapper.config", "pass", "loaded wrapper config", None)
    artifact_path = _resolve_package_path(path, config.artifact_path)
    _check(
        checks,
        "wrapper.artifact",
        "pass" if artifact_path.exists() else "fail",
        f"wrapper artifact {'exists' if artifact_path.exists() else 'is missing'}: {artifact_path}",
        None if artifact_path.exists() else "Copy or regenerate the carved artifact.",
    )
    if config.token_router_top_k is not None and not config.token_router_path:
        _check(
            checks,
            "wrapper.router_training",
            "warn",
            "learned-router top-k is configured but no learned router artifact is attached",
            "Run recovery training and `moe-forge recovery-export` to produce learned-router.safetensors.",
        )
    if config.token_router_path:
        router_path = _resolve_package_path(path, config.token_router_path)
        _check(
            checks,
            "wrapper.router_artifact",
            "pass" if router_path.exists() else "fail",
            f"learned router artifact {'exists' if router_path.exists() else 'is missing'}: {router_path}",
            None if router_path.exists() else "Re-export the recovered wrapper.",
        )
    return {
        "path": str(path),
        "layer_count": len(config.layers),
        "expert_count": config.expert_count,
        "token_router_top_k": config.token_router_top_k,
        "token_router_path": config.token_router_path,
    }


def _check_recovery_config(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _check_json_file(path, label="recovery_config", checks=checks)
    if not payload:
        return payload
    base_dir = path.parent
    wrapper = payload.get("wrapper")
    if wrapper:
        wrapper_path = _resolve_path(Path(str(wrapper)), base_dir=base_dir)
        _check(
            checks,
            "recovery.wrapper",
            "pass" if (wrapper_path / "moeforge_config.json").exists() else "fail",
            f"recovery wrapper {'exists' if (wrapper_path / 'moeforge_config.json').exists() else 'is missing'}: {wrapper_path}",
            None if (wrapper_path / "moeforge_config.json").exists() else "Create the wrapper before recovery.",
        )
    train = payload.get("train") if isinstance(payload.get("train"), dict) else {}
    has_samples = any(key in train for key in ("input_ids", "input_ids_file", "text", "texts", "text_file"))
    _check(
        checks,
        "recovery.samples",
        "pass" if has_samples else "fail",
        "recovery train samples configured" if has_samples else "recovery train samples are missing",
        None if has_samples else "Add input_ids or text_file under train.",
    )
    return payload


def _check_json_file(path: Path, *, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        _check(checks, f"{label}.exists", "fail", f"{label} file is missing: {path}", "Create or pass the correct path.")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _check(checks, f"{label}.json", "fail", f"{label} is not valid JSON: {exc}", "Fix JSON syntax.")
        return {}
    if not isinstance(payload, dict):
        _check(checks, f"{label}.json", "fail", f"{label} must contain a JSON object", "Use an object at the JSON root.")
        return {}
    _check(checks, f"{label}.json", "pass", f"loaded {label}: {path}", None)
    return payload


def _next_commands(
    *,
    model: str | None,
    recipe: Path | None,
    profile: Path | None,
    manifest: Path | None,
    artifact: Path | None,
    wrapper: Path | None,
    recovery_config: Path | None,
) -> list[str]:
    commands = []
    model_ref = model or "<model>"
    if recipe is None:
        commands.append(f"moe-forge plan {model_ref} --moe-layers all --output recipe.json")
    if recipe is not None and manifest is None:
        profile_arg = f" --profile {profile}" if profile is not None else ""
        commands.append(f"moe-forge carve-manifest {model_ref} --recipe {recipe}{profile_arg} --output carve-manifest.json")
    if manifest is not None and artifact is None:
        commands.append(f"moe-forge carve-apply --manifest {manifest} --output-dir carved-artifact")
    if manifest is not None and artifact is not None and wrapper is None:
        commands.append(
            f"moe-forge wrapper-export --manifest {manifest} --artifact {artifact} --copy-artifact --copy-source-model --output-dir wrapper"
        )
    if wrapper is not None and recovery_config is None:
        commands.append("moe-forge eval-batch --config eval-batch.json")
    if recovery_config is not None:
        commands.append(f"moe-forge recovery-experiment --config {recovery_config}")
    return commands


def _check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    message: str,
    next_action: str | None,
) -> None:
    checks.append(
        {
            "name": name,
            "status": status,
            "message": message,
            "next_action": next_action,
        }
    )


def _resolve_package_path(package_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return package_dir / path


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return base_dir / path


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
