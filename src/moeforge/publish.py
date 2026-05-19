from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .wrapper import load_wrapper_config


class PublishReadinessError(RuntimeError):
    """Raised when publish-readiness cannot be evaluated."""


def check_publish_readiness(
    *,
    wrapper: Path,
    output_path: Path | None = None,
    eval_reports: list[Path] | None = None,
    recovery_report: Path | None = None,
    validation_report: Path | None = None,
    require_recovery: bool = False,
    require_sparse_eval: bool = True,
    max_all_expert_error: float = 1e-4,
    max_all_expert_teacher_kl: float = 0.01,
    max_sparse_teacher_kl: float | None = None,
    max_sparse_nll_delta: float | None = None,
    trust_remote_code_load: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"wrapper": str(wrapper)}
    wrapper_config = _load_wrapper(wrapper, checks)

    _file_check(checks, wrapper / "config.json", "package.config_json", "Transformers config")
    _file_check(checks, wrapper / "moeforge_config.json", "package.moeforge_config", "MoE Forge config")
    _file_check(checks, wrapper / "MODEL_CARD.md", "package.model_card", "model card")
    _file_check(checks, wrapper / "configuration_moeforge.py", "package.auto_config_stub", "AutoConfig stub")
    _file_check(checks, wrapper / "modeling_moeforge.py", "package.auto_model_stub", "AutoModel stub")

    if wrapper_config is not None:
        artifact_path = _resolve_package_path(wrapper, wrapper_config.artifact_path)
        artifacts["artifact"] = str(artifact_path)
        _file_check(checks, artifact_path, "package.artifact", "carved or recovered safetensors artifact")
        if not wrapper_config.layers:
            _check(checks, "package.layers", "fail", "wrapper has no converted layers", "Export a wrapper with at least one converted FFN layer.")
        else:
            _check(checks, "package.layers", "pass", f"converted layers: {len(wrapper_config.layers)}", None)
        if wrapper_config.token_router_top_k is not None:
            if wrapper_config.token_router_path:
                router_path = _resolve_package_path(wrapper, wrapper_config.token_router_path)
                artifacts["learned_router"] = str(router_path)
                _file_check(checks, router_path, "package.learned_router", "learned router safetensors")
            else:
                _check(
                    checks,
                    "package.learned_router",
                    "fail",
                    "token-router top-k is configured but no learned router artifact is packaged",
                    "Run recovery training and recovery-export before publishing sparse routing.",
                )
    if trust_remote_code_load:
        artifacts["native_load"] = _native_load(wrapper, checks)

    eval_payloads = _load_reports(eval_reports or [], checks=checks, label="eval_report")
    if eval_payloads:
        artifacts["eval_reports"] = [str(path) for path in eval_reports or []]
        _eval_checks(
            checks,
            eval_payloads,
            require_sparse_eval=require_sparse_eval,
            max_all_expert_error=max_all_expert_error,
            max_all_expert_teacher_kl=max_all_expert_teacher_kl,
            max_sparse_teacher_kl=max_sparse_teacher_kl,
            max_sparse_nll_delta=max_sparse_nll_delta,
        )
    else:
        _check(
            checks,
            "eval.present",
            "fail",
            "no eval reports were provided",
            "Run eval-batch or convert --eval-smoke and include the JSON reports.",
        )

    recovery_payload = _load_optional_report(recovery_report, checks=checks, label="recovery_report")
    validation_payload = _load_optional_report(validation_report, checks=checks, label="validation_report")
    if recovery_payload is not None:
        artifacts["recovery_report"] = str(recovery_report)
        _recovery_checks(checks, recovery_payload)
    elif require_recovery:
        _check(
            checks,
            "recovery.present",
            "fail",
            "no recovery report was provided",
            "Run recovery-experiment and publish the recovered wrapper.",
        )
    else:
        _check(
            checks,
            "recovery.present",
            "warn",
            "no recovery report was provided",
            "Add recovery evidence before treating this as a quality candidate.",
        )
    if validation_payload is not None:
        artifacts["validation_report"] = str(validation_report)
        _validation_checks(checks, validation_payload)
    elif require_recovery:
        _check(
            checks,
            "validation.present",
            "fail",
            "no recovered-wrapper validation report was provided",
            "Run recovery-validate or recovery-experiment.",
        )

    failed = [check for check in checks if check["status"] == "fail"]
    report = {
        "format": "moeforge_publish_readiness",
        "status": "ready" if not failed else "blocked",
        "passed": not failed,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "warning_count": sum(1 for check in checks if check["status"] == "warn"),
        "checks": checks,
        "artifacts": artifacts,
        "thresholds": {
            "max_all_expert_error": max_all_expert_error,
            "max_all_expert_teacher_kl": max_all_expert_teacher_kl,
            "max_sparse_teacher_kl": max_sparse_teacher_kl,
            "max_sparse_nll_delta": max_sparse_nll_delta,
            "require_recovery": require_recovery,
            "require_sparse_eval": require_sparse_eval,
        },
        "next_actions": _next_actions(checks, wrapper=wrapper),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _eval_checks(
    checks: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    *,
    require_sparse_eval: bool,
    max_all_expert_error: float,
    max_all_expert_teacher_kl: float,
    max_sparse_teacher_kl: float | None,
    max_sparse_nll_delta: float | None,
) -> None:
    by_mode = {str(report.get("expert_mode") or _first_sample_mode(report)): report for report in reports}
    all_report = by_mode.get("all")
    if all_report is None:
        _check(checks, "eval.all_expert", "fail", "missing all-expert eval report", "Run eval-hf or eval-batch with expert_mode=all.")
    else:
        max_error = _float(all_report.get("max_abs_error"))
        teacher_kl = _float(_dict(all_report.get("summary")).get("average_teacher_kl_loss"))
        if max_error is not None and max_error <= max_all_expert_error:
            _check(checks, "eval.all_expert_error", "pass", f"all-expert max_abs_error={max_error:.6g}", None)
        elif teacher_kl is not None and teacher_kl <= max_all_expert_teacher_kl:
            _check(
                checks,
                "eval.all_expert_error",
                "pass",
                f"all-expert teacher_kl={teacher_kl:.6g}; max_abs_error={max_error}",
                None,
            )
        else:
            _check(
                checks,
                "eval.all_expert_error",
                "fail",
                f"all-expert max_abs_error={max_error}; teacher_kl={teacher_kl}",
                f"Keep all-expert max_abs_error <= {max_all_expert_error} or teacher_kl <= {max_all_expert_teacher_kl}.",
            )
    sparse_modes = [mode for mode in ("learned-router", "router", "default-pool") if mode in by_mode]
    if require_sparse_eval and not sparse_modes:
        _check(
            checks,
            "eval.sparse_modes",
            "fail",
            "missing sparse routing eval report",
            "Run eval-batch with learned-router or router mode.",
        )
        return
    if sparse_modes:
        _check(checks, "eval.sparse_modes", "pass", f"sparse modes: {', '.join(sparse_modes)}", None)
    for mode in sparse_modes:
        summary = _dict(by_mode[mode].get("summary"))
        teacher_kl = _float(summary.get("average_teacher_kl_loss"))
        nll_delta = _float(summary.get("average_nll_loss_delta"))
        if teacher_kl is None:
            _check(checks, f"eval.{mode}.teacher_kl", "warn", "teacher-KL was not recorded", "Enable recovery_eval metrics in eval-batch.")
        elif max_sparse_teacher_kl is None or teacher_kl <= max_sparse_teacher_kl:
            _check(checks, f"eval.{mode}.teacher_kl", "pass", f"teacher_kl={teacher_kl:.6g}", None)
        else:
            _check(
                checks,
                f"eval.{mode}.teacher_kl",
                "fail",
                f"teacher_kl={teacher_kl:.6g}",
                f"Improve recovery or routing until teacher_kl <= {max_sparse_teacher_kl}.",
            )
        if nll_delta is None:
            _check(checks, f"eval.{mode}.nll_delta", "warn", "NLL delta was not recorded", "Use tokenized samples with next-token labels.")
        elif max_sparse_nll_delta is None or nll_delta <= max_sparse_nll_delta:
            _check(checks, f"eval.{mode}.nll_delta", "pass", f"nll_delta={nll_delta:.6g}", None)
        else:
            _check(
                checks,
                f"eval.{mode}.nll_delta",
                "fail",
                f"nll_delta={nll_delta:.6g}",
                f"Improve recovery or routing until NLL delta <= {max_sparse_nll_delta}.",
            )


def _recovery_checks(checks: list[dict[str, Any]], report: dict[str, Any]) -> None:
    summary = _dict(report.get("summary"))
    validation = summary.get("recovered_wrapper_validation_status")
    if validation == "validated":
        _check(checks, "recovery.validation_status", "pass", "recovered wrapper validated", None)
    else:
        _check(checks, "recovery.validation_status", "fail", f"validation status: {validation}", "Validate the recovered wrapper before publishing.")
    router_tensors = int(summary.get("recovered_updated_router_tensor_count") or 0)
    if router_tensors > 0:
        _check(checks, "recovery.router_tensors", "pass", f"updated router tensors: {router_tensors}", None)
    else:
        _check(checks, "recovery.router_tensors", "warn", "no updated router tensors recorded", "Train learned routers for sparse routing candidates.")
    quality = _dict(report.get("quality_trends")).get("before_after_quality")
    if isinstance(quality, dict) and quality.get("mode_count", 0):
        _check(checks, "recovery.before_after_quality", "pass", f"compared modes: {quality.get('mode_count')}", None)
    else:
        _check(checks, "recovery.before_after_quality", "warn", "before/after quality comparison is missing", "Run recovery-experiment with eval modes.")


def _validation_checks(checks: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if report.get("status") == "validated" and report.get("passed") is not False:
        _check(checks, "validation.status", "pass", "validation report passed", None)
    else:
        _check(checks, "validation.status", "fail", f"validation status: {report.get('status')}", "Inspect recovered-wrapper validation errors.")
    reload_report = _dict(report.get("reload"))
    if int(reload_report.get("loaded_layer_count") or 0) > 0:
        _check(checks, "validation.native_reload", "pass", f"loaded layers: {reload_report.get('loaded_layer_count')}", None)
    else:
        _check(checks, "validation.native_reload", "fail", "validation did not reload converted layers", "Check wrapper package and source model refs.")


def _native_load(wrapper: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        _check(checks, "package.native_load", "warn", f"transformers unavailable: {exc}", "Install transformers to validate native loading.")
        return {"attempted": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(wrapper, trust_remote_code=True)
    except Exception as exc:
        _check(checks, "package.native_load", "fail", f"native load failed: {exc}", "Fix AutoModel package metadata.")
        return {"attempted": True, "loaded": False, "error": str(exc)}
    replaced = getattr(getattr(model, "replacement_report", None), "replaced", [])
    payload = {
        "attempted": True,
        "loaded": True,
        "class": model.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "replaced_layers": len(replaced),
    }
    _check(checks, "package.native_load", "pass", f"loaded {payload['class']} with {len(replaced)} replaced layers", None)
    return payload


def _load_wrapper(wrapper: Path, checks: list[dict[str, Any]]) -> Any | None:
    try:
        config = load_wrapper_config(wrapper / "moeforge_config.json")
    except Exception as exc:
        _check(checks, "package.wrapper_config_load", "fail", f"could not load wrapper config: {exc}", "Regenerate the wrapper package.")
        return None
    _check(checks, "package.wrapper_config_load", "pass", "loaded wrapper config", None)
    return config


def _load_reports(paths: list[Path], *, checks: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        payload = _load_optional_report(path, checks=checks, label=label)
        if payload is not None:
            reports.append(payload)
    return reports


def _load_optional_report(path: Path | None, *, checks: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        _check(checks, f"{label}.exists", "fail", f"missing report: {path}", "Pass the correct report path.")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _check(checks, f"{label}.json", "fail", f"invalid JSON in {path}: {exc}", "Fix or regenerate the report.")
        return None
    if not isinstance(payload, dict):
        _check(checks, f"{label}.json", "fail", f"report must be a JSON object: {path}", "Regenerate the report.")
        return None
    _check(checks, f"{label}.json", "pass", f"loaded report: {path}", None)
    return payload


def _file_check(checks: list[dict[str, Any]], path: Path, name: str, label: str) -> None:
    _check(
        checks,
        name,
        "pass" if path.exists() else "fail",
        f"{label} {'exists' if path.exists() else 'is missing'}: {path}",
        None if path.exists() else f"Create or include {label}.",
    )


def _check(checks: list[dict[str, Any]], name: str, status: str, message: str, next_action: str | None) -> None:
    checks.append({"name": name, "status": status, "message": message, "next_action": next_action})


def _next_actions(checks: list[dict[str, Any]], *, wrapper: Path) -> list[str]:
    actions = [str(check["next_action"]) for check in checks if check.get("status") == "fail" and check.get("next_action")]
    if not actions:
        actions.append(f"huggingface-cli upload <repo-id> {wrapper}")
    return _unique(actions)


def _resolve_package_path(package_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else package_dir / path


def _first_sample_mode(report: dict[str, Any]) -> str | None:
    samples = report.get("samples")
    if isinstance(samples, list) and samples and isinstance(samples[0], dict):
        value = samples[0].get("expert_mode")
        return str(value) if value is not None else None
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
