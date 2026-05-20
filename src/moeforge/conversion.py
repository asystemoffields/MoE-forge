from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from .carve import build_carve_manifest
from .evaluation import evaluate_hf_dense_vs_carved
from .inspectors import inspect_model
from .materialize import materialize_carve_manifest
from .model_card import write_model_card
from .planner import PlanOptions, plan_conversion
from .preflight import run_preflight
from .publish import check_publish_readiness
from .recipe import recipe_to_dict
from .recovery_experiment import run_recovery_experiment
from .runtime import verify_carved_artifact
from .wrapper import export_wrapper_package


class ConversionRunError(RuntimeError):
    """Raised when a dense-to-MoE conversion run cannot complete."""


@dataclass(slots=True)
class ConversionRunOptions:
    model: str
    output_dir: Path
    recipe: Path | None = None
    profile: Path | None = None
    goal: str = "balanced"
    target: str = "hf"
    hardware: str = "auto"
    experts: int | None = None
    top_k: int | None = None
    shared_ratio: float | None = None
    moe_layers: str | None = "all"
    calibration_samples: int | None = None
    recover_steps: int | None = None
    activation: str = "silu"
    token_router_top_k: int | None = None
    default_expert_mode: str | None = None
    copy_source_model: bool = True
    dry_run: bool = False
    eval_smoke: bool = False
    eval_expert_modes: list[str] | None = None
    eval_device: str = "cpu"
    eval_sequence_length: int = 128
    eval_atol: float = 1e-5
    eval_rtol: float = 1e-5
    recover: bool = False
    recover_experts: bool = False
    train_text: str | None = None
    train_text_file: Path | None = None
    train_input_ids_json: str | None = None
    eval_text: str | None = None
    eval_text_file: Path | None = None
    eval_input_ids_json: str | None = None
    max_all_expert_error: float = 1e-4
    max_all_expert_teacher_kl: float = 0.01
    max_sparse_teacher_kl: float | None = None
    max_sparse_nll_delta: float | None = None


def run_conversion(options: ConversionRunOptions) -> dict[str, Any]:
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"output_dir": str(output_dir)}
    warnings: list[str] = []

    model_ref = _local_model_ref(options.model)
    artifacts["model"] = model_ref
    start_preflight_path = output_dir / "preflight-start.json"
    start_preflight = run_preflight(
        model=model_ref,
        recipe=options.recipe,
        profile=options.profile,
        output_path=start_preflight_path,
    )
    artifacts["preflight_start"] = str(start_preflight_path)
    _stage(stages, "preflight-start", "completed", {"status": start_preflight["status"]})

    recipe_path = output_dir / "recipe.json"
    if options.recipe is None:
        info = inspect_model(model_ref)
        recipe_payload = recipe_to_dict(
            plan_conversion(
                info,
                PlanOptions(
                    goal=options.goal,
                    target=options.target,
                    hardware=options.hardware,
                    experts=options.experts,
                    top_k=options.top_k,
                    shared_ratio=options.shared_ratio,
                    moe_layers=options.moe_layers,
                    calibration_samples=options.calibration_samples,
                    recover_steps=options.recover_steps,
                ),
            )
        )
        _write_json(recipe_path, recipe_payload)
        warnings.extend(str(item) for item in recipe_payload.get("warnings", []))
        _stage(stages, "plan", "completed", {"recipe": str(recipe_path)})
    else:
        _copy_if_different(options.recipe, recipe_path)
        recipe_payload = _read_json(recipe_path)
        warnings.extend(str(item) for item in recipe_payload.get("warnings", []))
        _stage(stages, "plan", "skipped", {"recipe": str(recipe_path), "reason": "existing recipe supplied"})
    artifacts["recipe"] = str(recipe_path)

    manifest = build_carve_manifest(model=model_ref, recipe_path=recipe_path, profile_path=options.profile)
    manifest_path = output_dir / "carve-manifest.json"
    _write_json(manifest_path, manifest.to_dict())
    artifacts["manifest"] = str(manifest_path)
    _stage(
        stages,
        "carve-manifest",
        "completed",
        {"manifest": str(manifest_path), "layer_count": len(manifest.layers), "experts": manifest.experts},
    )

    carved_dir = output_dir / "carved"
    materialize_report = materialize_carve_manifest(
        manifest_path=manifest_path,
        output_dir=carved_dir,
        dry_run=options.dry_run,
    )
    materialize_payload = materialize_report.to_dict()
    artifacts["carve_apply_report"] = str(
        carved_dir / ("carve-apply-dry-run.json" if options.dry_run else "carve-apply-report.json")
    )
    _stage(
        stages,
        "carve-apply",
        "completed",
        {
            "dry_run": options.dry_run,
            "tensor_count": materialize_payload["tensor_count"],
            "output_files": materialize_payload.get("output_files", []),
        },
    )

    if options.dry_run:
        final_preflight_path = output_dir / "preflight-final.json"
        final_preflight = run_preflight(
            model=model_ref,
            recipe=recipe_path,
            profile=options.profile,
            manifest=manifest_path,
            output_path=final_preflight_path,
        )
        artifacts["preflight_final"] = str(final_preflight_path)
        return _conversion_report(
            status="dry_run",
            options=options,
            stages=stages,
            artifacts=artifacts,
            warnings=warnings,
            preflight=final_preflight,
            eval_reports=[],
            output_dir=output_dir,
        )

    artifact_path = carved_dir / "carved-experts.safetensors"
    artifacts["artifact"] = str(artifact_path)
    verify_path = output_dir / "carve-verify-report.json"
    verify_report = verify_carved_artifact(manifest_path=manifest_path, artifact_path=artifact_path)
    _write_json(verify_path, verify_report.to_dict())
    artifacts["verify_report"] = str(verify_path)
    _stage(stages, "carve-verify", "completed", {"passed": verify_report.passed, "report": str(verify_path)})

    wrapper_dir = output_dir / "wrapper"
    wrapper_config = export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        output_dir=wrapper_dir,
        activation=options.activation,
        copy_artifact=True,
        copy_source_model=options.copy_source_model,
        token_router_top_k=options.token_router_top_k,
        default_expert_mode=options.default_expert_mode,
    )
    artifacts["wrapper"] = str(wrapper_dir)
    _stage(
        stages,
        "wrapper-export",
        "completed",
        {
            "wrapper": str(wrapper_dir),
            "layer_count": len(wrapper_config.layers),
            "token_router_top_k": wrapper_config.token_router_top_k,
            "default_expert_mode": wrapper_config.default_expert_mode,
            "copy_source_model": options.copy_source_model,
        },
    )

    eval_reports = _run_optional_smoke_evals(options=options, model_ref=model_ref, wrapper_dir=wrapper_dir, stages=stages)
    if eval_reports:
        artifacts["eval_reports"] = [str(path) for path in eval_reports]

    recovery_report_path: Path | None = None
    validation_report_path: Path | None = None
    publish_wrapper_dir = wrapper_dir
    if options.recover:
        recovery = _run_recovery(
            options=options,
            model_ref=model_ref,
            wrapper_dir=wrapper_dir,
            output_dir=output_dir,
            stages=stages,
            artifacts=artifacts,
            warnings=warnings,
        )
        recovery_report_path = recovery["recovery_report_path"]
        validation_report_path = recovery["validation_report_path"]
        publish_wrapper_dir = recovery["recovered_wrapper"]
        eval_reports = recovery["after_eval_reports"]

    model_card_path = publish_wrapper_dir / "MODEL_CARD.md"
    write_model_card(
        wrapper_dir=publish_wrapper_dir,
        output_path=model_card_path,
        eval_reports=eval_reports,
        recovery_reports=[recovery_report_path] if recovery_report_path is not None else [],
        validation_reports=[validation_report_path] if validation_report_path is not None else [],
        commands=_reproduction_commands(options, output_dir=output_dir),
    )
    artifacts["model_card"] = str(model_card_path)
    _stage(stages, "model-card", "completed", {"model_card": str(model_card_path)})

    publish_path = output_dir / "publish-readiness.json"
    publish_report = check_publish_readiness(
        wrapper=publish_wrapper_dir,
        output_path=publish_path,
        eval_reports=eval_reports,
        recovery_report=recovery_report_path,
        validation_report=validation_report_path,
        require_recovery=options.recover,
        max_all_expert_error=options.max_all_expert_error,
        max_all_expert_teacher_kl=options.max_all_expert_teacher_kl,
        max_sparse_teacher_kl=options.max_sparse_teacher_kl,
        max_sparse_nll_delta=options.max_sparse_nll_delta,
    )
    artifacts["publish_readiness"] = str(publish_path)
    _stage(
        stages,
        "publish-check",
        "completed",
        {
            "status": publish_report["status"],
            "failed_check_count": publish_report["failed_check_count"],
            "warning_count": publish_report["warning_count"],
            "report": str(publish_path),
        },
    )

    final_preflight_path = output_dir / "preflight-final.json"
    final_preflight = run_preflight(
        model=model_ref,
        recipe=recipe_path,
        profile=options.profile,
        manifest=manifest_path,
        artifact=artifact_path,
        wrapper=publish_wrapper_dir,
        output_path=final_preflight_path,
    )
    artifacts["preflight_final"] = str(final_preflight_path)
    _stage(stages, "preflight-final", "completed", {"status": final_preflight["status"]})

    status = "publish_ready" if verify_report.passed and final_preflight.get("passed") and publish_report.get("passed") else "needs_attention"
    return _conversion_report(
        status=status,
        options=options,
        stages=stages,
        artifacts=artifacts,
        warnings=warnings,
        preflight=final_preflight,
        publish=publish_report,
        eval_reports=eval_reports,
        output_dir=output_dir,
    )


def _run_optional_smoke_evals(
    *,
    options: ConversionRunOptions,
    model_ref: str,
    wrapper_dir: Path,
    stages: list[dict[str, Any]],
) -> list[Path]:
    if not options.eval_smoke:
        _stage(stages, "eval-smoke", "skipped", {"reason": "pass --eval-smoke to run dense-vs-carved checks"})
        return []

    modes = options.eval_expert_modes or ["all"]
    if options.token_router_top_k is not None and "learned-router" not in modes:
        modes = [*modes, "learned-router"]
    eval_dir = options.output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for mode in modes:
        report = evaluate_hf_dense_vs_carved(
            model=model_ref,
            package_dir=wrapper_dir,
            sequence_length=options.eval_sequence_length,
            device=options.eval_device,
            atol=options.eval_atol,
            rtol=options.eval_rtol,
            expert_mode=mode,
        )
        report_path = eval_dir / f"eval-{mode}.json"
        _write_json(report_path, report.to_dict())
        reports.append(report_path)
        _stage(
            stages,
            f"eval-smoke:{mode}",
            "completed",
            {
                "passed": report.passed,
                "max_abs_error": report.max_abs_error,
                "average_teacher_kl_loss": report.summary.average_teacher_kl_loss,
                "report": str(report_path),
            },
        )
    return reports


def _run_recovery(
    *,
    options: ConversionRunOptions,
    model_ref: str,
    wrapper_dir: Path,
    output_dir: Path,
    stages: list[dict[str, Any]],
    artifacts: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    recovery_dir = output_dir / "recovery-experiment"
    recovery_config_path = output_dir / "recovery-experiment-config.json"
    recovery_config = _recovery_experiment_config(
        options=options,
        model_ref=model_ref,
        wrapper_dir=wrapper_dir,
        recovery_dir=recovery_dir,
        warnings=warnings,
    )
    _write_json(recovery_config_path, recovery_config)
    artifacts["recovery_config"] = str(recovery_config_path)
    report = run_recovery_experiment(config_path=recovery_config_path, output_dir=recovery_dir)
    recovery_report_path = Path(str(report["artifacts"]["json_report"]))
    validation_report_path = Path(str(report["artifacts"]["recovered_wrapper_validation"]))
    recovered_wrapper = Path(str(report["recovered_wrapper"]))
    after_eval_reports = _eval_report_paths(report.get("after_eval_batch"))
    artifacts["recovery_report"] = str(recovery_report_path)
    artifacts["validation_report"] = str(validation_report_path)
    artifacts["recovered_wrapper"] = str(recovered_wrapper)
    artifacts["after_recovery_eval_reports"] = [str(path) for path in after_eval_reports]
    _stage(
        stages,
        "recovery-experiment",
        "completed",
        {
            "report": str(recovery_report_path),
            "recovered_wrapper": str(recovered_wrapper),
            "validation": report["summary"].get("recovered_wrapper_validation_status"),
            "steps_completed": report["summary"].get("steps_completed"),
        },
    )
    return {
        "recovery_report_path": recovery_report_path,
        "validation_report_path": validation_report_path,
        "recovered_wrapper": recovered_wrapper,
        "after_eval_reports": after_eval_reports,
    }


def _recovery_experiment_config(
    *,
    options: ConversionRunOptions,
    model_ref: str,
    wrapper_dir: Path,
    recovery_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    train = _sample_config(
        text=options.train_text,
        text_file=options.train_text_file,
        input_ids_json=options.train_input_ids_json,
        sequence_length=options.eval_sequence_length,
    )
    eval_samples = _sample_config(
        text=options.eval_text,
        text_file=options.eval_text_file,
        input_ids_json=options.eval_input_ids_json,
        sequence_length=options.eval_sequence_length,
    )
    if not train:
        train = {"input_ids": [_smoke_input_ids()], "sequence_length": options.eval_sequence_length}
        warnings.append("recovery used deterministic smoke token ids because no train samples were provided")
    if not eval_samples:
        eval_samples = {"input_ids": [_smoke_input_ids()], "sequence_length": options.eval_sequence_length}
        warnings.append("recovery eval used deterministic smoke token ids because no eval samples were provided")
    modes = options.eval_expert_modes or ["all"]
    if options.token_router_top_k is not None and "learned-router" not in modes:
        modes = [*modes, "learned-router"]
    steps = int(options.recover_steps or 25)
    return {
        "model": model_ref,
        "wrapper": str(wrapper_dir),
        "output_dir": str(recovery_dir),
        "strict_validation": True,
        "eval": {
            **eval_samples,
            "expert_modes": modes,
            "device": options.eval_device,
            "atol": options.eval_atol,
            "rtol": options.eval_rtol,
            "write_html": True,
        },
        "train": train,
        "recovery": {
            "trainable": {"experts": options.recover_experts, "router": True, "shared": False, "dense_backbone": False},
            "loss": {"teacher_kl_weight": 1.0, "logits_mse_weight": 0.0, "router_balance_weight": 0.01},
            "optimizer": {"learning_rate": 5e-5, "weight_decay": 0.0},
            "schedule": {
                "steps": steps,
                "eval_every_steps": max(1, steps // 4),
                "save_every_steps": steps,
            },
        },
    }


def _sample_config(
    *,
    text: str | None,
    text_file: Path | None,
    input_ids_json: str | None,
    sequence_length: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"sequence_length": sequence_length}
    if text:
        payload["text"] = text
    if text_file is not None:
        payload["text_file"] = str(text_file.resolve())
    if input_ids_json:
        payload["input_ids_json"] = input_ids_json
    return payload if len(payload) > 1 else {}


def _smoke_input_ids() -> list[int]:
    return [1, 2, 3, 4, 5, 6, 7, 8]


def _eval_report_paths(manifest: Any) -> list[Path]:
    reports = []
    for run in _dict(manifest).get("runs", []):
        if isinstance(run, dict) and run.get("report_path"):
            reports.append(Path(str(run["report_path"])))
    return reports


def _conversion_report(
    *,
    status: str,
    options: ConversionRunOptions,
    stages: list[dict[str, Any]],
    artifacts: dict[str, Any],
    warnings: list[str],
    preflight: dict[str, Any],
    publish: dict[str, Any] | None = None,
    eval_reports: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    report = {
        "format": "moeforge_conversion_run",
        "status": status,
        "passed": status in {"publish_ready", "dry_run"} and bool(preflight.get("passed")),
        "model": options.model,
        "output_dir": str(output_dir),
        "dry_run": options.dry_run,
        "options": _options_payload(options),
        "stages": stages,
        "artifacts": artifacts,
        "warnings": _unique(warnings),
        "preflight": {
            "status": preflight.get("status"),
            "passed": preflight.get("passed"),
            "failed_check_count": preflight.get("failed_check_count"),
            "warning_count": preflight.get("warning_count"),
        },
        "publish_readiness": {
            "status": publish.get("status") if publish else None,
            "passed": publish.get("passed") if publish else None,
            "failed_check_count": publish.get("failed_check_count") if publish else None,
            "warning_count": publish.get("warning_count") if publish else None,
        },
        "eval_reports": [str(path) for path in eval_reports],
        "next_commands": _next_commands(status=status, artifacts=artifacts),
    }
    report_path = output_dir / "convert-report.json"
    _write_json(report_path, report)
    report["artifacts"]["convert_report"] = str(report_path)
    _write_json(report_path, report)
    return report


def _options_payload(options: ConversionRunOptions) -> dict[str, Any]:
    return {
        "recipe": str(options.recipe) if options.recipe is not None else None,
        "profile": str(options.profile) if options.profile is not None else None,
        "goal": options.goal,
        "target": options.target,
        "hardware": options.hardware,
        "experts": options.experts,
        "top_k": options.top_k,
        "shared_ratio": options.shared_ratio,
        "moe_layers": options.moe_layers,
        "calibration_samples": options.calibration_samples,
        "recover_steps": options.recover_steps,
        "activation": options.activation,
        "token_router_top_k": options.token_router_top_k,
        "default_expert_mode": options.default_expert_mode,
        "copy_source_model": options.copy_source_model,
        "eval_smoke": options.eval_smoke,
        "eval_expert_modes": options.eval_expert_modes,
        "eval_device": options.eval_device,
        "eval_sequence_length": options.eval_sequence_length,
        "eval_atol": options.eval_atol,
        "eval_rtol": options.eval_rtol,
        "recover": options.recover,
        "recover_experts": options.recover_experts,
        "train_text": options.train_text,
        "train_text_file": str(options.train_text_file) if options.train_text_file is not None else None,
        "train_input_ids_json": options.train_input_ids_json,
        "eval_text": options.eval_text,
        "eval_text_file": str(options.eval_text_file) if options.eval_text_file is not None else None,
        "eval_input_ids_json": options.eval_input_ids_json,
        "max_all_expert_error": options.max_all_expert_error,
        "max_all_expert_teacher_kl": options.max_all_expert_teacher_kl,
        "max_sparse_teacher_kl": options.max_sparse_teacher_kl,
        "max_sparse_nll_delta": options.max_sparse_nll_delta,
    }


def _stage(stages: list[dict[str, Any]], name: str, status: str, details: dict[str, Any]) -> None:
    stages.append({"name": name, "status": status, "details": details})


def _next_commands(*, status: str, artifacts: dict[str, Any]) -> list[str]:
    if status == "dry_run":
        return [
            f"moe-forge convert {artifacts.get('model', '<model>')} --output-dir {artifacts.get('output_dir', 'moeforge-run')}"
        ]
    wrapper = artifacts.get("recovered_wrapper") or artifacts.get("wrapper")
    commands = []
    if wrapper:
        commands.append(f"python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained(r'{wrapper}', trust_remote_code=True)\"")
        commands.append(f"moe-forge preflight --wrapper {wrapper} --output {Path(str(wrapper)).parent / 'preflight-wrapper.json'}")
        commands.append(f"moe-forge publish-check --wrapper {wrapper} --output {Path(str(wrapper)).parent / 'publish-readiness.json'}")
    return commands


def _reproduction_commands(options: ConversionRunOptions, *, output_dir: Path) -> list[str]:
    parts = ["moe-forge", "convert", str(options.model), "--output-dir", str(output_dir)]
    if options.moe_layers:
        parts.extend(["--moe-layers", str(options.moe_layers)])
    if options.experts is not None:
        parts.extend(["--experts", str(options.experts)])
    if options.top_k is not None:
        parts.extend(["--top-k", str(options.top_k)])
    if options.shared_ratio is not None:
        parts.extend(["--shared-ratio", str(options.shared_ratio)])
    if options.token_router_top_k is not None:
        parts.extend(["--token-router-top-k", str(options.token_router_top_k)])
    if options.default_expert_mode is not None:
        parts.extend(["--default-expert-mode", options.default_expert_mode])
    if options.eval_smoke:
        parts.append("--eval-smoke")
    if options.recover:
        parts.append("--recover")
    if options.recover_experts:
        parts.append("--recover-experts")
    if not options.copy_source_model:
        parts.append("--skip-source-model-copy")
    return [" ".join(parts)]


def _local_model_ref(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path.resolve())
    return model


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConversionRunError(f"expected JSON object in {path}")
    return payload


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
