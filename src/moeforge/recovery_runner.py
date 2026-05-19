from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from .hf_runtime import MoEForgeCarvedMLPModule, MoEForgeConfig, replace_hf_mlp_modules
from .recovery import compare_eval_batch_manifests
from .wrapper import load_wrapper_config


class RecoveryRunError(RuntimeError):
    """Raised when a recovery run cannot execute."""


def run_recovery(
    *,
    plan_path: Path,
    output_path: Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RecoveryRunError("recovery-run requires torch and transformers") from exc

    plan = _load_plan(plan_path)
    output_dir = Path(str(plan.get("output_dir", plan_path.parent / "recovery-run")))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_path or output_dir / "recovery-run-report.json"
    teacher_config = _dict(plan.get("teacher"))
    student_config = _dict(plan.get("student"))
    schedule = _dict(plan.get("schedule"))
    loss_config = _dict(plan.get("loss"))
    optimizer_config = _dict(plan.get("optimizer"))
    checkpoints = _dict(plan.get("checkpoints"))
    warnings = list(plan.get("warnings") if isinstance(plan.get("warnings"), list) else [])

    device = _resolve_device(str(teacher_config.get("device", "auto")), torch=torch)
    train_samples, train_sample_source = _training_samples(
        plan,
        model_ref=str(teacher_config["model"]),
        tokenizer_cls=AutoTokenizer,
    )
    steps = int(max_steps or schedule.get("steps", 1))
    batch_size = max(1, int(schedule.get("batch_size", 1)))
    save_every = max(1, int(schedule.get("save_every_steps", steps)))

    teacher = AutoModelForCausalLM.from_pretrained(str(teacher_config["model"])).to(device)
    student = AutoModelForCausalLM.from_pretrained(str(student_config["model"])).to(device)
    teacher.eval()
    _freeze_student(student=student, trainable=_dict(student_config.get("trainable")))
    replacement_report = replace_hf_mlp_modules(student, Path(str(student_config["wrapper"])))
    promoted = _promote_carved_parameters(
        student=student,
        trainable=_dict(student_config.get("trainable")),
        torch=torch,
    )
    router_parameters = _configure_router_parameters(
        student=student,
        trainable=_dict(student_config.get("trainable")),
    )
    if _dict(student_config.get("trainable")).get("router", True) and not router_parameters:
        warnings.append("router training was requested, but the wrapper package does not define learned token routers")
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RecoveryRunError("recovery-run found no trainable student parameters")

    optimizer = _optimizer(optimizer_config, parameters=trainable_parameters, torch=torch)
    losses = []
    checkpoint_records = []
    student.train()
    for step in range(1, steps + 1):
        batch = _batch(train_samples, step=step, batch_size=batch_size, device=device, torch=torch)
        with torch.no_grad():
            teacher_logits = teacher(**batch).logits.detach()
        student_logits = student(**batch).logits
        loss_parts = _loss_parts(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            loss_config=loss_config,
            torch=torch,
            F=F,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_parts["total_loss"].backward()
        max_grad_norm = optimizer_config.get("max_grad_norm")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(max_grad_norm))
        optimizer.step()
        losses.append(_loss_record(step=step, loss_parts=loss_parts, optimizer=optimizer))
        if step == steps or step % save_every == 0:
            checkpoint_records.append(
                _write_checkpoint(
                    step=step,
                    plan=plan,
                    checkpoint_dir=Path(str(checkpoints.get("output_dir", output_dir / "checkpoints"))),
                    student=student,
                    optimizer=optimizer,
                    trainable_parameters=trainable_parameters,
                    promoted=promoted,
                    router_parameters=router_parameters,
                    torch=torch,
                )
            )

    report = {
        "format": "moeforge_recovery_run",
        "plan_path": str(plan_path),
        "output_dir": str(output_dir),
        "teacher_model": str(teacher_config["model"]),
        "student_model": str(student_config["model"]),
        "wrapper": str(student_config["wrapper"]),
        "device": str(device),
        "steps_requested": steps,
        "steps_completed": len(losses),
        "initial_loss": losses[0]["total_loss"] if losses else None,
        "final_loss": losses[-1]["total_loss"] if losses else None,
        "losses": losses,
        "loss_config": loss_config,
        "optimizer": optimizer_config,
        "replacement_report": replacement_report.to_dict(),
        "train_sample_source": train_sample_source,
        "trainable_parameter_count": _parameter_count(trainable_parameters),
        "promoted_carved_parameters": promoted,
        "promoted_router_parameters": router_parameters,
        "checkpoints": checkpoint_records,
        "before_after_eval": _before_after_from_plan(plan),
        "warnings": warnings,
    }
    _write_json(target, report)
    return report


def export_recovered_wrapper(
    *,
    checkpoint_path: Path,
    wrapper_dir: Path,
    output_dir: Path,
    artifact_name: str = "recovered-carved-experts.safetensors",
    router_artifact_name: str = "learned-router.safetensors",
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RecoveryRunError("recovery-export requires torch and safetensors") from exc

    checkpoint = _load_checkpoint_metadata(checkpoint_path)
    wrapper_config = load_wrapper_config(wrapper_dir / "moeforge_config.json")
    source_artifact = _resolve_package_path(wrapper_dir, wrapper_config.artifact_path)
    state_path = Path(str(checkpoint["state_path"]))
    if not state_path.is_absolute():
        state_path = checkpoint_path.parent / state_path
    state = torch.load(str(state_path), map_location="cpu")
    trainable_state = state.get("trainable_state")
    if not isinstance(trainable_state, dict):
        raise RecoveryRunError("recovery checkpoint state must include trainable_state")

    tensors = dict(load_file(str(source_artifact), device="cpu"))
    tensor_map = _promoted_tensor_map(checkpoint)
    updated = []
    router_tensor_map = _promoted_router_tensor_map(checkpoint)
    source_router_artifact = (
        _resolve_package_path(wrapper_dir, wrapper_config.token_router_path)
        if getattr(wrapper_config, "token_router_path", None)
        else None
    )
    router_tensors = (
        dict(load_file(str(source_router_artifact), device="cpu"))
        if source_router_artifact is not None and source_router_artifact.exists()
        else {}
    )
    updated_router = []
    for parameter_name, value in trainable_state.items():
        tensor_name = _tensor_name_for_parameter(str(parameter_name), tensor_map)
        if tensor_name is not None:
            if tensor_name not in tensors:
                raise RecoveryRunError(f"checkpoint tensor is not present in wrapper artifact: {tensor_name}")
            source_tensor = tensors[tensor_name]
            checkpoint_value = value.detach().cpu()
            exported_value = checkpoint_value.to(dtype=source_tensor.dtype).contiguous()
            tensors[tensor_name] = exported_value
            updated.append(
                {
                    "parameter": str(parameter_name),
                    "tensor": tensor_name,
                    "shape": list(exported_value.shape),
                    "source_dtype": _torch_dtype_name(source_tensor),
                    "checkpoint_dtype": _torch_dtype_name(checkpoint_value),
                    "export_dtype": _torch_dtype_name(exported_value),
                    "dtype_cast": bool(checkpoint_value.dtype != exported_value.dtype),
                }
            )
            continue
        router_tensor_name = _tensor_name_for_parameter(str(parameter_name), router_tensor_map)
        if router_tensor_name is not None:
            checkpoint_value = value.detach().cpu().contiguous()
            router_tensors[router_tensor_name] = checkpoint_value
            updated_router.append(
                {
                    "parameter": str(parameter_name),
                    "tensor": router_tensor_name,
                    "shape": list(checkpoint_value.shape),
                    "checkpoint_dtype": _torch_dtype_name(checkpoint_value),
                    "export_dtype": _torch_dtype_name(checkpoint_value),
                }
            )
    if not updated and not updated_router:
        raise RecoveryRunError("recovery checkpoint did not contain carved tensor or router parameters to export")

    _copy_wrapper_scaffold(source_dir=wrapper_dir, output_dir=output_dir, skip_files={source_artifact.name})
    artifact_path = output_dir / artifact_name
    if artifact_path.exists():
        artifact_path.unlink()
    save_file(
        tensors,
        str(artifact_path),
        metadata={
            "moe_forge": "recovered_carved_experts",
            "source_artifact": str(source_artifact),
            "checkpoint": str(checkpoint_path),
        },
    )
    router_artifact_path = None
    if updated_router:
        router_artifact_path = output_dir / router_artifact_name
        if router_artifact_path.exists():
            router_artifact_path.unlink()
        save_file(
            router_tensors,
            str(router_artifact_path),
            metadata={
                "moe_forge": "learned_token_router",
                "checkpoint": str(checkpoint_path),
            },
        )
    _rewrite_wrapper_artifact_refs(
        output_dir=output_dir,
        artifact_name=artifact_name,
        token_router_name=router_artifact_name if updated_router else None,
        checkpoint_path=checkpoint_path,
    )
    report = {
        "format": "moeforge_recovery_export",
        "checkpoint_path": str(checkpoint_path),
        "state_path": str(state_path),
        "source_wrapper": str(wrapper_dir),
        "output_dir": str(output_dir),
        "artifact_path": str(artifact_path),
        "router_artifact_path": str(router_artifact_path) if router_artifact_path is not None else None,
        "updated_tensor_count": len(updated),
        "updated_tensors": updated,
        "updated_router_tensor_count": len(updated_router),
        "updated_router_tensors": updated_router,
        "wrapper_config": str(output_dir / "moeforge_config.json"),
    }
    _write_json(output_dir / "recovery-export-report.json", report)
    return report


def validate_recovered_wrapper(
    *,
    source_wrapper: Path,
    recovered_wrapper: Path,
    checkpoint_path: Path | None = None,
    export_report_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "format": "moeforge_recovered_wrapper_validation",
        "source_wrapper": str(source_wrapper),
        "recovered_wrapper": str(recovered_wrapper),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "export_report_path": str(export_report_path) if export_report_path is not None else None,
        "status": "validated",
        "passed": True,
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    source_config_path = source_wrapper / "moeforge_config.json"
    recovered_config_path = recovered_wrapper / "moeforge_config.json"
    source_config = _load_wrapper_config_for_validation(source_config_path, errors=errors)
    recovered_config = _load_wrapper_config_for_validation(recovered_config_path, errors=errors)
    if source_config is None or recovered_config is None:
        return _finish_validation_report(report=report, output_path=output_path)

    source_artifact = _resolve_package_path(source_wrapper, source_config.artifact_path)
    recovered_artifact = _resolve_package_path(recovered_wrapper, recovered_config.artifact_path)
    recovered_router_artifact = (
        _resolve_package_path(recovered_wrapper, recovered_config.token_router_path)
        if recovered_config.token_router_path
        else None
    )
    _require_file(source_artifact, label="source wrapper artifact", errors=errors)
    _require_file(recovered_artifact, label="recovered wrapper artifact", errors=errors)
    if recovered_config.token_router_path:
        _require_file(recovered_router_artifact, label="recovered token router artifact", errors=errors)
    report["wrapper_configs"] = {
        "source_config": str(source_config_path),
        "recovered_config": str(recovered_config_path),
        "source_artifact": str(source_artifact),
        "recovered_artifact": str(recovered_artifact),
        "source_artifact_ref": source_config.artifact_path,
        "recovered_artifact_ref": recovered_config.artifact_path,
        "source_token_router_ref": source_config.token_router_path,
        "recovered_token_router_ref": recovered_config.token_router_path,
        "source_model": source_config.source_model,
        "recovered_source_model": recovered_config.source_model,
        "source_layers": [item.layer for item in source_config.layers],
        "recovered_layers": [item.layer for item in recovered_config.layers],
    }

    config_checks = _wrapper_config_checks(source_config=source_config, recovered_config=recovered_config)
    report["config_checks"] = config_checks
    for check, passed in config_checks.items():
        if not passed:
            errors.append(f"wrapper config check failed: {check}")

    resolved_export_report_path = export_report_path or recovered_wrapper / "recovery-export-report.json"
    export_report = _load_optional_json(resolved_export_report_path, label="recovery export report", warnings=warnings)
    report["export_report_path"] = str(resolved_export_report_path)
    if export_report is not None:
        report["export"] = _export_validation(
            export_report=export_report,
            source_wrapper=source_wrapper,
            recovered_artifact=recovered_artifact,
            warnings=warnings,
            errors=errors,
        )
        checkpoint_path = checkpoint_path or _path_or_none(export_report.get("checkpoint_path"))

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = _load_optional_json(checkpoint_path, label="recovery checkpoint metadata", warnings=warnings)
        report["checkpoint_path"] = str(checkpoint_path)
    if checkpoint is not None:
        report["checkpoint"] = _checkpoint_validation(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path or Path("."),
            export_report=export_report,
            errors=errors,
        )

    if source_artifact.exists() and recovered_artifact.exists():
        report["tensor_comparison"] = _compare_wrapper_tensors(
            source_artifact=source_artifact,
            recovered_artifact=recovered_artifact,
            export_report=export_report,
            errors=errors,
        )
    if recovered_router_artifact is not None and recovered_router_artifact.exists():
        source_router_artifact = (
            _resolve_package_path(source_wrapper, source_config.token_router_path)
            if source_config.token_router_path
            else None
        )
        report["router_tensor_validation"] = _validate_router_tensors(
            recovered_artifact=recovered_artifact,
            recovered_router_artifact=recovered_router_artifact,
            source_router_artifact=source_router_artifact,
            recovered_config=recovered_config,
            errors=errors,
        )

    if recovered_artifact.exists() and not errors:
        report["reload"] = _reload_recovered_layers(
            recovered_wrapper=recovered_wrapper,
            errors=errors,
        )
    if recovered_artifact.exists() and not errors:
        report["native_load"] = _reload_native_auto_model(
            recovered_wrapper=recovered_wrapper,
            errors=errors,
            warnings=warnings,
        )

    return _finish_validation_report(report=report, output_path=output_path)


def _loss_parts(
    *,
    student_logits: Any,
    teacher_logits: Any,
    loss_config: dict[str, Any],
    torch: Any,
    F: Any,
) -> dict[str, Any]:
    temperature = float(loss_config.get("temperature", 1.0))
    student_flat = student_logits.reshape(-1, student_logits.shape[-1])
    teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    teacher_kl = F.kl_div(
        F.log_softmax(student_flat / temperature, dim=-1),
        F.softmax(teacher_flat / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    teacher_kl = teacher_kl.clamp_min(0.0)
    logits_mse = F.mse_loss(student_flat, teacher_flat)
    z_loss = torch.logsumexp(student_flat, dim=-1).pow(2).mean()
    total = (
        float(loss_config.get("teacher_kl_weight", 1.0)) * teacher_kl
        + float(loss_config.get("logits_mse_weight", 0.0)) * logits_mse
        + float(loss_config.get("z_loss_weight", 0.0)) * z_loss
    )
    return {
        "total_loss": total,
        "teacher_kl": teacher_kl,
        "logits_mse": logits_mse,
        "z_loss": z_loss,
    }


def _training_samples(
    plan: dict[str, Any],
    *,
    model_ref: str,
    tokenizer_cls: Any,
) -> tuple[list[list[int]], dict[str, Any]]:
    train = _dict(_dict(plan.get("samples")).get("train"))
    if train.get("kind") == "text":
        return _text_training_samples(train, model_ref=model_ref, tokenizer_cls=tokenizer_cls)
    if train.get("kind") != "input_ids":
        raise RecoveryRunError("recovery-run requires train samples with kind=input_ids or kind=text")
    samples = train.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RecoveryRunError("recovery-run requires at least one train input-id sample")
    input_ids = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("input_ids"), list):
            raise RecoveryRunError("train input-id samples must include input_ids")
        input_ids.append([int(item) for item in sample["input_ids"]])
    return input_ids, {
        "kind": "input_ids",
        "sample_count": len(input_ids),
        "sequence_length": train.get("sequence_length"),
        "source": train.get("source"),
    }


def _text_training_samples(
    train: dict[str, Any],
    *,
    model_ref: str,
    tokenizer_cls: Any,
) -> tuple[list[list[int]], dict[str, Any]]:
    texts = _text_values_from_manifest(train)
    if not texts:
        raise RecoveryRunError("recovery-run requires at least one train text sample")
    try:
        tokenizer = tokenizer_cls.from_pretrained(model_ref)
    except Exception as exc:  # pragma: no cover - depends on optional tokenizer assets
        raise RecoveryRunError("text recovery requires a loadable tokenizer") from exc
    sequence_length = max(1, int(train.get("sequence_length") or 128))
    input_ids = []
    token_counts = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            return_attention_mask=False,
        )
        sample_ids = _encoded_input_ids(encoded)
        if not sample_ids:
            raise RecoveryRunError("text recovery tokenizer produced an empty input-id sample")
        input_ids.append(sample_ids)
        token_counts.append(len(sample_ids))
    return input_ids, {
        "kind": "text",
        "sample_count": len(input_ids),
        "sequence_length": sequence_length,
        "source": train.get("source"),
        "token_counts": token_counts,
    }


def _text_values_from_manifest(train: dict[str, Any]) -> list[str]:
    source = _dict(train.get("source"))
    text_file = _dict(source.get("text_file"))
    path_value = text_file.get("resolved_path") or text_file.get("path")
    if not path_value:
        raise RecoveryRunError("text recovery currently requires train.source.text_file provenance")
    path = Path(str(path_value))
    content = path.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]


def _encoded_input_ids(encoded: Any) -> list[int]:
    raw = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise RecoveryRunError("tokenizer input_ids must be integer token ids") from exc


def _batch(
    samples: list[list[int]],
    *,
    step: int,
    batch_size: int,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    selected = [samples[(step - 1 + offset) % len(samples)] for offset in range(batch_size)]
    max_len = max(len(sample) for sample in selected)
    padded = [sample + [0] * (max_len - len(sample)) for sample in selected]
    mask = [[1] * len(sample) + [0] * (max_len - len(sample)) for sample in selected]
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask, dtype=torch.long, device=device),
    }


def _freeze_student(*, student: Any, trainable: dict[str, Any]) -> None:
    dense_trainable = bool(trainable.get("dense_backbone", False))
    for parameter in student.parameters():
        parameter.requires_grad = dense_trainable


def _promote_carved_parameters(*, student: Any, trainable: dict[str, Any], torch: Any) -> list[dict[str, Any]]:
    promoted = []
    train_experts = bool(trainable.get("experts", True))
    train_shared = bool(trainable.get("shared", False))
    for module_name, module in student.named_modules():
        if not isinstance(module, MoEForgeCarvedMLPModule):
            continue
        for tensor_name, attribute in list(module._tensor_buffers.items()):
            should_train = (
                (".experts." in tensor_name and train_experts)
                or (".shared." in tensor_name and train_shared)
            )
            if not should_train or attribute not in module._buffers:
                continue
            value = module._buffers.pop(attribute)
            parameter = torch.nn.Parameter(value.detach().clone())
            module.register_parameter(attribute, parameter)
            promoted.append(
                {
                    "module": module_name,
                    "layer": module.layer,
                    "tensor": tensor_name,
                    "parameter": attribute,
                    "shape": list(parameter.shape),
                }
            )
    return promoted


def _configure_router_parameters(*, student: Any, trainable: dict[str, Any]) -> list[dict[str, Any]]:
    train_router = bool(trainable.get("router", True))
    records = []
    for module_name, module in student.named_modules():
        if not isinstance(module, MoEForgeCarvedMLPModule) or module.token_router is None:
            continue
        for parameter_name, parameter in module.token_router.named_parameters():
            parameter.requires_grad = train_router
            tensor_name = f"moe.layers.{module.layer}.mlp.router.{parameter_name}"
            records.append(
                {
                    "module": module_name,
                    "layer": module.layer,
                    "tensor": tensor_name,
                    "parameter": f"{module_name}.token_router.{parameter_name}",
                    "shape": list(parameter.shape),
                    "trainable": train_router,
                    "top_k": module.token_router_top_k,
                }
            )
    return [record for record in records if record["trainable"]]


def _optimizer(config: dict[str, Any], *, parameters: list[Any], torch: Any) -> Any:
    name = str(config.get("name", "adamw")).lower()
    if name != "adamw":
        raise RecoveryRunError(f"unsupported optimizer {name}")
    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 5e-5)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        betas=tuple(float(item) for item in config.get("betas", [0.9, 0.95])),
    )


def _loss_record(*, step: int, loss_parts: dict[str, Any], optimizer: Any) -> dict[str, Any]:
    return {
        "step": step,
        "total_loss": float(loss_parts["total_loss"].detach().cpu().item()),
        "teacher_kl": float(loss_parts["teacher_kl"].detach().cpu().item()),
        "logits_mse": float(loss_parts["logits_mse"].detach().cpu().item()),
        "z_loss": float(loss_parts["z_loss"].detach().cpu().item()),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


def _write_checkpoint(
    *,
    step: int,
    plan: dict[str, Any],
    checkpoint_dir: Path,
    student: Any,
    optimizer: Any,
    trainable_parameters: list[Any],
    promoted: list[dict[str, Any]],
    router_parameters: list[dict[str, Any]],
    torch: Any,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / f"trainable-state-step-{step}.pt"
    metadata_path = checkpoint_dir / f"checkpoint-step-{step}.json"
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "step": step,
            "trainable_state": trainable_state,
            "optimizer": optimizer.state_dict(),
        },
        state_path,
    )
    metadata = {
        "format": "moeforge_recovery_checkpoint",
        "step": step,
        "metadata_path": str(metadata_path),
        "state_path": str(state_path),
        "plan_path": plan.get("artifacts", {}).get("plan_path"),
        "trainable_parameter_count": _parameter_count(trainable_parameters),
        "saved_tensor_count": len(trainable_state),
        "promoted_carved_parameter_count": len(promoted),
        "promoted_carved_parameters": promoted,
        "promoted_router_parameter_count": len(router_parameters),
        "promoted_router_parameters": router_parameters,
    }
    _write_json(metadata_path, metadata)
    return metadata


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecoveryRunError("recovery checkpoint metadata must be a JSON object")
    if payload.get("format") != "moeforge_recovery_checkpoint":
        raise RecoveryRunError("recovery-export requires moeforge_recovery_checkpoint metadata")
    if not payload.get("state_path"):
        raise RecoveryRunError("recovery checkpoint metadata is missing state_path")
    return payload


def _load_wrapper_config_for_validation(path: Path, *, errors: list[str]) -> Any | None:
    if not path.exists():
        errors.append(f"wrapper config not found: {path}")
        return None
    try:
        return load_wrapper_config(path)
    except Exception as exc:
        errors.append(f"could not load wrapper config {path}: {exc}")
        return None


def _wrapper_config_checks(*, source_config: Any, recovered_config: Any) -> dict[str, bool]:
    return {
        "format_version_match": source_config.format_version == recovered_config.format_version,
        "model_type_match": source_config.model_type == recovered_config.model_type,
        "adapter_family_match": source_config.adapter_family == recovered_config.adapter_family,
        "source_model_match": source_config.source_model == recovered_config.source_model,
        "activation_match": source_config.activation == recovered_config.activation,
        "expert_count_match": source_config.expert_count == recovered_config.expert_count,
        "token_router_top_k_match": source_config.token_router_top_k == recovered_config.token_router_top_k,
        "layer_signature_match": _layer_signature(source_config.layers) == _layer_signature(recovered_config.layers),
    }


def _layer_signature(layers: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(layer.layer),
            "width": layer.width,
            "tensor_prefix": str(layer.tensor_prefix),
            "expert_count": int(layer.expert_count),
            "shared_channels": int(layer.shared_channels),
            "expert_channels": [int(value) for value in layer.expert_channels],
        }
        for layer in layers
    ]


def _export_validation(
    *,
    export_report: dict[str, Any],
    source_wrapper: Path,
    recovered_artifact: Path,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    checks = {
        "format": export_report.get("format"),
        "updated_tensor_count": export_report.get("updated_tensor_count"),
        "updated_tensors": export_report.get("updated_tensors") if isinstance(export_report.get("updated_tensors"), list) else [],
        "updated_router_tensor_count": export_report.get("updated_router_tensor_count", 0),
        "updated_router_tensors": export_report.get("updated_router_tensors") if isinstance(export_report.get("updated_router_tensors"), list) else [],
    }
    if export_report.get("format") != "moeforge_recovery_export":
        errors.append("recovery export report has unexpected format")
    artifact_path = _path_or_none(export_report.get("artifact_path"))
    if artifact_path is None:
        errors.append("recovery export report is missing artifact_path")
    elif artifact_path.exists() and recovered_artifact.exists() and artifact_path.resolve() != recovered_artifact.resolve():
        errors.append("recovery export report artifact_path does not match recovered wrapper config")
    source_report_wrapper = _path_or_none(export_report.get("source_wrapper"))
    if source_report_wrapper is not None and source_report_wrapper.exists() and source_wrapper.exists():
        if source_report_wrapper.resolve() != source_wrapper.resolve():
            errors.append("recovery export report source_wrapper does not match validation source_wrapper")
    updated_tensors = checks["updated_tensors"]
    if checks["updated_tensor_count"] != len(updated_tensors):
        errors.append("recovery export report updated_tensor_count does not match updated_tensors length")
    updated_router_tensors = checks["updated_router_tensors"]
    if checks["updated_router_tensor_count"] != len(updated_router_tensors):
        errors.append("recovery export report updated_router_tensor_count does not match updated_router_tensors length")
    if not updated_tensors and not updated_router_tensors:
        warnings.append("recovery export report did not record updated tensors")
    return checks


def _checkpoint_validation(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    export_report: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    state_path = _path_or_none(checkpoint.get("state_path"))
    if state_path is not None and not state_path.is_absolute():
        state_path = checkpoint_path.parent / state_path
    promoted = checkpoint.get("promoted_carved_parameters")
    promoted_router = checkpoint.get("promoted_router_parameters")
    if checkpoint.get("format") != "moeforge_recovery_checkpoint":
        errors.append("recovery checkpoint metadata has unexpected format")
    if state_path is None:
        errors.append("recovery checkpoint metadata is missing state_path")
    elif not state_path.exists():
        errors.append(f"recovery checkpoint state file not found: {state_path}")
    if not isinstance(promoted, list):
        errors.append("recovery checkpoint metadata does not include promoted carved parameters")
        promoted = []
    if not isinstance(promoted_router, list):
        promoted_router = []
    if not promoted and not promoted_router:
        errors.append("recovery checkpoint metadata does not include promoted carved or router parameters")
    promoted_tensors = {
        str(item["tensor"])
        for item in promoted
        if isinstance(item, dict) and item.get("tensor") is not None
    }
    promoted_router_tensors = {
        str(item["tensor"])
        for item in promoted_router
        if isinstance(item, dict) and item.get("tensor") is not None
    }
    updated_tensors = set()
    if export_report is not None and isinstance(export_report.get("updated_tensors"), list):
        updated_tensors = {
            str(item["tensor"])
            for item in export_report["updated_tensors"]
            if isinstance(item, dict) and item.get("tensor") is not None
        }
        if promoted_tensors and updated_tensors != promoted_tensors:
            errors.append("recovery export updated tensors do not match checkpoint promoted tensors")
        updated_router_tensors = {
            str(item["tensor"])
            for item in export_report.get("updated_router_tensors", [])
            if isinstance(item, dict) and item.get("tensor") is not None
        }
        if promoted_router_tensors and updated_router_tensors != promoted_router_tensors:
            errors.append("recovery export updated router tensors do not match checkpoint promoted router tensors")
    return {
        "format": checkpoint.get("format"),
        "step": checkpoint.get("step"),
        "state_path": str(state_path) if state_path is not None else None,
        "state_exists": bool(state_path is not None and state_path.exists()),
        "promoted_carved_parameter_count": len(promoted),
        "promoted_tensor_count": len(promoted_tensors),
        "promoted_router_parameter_count": len(promoted_router),
        "promoted_router_tensor_count": len(promoted_router_tensors),
        "exported_tensor_count": len(updated_tensors),
    }


def _compare_wrapper_tensors(
    *,
    source_artifact: Path,
    recovered_artifact: Path,
    export_report: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RecoveryRunError("recovered wrapper validation requires torch and safetensors") from exc

    source_tensors = load_file(str(source_artifact), device="cpu")
    recovered_tensors = load_file(str(recovered_artifact), device="cpu")
    source_names = set(source_tensors)
    recovered_names = set(recovered_tensors)
    missing = sorted(source_names - recovered_names)
    extra = sorted(recovered_names - source_names)
    if missing:
        errors.append(f"recovered artifact is missing {len(missing)} source tensors")
    if extra:
        errors.append(f"recovered artifact has {len(extra)} unexpected tensors")

    updated_names = _updated_tensor_names(export_report)
    shape_mismatches = []
    dtype_mismatches = []
    changed_tensors = []
    updated_tensor_metadata = []
    for name in sorted(source_names & recovered_names):
        source = source_tensors[name]
        recovered = recovered_tensors[name]
        source_shape = list(source.shape)
        recovered_shape = list(recovered.shape)
        source_dtype = str(source.dtype).replace("torch.", "")
        recovered_dtype = str(recovered.dtype).replace("torch.", "")
        if source_shape != recovered_shape:
            shape_mismatches.append({"tensor": name, "source_shape": source_shape, "recovered_shape": recovered_shape})
            continue
        if source_dtype != recovered_dtype:
            dtype_mismatches.append({"tensor": name, "source_dtype": source_dtype, "recovered_dtype": recovered_dtype})
        diff = (source - recovered).abs() if source.is_floating_point() else (source != recovered).to(torch.float32)
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        changed = bool(max_abs != 0.0)
        record = {
            "tensor": name,
            "shape": recovered_shape,
            "source_dtype": source_dtype,
            "recovered_dtype": recovered_dtype,
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
        }
        if changed:
            changed_tensors.append(record)
        if name in updated_names:
            updated_tensor_metadata.append(record)
    for item in shape_mismatches:
        errors.append(f"recovered tensor shape mismatch: {item['tensor']}")
    for item in dtype_mismatches:
        errors.append(f"recovered tensor dtype mismatch: {item['tensor']}")
    updated_missing = sorted(updated_names - recovered_names)
    if updated_missing:
        errors.append(f"export report references {len(updated_missing)} tensors missing from recovered artifact")
    return {
        "source_artifact": str(source_artifact),
        "recovered_artifact": str(recovered_artifact),
        "source_tensor_count": len(source_tensors),
        "recovered_tensor_count": len(recovered_tensors),
        "common_tensor_count": len(source_names & recovered_names),
        "missing_from_recovered": missing,
        "extra_in_recovered": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "updated_tensor_count": len(updated_names),
        "updated_tensors": updated_tensor_metadata,
        "changed_tensor_count": len(changed_tensors),
        "changed_tensors": changed_tensors,
    }


def _validate_router_tensors(
    *,
    recovered_artifact: Path,
    recovered_router_artifact: Path,
    source_router_artifact: Path | None,
    recovered_config: Any,
    errors: list[str],
) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RecoveryRunError("router tensor validation requires safetensors") from exc

    recovered_tensors = load_file(str(recovered_router_artifact), device="cpu")
    carved_tensors = load_file(str(recovered_artifact), device="cpu")
    source_tensors = (
        load_file(str(source_router_artifact), device="cpu")
        if source_router_artifact is not None and source_router_artifact.exists()
        else {}
    )
    expected = _expected_router_tensor_shapes(recovered_config, carved_tensors=carved_tensors)
    missing_expected = sorted(name for name in expected if name not in recovered_tensors)
    extra = sorted(set(recovered_tensors) - set(expected))
    shape_mismatches = []
    dtype_changes = []
    changed_tensors = []
    tensor_metadata = []
    for name in sorted(recovered_tensors):
        recovered = recovered_tensors[name]
        recovered_shape = list(recovered.shape)
        recovered_dtype = _torch_dtype_name(recovered)
        expected_shape = expected.get(name)
        if expected_shape is not None and recovered_shape != expected_shape:
            shape_mismatches.append(
                {"tensor": name, "expected_shape": expected_shape, "recovered_shape": recovered_shape}
            )
        record = {
            "tensor": name,
            "shape": recovered_shape,
            "dtype": recovered_dtype,
            "expected": name in expected,
        }
        if name in source_tensors:
            source = source_tensors[name]
            source_dtype = _torch_dtype_name(source)
            record["source_dtype"] = source_dtype
            if source_dtype != recovered_dtype:
                dtype_changes.append({"tensor": name, "source_dtype": source_dtype, "recovered_dtype": recovered_dtype})
            if list(source.shape) == recovered_shape:
                diff = (source - recovered).abs() if source.is_floating_point() else (source != recovered).float()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
                mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
                record["max_abs_delta"] = max_abs
                record["mean_abs_delta"] = mean_abs
                if max_abs != 0.0:
                    changed_tensors.append(record)
        tensor_metadata.append(record)
    for name in missing_expected:
        errors.append(f"recovered token router artifact is missing expected tensor: {name}")
    for item in shape_mismatches:
        errors.append(f"recovered token router tensor shape mismatch: {item['tensor']}")
    return {
        "recovered_router_artifact": str(recovered_router_artifact),
        "source_router_artifact": str(source_router_artifact) if source_router_artifact is not None else None,
        "source_router_exists": bool(source_router_artifact is not None and source_router_artifact.exists()),
        "tensor_count": len(recovered_tensors),
        "expected_tensor_count": len(expected),
        "missing_expected": missing_expected,
        "extra_tensors": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_changes": dtype_changes,
        "changed_tensor_count": len(changed_tensors),
        "changed_tensors": changed_tensors,
        "tensors": tensor_metadata,
    }


def _expected_router_tensor_shapes(config: Any, *, carved_tensors: dict[str, Any]) -> dict[str, list[int]]:
    expected = {}
    for layer in config.layers:
        hidden_size = _router_hidden_size(layer, carved_tensors=carved_tensors)
        layer_id = int(layer.layer)
        expert_count = int(config.expert_count)
        expected[f"moe.layers.{layer_id}.mlp.router.weight"] = [expert_count, hidden_size]
        expected[f"moe.layers.{layer_id}.mlp.router.bias"] = [expert_count]
    return expected


def _router_hidden_size(layer: Any, *, carved_tensors: dict[str, Any]) -> int:
    layer_id = int(layer.layer)
    candidates = [
        f"moe.layers.{layer_id}.mlp.shared.gate.weight",
        f"moe.layers.{layer_id}.mlp.shared.up.weight",
        f"moe.layers.{layer_id}.mlp.experts.0.gate.weight",
        f"moe.layers.{layer_id}.mlp.experts.0.up.weight",
    ]
    for name in candidates:
        tensor = carved_tensors.get(name)
        if tensor is not None and len(tensor.shape) == 2:
            return int(tensor.shape[1])
    for name, tensor in sorted(carved_tensors.items()):
        if name.startswith(f"moe.layers.{layer_id}.mlp.") and name.endswith(".weight") and len(tensor.shape) == 2:
            return int(tensor.shape[1])
    raise RecoveryRunError(f"could not infer router hidden size for layer {layer_id}")


def _reload_recovered_layers(*, recovered_wrapper: Path, errors: list[str]) -> dict[str, Any]:
    loaded_layers = []
    try:
        config = MoEForgeConfig.from_package(recovered_wrapper)
        for layer in config.layer_ids():
            module = MoEForgeCarvedMLPModule.from_package(recovered_wrapper, layer=layer, config=config)
            router_parameters = []
            if module.token_router is not None:
                for name, parameter in module.token_router.named_parameters():
                    router_parameters.append(
                        {
                            "name": name,
                            "shape": list(parameter.shape),
                            "dtype": _torch_dtype_name(parameter),
                            "requires_grad": bool(parameter.requires_grad),
                        }
                    )
            loaded_layers.append(
                {
                    "layer": layer,
                    "expert_count": module.expert_count,
                    "tensor_buffer_count": len(module._tensor_buffers),
                    "token_router_top_k": module.token_router_top_k,
                    "token_router_loaded": module.token_router is not None,
                    "token_router_parameter_count": len(router_parameters),
                    "token_router_parameters": router_parameters,
                }
            )
            if config.token_router_path and module.token_router is None:
                errors.append(f"recovered wrapper layer {layer} did not load token router parameters")
    except Exception as exc:
        errors.append(f"could not reload recovered wrapper layers: {exc}")
    return {
        "loaded_layer_count": len(loaded_layers),
        "loaded_layers": loaded_layers,
    }


def _reload_native_auto_model(*, recovered_wrapper: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        warnings.append("native AutoModel reload skipped because transformers is not installed")
        return {"status": "skipped", "reason": "transformers_not_installed"}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(recovered_wrapper))
        replacement_report = getattr(model, "replacement_report", None)
        replacement_payload = (
            replacement_report.to_dict()
            if replacement_report is not None and hasattr(replacement_report, "to_dict")
            else {}
        )
        replaced = replacement_payload.get("replaced") if isinstance(replacement_payload.get("replaced"), list) else []
        dense_model = getattr(model, "dense_model", model)
        config = getattr(model, "config", None)
        configured_layers = config.layer_ids() if hasattr(config, "layer_ids") else []
        token_router_layers = []
        replaced_layers = []
        for item in replaced:
            if not isinstance(item, dict):
                continue
            layer = int(item.get("layer"))
            replaced_layers.append(layer)
            module_path = str(item.get("module_path", ""))
            module = dense_model.get_submodule(module_path) if module_path else None
            if module is not None and getattr(module, "token_router", None) is not None:
                token_router_layers.append(
                    {
                        "layer": layer,
                        "module_path": module_path,
                        "top_k": getattr(module, "token_router_top_k", None),
                        "parameter_count": sum(parameter.numel() for parameter in module.token_router.parameters()),
                    }
                )
        if configured_layers and sorted(replaced_layers) != sorted(int(layer) for layer in configured_layers):
            errors.append("native AutoModel reload did not replace every configured layer")
        if getattr(config, "token_router_path", None) and len(token_router_layers) != len(replaced_layers):
            errors.append("native AutoModel reload did not load token routers for every replaced layer")
        return {
            "status": "loaded",
            "model_class": model.__class__.__name__,
            "dense_model_class": dense_model.__class__.__name__,
            "configured_layers": [int(layer) for layer in configured_layers],
            "replaced_layer_count": len(replaced_layers),
            "replaced_layers": replaced_layers,
            "token_router_layer_count": len(token_router_layers),
            "token_router_layers": token_router_layers,
        }
    except Exception as exc:
        errors.append(f"native AutoModel reload failed: {exc}")
        return {"status": "error", "error": str(exc)}


def _updated_tensor_names(export_report: dict[str, Any] | None) -> set[str]:
    if export_report is None or not isinstance(export_report.get("updated_tensors"), list):
        return set()
    return {
        str(item["tensor"])
        for item in export_report["updated_tensors"]
        if isinstance(item, dict) and item.get("tensor") is not None
    }


def _finish_validation_report(*, report: dict[str, Any], output_path: Path | None) -> dict[str, Any]:
    report["passed"] = not bool(report.get("errors"))
    report["status"] = "validated" if report["passed"] else "error"
    if output_path is not None:
        _write_json(output_path, report)
    return report


def _load_optional_json(path: Path, *, label: str, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        warnings.append(f"{label} not found: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        warnings.append(f"{label} is not a JSON object: {path}")
        return None
    return payload


def _path_or_none(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _torch_dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _require_file(path: Path, *, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} not found: {path}")


def _promoted_tensor_map(checkpoint: dict[str, Any]) -> dict[str, str]:
    promoted = checkpoint.get("promoted_carved_parameters")
    if not isinstance(promoted, list):
        raise RecoveryRunError("recovery checkpoint metadata is missing promoted_carved_parameters")
    tensor_map = {}
    for item in promoted:
        if isinstance(item, dict) and item.get("parameter") and item.get("tensor"):
            tensor_map[str(item["parameter"])] = str(item["tensor"])
    return tensor_map


def _promoted_router_tensor_map(checkpoint: dict[str, Any]) -> dict[str, str]:
    promoted = checkpoint.get("promoted_router_parameters")
    if not isinstance(promoted, list):
        return {}
    tensor_map = {}
    for item in promoted:
        if isinstance(item, dict) and item.get("parameter") and item.get("tensor"):
            tensor_map[str(item["parameter"])] = str(item["tensor"])
    return tensor_map


def _tensor_name_for_parameter(parameter_name: str, tensor_map: dict[str, str]) -> str | None:
    for parameter, tensor in tensor_map.items():
        if parameter_name == parameter or parameter_name.endswith(f".{parameter}"):
            return tensor
    return None


def _copy_wrapper_scaffold(*, source_dir: Path, output_dir: Path, skip_files: set[str]) -> None:
    if source_dir.resolve() == output_dir.resolve():
        raise RecoveryRunError("recovery-export output_dir must differ from source wrapper")
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name in skip_files:
            continue
        destination = output_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def _rewrite_wrapper_artifact_refs(
    *,
    output_dir: Path,
    artifact_name: str,
    token_router_name: str | None,
    checkpoint_path: Path,
) -> None:
    config_path = output_dir / "moeforge_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["artifact_path"] = artifact_name
    if token_router_name is not None:
        config["token_router_path"] = token_router_name
    config.setdefault("warnings", [])
    if isinstance(config["warnings"], list):
        config["warnings"].append(f"recovered tensors applied from {checkpoint_path}")
    _write_json(config_path, config)

    hf_config_path = output_dir / "config.json"
    if hf_config_path.exists():
        hf_config = json.loads(hf_config_path.read_text(encoding="utf-8"))
        hf_config["artifact_path"] = artifact_name
        if token_router_name is not None:
            hf_config["token_router_path"] = token_router_name
        _write_json(hf_config_path, hf_config)


def _before_after_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    before_after = _dict(plan.get("before_after_eval"))
    if before_after.get("status") == "compared":
        before = before_after.get("before_manifest")
        after = before_after.get("after_manifest")
        if before and after:
            return compare_eval_batch_manifests(before_path=Path(str(before)), after_path=Path(str(after)))
    return before_after or {"status": "not_configured"}


def _resolve_device(device: str, *, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_package_path(package_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return package_dir / path


def _parameter_count(parameters: list[Any]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecoveryRunError("recovery plan must be a JSON object")
    if payload.get("format") != "moeforge_recovery_plan":
        raise RecoveryRunError("recovery-run requires a moeforge_recovery_plan artifact")
    return payload


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
