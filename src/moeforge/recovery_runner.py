from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from .hf_runtime import MoEForgeCarvedMLPModule, replace_hf_mlp_modules
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
        from transformers import AutoModelForCausalLM
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
    train_samples = _input_id_samples(plan)
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
    if _dict(student_config.get("trainable")).get("router", True):
        warnings.append("router training is recorded in the plan; the current runner trains carved tensor parameters")
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
        "trainable_parameter_count": _parameter_count(trainable_parameters),
        "promoted_carved_parameters": promoted,
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
    for parameter_name, value in trainable_state.items():
        tensor_name = _tensor_name_for_parameter(str(parameter_name), tensor_map)
        if tensor_name is None:
            continue
        if tensor_name not in tensors:
            raise RecoveryRunError(f"checkpoint tensor is not present in wrapper artifact: {tensor_name}")
        tensors[tensor_name] = value.detach().cpu().contiguous()
        updated.append(
            {
                "parameter": str(parameter_name),
                "tensor": tensor_name,
                "shape": list(value.shape),
            }
        )
    if not updated:
        raise RecoveryRunError("recovery checkpoint did not contain carved tensor parameters to export")

    _copy_wrapper_scaffold(source_dir=wrapper_dir, output_dir=output_dir, skip_files={source_artifact.name})
    artifact_path = output_dir / artifact_name
    save_file(
        tensors,
        str(artifact_path),
        metadata={
            "moe_forge": "recovered_carved_experts",
            "source_artifact": str(source_artifact),
            "checkpoint": str(checkpoint_path),
        },
    )
    _rewrite_wrapper_artifact_refs(
        output_dir=output_dir,
        artifact_name=artifact_name,
        checkpoint_path=checkpoint_path,
    )
    report = {
        "format": "moeforge_recovery_export",
        "checkpoint_path": str(checkpoint_path),
        "state_path": str(state_path),
        "source_wrapper": str(wrapper_dir),
        "output_dir": str(output_dir),
        "artifact_path": str(artifact_path),
        "updated_tensor_count": len(updated),
        "updated_tensors": updated,
        "wrapper_config": str(output_dir / "moeforge_config.json"),
    }
    _write_json(output_dir / "recovery-export-report.json", report)
    return report


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


def _input_id_samples(plan: dict[str, Any]) -> list[list[int]]:
    train = _dict(_dict(plan.get("samples")).get("train"))
    if train.get("kind") != "input_ids":
        raise RecoveryRunError("recovery-run currently requires train samples with kind=input_ids")
    samples = train.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RecoveryRunError("recovery-run requires at least one train input-id sample")
    input_ids = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("input_ids"), list):
            raise RecoveryRunError("train input-id samples must include input_ids")
        input_ids.append([int(item) for item in sample["input_ids"]])
    return input_ids


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


def _promoted_tensor_map(checkpoint: dict[str, Any]) -> dict[str, str]:
    promoted = checkpoint.get("promoted_carved_parameters")
    if not isinstance(promoted, list):
        raise RecoveryRunError("recovery checkpoint metadata is missing promoted_carved_parameters")
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


def _rewrite_wrapper_artifact_refs(*, output_dir: Path, artifact_name: str, checkpoint_path: Path) -> None:
    config_path = output_dir / "moeforge_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["artifact_path"] = artifact_name
    config.setdefault("warnings", [])
    if isinstance(config["warnings"], list):
        config["warnings"].append(f"recovered tensors applied from {checkpoint_path}")
    _write_json(config_path, config)

    hf_config_path = output_dir / "config.json"
    if hf_config_path.exists():
        hf_config = json.loads(hf_config_path.read_text(encoding="utf-8"))
        hf_config["artifact_path"] = artifact_name
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
