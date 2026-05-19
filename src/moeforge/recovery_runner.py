from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hf_runtime import MoEForgeCarvedMLPModule, replace_hf_mlp_modules
from .recovery import compare_eval_batch_manifests


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
        "state_path": str(state_path),
        "plan_path": plan.get("artifacts", {}).get("plan_path"),
        "trainable_parameter_count": _parameter_count(trainable_parameters),
        "saved_tensor_count": len(trainable_state),
        "promoted_carved_parameter_count": len(promoted),
    }
    _write_json(metadata_path, metadata)
    return metadata


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
