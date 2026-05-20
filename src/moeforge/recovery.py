from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class RecoveryPlanError(RuntimeError):
    """Raised when a recovery-training plan cannot be built."""


def build_recovery_plan(*, config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    base_dir = config_path.parent
    output_dir = _output_dir(config=config, base_dir=base_dir)
    teacher_model = _required_model_ref(config, "teacher_model", base_dir=base_dir)
    wrapper = _required_path(config, "wrapper", base_dir=base_dir)
    student_model = _model_ref(config.get("student_model", teacher_model), base_dir=base_dir)
    warnings: list[str] = []
    train_samples = _sample_manifest(
        section=_dict(config.get("train")),
        split="train",
        base_dir=base_dir,
        warnings=warnings,
    )
    eval_samples = _sample_manifest(
        section=_dict(config.get("eval")),
        split="eval",
        base_dir=base_dir,
        warnings=warnings,
    )
    before_after = _before_after_eval(config=config, base_dir=base_dir)
    return {
        "format": "moeforge_recovery_plan",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "teacher": _teacher_config(config=config, teacher_model=teacher_model),
        "student": {
            "model": str(student_model),
            "wrapper": str(wrapper),
            "trainable": _trainable_config(config),
        },
        "loss": _loss_config(config),
        "optimizer": _optimizer_config(config),
        "schedule": _schedule_config(config),
        "checkpoints": _checkpoint_config(config=config, output_dir=output_dir),
        "samples": {
            "train": train_samples,
            "eval": eval_samples,
        },
        "before_after_eval": before_after,
        "artifacts": {
            "plan_path": None,
            "before_after_comparison_path": None,
        },
        "status": "planned",
        "warnings": warnings,
    }


def write_recovery_plan(*, config_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    plan = build_recovery_plan(config_path=config_path)
    target = output_path or Path(str(plan["output_dir"])) / "recovery-plan.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    comparison = _dict(plan.get("before_after_eval"))
    if comparison.get("status") == "compared":
        comparison_path = target.parent / "recovery-before-after.json"
        _write_json(comparison_path, comparison)
        plan["artifacts"]["before_after_comparison_path"] = str(comparison_path)
    plan["artifacts"]["plan_path"] = str(target)
    _write_json(target, plan)
    return plan


def compare_eval_batch_manifests(*, before_path: Path, after_path: Path) -> dict[str, Any]:
    before = _load_json_object(before_path, label="before eval-batch manifest")
    after = _load_json_object(after_path, label="after eval-batch manifest")
    before_runs = _runs_by_mode(before)
    after_runs = _runs_by_mode(after)
    modes = sorted(set(before_runs) | set(after_runs))
    deltas = [
        _mode_delta(
            mode=mode,
            before=before_runs.get(mode),
            after=after_runs.get(mode),
        )
        for mode in modes
    ]
    comparable = [item for item in deltas if item["status"] == "compared"]
    improved = [
        item
        for item in comparable
        if _numeric(item.get("max_abs_error_delta")) is not None
        and _numeric(item.get("max_abs_error_delta")) < 0
    ]
    regressed = [
        item
        for item in comparable
        if _numeric(item.get("max_abs_error_delta")) is not None
        and _numeric(item.get("max_abs_error_delta")) > 0
    ]
    kl_improved = [
        item
        for item in comparable
        if _numeric(item.get("teacher_kl_loss_delta")) is not None
        and _numeric(item.get("teacher_kl_loss_delta")) < 0
    ]
    kl_regressed = [
        item
        for item in comparable
        if _numeric(item.get("teacher_kl_loss_delta")) is not None
        and _numeric(item.get("teacher_kl_loss_delta")) > 0
    ]
    return {
        "format": "moeforge_recovery_before_after_eval",
        "status": "compared",
        "before_manifest": str(before_path),
        "after_manifest": str(after_path),
        "mode_count": len(modes),
        "compared_mode_count": len(comparable),
        "summary": {
            "improved_modes_by_max_abs_error": len(improved),
            "regressed_modes_by_max_abs_error": len(regressed),
            "unchanged_or_unscored_modes": len(comparable) - len(improved) - len(regressed),
            "improved_modes_by_teacher_kl": len(kl_improved),
            "regressed_modes_by_teacher_kl": len(kl_regressed),
        },
        "mode_deltas": deltas,
    }


def _teacher_config(*, config: dict[str, Any], teacher_model: str) -> dict[str, Any]:
    teacher = _dict(config.get("teacher"))
    return {
        "model": str(teacher.get("model", teacher_model)),
        "dtype": str(teacher.get("dtype", config.get("dtype", "auto"))),
        "device": str(teacher.get("device", config.get("device", "auto"))),
        "logits_dtype": str(teacher.get("logits_dtype", "fp32")),
    }


def _trainable_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _dict(config.get("trainable"))
    return {
        "experts": bool(raw.get("experts", True)),
        "router": bool(raw.get("router", True)),
        "shared": bool(raw.get("shared", False)),
        "dense_backbone": bool(raw.get("dense_backbone", False)),
    }


def _loss_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _dict(config.get("loss"))
    loss = {
        "teacher_kl_weight": float(raw.get("teacher_kl_weight", 1.0)),
        "logits_mse_weight": float(raw.get("logits_mse_weight", 0.0)),
        "router_oracle_weight": float(raw.get("router_oracle_weight", 0.0)),
        "router_balance_weight": float(raw.get("router_balance_weight", 0.01)),
        "z_loss_weight": float(raw.get("z_loss_weight", 0.0)),
        "temperature": float(raw.get("temperature", 1.0)),
    }
    if loss["temperature"] <= 0:
        raise RecoveryPlanError("loss.temperature must be greater than zero")
    for key, value in loss.items():
        if key != "temperature" and value < 0:
            raise RecoveryPlanError(f"loss.{key} must be non-negative")
    return loss


def _optimizer_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _dict(config.get("optimizer"))
    return {
        "name": str(raw.get("name", "adamw")),
        "learning_rate": float(raw.get("learning_rate", 5e-5)),
        "weight_decay": float(raw.get("weight_decay", 0.0)),
        "betas": _float_list(raw.get("betas", [0.9, 0.95]), expected=2),
        "max_grad_norm": float(raw.get("max_grad_norm", 1.0)),
    }


def _schedule_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = _dict(config.get("schedule"))
    steps = int(raw.get("steps", config.get("steps", 100)))
    if steps <= 0:
        raise RecoveryPlanError("schedule.steps must be greater than zero")
    return {
        "steps": steps,
        "warmup_steps": int(raw.get("warmup_steps", 0)),
        "batch_size": int(raw.get("batch_size", 1)),
        "gradient_accumulation_steps": int(raw.get("gradient_accumulation_steps", 1)),
        "eval_every_steps": int(raw.get("eval_every_steps", max(1, steps // 4))),
        "save_every_steps": int(raw.get("save_every_steps", max(1, steps // 4))),
    }


def _checkpoint_config(*, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    raw = _dict(config.get("checkpoints"))
    checkpoint_dir = _resolve_path(Path(str(raw.get("output_dir", "checkpoints"))), base_dir=output_dir)
    return {
        "output_dir": str(checkpoint_dir),
        "keep_last": int(raw.get("keep_last", 2)),
        "save_format": str(raw.get("save_format", "hf_wrapper")),
        "save_optimizer": bool(raw.get("save_optimizer", True)),
    }


def _sample_manifest(
    *,
    section: dict[str, Any],
    split: str,
    base_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    input_ids = _input_ids(section, base_dir=base_dir)
    texts = _texts(section, base_dir=base_dir)
    if input_ids is not None and texts is not None:
        raise RecoveryPlanError(f"{split} samples can provide input_ids or text samples, not both")
    sequence_length = int(section.get("sequence_length", 128))
    max_samples = section.get("max_samples")
    if input_ids is not None:
        samples = [
            {
                "index": index,
                "source": f"input_ids:{index}",
                "input_ids": sample,
                "token_count": len(sample),
                "sha256": _sha256_json(sample),
            }
            for index, sample in enumerate(input_ids)
        ]
        return {
            "kind": "input_ids",
            "split": split,
            "sample_count": len(samples),
            "sequence_length": sequence_length,
            "max_samples": max_samples,
            "source": _input_id_source(section, input_ids=input_ids, base_dir=base_dir),
            "samples": samples,
        }
    if texts is not None:
        samples = [
            {
                "index": index,
                "source": item["source"],
                "char_count": len(item["text"]),
                "sha256": _sha256_text(item["text"]),
            }
            for index, item in enumerate(texts)
        ]
        return {
            "kind": "text",
            "split": split,
            "sample_count": len(samples),
            "sequence_length": sequence_length,
            "max_samples": max_samples,
            "source": _text_source(section, texts=texts, base_dir=base_dir),
            "samples": samples,
        }
    warnings.append(f"{split} samples are unspecified; recovery runner should provide data before training")
    return {
        "kind": "unspecified",
        "split": split,
        "sample_count": 0,
        "sequence_length": sequence_length,
        "max_samples": max_samples,
        "samples": [],
    }


def _before_after_eval(*, config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    before = config.get("before_eval_batch")
    after = config.get("after_eval_batch")
    if before is None and after is None:
        return {
            "status": "not_configured",
            "notes": ["Add before_eval_batch and after_eval_batch manifest paths to compare recovery impact."],
        }
    if before is None or after is None:
        raise RecoveryPlanError("before_after evaluation requires both before_eval_batch and after_eval_batch")
    return compare_eval_batch_manifests(
        before_path=_resolve_path(Path(str(before)), base_dir=base_dir),
        after_path=_resolve_path(Path(str(after)), base_dir=base_dir),
    )


def _mode_delta(
    *,
    mode: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    if before is None or after is None:
        return {
            "expert_mode": mode,
            "status": "missing",
            "before_status": before.get("status") if before else None,
            "after_status": after.get("status") if after else None,
        }
    return {
        "expert_mode": mode,
        "status": "compared",
        "before_status": before.get("status"),
        "after_status": after.get("status"),
        "max_abs_error_before": before.get("max_abs_error"),
        "max_abs_error_after": after.get("max_abs_error"),
        "max_abs_error_delta": _delta(before.get("max_abs_error"), after.get("max_abs_error")),
        "mean_abs_error_before": before.get("mean_abs_error"),
        "mean_abs_error_after": after.get("mean_abs_error"),
        "mean_abs_error_delta": _delta(before.get("mean_abs_error"), after.get("mean_abs_error")),
        "latency_ratio_before": before.get("latency_ratio"),
        "latency_ratio_after": after.get("latency_ratio"),
        "latency_ratio_delta": _delta(before.get("latency_ratio"), after.get("latency_ratio")),
        "teacher_kl_loss_before": before.get("teacher_kl_loss"),
        "teacher_kl_loss_after": after.get("teacher_kl_loss"),
        "teacher_kl_loss_delta": _delta(before.get("teacher_kl_loss"), after.get("teacher_kl_loss")),
        "dense_nll_loss_before": before.get("dense_nll_loss"),
        "dense_nll_loss_after": after.get("dense_nll_loss"),
        "dense_nll_loss_delta": _delta(before.get("dense_nll_loss"), after.get("dense_nll_loss")),
        "carved_nll_loss_before": before.get("carved_nll_loss"),
        "carved_nll_loss_after": after.get("carved_nll_loss"),
        "carved_nll_loss_delta": _delta(before.get("carved_nll_loss"), after.get("carved_nll_loss")),
        "nll_loss_delta_before": before.get("nll_loss_delta"),
        "nll_loss_delta_after": after.get("nll_loss_delta"),
        "nll_loss_delta_delta": _delta(before.get("nll_loss_delta"), after.get("nll_loss_delta")),
        "loss_token_count_before": before.get("loss_token_count"),
        "loss_token_count_after": after.get("loss_token_count"),
    }


def _runs_by_mode(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise RecoveryPlanError("eval-batch manifest must include a runs list")
    result = {}
    for run in runs:
        if isinstance(run, dict) and run.get("expert_mode") is not None:
            result[str(run["expert_mode"])] = run
    return result


def _load_config(path: Path) -> dict[str, Any]:
    return _load_json_object(path, label="recovery config")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecoveryPlanError(f"{label} must be a JSON object")
    return payload


def _required_model_ref(config: dict[str, Any], key: str, *, base_dir: Path) -> str:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise RecoveryPlanError(f"recovery config requires {key}")
    return _model_ref(value, base_dir=base_dir)


def _model_ref(value: Any, *, base_dir: Path) -> str:
    raw = str(value)
    path = Path(raw)
    if path.is_absolute() or path.exists() or (base_dir / path).exists():
        return str(_resolve_path(path, base_dir=base_dir))
    return raw


def _required_path(config: dict[str, Any], key: str, *, base_dir: Path) -> Path:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise RecoveryPlanError(f"recovery config requires {key}")
    return _resolve_path(Path(str(value)), base_dir=base_dir)


def _output_dir(*, config: dict[str, Any], base_dir: Path) -> Path:
    value = config.get("output_dir", "recovery-run")
    return _resolve_path(Path(str(value)), base_dir=base_dir)


def _input_ids(section: dict[str, Any], *, base_dir: Path) -> list[list[int]] | None:
    configured = [
        key
        for key in ("input_ids", "input_ids_json", "input_ids_file")
        if section.get(key) is not None
    ]
    if len(configured) > 1:
        raise RecoveryPlanError(
            "samples can provide only one of input_ids, input_ids_json, or input_ids_file"
        )
    if section.get("input_ids_file") is not None:
        path = _resolve_path(Path(str(section["input_ids_file"])), base_dir=base_dir)
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif "input_ids_json" in section:
        raw = json.loads(str(section["input_ids_json"]))
    else:
        raw = section.get("input_ids")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise RecoveryPlanError("input_ids must be a list of token-id lists")
    normalized = []
    for sample in raw:
        if not isinstance(sample, list):
            raise RecoveryPlanError("input_ids samples must be token-id lists")
        normalized.append([int(item) for item in sample])
    return normalized


def _texts(section: dict[str, Any], *, base_dir: Path) -> list[dict[str, str]] | None:
    samples: list[dict[str, str]] = []
    if section.get("text"):
        samples.append({"source": "text:0", "text": str(section["text"])})
    raw_texts = section.get("texts")
    if raw_texts is not None:
        if not isinstance(raw_texts, list):
            raise RecoveryPlanError("texts must be a list of strings")
        offset = len(samples)
        samples.extend(
            {"source": f"text:{index + offset}", "text": str(item)}
            for index, item in enumerate(raw_texts)
        )
    if section.get("text_file"):
        path = _resolve_path(Path(str(section["text_file"])), base_dir=base_dir)
        content = path.read_text(encoding="utf-8")
        offset = len(samples)
        samples.extend(
            {"source": f"{path}:chunk:{index}", "text": chunk.strip()}
            for index, chunk in enumerate(content.split("\n\n"), start=offset)
            if chunk.strip()
        )
    return samples or None


def _float_list(value: Any, *, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise RecoveryPlanError(f"expected a list of {expected} floats")
    return [float(item) for item in value]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _input_id_source(
    section: dict[str, Any],
    *,
    input_ids: list[list[int]],
    base_dir: Path,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "kind": "input_ids",
        "sample_count": len(input_ids),
        "sha256": _sha256_json(input_ids),
        "sample_sha256": [_sha256_json(sample) for sample in input_ids],
    }
    if section.get("input_ids_file") is not None:
        source["input_ids_file"] = _file_identity(
            Path(str(section["input_ids_file"])),
            base_dir=base_dir,
        )
    elif "input_ids_json" in section:
        source["source"] = "input_ids_json"
    else:
        source["source"] = "inline_input_ids"
    return source


def _file_identity(path: Path, *, base_dir: Path) -> dict[str, Any]:
    resolved = _resolve_path(path, base_dir=base_dir)
    data = resolved.read_bytes()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _text_source(
    section: dict[str, Any],
    *,
    texts: list[dict[str, str]],
    base_dir: Path,
) -> dict[str, Any]:
    values = [item["text"] for item in texts]
    source: dict[str, Any] = {
        "kind": "text",
        "sample_count": len(values),
        "sha256": _sha256_json(values),
        "sample_sha256": [_sha256_text(value) for value in values],
    }
    if section.get("text_file") is not None:
        source["text_file"] = _file_identity(Path(str(section["text_file"])), base_dir=base_dir)
    elif section.get("texts") is not None:
        source["source"] = "inline_texts"
    else:
        source["source"] = "inline_text"
    return source


def _delta(before: Any, after: Any) -> float | None:
    before_number = _numeric(before)
    after_number = _numeric(after)
    if before_number is None or after_number is None:
        return None
    return after_number - before_number


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
