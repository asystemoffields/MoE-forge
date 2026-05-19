from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_hf_dense_vs_carved
from .reports import write_eval_comparison_report, write_eval_html_report_payload


class EvalBatchError(RuntimeError):
    """Raised when an eval-batch config cannot be run."""


EXPERT_MODES = ("all", "default-pool", "router", "learned-router")


def run_eval_batch(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    strict: bool | None = None,
    evaluator: Any = evaluate_hf_dense_vs_carved,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    base_dir = config_path.parent
    model = _required_path(config, "model", base_dir=base_dir)
    wrapper = _required_path(config, "wrapper", base_dir=base_dir)
    run_dir = _output_dir(config, output_dir=output_dir, base_dir=base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    expert_modes = _expert_modes(config)
    batch_strict = bool(config.get("strict", False)) if strict is None else strict
    write_html = bool(config.get("write_html", config.get("html", True)))
    stop_on_error = bool(config.get("stop_on_error", False))
    input_ids = _input_ids(config, base_dir=base_dir)
    texts = _texts(config, base_dir=base_dir)
    if input_ids is not None and texts is not None:
        raise EvalBatchError("eval-batch config can provide input_ids or text samples, not both")

    eval_options = {
        "sequence_length": int(config.get("sequence_length", 128)),
        "device": str(config.get("device", "cpu")),
        "atol": float(config.get("atol", 1e-5)),
        "rtol": float(config.get("rtol", 1e-5)),
    }
    runs: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    warnings: list[str] = []

    for mode in expert_modes:
        slug = _mode_slug(mode)
        report_path = run_dir / f"eval-{slug}.json"
        html_path = run_dir / f"eval-{slug}.html"
        try:
            report = evaluator(
                model=model,
                package_dir=wrapper,
                texts=texts,
                input_ids=input_ids,
                expert_mode=mode,
                **eval_options,
            ).to_dict()
            _write_json(report_path, report)
            if write_html:
                write_eval_html_report_payload(report=report, output_path=html_path)
            report_paths.append(report_path)
            runs.append(
                _run_record(
                    mode=mode,
                    report=report,
                    report_path=report_path,
                    html_path=html_path if write_html else None,
                )
            )
        except Exception as exc:
            error_path = run_dir / f"eval-{slug}-error.json"
            error_payload = {
                "format": "moeforge_eval_batch_error",
                "expert_mode": mode,
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            }
            _write_json(error_path, error_payload)
            runs.append(
                {
                    "expert_mode": mode,
                    "status": "error",
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                    "error_path": str(error_path),
                }
            )
            if stop_on_error:
                raise EvalBatchError(f"expert_mode={mode} failed: {exc}") from exc

    comparison = _write_comparison(
        run_dir=run_dir,
        report_paths=report_paths,
        write_html=write_html,
        warnings=warnings,
    )
    manifest = {
        "format": "moeforge_eval_batch",
        "config_path": str(config_path),
        "model": str(model),
        "wrapper": str(wrapper),
        "output_dir": str(run_dir),
        "expert_modes": expert_modes,
        "sample_source": _sample_source(
            config=config,
            input_ids=input_ids,
            texts=texts,
            base_dir=base_dir,
        ),
        "evaluation": {
            **eval_options,
            "strict": batch_strict,
            "write_html": write_html,
            "stop_on_error": stop_on_error,
        },
        "recovery_eval": _recovery_eval_plan(config=config, model=model),
        "run_count": len(runs),
        "completed_report_count": len(report_paths),
        "runs": runs,
        "comparison": comparison,
        "passed": bool(runs) and all(run.get("status") == "passed" for run in runs),
        "warnings": warnings,
    }
    _write_json(run_dir / "eval-batch-manifest.json", manifest)
    return manifest


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvalBatchError("eval-batch config must be a JSON object")
    return payload


def _required_path(config: dict[str, Any], key: str, *, base_dir: Path) -> Path:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise EvalBatchError(f"eval-batch config requires {key}")
    return _resolve_path(Path(str(value)), base_dir=base_dir)


def _output_dir(config: dict[str, Any], *, output_dir: Path | None, base_dir: Path) -> Path:
    if output_dir is not None:
        return _resolve_path(output_dir, base_dir=Path.cwd())
    configured = config.get("output_dir")
    if configured is not None and str(configured).strip():
        return _resolve_path(Path(str(configured)), base_dir=base_dir)
    return base_dir / "eval-batch"


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _expert_modes(config: dict[str, Any]) -> list[str]:
    raw_modes = config.get("expert_modes", list(EXPERT_MODES))
    if isinstance(raw_modes, str):
        raw_modes = [raw_modes]
    if not isinstance(raw_modes, list) or not raw_modes:
        raise EvalBatchError("expert_modes must be a non-empty list")
    modes = [str(mode) for mode in raw_modes]
    unknown = sorted(set(modes) - set(EXPERT_MODES))
    if unknown:
        raise EvalBatchError(f"expert_modes contains unsupported modes: {', '.join(unknown)}")
    return modes


def _input_ids(config: dict[str, Any], *, base_dir: Path) -> list[list[int]] | None:
    configured = [
        key
        for key in ("input_ids", "input_ids_json", "input_ids_file")
        if config.get(key) is not None
    ]
    if len(configured) > 1:
        raise EvalBatchError(
            "eval-batch config can provide only one of input_ids, input_ids_json, or input_ids_file"
        )
    if config.get("input_ids_file") is not None:
        path = _resolve_path(Path(str(config["input_ids_file"])), base_dir=base_dir)
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif "input_ids_json" in config:
        raw = json.loads(str(config["input_ids_json"]))
    else:
        raw = config.get("input_ids")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise EvalBatchError("input_ids must be a list of token-id lists")
    normalized = []
    for sample in raw:
        if not isinstance(sample, list):
            raise EvalBatchError("input_ids samples must be token-id lists")
        try:
            normalized.append([int(item) for item in sample])
        except (TypeError, ValueError) as exc:
            raise EvalBatchError("input_ids samples must contain integer token ids") from exc
    return normalized


def _texts(config: dict[str, Any], *, base_dir: Path) -> list[str] | None:
    samples: list[str] = []
    if config.get("text"):
        samples.append(str(config["text"]))
    raw_texts = config.get("texts")
    if raw_texts is not None:
        if not isinstance(raw_texts, list):
            raise EvalBatchError("texts must be a list of strings")
        samples.extend(str(item) for item in raw_texts)
    if config.get("text_file"):
        path = _resolve_path(Path(str(config["text_file"])), base_dir=base_dir)
        content = path.read_text(encoding="utf-8")
        samples.extend(chunk.strip() for chunk in content.split("\n\n") if chunk.strip())
    return samples or None


def _run_record(
    *,
    mode: str,
    report: dict[str, Any],
    report_path: Path,
    html_path: Path | None,
) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    record = {
        "expert_mode": mode,
        "status": "passed" if report.get("passed") else "failed",
        "report_path": str(report_path),
        "html_path": str(html_path) if html_path is not None else None,
        "passed": bool(report.get("passed")),
        "sample_count": int(report.get("sample_count") or 0),
        "max_abs_error": report.get("max_abs_error"),
        "mean_abs_error": report.get("mean_abs_error"),
        "latency_ratio": summary.get("average_carved_vs_dense_latency_ratio"),
        "teacher_kl_loss": summary.get("average_teacher_kl_loss"),
        "dense_nll_loss": summary.get("average_dense_nll_loss"),
        "carved_nll_loss": summary.get("average_carved_nll_loss"),
        "nll_loss_delta": summary.get("average_nll_loss_delta"),
        "loss_token_count": summary.get("loss_token_count"),
        "worst_layer": summary.get("worst_layer"),
        "worst_layer_selected_vs_all_max_abs_error": summary.get(
            "worst_layer_selected_vs_all_max_abs_error"
        ),
    }
    return record


def _write_comparison(
    *,
    run_dir: Path,
    report_paths: list[Path],
    write_html: bool,
    warnings: list[str],
) -> dict[str, Any]:
    if len(report_paths) < 2:
        warnings.append("eval comparison requires at least two completed eval reports")
        return {
            "status": "skipped",
            "reason": "fewer than two completed eval reports",
            "completed_report_count": len(report_paths),
        }
    comparison_path = run_dir / "eval-compare.json"
    comparison_html_path = run_dir / "eval-compare.html" if write_html else None
    comparison = write_eval_comparison_report(
        report_paths=report_paths,
        output_path=comparison_path,
        html_output_path=comparison_html_path,
    )
    return {
        "status": "written",
        "json_path": str(comparison_path),
        "html_path": str(comparison_html_path) if comparison_html_path is not None else None,
        "report_count": comparison.get("report_count"),
        "best": comparison.get("best"),
        "fastest": comparison.get("fastest"),
        "lowest_error": comparison.get("lowest_error"),
    }


def _sample_source(
    *,
    config: dict[str, Any],
    input_ids: list[list[int]] | None,
    texts: list[str] | None,
    base_dir: Path,
) -> dict[str, Any]:
    if input_ids is not None:
        source: dict[str, Any] = {
            "kind": "input_ids",
            "sample_count": len(input_ids),
            "sha256": _sha256_json(input_ids),
            "sample_sha256": [_sha256_json(sample) for sample in input_ids],
        }
        if config.get("input_ids_file") is not None:
            source["input_ids_file"] = _file_identity(
                Path(str(config["input_ids_file"])),
                base_dir=base_dir,
            )
        elif "input_ids_json" in config:
            source["source"] = "input_ids_json"
        else:
            source["source"] = "inline_input_ids"
        return source
    if texts is not None:
        source = {
            "kind": "text",
            "sample_count": len(texts),
            "sha256": _sha256_json(texts),
            "sample_sha256": [_sha256_text(text) for text in texts],
        }
        if config.get("text_file"):
            source["text_file"] = _file_identity(Path(str(config["text_file"])), base_dir=base_dir)
            source["chunk_count"] = len(texts)
        return source
    return {"kind": "generated_smoke_input_ids", "sample_count": None}


def _recovery_eval_plan(*, config: dict[str, Any], model: Path) -> dict[str, Any]:
    raw = config.get("recovery_eval", config.get("recovery", {}))
    recovery = raw if isinstance(raw, dict) else {}
    metrics = recovery.get("metrics", ["logits_parity", "teacher_kl"])
    if not isinstance(metrics, list):
        metrics = ["logits_parity", "teacher_kl"]
    return {
        "enabled": bool(recovery.get("enabled", False)),
        "teacher_model": str(recovery.get("teacher_model", model)),
        "metrics": [str(metric) for metric in metrics],
        "notes": [
            "Batch eval records dense-teacher settings for recovery-training runs.",
            "Eval reports include dense-teacher KL and next-token NLL summaries for before/after recovery comparisons.",
        ],
    }


def _mode_slug(mode: str) -> str:
    return mode.replace("-", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_identity(path: Path, *, base_dir: Path) -> dict[str, Any]:
    resolved = _resolve_path(path, base_dir=base_dir)
    data = resolved.read_bytes()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "byte_count": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
