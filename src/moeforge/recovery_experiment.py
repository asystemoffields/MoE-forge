from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from .batch import run_eval_batch
from .evaluation import evaluate_hf_dense_vs_carved
from .recovery import compare_eval_batch_manifests, write_recovery_plan
from .recovery_runner import export_recovered_wrapper, run_recovery, validate_recovered_wrapper


class RecoveryExperimentError(RuntimeError):
    """Raised when a recovery experiment cannot be orchestrated."""


def run_recovery_experiment(
    *,
    config_path: Path,
    output_dir: Path | None = None,
    max_steps: int | None = None,
    evaluator: Any = evaluate_hf_dense_vs_carved,
    recovery_runner: Any = run_recovery,
    exporter: Any = export_recovered_wrapper,
    validator: Any = validate_recovered_wrapper,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_config(config_path)
    base_dir = config_path.parent
    run_dir = _output_dir(config=config, output_dir=output_dir, base_dir=base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _model_ref(config, base_dir=base_dir)
    wrapper = _required_path(config, "wrapper", base_dir=base_dir)
    recovered_wrapper = run_dir / "recovered-wrapper"
    eval_config = _resolve_sample_file_refs(_dict(config.get("eval")), base_dir=base_dir)
    train_config = _resolve_sample_file_refs(_dict(config.get("train")), base_dir=base_dir)

    before_config_path = run_dir / "before" / "eval-batch-config.json"
    before_config = _eval_batch_config(
        config=eval_config,
        model=model,
        wrapper=wrapper,
        output_dir=run_dir / "before",
        recovery_teacher_model=model,
    )
    _write_json(before_config_path, before_config)
    before_manifest = run_eval_batch(config_path=before_config_path, evaluator=evaluator)

    recovery_config_path = run_dir / "recovery" / "recovery-config.json"
    recovery_plan_path = run_dir / "recovery" / "recovery-plan.json"
    recovery_config = _recovery_config(
        config=config,
        eval_config=eval_config,
        train_config=train_config,
        model=model,
        wrapper=wrapper,
        output_dir=run_dir / "recovery",
    )
    _write_json(recovery_config_path, recovery_config)
    recovery_plan = write_recovery_plan(config_path=recovery_config_path, output_path=recovery_plan_path)
    recovery_report = recovery_runner(
        plan_path=Path(str(recovery_plan["artifacts"]["plan_path"])),
        output_path=run_dir / "recovery" / "recovery-run-report.json",
        max_steps=max_steps,
    )
    checkpoint_path = _last_checkpoint_path(recovery_report)
    export_report = exporter(
        checkpoint_path=checkpoint_path,
        wrapper_dir=wrapper,
        output_dir=recovered_wrapper,
    )
    validation_path = run_dir / "recovered-wrapper-validation.json"
    validation_report = validator(
        source_wrapper=wrapper,
        recovered_wrapper=recovered_wrapper,
        checkpoint_path=checkpoint_path,
        export_report_path=Path(str(export_report["output_dir"])) / "recovery-export-report.json",
        output_path=validation_path,
    )
    if validation_report.get("status") != "validated" and bool(config.get("strict_validation", True)):
        raise RecoveryExperimentError(
            f"recovered wrapper validation failed; see {validation_path}"
        )

    after_config_path = run_dir / "after" / "eval-batch-config.json"
    after_config = _eval_batch_config(
        config=eval_config,
        model=model,
        wrapper=recovered_wrapper,
        output_dir=run_dir / "after",
        recovery_teacher_model=model,
    )
    _write_json(after_config_path, after_config)
    after_manifest = run_eval_batch(config_path=after_config_path, evaluator=evaluator)

    comparison = compare_eval_batch_manifests(
        before_path=Path(str(before_manifest["output_dir"])) / "eval-batch-manifest.json",
        after_path=Path(str(after_manifest["output_dir"])) / "eval-batch-manifest.json",
    )
    comparison_path = run_dir / "recovery-before-after.json"
    html_path = run_dir / "recovery-experiment.html"
    report_path = run_dir / "recovery-experiment-report.json"
    _write_json(comparison_path, comparison)
    report = _experiment_report(
        config_path=config_path,
        run_dir=run_dir,
        model=model,
        wrapper=wrapper,
        recovered_wrapper=recovered_wrapper,
        before_config_path=before_config_path,
        before_manifest=before_manifest,
        recovery_config_path=recovery_config_path,
        recovery_plan=recovery_plan,
        recovery_report=recovery_report,
        export_report=export_report,
        validation_report=validation_report,
        validation_path=validation_path,
        after_config_path=after_config_path,
        after_manifest=after_manifest,
        comparison=comparison,
        comparison_path=comparison_path,
        html_path=html_path,
        report_path=report_path,
    )
    _write_json(report_path, report)
    html_path.write_text(render_recovery_experiment_html(report), encoding="utf-8")
    return report


def render_recovery_experiment_html(report: dict[str, Any]) -> str:
    title = "MoE Forge Recovery Experiment"
    summary = _dict(report.get("summary"))
    rows = []
    comparison = _dict(report.get("before_after_eval"))
    validation = _dict(report.get("recovered_wrapper_validation"))
    tensor_comparison = _dict(validation.get("tensor_comparison"))
    router_validation = _dict(validation.get("router_tensor_validation"))
    reload_report = _dict(validation.get("reload"))
    artifacts = _dict(report.get("artifacts"))
    before_batch = _dict(report.get("before_eval_batch"))
    after_batch = _dict(report.get("after_eval_batch"))
    quality_trends = _dict(report.get("quality_trends"))
    training_trend = _dict(quality_trends.get("training"))
    before_after_quality = _dict(quality_trends.get("before_after_quality"))
    for item in comparison.get("mode_deltas", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('expert_mode')))}</td>"
            f"<td>{escape(_text(item.get('status')))}</td>"
            f"<td>{escape(_number(item.get('max_abs_error_before')))}</td>"
            f"<td>{escape(_number(item.get('max_abs_error_after')))}</td>"
            f"<td>{escape(_number(item.get('max_abs_error_delta')))}</td>"
            f"<td>{escape(_number(item.get('teacher_kl_loss_before')))}</td>"
            f"<td>{escape(_number(item.get('teacher_kl_loss_after')))}</td>"
            f"<td>{escape(_number(item.get('teacher_kl_loss_delta')))}</td>"
            f"<td>{escape(_number(item.get('latency_ratio_delta')))}</td>"
            "</tr>"
        )
    updated_rows = []
    for item in tensor_comparison.get("updated_tensors", [])[:20]:
        if not isinstance(item, dict):
            continue
        updated_rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('tensor')))}</td>"
            f"<td>{escape(_text(item.get('shape')))}</td>"
            f"<td>{escape(_text(item.get('source_dtype')))}</td>"
            f"<td>{escape(_text(item.get('recovered_dtype')))}</td>"
            f"<td>{escape(_number(item.get('max_abs_delta')))}</td>"
            f"<td>{escape(_number(item.get('mean_abs_delta')))}</td>"
            "</tr>"
        )
    check_rows = []
    for check, passed in _dict(validation.get("config_checks")).items():
        check_rows.append(
            "<tr>"
            f"<td>{escape(_text(check))}</td>"
            f"<td>{escape('pass' if passed else 'fail')}</td>"
            "</tr>"
        )
    quality_rows = []
    for item in before_after_quality.get("modes", []):
        if not isinstance(item, dict):
            continue
        quality_rows.append(
            "<tr>"
            f"<td>{escape(_text(item.get('expert_mode')))}</td>"
            f"<td>{escape(_number(item.get('teacher_kl_loss_before')))}</td>"
            f"<td>{escape(_number(item.get('teacher_kl_loss_after')))}</td>"
            f"<td>{escape(_number(item.get('teacher_kl_loss_delta')))}</td>"
            f"<td>{escape(_number(item.get('nll_loss_delta_before')))}</td>"
            f"<td>{escape(_number(item.get('nll_loss_delta_after')))}</td>"
            f"<td>{escape(_number(item.get('nll_loss_delta_delta')))}</td>"
            f"<td>{escape(_text(item.get('loss_token_count_after')))}</td>"
            "</tr>"
        )
    artifact_rows = [
        ("Before Eval Manifest", artifacts.get("before_eval_manifest")),
        ("After Eval Manifest", artifacts.get("after_eval_manifest")),
        ("Before/After Comparison", artifacts.get("before_after_comparison")),
        ("Recovery Run Report", artifacts.get("recovery_run_report")),
        ("Recovery Export Report", artifacts.get("recovery_export_report")),
        ("Recovered Wrapper Validation", artifacts.get("recovered_wrapper_validation")),
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{title}</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            f"<h1>{title}</h1>",
            '<section class="cards">',
            _card("Improved Modes", summary.get("improved_modes_by_max_abs_error")),
            _card("Regressed Modes", summary.get("regressed_modes_by_max_abs_error")),
            _card("Improved KL Modes", summary.get("improved_modes_by_teacher_kl")),
            _card("Initial Loss", summary.get("initial_loss")),
            _card("Final Loss", summary.get("final_loss")),
            _card("Total Loss Delta", summary.get("total_loss_delta")),
            _card("Avg KL Delta", summary.get("average_teacher_kl_delta")),
            _card("Avg NLL Delta", summary.get("average_nll_delta_delta")),
            _card("Validation", validation.get("status")),
            _card("Updated Tensors", tensor_comparison.get("updated_tensor_count")),
            _card("Updated Router Tensors", summary.get("recovered_updated_router_tensor_count")),
            _card("Router Tensors", router_validation.get("tensor_count")),
            _card("Changed Tensors", tensor_comparison.get("changed_tensor_count")),
            _card("Reloaded Layers", reload_report.get("loaded_layer_count")),
            "</section>",
            "<section>",
            "<h2>Recovered Wrapper</h2>",
            '<div class="grid-2">',
            _fact_panel(
                "Reload",
                [
                    ("Status", validation.get("status")),
                    ("Source tensors", tensor_comparison.get("source_tensor_count")),
                    ("Recovered tensors", tensor_comparison.get("recovered_tensor_count")),
                    ("Router tensors", router_validation.get("tensor_count")),
                    ("Expected router tensors", router_validation.get("expected_tensor_count")),
                    ("Missing tensors", len(tensor_comparison.get("missing_from_recovered", []))),
                    ("Extra tensors", len(tensor_comparison.get("extra_in_recovered", []))),
                ],
            ),
            _fact_panel(
                "Before/After Evidence",
                [
                    ("Before runs", before_batch.get("completed_report_count")),
                    ("After runs", after_batch.get("completed_report_count")),
                    ("Compared modes", comparison.get("compared_mode_count")),
                    ("Steps completed", summary.get("steps_completed")),
                ],
            ),
            "</div>",
            '<div class="table-wrap"><table>',
            "<thead><tr><th>Config Check</th><th>Status</th></tr></thead>",
            f"<tbody>{''.join(check_rows)}</tbody>",
            "</table></div>",
            "</section>",
            "<section>",
            "<h2>Quality Trends</h2>",
            '<div class="grid-2">',
            _fact_panel(
                "Recovery Training",
                [
                    ("Steps", training_trend.get("step_count")),
                    ("Initial total loss", training_trend.get("initial_total_loss")),
                    ("Final total loss", training_trend.get("final_total_loss")),
                    ("Total loss delta", training_trend.get("total_loss_delta")),
                    ("Final teacher KL", training_trend.get("final_teacher_kl")),
                ],
            ),
            _fact_panel(
                "Before/After Quality",
                [
                    ("Compared modes", before_after_quality.get("mode_count")),
                    ("Average teacher KL delta", before_after_quality.get("average_teacher_kl_delta")),
                    ("Average NLL delta delta", before_after_quality.get("average_nll_delta_delta")),
                    ("Best KL mode", _dict(before_after_quality.get("best_teacher_kl_mode")).get("expert_mode")),
                    ("Best NLL mode", _dict(before_after_quality.get("best_nll_delta_mode")).get("expert_mode")),
                ],
            ),
            "</div>",
            '<div class="table-wrap"><table>',
            "<thead><tr>"
            "<th>Mode</th><th>Before KL</th><th>After KL</th><th>Delta KL</th>"
            "<th>Before NLL Delta</th><th>After NLL Delta</th><th>Delta</th><th>Tokens</th>"
            "</tr></thead>",
            f"<tbody>{''.join(quality_rows)}</tbody>",
            "</table></div>",
            "</section>",
            "<section>",
            "<h2>Mode Deltas</h2>",
            '<div class="table-wrap"><table>',
            "<thead><tr>"
            "<th>Mode</th><th>Status</th><th>Before Max</th><th>After Max</th>"
            "<th>Delta Max</th><th>Before KL</th><th>After KL</th><th>Delta KL</th><th>Delta Latency</th>"
            "</tr></thead>",
            f"<tbody>{''.join(rows)}</tbody>",
            "</table></div>",
            "</section>",
            "<section>",
            "<h2>Updated Tensor Metadata</h2>",
            '<div class="table-wrap"><table>',
            "<thead><tr>"
            "<th>Tensor</th><th>Shape</th><th>Source Dtype</th><th>Recovered Dtype</th>"
            "<th>Max Delta</th><th>Mean Delta</th>"
            "</tr></thead>",
            f"<tbody>{''.join(updated_rows)}</tbody>",
            "</table></div>",
            "</section>",
            "<section>",
            "<h2>Artifacts</h2>",
            '<div class="table-wrap"><table>',
            "<thead><tr><th>Artifact</th><th>Path</th></tr></thead>",
            f"<tbody>{''.join(_artifact_row(label, value) for label, value in artifact_rows)}</tbody>",
            "</table></div>",
            "</section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _experiment_report(
    *,
    config_path: Path,
    run_dir: Path,
    model: Path,
    wrapper: Path,
    recovered_wrapper: Path,
    before_config_path: Path,
    before_manifest: dict[str, Any],
    recovery_config_path: Path,
    recovery_plan: dict[str, Any],
    recovery_report: dict[str, Any],
    export_report: dict[str, Any],
    validation_report: dict[str, Any],
    validation_path: Path,
    after_config_path: Path,
    after_manifest: dict[str, Any],
    comparison: dict[str, Any],
    comparison_path: Path,
    html_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    comparison_summary = _dict(comparison.get("summary"))
    tensor_comparison = _dict(validation_report.get("tensor_comparison"))
    router_validation = _dict(validation_report.get("router_tensor_validation"))
    reload_report = _dict(validation_report.get("reload"))
    quality_trends = _quality_trends(comparison=comparison, recovery_report=recovery_report)
    return {
        "format": "moeforge_recovery_experiment",
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "model": str(model),
        "source_wrapper": str(wrapper),
        "recovered_wrapper": str(recovered_wrapper),
        "summary": {
            "initial_loss": recovery_report.get("initial_loss"),
            "final_loss": recovery_report.get("final_loss"),
            "steps_completed": recovery_report.get("steps_completed"),
            "recovered_wrapper_validation_status": validation_report.get("status"),
            "recovered_updated_tensor_count": tensor_comparison.get("updated_tensor_count"),
            "recovered_changed_tensor_count": tensor_comparison.get("changed_tensor_count"),
            "recovered_updated_router_tensor_count": export_report.get("updated_router_tensor_count"),
            "recovered_router_tensor_count": router_validation.get("tensor_count"),
            "recovered_expected_router_tensor_count": router_validation.get("expected_tensor_count"),
            "recovered_reload_layer_count": reload_report.get("loaded_layer_count"),
            "average_teacher_kl_delta": _dict(quality_trends.get("before_after_quality")).get("average_teacher_kl_delta"),
            "average_nll_delta_delta": _dict(quality_trends.get("before_after_quality")).get("average_nll_delta_delta"),
            "total_loss_delta": _dict(quality_trends.get("training")).get("total_loss_delta"),
            **comparison_summary,
        },
        "artifacts": {
            "before_eval_config": str(before_config_path),
            "before_eval_manifest": str(
                Path(str(before_manifest["output_dir"])) / "eval-batch-manifest.json"
            ),
            "recovery_config": str(recovery_config_path),
            "recovery_plan": str(recovery_plan["artifacts"]["plan_path"]),
            "recovery_run_report": str(
                Path(str(recovery_report["output_dir"])) / "recovery-run-report.json"
            ),
            "recovery_checkpoint": str(_last_checkpoint_path(recovery_report)),
            "recovery_export_report": str(
                Path(str(export_report["output_dir"])) / "recovery-export-report.json"
            ),
            "recovered_wrapper_validation": str(validation_path),
            "after_eval_config": str(after_config_path),
            "after_eval_manifest": str(
                Path(str(after_manifest["output_dir"])) / "eval-batch-manifest.json"
            ),
            "before_after_comparison": str(comparison_path),
            "html_report": str(html_path),
            "json_report": str(report_path),
        },
        "before_eval_batch": before_manifest,
        "after_eval_batch": after_manifest,
        "recovery_run": recovery_report,
        "recovery_export": export_report,
        "recovered_wrapper_validation": validation_report,
        "before_after_eval": comparison,
        "quality_trends": quality_trends,
    }


def _quality_trends(*, comparison: dict[str, Any], recovery_report: dict[str, Any]) -> dict[str, Any]:
    mode_deltas = [
        item
        for item in comparison.get("mode_deltas", [])
        if isinstance(item, dict) and item.get("status") == "compared"
    ]
    return {
        "training": _training_trend(recovery_report),
        "before_after_quality": {
            "mode_count": len(mode_deltas),
            "average_teacher_kl_delta": _average_numeric(
                item.get("teacher_kl_loss_delta") for item in mode_deltas
            ),
            "average_carved_nll_delta": _average_numeric(
                item.get("carved_nll_loss_delta") for item in mode_deltas
            ),
            "average_nll_delta_delta": _average_numeric(
                item.get("nll_loss_delta_delta") for item in mode_deltas
            ),
            "best_teacher_kl_mode": _mode_extreme(mode_deltas, key="teacher_kl_loss_delta", prefer="min"),
            "worst_teacher_kl_mode": _mode_extreme(mode_deltas, key="teacher_kl_loss_delta", prefer="max"),
            "best_nll_delta_mode": _mode_extreme(mode_deltas, key="nll_loss_delta_delta", prefer="min"),
            "modes": [
                {
                    "expert_mode": item.get("expert_mode"),
                    "max_abs_error_delta": item.get("max_abs_error_delta"),
                    "teacher_kl_loss_before": item.get("teacher_kl_loss_before"),
                    "teacher_kl_loss_after": item.get("teacher_kl_loss_after"),
                    "teacher_kl_loss_delta": item.get("teacher_kl_loss_delta"),
                    "carved_nll_loss_before": item.get("carved_nll_loss_before"),
                    "carved_nll_loss_after": item.get("carved_nll_loss_after"),
                    "carved_nll_loss_delta": item.get("carved_nll_loss_delta"),
                    "nll_loss_delta_before": item.get("nll_loss_delta_before"),
                    "nll_loss_delta_after": item.get("nll_loss_delta_after"),
                    "nll_loss_delta_delta": item.get("nll_loss_delta_delta"),
                    "loss_token_count_before": item.get("loss_token_count_before"),
                    "loss_token_count_after": item.get("loss_token_count_after"),
                }
                for item in mode_deltas
            ],
        },
    }


def _training_trend(recovery_report: dict[str, Any]) -> dict[str, Any]:
    losses = [item for item in recovery_report.get("losses", []) if isinstance(item, dict)]
    first = losses[0] if losses else {}
    last = losses[-1] if losses else {}
    total_values = [_numeric(item.get("total_loss")) for item in losses]
    teacher_kl_values = [_numeric(item.get("teacher_kl")) for item in losses]
    total_values = [value for value in total_values if value is not None]
    teacher_kl_values = [value for value in teacher_kl_values if value is not None]
    return {
        "step_count": len(losses),
        "initial_total_loss": first.get("total_loss", recovery_report.get("initial_loss")),
        "final_total_loss": last.get("total_loss", recovery_report.get("final_loss")),
        "total_loss_delta": _delta(first.get("total_loss"), last.get("total_loss")),
        "min_total_loss": min(total_values) if total_values else None,
        "initial_teacher_kl": first.get("teacher_kl"),
        "final_teacher_kl": last.get("teacher_kl"),
        "teacher_kl_delta": _delta(first.get("teacher_kl"), last.get("teacher_kl")),
        "min_teacher_kl": min(teacher_kl_values) if teacher_kl_values else None,
        "loss_points": [
            {
                "step": item.get("step"),
                "total_loss": item.get("total_loss"),
                "teacher_kl": item.get("teacher_kl"),
                "logits_mse": item.get("logits_mse"),
                "z_loss": item.get("z_loss"),
                "learning_rate": item.get("learning_rate"),
            }
            for item in losses
        ],
    }


def _mode_extreme(items: list[dict[str, Any]], *, key: str, prefer: str) -> dict[str, Any] | None:
    scored = [(item, _numeric(item.get(key))) for item in items]
    scored = [(item, value) for item, value in scored if value is not None]
    if not scored:
        return None
    selected = min(scored, key=lambda pair: pair[1]) if prefer == "min" else max(scored, key=lambda pair: pair[1])
    return {
        "expert_mode": selected[0].get("expert_mode"),
        key: selected[1],
    }


def _average_numeric(values: Any) -> float | None:
    numbers = [_numeric(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return float(sum(numbers) / len(numbers))


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


def _eval_batch_config(
    *,
    config: dict[str, Any],
    model: Path,
    wrapper: Path,
    output_dir: Path,
    recovery_teacher_model: Path,
) -> dict[str, Any]:
    payload = dict(config)
    payload["model"] = str(model)
    payload["wrapper"] = str(wrapper)
    payload["output_dir"] = str(output_dir)
    payload.setdefault("expert_modes", ["all", "default-pool", "router"])
    payload.setdefault("write_html", True)
    payload.setdefault(
        "recovery_eval",
        {"enabled": True, "teacher_model": str(recovery_teacher_model)},
    )
    return payload


def _recovery_config(
    *,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    train_config: dict[str, Any],
    model: Path,
    wrapper: Path,
    output_dir: Path,
) -> dict[str, Any]:
    recovery = _dict(config.get("recovery"))
    payload = dict(recovery)
    payload["teacher_model"] = str(config.get("teacher_model", model))
    payload["student_model"] = str(config.get("student_model", model))
    payload["wrapper"] = str(wrapper)
    payload["output_dir"] = str(output_dir)
    if train_config:
        payload.setdefault("train", train_config)
    if eval_config and "eval" not in payload:
        payload["eval"] = _eval_samples_from_batch(eval_config)
    return payload


def _eval_samples_from_batch(eval_config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "input_ids",
        "input_ids_json",
        "input_ids_file",
        "text",
        "texts",
        "text_file",
        "sequence_length",
        "max_samples",
    ]
    return {key: eval_config[key] for key in keys if key in eval_config}


def _resolve_sample_file_refs(config: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    payload = dict(config)
    for key in ("input_ids_file", "text_file"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            payload[key] = str(_resolve_path(Path(str(value)), base_dir=base_dir))
    return payload


def _last_checkpoint_path(recovery_report: dict[str, Any]) -> Path:
    checkpoints = recovery_report.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RecoveryExperimentError("recovery run did not produce checkpoints")
    checkpoint = checkpoints[-1]
    if not isinstance(checkpoint, dict):
        raise RecoveryExperimentError("recovery checkpoint record must be a JSON object")
    value = checkpoint.get("metadata_path") or checkpoint.get("checkpoint_path")
    if not value:
        raise RecoveryExperimentError("recovery checkpoint record is missing metadata_path")
    return Path(str(value))


def _model_ref(config: dict[str, Any], *, base_dir: Path) -> Path:
    value = config.get("model", config.get("teacher_model"))
    if value is None or str(value).strip() == "":
        raise RecoveryExperimentError("recovery experiment config requires model or teacher_model")
    return _resolve_path(Path(str(value)), base_dir=base_dir)


def _required_path(config: dict[str, Any], key: str, *, base_dir: Path) -> Path:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise RecoveryExperimentError(f"recovery experiment config requires {key}")
    return _resolve_path(Path(str(value)), base_dir=base_dir)


def _output_dir(*, config: dict[str, Any], output_dir: Path | None, base_dir: Path) -> Path:
    if output_dir is not None:
        return _resolve_path(output_dir, base_dir=Path.cwd())
    return _resolve_path(Path(str(config.get("output_dir", "recovery-experiment"))), base_dir=base_dir)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecoveryExperimentError("recovery experiment config must be a JSON object")
    return payload


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _card(label: str, value: Any) -> str:
    return (
        '<article class="card">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(_number(value))}</div>'
        "</article>"
    )


def _fact_panel(title: str, items: list[tuple[str, Any]]) -> str:
    rows = "".join(
        f'<div class="fact"><span>{escape(label)}</span><strong>{escape(_number(value))}</strong></div>'
        for label, value in items
    )
    return f'<article class="panel"><h3>{escape(title)}</h3>{rows}</article>'


def _artifact_row(label: str, value: Any) -> str:
    return (
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td><code>{escape(_text(value))}</code></td>"
        "</tr>"
    )


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _text(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _css() -> str:
    return """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f6f7f9; color: #1d2430; }
.page { max-width: 1100px; margin: 0 auto; padding: 32px 20px 56px; }
h1 { margin: 0 0 24px; font-size: 28px; }
h2 { margin: 28px 0 12px; font-size: 18px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }
.label { color: #64748b; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }
.value { margin-top: 6px; font-size: 20px; font-weight: 750; overflow-wrap: anywhere; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 12px; }
.panel { background: #fff; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }
.panel h3 { margin: 0 0 12px; font-size: 15px; }
.fact { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 8px 0; border-top: 1px solid #edf1f5; }
.fact:first-of-type { border-top: 0; }
.fact span { color: #64748b; font-size: 13px; }
.fact strong { font-size: 14px; text-align: right; overflow-wrap: anywhere; }
.table-wrap { overflow-x: auto; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e7ebf0; text-align: left; vertical-align: top; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; overflow-wrap: anywhere; }
thead th { background: #edf1f5; color: #334155; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
tbody tr:last-child td { border-bottom: 0; }
"""
