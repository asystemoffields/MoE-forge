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
    config = _load_config(config_path)
    base_dir = config_path.parent
    run_dir = _output_dir(config=config, output_dir=output_dir, base_dir=base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = _model_ref(config, base_dir=base_dir)
    wrapper = _required_path(config, "wrapper", base_dir=base_dir)
    recovered_wrapper = run_dir / "recovered-wrapper"
    eval_config = _dict(config.get("eval"))

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
    reload_report = _dict(validation.get("reload"))
    artifacts = _dict(report.get("artifacts"))
    before_batch = _dict(report.get("before_eval_batch"))
    after_batch = _dict(report.get("after_eval_batch"))
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
            _card("Initial Loss", summary.get("initial_loss")),
            _card("Final Loss", summary.get("final_loss")),
            _card("Validation", validation.get("status")),
            _card("Updated Tensors", tensor_comparison.get("updated_tensor_count")),
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
            "<h2>Mode Deltas</h2>",
            '<div class="table-wrap"><table>',
            "<thead><tr>"
            "<th>Mode</th><th>Status</th><th>Before Max</th><th>After Max</th>"
            "<th>Delta Max</th><th>Delta Latency</th>"
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
    reload_report = _dict(validation_report.get("reload"))
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
            "recovered_reload_layer_count": reload_report.get("loaded_layer_count"),
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
    }


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


def _recovery_config(*, config: dict[str, Any], model: Path, wrapper: Path, output_dir: Path) -> dict[str, Any]:
    recovery = _dict(config.get("recovery"))
    payload = dict(recovery)
    payload["teacher_model"] = str(config.get("teacher_model", model))
    payload["student_model"] = str(config.get("student_model", model))
    payload["wrapper"] = str(wrapper)
    payload["output_dir"] = str(output_dir)
    if "train" in config:
        payload.setdefault("train", config["train"])
    if "eval" in config and "eval" not in payload:
        payload["eval"] = _eval_samples_from_batch(_dict(config["eval"]))
    return payload


def _eval_samples_from_batch(eval_config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "input_ids",
        "input_ids_json",
        "text",
        "texts",
        "text_file",
        "sequence_length",
        "max_samples",
    ]
    return {key: eval_config[key] for key in keys if key in eval_config}


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
