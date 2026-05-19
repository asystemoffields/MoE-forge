from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from .batch import run_eval_batch
from .evaluation import evaluate_hf_dense_vs_carved
from .recovery import compare_eval_batch_manifests, write_recovery_plan
from .recovery_runner import export_recovered_wrapper, run_recovery


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
    after_config_path: Path,
    after_manifest: dict[str, Any],
    comparison: dict[str, Any],
    comparison_path: Path,
    html_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    comparison_summary = _dict(comparison.get("summary"))
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
.table-wrap { overflow-x: auto; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 12px; border-bottom: 1px solid #e7ebf0; text-align: left; vertical-align: top; }
thead th { background: #edf1f5; color: #334155; font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
tbody tr:last-child td { border-bottom: 0; }
"""
