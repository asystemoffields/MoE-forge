from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SmokeAssertionError(RuntimeError):
    """Raised when a smoke run assertion cannot be evaluated."""


def assert_tiny_hf_smoke_run(*, run_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    checks: list[dict[str, Any]] = []
    artifacts = _artifact_paths(run_dir)
    payloads = {
        name: _load_json(path, label=name, checks=checks)
        for name, path in artifacts.items()
        if path.suffix == ".json"
    }

    eval_manifest = _dict(payloads.get("eval_manifest"))
    recovery_report = _dict(payloads.get("recovery_report"))
    validation = _dict(payloads.get("validation"))
    cli_validation = _dict(payloads.get("cli_validation"))
    before_after = _dict(recovery_report.get("before_after_eval"))
    recovery_summary = _dict(recovery_report.get("summary"))

    _check(
        checks,
        name="eval batch completed expected modes",
        passed=int(eval_manifest.get("completed_report_count") or 0) >= 3,
        evidence={"completed_report_count": eval_manifest.get("completed_report_count")},
    )
    _check(
        checks,
        name="eval batch reports teacher quality metrics",
        passed=_runs_have_quality_metrics(eval_manifest),
        evidence={"run_count": len(_list(eval_manifest.get("runs")))},
    )
    _check(
        checks,
        name="recovery experiment compared before and after eval batches",
        passed=(
            before_after.get("status") == "compared"
            and int(before_after.get("compared_mode_count") or 0) >= 1
        ),
        evidence={
            "status": before_after.get("status"),
            "compared_mode_count": before_after.get("compared_mode_count"),
        },
    )
    _check(
        checks,
        name="recovery experiment records quality deltas",
        passed=_mode_deltas_have_quality_metrics(before_after),
        evidence={"mode_count": len(_list(before_after.get("mode_deltas")))},
    )
    _check(
        checks,
        name="recovered wrapper validation passed",
        passed=validation.get("status") == "validated" and bool(validation.get("passed")),
        evidence={"status": validation.get("status"), "passed": validation.get("passed")},
    )
    if cli_validation:
        _check(
            checks,
            name="standalone recovery-validate passed",
            passed=cli_validation.get("status") == "validated" and bool(cli_validation.get("passed")),
            evidence={"status": cli_validation.get("status"), "passed": cli_validation.get("passed")},
        )
    _check(
        checks,
        name="recovered tensors were updated",
        passed=int(_dict(validation.get("tensor_comparison")).get("updated_tensor_count") or 0) > 0,
        evidence={
            "updated_tensor_count": _dict(validation.get("tensor_comparison")).get("updated_tensor_count"),
            "changed_tensor_count": _dict(validation.get("tensor_comparison")).get("changed_tensor_count"),
        },
    )
    _check(
        checks,
        name="html reports exist",
        passed=(
            (run_dir / "eval-runs" / "eval-compare.html").exists()
            and (run_dir / "recovery-experiment" / "recovery-experiment.html").exists()
        ),
        evidence={
            "eval_compare_html": str(run_dir / "eval-runs" / "eval-compare.html"),
            "recovery_experiment_html": str(run_dir / "recovery-experiment" / "recovery-experiment.html"),
        },
    )

    report = {
        "format": "moeforge_tiny_hf_smoke_assertions",
        "run_dir": str(run_dir),
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "passed": all(check["passed"] for check in checks),
        "check_count": len(checks),
        "passed_check_count": sum(1 for check in checks if check["passed"]),
        "checks": checks,
        "metrics": {
            "eval_batch": _eval_metrics(eval_manifest),
            "recovery_quality": _quality_delta_summary(before_after),
            "recovery_training": _training_summary(recovery_report),
            "recovered_wrapper": {
                "validation_status": validation.get("status"),
                "updated_tensor_count": _dict(validation.get("tensor_comparison")).get("updated_tensor_count"),
                "changed_tensor_count": _dict(validation.get("tensor_comparison")).get("changed_tensor_count"),
                "loaded_layer_count": _dict(validation.get("reload")).get("loaded_layer_count"),
            },
        },
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "summary": recovery_summary,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "eval_manifest": run_dir / "eval-runs" / "eval-batch-manifest.json",
        "eval_compare_html": run_dir / "eval-runs" / "eval-compare.html",
        "recovery_report": run_dir / "recovery-experiment" / "recovery-experiment-report.json",
        "recovery_html": run_dir / "recovery-experiment" / "recovery-experiment.html",
        "validation": run_dir / "recovery-experiment" / "recovered-wrapper-validation.json",
        "cli_validation": run_dir / "recovery-experiment" / "recovered-wrapper-validation-cli.json",
    }


def _load_json(path: Path, *, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        if label == "cli_validation":
            return {}
        _check(checks, name=f"artifact exists: {label}", passed=False, evidence={"path": str(path)})
        return {}
    if path.suffix != ".json":
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmokeAssertionError(f"{label} must be a JSON object: {path}")
    _check(checks, name=f"artifact exists: {label}", passed=True, evidence={"path": str(path)})
    return payload


def _runs_have_quality_metrics(manifest: dict[str, Any]) -> bool:
    runs = [run for run in _list(manifest.get("runs")) if isinstance(run, dict) and run.get("status") != "error"]
    return bool(runs) and all(
        run.get("teacher_kl_loss") is not None
        and run.get("carved_nll_loss") is not None
        and run.get("nll_loss_delta") is not None
        and int(run.get("loss_token_count") or 0) > 0
        for run in runs
    )


def _mode_deltas_have_quality_metrics(comparison: dict[str, Any]) -> bool:
    deltas = [
        item
        for item in _list(comparison.get("mode_deltas"))
        if isinstance(item, dict) and item.get("status") == "compared"
    ]
    return bool(deltas) and all(
        item.get("teacher_kl_loss_delta") is not None
        and item.get("carved_nll_loss_delta") is not None
        and item.get("nll_loss_delta_delta") is not None
        for item in deltas
    )


def _eval_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    runs = [run for run in _list(manifest.get("runs")) if isinstance(run, dict)]
    return {
        "run_count": manifest.get("run_count"),
        "completed_report_count": manifest.get("completed_report_count"),
        "modes": [run.get("expert_mode") for run in runs],
        "teacher_kl_loss_by_mode": {
            str(run.get("expert_mode")): run.get("teacher_kl_loss") for run in runs
        },
        "nll_loss_delta_by_mode": {
            str(run.get("expert_mode")): run.get("nll_loss_delta") for run in runs
        },
    }


def _quality_delta_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": comparison.get("status"),
        "compared_mode_count": comparison.get("compared_mode_count"),
        "summary": comparison.get("summary"),
        "teacher_kl_delta_by_mode": {
            str(item.get("expert_mode")): item.get("teacher_kl_loss_delta")
            for item in _list(comparison.get("mode_deltas"))
            if isinstance(item, dict)
        },
        "nll_delta_delta_by_mode": {
            str(item.get("expert_mode")): item.get("nll_loss_delta_delta")
            for item in _list(comparison.get("mode_deltas"))
            if isinstance(item, dict)
        },
        "max_abs_delta_by_mode": {
            str(item.get("expert_mode")): item.get("max_abs_error_delta")
            for item in _list(comparison.get("mode_deltas"))
            if isinstance(item, dict)
        },
    }


def _training_summary(recovery_report: dict[str, Any]) -> dict[str, Any]:
    losses = [
        item
        for item in _list(_dict(recovery_report.get("recovery_run")).get("losses"))
        if isinstance(item, dict)
    ]
    return {
        "initial_loss": _dict(recovery_report.get("summary")).get("initial_loss"),
        "final_loss": _dict(recovery_report.get("summary")).get("final_loss"),
        "steps_completed": _dict(recovery_report.get("summary")).get("steps_completed"),
        "loss_points": [
            {
                "step": item.get("step"),
                "total_loss": item.get("total_loss"),
                "teacher_kl": item.get("teacher_kl"),
                "logits_mse": item.get("logits_mse"),
            }
            for item in losses
        ],
    }


def _check(checks: list[dict[str, Any]], *, name: str, passed: bool, evidence: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
