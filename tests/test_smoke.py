from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.smoke import assert_tiny_hf_smoke_run


def test_assert_tiny_hf_smoke_run_writes_lab_notebook_report(tmp_path: Path) -> None:
    _write_smoke_artifacts(tmp_path)

    report = assert_tiny_hf_smoke_run(run_dir=tmp_path, output_path=tmp_path / "smoke-assertions.json")

    saved = json.loads((tmp_path / "smoke-assertions.json").read_text(encoding="utf-8"))
    assert report["format"] == "moeforge_tiny_hf_smoke_assertions"
    assert report["passed"] is True
    assert saved["metrics"]["eval_batch"]["teacher_kl_loss_by_mode"]["router"] == 0.03
    assert saved["metrics"]["recovery_quality"]["teacher_kl_delta_by_mode"]["router"] == -0.02
    assert saved["metrics"]["recovered_wrapper"]["updated_tensor_count"] == 18


def test_smoke_assert_cli_returns_nonzero_on_missing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "assertions.json"

    status = main(["smoke-assert", "--run-dir", str(tmp_path), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert payload["passed"] is False
    assert payload["passed_check_count"] < payload["check_count"]


def _write_smoke_artifacts(path: Path) -> None:
    eval_dir = path / "eval-runs"
    experiment_dir = path / "recovery-experiment"
    eval_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)
    (eval_dir / "eval-compare.html").write_text("<!doctype html>", encoding="utf-8")
    (experiment_dir / "recovery-experiment.html").write_text("<!doctype html>", encoding="utf-8")
    _write_json(
        eval_dir / "eval-batch-manifest.json",
        {
            "format": "moeforge_eval_batch",
            "run_count": 3,
            "completed_report_count": 3,
            "runs": [
                _run("all", teacher_kl=0.0, nll_delta=0.0),
                _run("default-pool", teacher_kl=0.02, nll_delta=0.1),
                _run("router", teacher_kl=0.03, nll_delta=0.2),
            ],
        },
    )
    comparison = {
        "status": "compared",
        "compared_mode_count": 3,
        "summary": {
            "improved_modes_by_max_abs_error": 2,
            "regressed_modes_by_max_abs_error": 1,
            "improved_modes_by_teacher_kl": 2,
            "regressed_modes_by_teacher_kl": 1,
        },
        "mode_deltas": [
            _delta("all", teacher_kl_delta=0.0, nll_delta_delta=0.0),
            _delta("default-pool", teacher_kl_delta=-0.01, nll_delta_delta=-0.02),
            _delta("router", teacher_kl_delta=-0.02, nll_delta_delta=-0.03),
        ],
    }
    recovery_report = {
        "format": "moeforge_recovery_experiment",
        "summary": {
            "initial_loss": 1.0,
            "final_loss": 0.5,
            "steps_completed": 2,
            "recovered_wrapper_validation_status": "validated",
        },
        "before_after_eval": comparison,
        "recovery_run": {
            "losses": [
                {"step": 1, "total_loss": 1.0, "teacher_kl": 0.2, "logits_mse": 0.1},
                {"step": 2, "total_loss": 0.5, "teacher_kl": 0.1, "logits_mse": 0.05},
            ]
        },
    }
    validation = {
        "format": "moeforge_recovered_wrapper_validation",
        "status": "validated",
        "passed": True,
        "tensor_comparison": {
            "updated_tensor_count": 18,
            "changed_tensor_count": 18,
        },
        "reload": {"loaded_layer_count": 2},
    }
    _write_json(experiment_dir / "recovery-experiment-report.json", recovery_report)
    _write_json(experiment_dir / "recovered-wrapper-validation.json", validation)
    _write_json(experiment_dir / "recovered-wrapper-validation-cli.json", validation)


def _run(mode: str, *, teacher_kl: float, nll_delta: float) -> dict:
    return {
        "expert_mode": mode,
        "status": "passed" if teacher_kl == 0.0 else "failed",
        "teacher_kl_loss": teacher_kl,
        "dense_nll_loss": 2.0,
        "carved_nll_loss": 2.0 + nll_delta,
        "nll_loss_delta": nll_delta,
        "loss_token_count": 4,
    }


def _delta(mode: str, *, teacher_kl_delta: float, nll_delta_delta: float) -> dict:
    return {
        "expert_mode": mode,
        "status": "compared",
        "max_abs_error_delta": teacher_kl_delta,
        "teacher_kl_loss_delta": teacher_kl_delta,
        "carved_nll_loss_delta": nll_delta_delta,
        "nll_loss_delta_delta": nll_delta_delta,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
