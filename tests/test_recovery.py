from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.cli import main
from moeforge.recovery import (
    RecoveryPlanError,
    compare_eval_batch_manifests,
    write_recovery_plan,
)


def test_write_recovery_plan_records_training_settings_and_before_after(tmp_path: Path) -> None:
    train_text = tmp_path / "train.txt"
    train_text.write_text("alpha sample\n\nbeta sample", encoding="utf-8")
    before = _write_batch_manifest(
        tmp_path / "before.json",
        all_error=0.2,
        router_error=0.4,
        router_latency=0.7,
    )
    after = _write_batch_manifest(
        tmp_path / "after.json",
        all_error=0.1,
        router_error=0.25,
        router_latency=0.8,
    )
    config_path = tmp_path / "recovery.json"
    output_path = tmp_path / "run" / "plan.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": "dense-teacher",
                "student_model": "dense-student",
                "wrapper": "wrapper",
                "output_dir": "run",
                "train": {"text_file": str(train_text), "sequence_length": 64},
                "eval": {"input_ids": [[1, 2, 3, 4]]},
                "loss": {"temperature": 2.0, "teacher_kl_weight": 0.8},
                "optimizer": {"learning_rate": 0.0001},
                "schedule": {"steps": 12, "eval_every_steps": 3},
                "checkpoints": {"keep_last": 3},
                "before_eval_batch": str(before),
                "after_eval_batch": str(after),
            }
        ),
        encoding="utf-8",
    )

    plan = write_recovery_plan(config_path=config_path, output_path=output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    before_after = json.loads(
        (output_path.parent / "recovery-before-after.json").read_text(encoding="utf-8")
    )

    assert plan["format"] == "moeforge_recovery_plan"
    assert saved["teacher"]["model"] == "dense-teacher"
    assert saved["student"]["wrapper"] == str(tmp_path / "wrapper")
    assert saved["loss"]["temperature"] == 2.0
    assert saved["schedule"]["steps"] == 12
    assert saved["checkpoints"]["keep_last"] == 3
    assert saved["samples"]["train"]["sample_count"] == 2
    assert saved["samples"]["eval"]["samples"][0]["token_count"] == 4
    assert saved["before_after_eval"]["summary"]["improved_modes_by_max_abs_error"] == 2
    assert before_after["mode_deltas"][1]["expert_mode"] == "router"
    assert before_after["mode_deltas"][1]["max_abs_error_delta"] < 0


def test_recovery_plan_cli_writes_output(tmp_path: Path) -> None:
    config_path = tmp_path / "recovery.json"
    output_path = tmp_path / "plan.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": "dense-teacher",
                "wrapper": "wrapper",
                "train": {"input_ids": [[1, 2, 3]]},
            }
        ),
        encoding="utf-8",
    )

    status = main(["recovery-plan", "--config", str(config_path), "--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["artifacts"]["plan_path"] == str(output_path)
    assert payload["samples"]["train"]["kind"] == "input_ids"
    assert payload["before_after_eval"]["status"] == "not_configured"


def test_compare_eval_batch_manifests_reports_missing_modes(tmp_path: Path) -> None:
    before = _write_batch_manifest(
        tmp_path / "before.json",
        all_error=0.2,
        router_error=0.4,
    )
    after = _write_batch_manifest(
        tmp_path / "after.json",
        all_error=0.3,
        include_router=False,
    )

    comparison = compare_eval_batch_manifests(before_path=before, after_path=after)

    assert comparison["summary"]["regressed_modes_by_max_abs_error"] == 1
    assert comparison["mode_deltas"][1]["expert_mode"] == "router"
    assert comparison["mode_deltas"][1]["status"] == "missing"


def test_recovery_plan_validates_loss(tmp_path: Path) -> None:
    config_path = tmp_path / "recovery.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": "dense-teacher",
                "wrapper": "wrapper",
                "loss": {"temperature": 0.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RecoveryPlanError, match="temperature"):
        write_recovery_plan(config_path=config_path)


def _write_batch_manifest(
    path: Path,
    *,
    all_error: float,
    router_error: float = 0.0,
    router_latency: float = 1.0,
    include_router: bool = True,
) -> Path:
    runs = [
        {
            "expert_mode": "all",
            "status": "failed" if all_error else "passed",
            "max_abs_error": all_error,
            "mean_abs_error": all_error / 2,
            "latency_ratio": 1.0,
        }
    ]
    if include_router:
        runs.append(
            {
                "expert_mode": "router",
                "status": "failed" if router_error else "passed",
                "max_abs_error": router_error,
                "mean_abs_error": router_error / 2,
                "latency_ratio": router_latency,
            }
        )
    path.write_text(
        json.dumps({"format": "moeforge_eval_batch", "runs": runs}),
        encoding="utf-8",
    )
    return path
