from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.summary import summarize_run


def _write_report(path: Path, *, modes: list[dict], summary: dict, losses: list[dict]) -> Path:
    payload = {
        "format": "moeforge_recovery_experiment_report",
        "summary": summary,
        "recovery_run": {"trainable_parameter_count": 66424440, "losses": losses},
        "quality_trends": {"before_after_quality": {"modes": modes}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarize_joint_run_improving_but_undertrained(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path / "report.json",
        summary={
            "initial_loss": 4.125,
            "final_loss": 1.90625,
            "steps_completed": 1000,
            "recovered_updated_tensor_count": 360,
            "recovered_updated_router_tensor_count": 60,
        },
        losses=[
            {"step": 250, "total_loss": 3.0},
            {"step": 500, "total_loss": 2.4},
            {"step": 750, "total_loss": 2.1},
            {"step": 1000, "total_loss": 1.90625},
        ],
        modes=[
            {
                "expert_mode": "learned-router",
                "teacher_kl_loss_before": 1.1472,
                "teacher_kl_loss_after": 1.0954,
                "teacher_kl_loss_delta": -0.0518,
                "carved_nll_loss_before": 3.8137,
                "carved_nll_loss_after": 3.6945,
                "carved_nll_loss_delta": -0.1192,
            },
            {
                "expert_mode": "all",
                "teacher_kl_loss_before": 0.0022,
                "teacher_kl_loss_after": 0.0888,
                "teacher_kl_loss_delta": 0.0866,
            },
        ],
    )

    report = summarize_run(report_path=report_path)

    assert report["status"] == "summarized"
    v = report["verdicts"]
    assert v["experts_trained"] is True
    assert v["carve_lossless"] is False
    assert v["routing_gap"] == "large"
    assert v["direction"] == "improving"
    assert v["undertrained"] is True
    assert "undertrained" in report["headline"].lower()
    assert any("Train longer" in cmd for cmd in report["next_commands"])


def test_summarize_plateau_recommends_pivot(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path / "report.json",
        summary={
            "initial_loss": 4.0,
            "final_loss": 2.05,
            "steps_completed": 1000,
            "recovered_updated_tensor_count": 0,
            "recovered_updated_router_tensor_count": 60,
        },
        losses=[
            {"step": 250, "total_loss": 2.0},
            {"step": 500, "total_loss": 1.95},
            {"step": 750, "total_loss": 1.97},
            {"step": 1000, "total_loss": 2.05},
        ],
        modes=[
            {
                "expert_mode": "learned-router",
                "teacher_kl_loss_before": 1.12,
                "teacher_kl_loss_after": 1.121,
                "teacher_kl_loss_delta": 0.001,
                "carved_nll_loss_before": 3.8,
                "carved_nll_loss_after": 3.85,
                "carved_nll_loss_delta": 0.05,
            },
        ],
    )

    report = summarize_run(report_path=report_path)

    v = report["verdicts"]
    assert v["undertrained"] is False
    assert v["routing_gap"] == "large"
    assert v["direction"] == "regressing"
    assert any("upcycle" in cmd for cmd in report["next_commands"])


def test_summarize_limited_when_no_eval(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path / "report.json",
        summary={"initial_loss": 4.0, "final_loss": 3.0, "steps_completed": 500},
        losses=[{"step": 500, "total_loss": 3.0}],
        modes=[],
    )

    report = summarize_run(report_path=report_path)

    assert report["status"] == "limited"
    assert "no before/after eval" in report["headline"].lower()


def test_summarize_cli_writes_output(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path / "report.json",
        summary={
            "initial_loss": 4.0,
            "final_loss": 0.05,
            "steps_completed": 2000,
            "recovered_updated_tensor_count": 360,
        },
        losses=[{"step": 1000, "total_loss": 0.1}, {"step": 2000, "total_loss": 0.05}],
        modes=[
            {
                "expert_mode": "learned-router",
                "teacher_kl_loss_before": 0.5,
                "teacher_kl_loss_after": 0.05,
                "teacher_kl_loss_delta": -0.45,
                "carved_nll_loss_before": 3.0,
                "carved_nll_loss_after": 2.8,
                "carved_nll_loss_delta": -0.2,
            },
            {"expert_mode": "all", "teacher_kl_loss_after": 0.002, "teacher_kl_loss_before": 0.002, "teacher_kl_loss_delta": 0.0},
        ],
    )
    out = tmp_path / "summary.json"

    status = main(["summarize", str(report_path), "--output", str(out), "--json"])

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert status == 0
    assert saved["verdicts"]["routing_gap"] == "small"
    assert "near dense quality" in saved["headline"].lower()
