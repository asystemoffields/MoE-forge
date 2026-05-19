from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.recovery_compare import (
    build_recovery_comparison,
    render_recovery_comparison_html,
    write_recovery_comparison_report,
)


def test_build_recovery_comparison_ranks_quality_and_renders_html(tmp_path: Path) -> None:
    token_report = tmp_path / "recovery-experiment" / "recovery-experiment-report.json"
    text_report = tmp_path / "recovery-experiment-text" / "recovery-experiment-report.json"
    _write_json(
        token_report,
        _report(
            label="token",
            train_kind="input_ids",
            avg_kl_delta=0.02,
            avg_nll_delta=0.01,
            after_latency=0.7,
        ),
    )
    _write_json(
        text_report,
        _report(
            label="text",
            train_kind="text",
            avg_kl_delta=-0.03,
            avg_nll_delta=-0.02,
            after_latency=1.1,
        ),
    )

    comparison = build_recovery_comparison(report_paths=[token_report, text_report])
    html = render_recovery_comparison_html(comparison)

    assert comparison["format"] == "moeforge_recovery_experiment_comparison"
    assert comparison["best"]["label"] == "recovery-experiment-text"
    assert comparison["fastest_after"]["label"] == "recovery-experiment"
    assert comparison["fastest_after"]["average_after_latency_ratio"] == 0.7
    assert comparison["ranked"][0]["train_sample_kind"] == "text"
    assert comparison["ranked"][0]["modes"][0]["teacher_kl_loss_delta"] == -0.03
    assert comparison["warnings"] == ["Reports use different training sample kinds: input_ids, text"]
    assert "Ranked Experiments" in html
    assert "Mode Deltas" in html


def test_recovery_compare_cli_writes_json_and_html(tmp_path: Path) -> None:
    first = tmp_path / "first-report.json"
    second = tmp_path / "second-report.json"
    output = tmp_path / "compare.json"
    html_output = tmp_path / "compare.html"
    _write_json(first, _report(label="first", train_kind="input_ids", avg_kl_delta=0.01))
    _write_json(second, _report(label="second", train_kind="input_ids", avg_kl_delta=0.02))

    comparison = write_recovery_comparison_report(
        report_paths=[second, first],
        output_path=tmp_path / "api-compare.json",
        html_output_path=tmp_path / "api-compare.html",
    )
    status = main(
        [
            "recovery-compare",
            str(second),
            str(first),
            "--output",
            str(output),
            "--html-output",
            str(html_output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert comparison["best"]["label"] == "first-report"
    assert payload["report_count"] == 2
    assert payload["best"]["label"] == "first-report"
    assert html_output.read_text(encoding="utf-8").startswith("<!doctype html>")


def _report(
    *,
    label: str,
    train_kind: str,
    avg_kl_delta: float,
    avg_nll_delta: float = 0.0,
    after_latency: float = 1.0,
) -> dict:
    return {
        "format": "moeforge_recovery_experiment",
        "model": f"{label}-model",
        "source_wrapper": "wrapper",
        "recovered_wrapper": f"{label}-recovered-wrapper",
        "summary": {
            "initial_loss": 0.5,
            "final_loss": 0.2,
            "steps_completed": 2,
            "recovered_wrapper_validation_status": "validated",
            "average_teacher_kl_delta": avg_kl_delta,
            "average_nll_delta_delta": avg_nll_delta,
            "total_loss_delta": -0.3,
            "improved_modes_by_max_abs_error": 1,
            "regressed_modes_by_max_abs_error": 0,
            "improved_modes_by_teacher_kl": 1 if avg_kl_delta < 0 else 0,
            "regressed_modes_by_teacher_kl": 0 if avg_kl_delta < 0 else 1,
        },
        "artifacts": {"json_report": f"{label}.json"},
        "before_eval_batch": {"sample_source": {"kind": train_kind, "sample_count": 2}},
        "after_eval_batch": {"sample_source": {"kind": train_kind, "sample_count": 2}},
        "recovery_run": {
            "train_sample_source": {
                "kind": train_kind,
                "sample_count": 2,
                "token_counts": [4, 5] if train_kind == "text" else None,
            }
        },
        "quality_trends": {
            "before_after_quality": {
                "average_teacher_kl_delta": avg_kl_delta,
                "average_nll_delta_delta": avg_nll_delta,
                "modes": [
                    {
                        "expert_mode": "router",
                        "max_abs_error_delta": avg_kl_delta / 2,
                        "teacher_kl_loss_before": 0.1,
                        "teacher_kl_loss_after": 0.1 + avg_kl_delta,
                        "teacher_kl_loss_delta": avg_kl_delta,
                        "nll_loss_delta_before": 0.2,
                        "nll_loss_delta_after": 0.2 + avg_nll_delta,
                        "nll_loss_delta_delta": avg_nll_delta,
                        "latency_ratio_before": 1.2,
                        "latency_ratio_after": after_latency,
                        "latency_ratio_delta": after_latency - 1.2,
                        "loss_token_count_after": 8,
                    }
                ],
            }
        },
        "before_after_eval": {
            "status": "compared",
            "mode_deltas": [
                {
                    "expert_mode": "router",
                    "max_abs_error_delta": avg_kl_delta / 2,
                    "teacher_kl_loss_before": 0.1,
                    "teacher_kl_loss_after": 0.1 + avg_kl_delta,
                    "teacher_kl_loss_delta": avg_kl_delta,
                    "nll_loss_delta_before": 0.2,
                    "nll_loss_delta_after": 0.2 + avg_nll_delta,
                    "nll_loss_delta_delta": avg_nll_delta,
                    "latency_ratio_before": 1.2,
                    "latency_ratio_after": after_latency,
                    "latency_ratio_delta": after_latency - 1.2,
                    "loss_token_count_after": 8,
                }
            ],
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
