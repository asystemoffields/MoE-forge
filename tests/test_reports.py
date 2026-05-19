from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.reports import (
    build_eval_comparison,
    render_eval_comparison_html,
    render_eval_html_report,
    write_eval_comparison_report,
    write_eval_html_report,
)


def test_render_eval_html_report_escapes_and_summarizes() -> None:
    html = render_eval_html_report(_report())

    assert "MoE Forge Eval Report" in html
    assert "tiny &lt;model&gt;" in html
    assert "Layer Attribution" in html
    assert "Selected-All Max" in html
    assert "Teacher KL" in html
    assert "E0" in html
    assert "quality/speed tradeoff" in html


def test_write_eval_html_report_and_cli(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    html_path = tmp_path / "report.html"
    cli_html_path = tmp_path / "cli-report.html"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")

    write_eval_html_report(report_path=report_path, output_path=html_path)
    status = main(
        [
            "eval-report-html",
            "--input",
            str(report_path),
            "--output",
            str(cli_html_path),
        ]
    )

    assert status == 0
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "Active Experts" in cli_html_path.read_text(encoding="utf-8")


def test_build_eval_comparison_ranks_reports_and_renders_html(tmp_path: Path) -> None:
    all_path = tmp_path / "all.json"
    router_path = tmp_path / "router.json"
    all_path.write_text(
        json.dumps(_report(label="all", mode="all", passed=True, max_abs=0.0, latency_ratio=2.0)),
        encoding="utf-8",
    )
    router_path.write_text(
        json.dumps(_report(label="router", mode="router", passed=False, max_abs=0.2, latency_ratio=0.6)),
        encoding="utf-8",
    )

    comparison = build_eval_comparison(report_paths=[router_path, all_path])
    html = render_eval_comparison_html(comparison)

    assert comparison["best"]["label"] == "all"
    assert comparison["ranked"][0]["rank"] == 1
    assert comparison["ranked"][1]["expert_modes"] == ["router"]
    assert comparison["ranked"][1]["active_experts"][0]["expert_sets"] == [[0]]
    assert "Ranked Reports" in html
    assert "Active Expert Sets" in html


def test_eval_compare_cli_writes_json_and_html(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "compare.json"
    html_output = tmp_path / "compare.html"
    first.write_text(
        json.dumps(_report(label="first", mode="all", passed=True, max_abs=0.0)),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(_report(label="second", mode="router", passed=False, max_abs=0.1)),
        encoding="utf-8",
    )

    comparison = write_eval_comparison_report(
        report_paths=[second, first],
        output_path=tmp_path / "api-compare.json",
        html_output_path=tmp_path / "api-compare.html",
    )
    status = main(
        [
            "eval-compare",
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
    assert comparison["best"]["label"] == "first"
    assert payload["report_count"] == 2
    assert payload["best"]["label"] == "first"
    assert html_output.read_text(encoding="utf-8").startswith("<!doctype html>")


def _report(
    *,
    label: str = "tiny-report",
    mode: str = "router",
    passed: bool = False,
    max_abs: float = 0.25,
    latency_ratio: float = 2.0,
) -> dict:
    return _report_with(
        label=label,
        mode=mode,
        passed=passed,
        max_abs=max_abs,
        latency_ratio=latency_ratio,
    )


def _report_with(
    *,
    label: str = "tiny-report",
    mode: str = "router",
    passed: bool = False,
    max_abs: float = 0.25,
    latency_ratio: float = 2.0,
) -> dict:
    return {
        "model": f"tiny <model> {label}",
        "package_dir": "wrapper",
        "source_model": "dense-source",
        "adapter_family": "llama",
        "sample_count": 1,
        "passed": passed,
        "max_abs_error": max_abs,
        "mean_abs_error": 0.05,
        "warnings": ["approximate subset comparison"],
        "summary": {
            "average_dense_latency_s": 0.01,
            "average_carved_latency_s": 0.02,
            "average_carved_vs_dense_latency_ratio": latency_ratio,
            "average_teacher_kl_loss": max_abs / 4,
            "average_dense_nll_loss": 2.0,
            "average_carved_nll_loss": 2.0 + max_abs,
            "average_nll_loss_delta": max_abs,
            "loss_token_count": 4,
            "worst_sample_index": 0,
            "worst_sample_max_abs_error": max_abs,
            "worst_layer_sample_index": 0,
            "worst_layer": 1,
            "worst_layer_selected_vs_all_max_abs_error": 0.2,
        },
        "samples": [
            {
                "index": 0,
                "source": "input_ids:0",
                "expert_mode": mode,
                "max_abs_error": max_abs,
                "mean_abs_error": 0.05,
                "teacher_kl_loss": max_abs / 4,
                "dense_nll_loss": 2.0,
                "carved_nll_loss": 2.0 + max_abs,
                "nll_loss_delta": max_abs,
                "loss_token_count": 4,
                "carved_vs_dense_latency_ratio": latency_ratio,
                "allclose": passed,
            }
        ],
        "active_experts": [
            {"sample_index": 0, "layer": 0, "mode": mode, "experts": [0]},
            {"sample_index": 0, "layer": 1, "mode": mode, "experts": [1]},
        ],
        "layer_attribution": [
            {
                "sample_index": 0,
                "layer": 1,
                "experts": [1],
                "dense_vs_all_max_abs_error": 0.0,
                "dense_vs_selected_max_abs_error": 0.2,
                "selected_vs_all_max_abs_error": 0.2,
                "selected_vs_all_mean_abs_error": 0.04,
            }
        ],
        "memory": {
            "dense_parameter_count": 100,
            "carved_parameter_count": 100,
            "carved_buffer_count": 20,
        },
        "package": {"expert_count": 3},
        "replacements": {"replaced": [{"module_path": "model.layers.0.mlp"}]},
    }
