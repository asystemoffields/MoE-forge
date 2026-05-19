from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.reports import render_eval_html_report, write_eval_html_report


def test_render_eval_html_report_escapes_and_summarizes() -> None:
    html = render_eval_html_report(_report())

    assert "MoE Forge Eval Report" in html
    assert "tiny &lt;model&gt;" in html
    assert "Layer Attribution" in html
    assert "Selected-All Max" in html
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


def _report() -> dict:
    return {
        "model": "tiny <model>",
        "package_dir": "wrapper",
        "source_model": "dense-source",
        "adapter_family": "llama",
        "sample_count": 1,
        "passed": False,
        "max_abs_error": 0.25,
        "mean_abs_error": 0.05,
        "warnings": ["approximate subset comparison"],
        "summary": {
            "average_dense_latency_s": 0.01,
            "average_carved_latency_s": 0.02,
            "average_carved_vs_dense_latency_ratio": 2.0,
            "worst_sample_index": 0,
            "worst_sample_max_abs_error": 0.25,
            "worst_layer_sample_index": 0,
            "worst_layer": 1,
            "worst_layer_selected_vs_all_max_abs_error": 0.2,
        },
        "samples": [
            {
                "index": 0,
                "source": "input_ids:0",
                "expert_mode": "router",
                "max_abs_error": 0.25,
                "mean_abs_error": 0.05,
                "carved_vs_dense_latency_ratio": 2.0,
                "allclose": False,
            }
        ],
        "active_experts": [
            {"sample_index": 0, "layer": 0, "mode": "router", "experts": [0]},
            {"sample_index": 0, "layer": 1, "mode": "router", "experts": [1]},
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
