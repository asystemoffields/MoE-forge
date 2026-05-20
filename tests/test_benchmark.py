from __future__ import annotations

import json
from pathlib import Path

from moeforge.benchmark import BenchmarkCompareOptions, BenchmarkPlanOptions, compare_benchmark_reports, write_benchmark_plan
from moeforge.cli import main


def test_benchmark_plan_writes_smollm_base_commands(tmp_path: Path) -> None:
    output = tmp_path / "benchmark-plan.json"

    plan = write_benchmark_plan(
        BenchmarkPlanOptions(
            source_model="HuggingFaceTB/SmolLM-135M",
            moe_model="outputs/smollm-moe/recovered-wrapper",
            output_path=output,
            output_dir=tmp_path / "benchmarks",
            max_samples=25,
            batch_size=2,
        )
    )

    assert output.exists()
    assert plan["suite"] == "smollm-base"
    assert "custom|hellaswag|0|1" in plan["task_spec"]
    assert "custom|mmlu_cloze:abstract_algebra|0|1" in plan["task_spec"]
    assert plan["commands"]["dense"].startswith("lighteval accelerate")
    assert "model_name=HuggingFaceTB/SmolLM-135M,batch_size=2" in plan["commands"]["dense"]
    assert "trust_remote_code=True" in plan["commands"]["moe"]
    assert plan["release_gate"]["required_artifact"].endswith("benchmark-compare.json")


def test_benchmark_plan_supports_instruct_chat_template(tmp_path: Path) -> None:
    output = tmp_path / "benchmark-plan.json"
    status = main(
        [
            "benchmark-plan",
            "--source-model",
            "HuggingFaceTB/SmolLM-135M-Instruct",
            "--moe-model",
            "outputs/smollm-instruct-moe/recovered-wrapper",
            "--suite",
            "smollm-instruct",
            "--output",
            str(output),
        ]
    )

    plan = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert "lighteval|ifeval|0|0" in plan["task_spec"]
    assert "lighteval|mt_bench|0|0" in plan["task_spec"]
    assert "--use-chat-template" in plan["commands"]["dense"]
    assert any(task["priority"] == "chat_core" for task in plan["tasks"])


def test_benchmark_compare_passes_when_moe_retains_scores(tmp_path: Path) -> None:
    dense = _write_results(
        tmp_path / "dense.json",
        {
            "hellaswag": {"loglikelihood_acc_norm_nospace": 0.412},
            "piqa": {"loglikelihood_acc_norm_nospace": 0.684},
            "mmlu_cloze:abstract_algebra": {"loglikelihood_acc_norm_nospace": 0.30},
            "mmlu_cloze:anatomy": {"loglikelihood_acc_norm_nospace": 0.32},
        },
    )
    moe = _write_results(
        tmp_path / "moe.json",
        {
            "hellaswag": {"loglikelihood_acc_norm_nospace": 0.405},
            "piqa": {"loglikelihood_acc_norm_nospace": 0.680},
            "mmlu_cloze:abstract_algebra": {"loglikelihood_acc_norm_nospace": 0.29},
            "mmlu_cloze:anatomy": {"loglikelihood_acc_norm_nospace": 0.31},
        },
    )

    report = compare_benchmark_reports(
        BenchmarkCompareOptions(dense_report=dense, moe_report=moe, output_path=tmp_path / "compare.json")
    )

    assert report["passed"]
    assert report["summary"]["average_retention"] > 0.95
    assert any(row["task"] == "hellaswag" for row in report["results"])


def test_benchmark_compare_blocks_large_chat_core_drop(tmp_path: Path) -> None:
    dense = _write_results(
        tmp_path / "dense.json",
        {
            "ifeval": {"prompt_level_strict_acc": 0.172},
            "mt_bench": {"score": 1.68},
        },
    )
    moe = _write_results(
        tmp_path / "moe.json",
        {
            "ifeval": {"prompt_level_strict_acc": 0.05},
            "mt_bench": {"score": 0.40},
        },
    )

    status = main(
        [
            "benchmark-compare",
            "--dense-report",
            str(dense),
            "--moe-report",
            str(moe),
            "--suite",
            "smollm-instruct",
            "--output",
            str(tmp_path / "compare.json"),
        ]
    )
    report = json.loads((tmp_path / "compare.json").read_text(encoding="utf-8"))

    assert status == 1
    assert report["status"] == "blocked"
    assert report["summary"]["worst_core_retention"] < 0.90


def _write_results(path: Path, results: dict[str, dict[str, float]]) -> Path:
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return path
