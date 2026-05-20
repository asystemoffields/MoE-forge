from __future__ import annotations

import json
from pathlib import Path

from moeforge.cli import main
from moeforge.publish import check_publish_readiness


def test_publish_check_blocks_untrained_token_router(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper", token_router_top_k=1)
    eval_report = _write_eval_report(tmp_path / "eval-all.json", mode="all", max_abs_error=0.0)
    output = tmp_path / "publish.json"

    status = main(
        [
            "publish-check",
            "--wrapper",
            str(wrapper),
            "--eval-report",
            str(eval_report),
            "--allow-missing-sparse-eval",
            "--skip-native-load",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert payload["status"] == "blocked"
    assert any(check["name"] == "package.learned_router" for check in payload["checks"])


def test_publish_check_accepts_eval_and_recovery_evidence(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper", token_router_top_k=1, token_router_path="learned-router.safetensors")
    (wrapper / "learned-router.safetensors").write_bytes(b"placeholder")
    eval_all = _write_eval_report(tmp_path / "eval-all.json", mode="all", max_abs_error=0.0)
    eval_sparse = _write_eval_report(tmp_path / "eval-learned-router.json", mode="learned-router", max_abs_error=0.1)
    recovery_report = _write_recovery_report(tmp_path / "recovery.json")
    validation_report = _write_validation_report(tmp_path / "validation.json")

    report = check_publish_readiness(
        wrapper=wrapper,
        eval_reports=[eval_all, eval_sparse],
        recovery_report=recovery_report,
        validation_report=validation_report,
        require_recovery=True,
        trust_remote_code_load=False,
    )

    assert report["status"] == "ready"
    assert report["passed"]
    assert any(check["name"] == "recovery.validation_status" for check in report["checks"])
    assert any(check["name"] == "eval.learned-router.teacher_kl" for check in report["checks"])


def test_publish_check_requires_benchmark_when_requested(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper", token_router_top_k=None)
    eval_all = _write_eval_report(tmp_path / "eval-all.json", mode="all", max_abs_error=0.0)
    output = tmp_path / "publish.json"

    status = main(
        [
            "publish-check",
            "--wrapper",
            str(wrapper),
            "--eval-report",
            str(eval_all),
            "--allow-missing-sparse-eval",
            "--require-benchmark",
            "--skip-native-load",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert payload["status"] == "blocked"
    assert any(check["name"] == "benchmark.present" for check in payload["checks"])


def test_publish_check_accepts_benchmark_evidence(tmp_path: Path) -> None:
    wrapper = _write_wrapper(tmp_path / "wrapper", token_router_top_k=None)
    eval_all = _write_eval_report(tmp_path / "eval-all.json", mode="all", max_abs_error=0.0)
    benchmark = _write_benchmark_report(tmp_path / "benchmark.json")

    report = check_publish_readiness(
        wrapper=wrapper,
        eval_reports=[eval_all],
        benchmark_report=benchmark,
        require_benchmark=True,
        require_sparse_eval=False,
        trust_remote_code_load=False,
    )

    assert report["status"] == "ready"
    assert any(check["name"] == "benchmark.status" for check in report["checks"])
    assert any(check["name"] == "benchmark.core_retention" for check in report["checks"])


def _write_wrapper(path: Path, *, token_router_top_k: int | None, token_router_path: str | None = None) -> Path:
    path.mkdir(parents=True)
    (path / "carved-experts.safetensors").write_bytes(b"placeholder")
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "moeforge_carved_moe",
                "architectures": ["MoEForgeForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_moeforge.MoEForgeConfig",
                    "AutoModelForCausalLM": "modeling_moeforge.MoEForgeForCausalLM",
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "configuration_moeforge.py").write_text("from moeforge.hf_runtime import MoEForgeConfig\n", encoding="utf-8")
    (path / "modeling_moeforge.py").write_text("from moeforge.hf_runtime import MoEForgeForCausalLM\n", encoding="utf-8")
    (path / "MODEL_CARD.md").write_text("# Test Model\n", encoding="utf-8")
    (path / "moeforge_config.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model_type": "moeforge_carved_moe",
                "adapter_family": "llama",
                "source_model": "source-model",
                "manifest_path": "carve-manifest.json",
                "artifact_path": "carved-experts.safetensors",
                "router_plan_path": None,
                "token_router_top_k": token_router_top_k,
                "token_router_path": token_router_path,
                "activation": "silu",
                "expert_count": 2,
                "layers": [
                    {
                        "layer": 0,
                        "width": 16,
                        "tensor_prefix": "moe.layers.0.mlp",
                        "expert_count": 2,
                        "shared_channels": 4,
                        "expert_channels": [6, 6],
                    }
                ],
                "warnings": [],
                "references": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_eval_report(path: Path, *, mode: str, max_abs_error: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "moeforge_eval_report",
                "sample_count": 1,
                "max_abs_error": max_abs_error,
                "samples": [{"expert_mode": mode}],
                "summary": {
                    "average_teacher_kl_loss": 0.01,
                    "average_nll_loss_delta": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_recovery_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "moeforge_recovery_experiment",
                "summary": {
                    "recovered_wrapper_validation_status": "validated",
                    "recovered_updated_router_tensor_count": 2,
                },
                "quality_trends": {"before_after_quality": {"mode_count": 2}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_validation_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "moeforge_recovered_wrapper_validation",
                "status": "validated",
                "passed": True,
                "reload": {"loaded_layer_count": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_benchmark_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "moeforge_benchmark_comparison",
                "status": "passed",
                "passed": True,
                "comparable_task_count": 2,
                "summary": {
                    "average_retention": 0.98,
                    "worst_core_retention": 0.95,
                },
            }
        ),
        encoding="utf-8",
    )
    return path
