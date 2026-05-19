from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.cli import main
from moeforge.evaluation import EvaluationError, evaluate_hf_dense_vs_carved
from moeforge.materialize import materialize_carve_manifest
from moeforge.wrapper import export_wrapper_package

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")


def test_evaluate_hf_dense_vs_carved_reports_logits_parity(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)

    report = evaluate_hf_dense_vs_carved(
        model=model_dir,
        package_dir=package_dir,
        input_ids=[[1, 2, 3, 4], [4, 3, 2, 1]],
    )
    payload = report.to_dict()

    assert report.passed
    assert report.sample_count == 2
    assert report.max_abs_error <= 1e-5
    assert payload["replacements"]["replaced"][0]["module_path"] == "model.layers.0.mlp"
    assert payload["active_experts"][0]["experts"] == [0, 1, 2]
    assert len(payload["layer_attribution"]) == 4
    assert payload["layer_attribution"][0]["dense_vs_all_max_abs_error"] <= 1e-6
    assert payload["layer_attribution"][0]["selected_vs_all_max_abs_error"] <= 1e-6
    assert payload["summary"]["average_carved_latency_s"] >= 0.0
    assert payload["summary"]["average_teacher_kl_loss"] <= 1e-7
    assert abs(payload["summary"]["average_nll_loss_delta"]) <= 1e-6
    assert payload["summary"]["loss_token_count"] == 6
    assert payload["samples"][0]["teacher_kl_loss"] <= 1e-7
    assert payload["samples"][0]["loss_token_count"] == 3
    assert payload["summary"]["worst_layer_selected_vs_all_max_abs_error"] <= 1e-6
    assert payload["memory"]["dense_parameter_count"] > 0
    assert payload["package"]["model_type"] == "moeforge_carved_moe"


def test_eval_hf_cli_writes_report(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)
    output = tmp_path / "eval-report.json"
    html_output = tmp_path / "eval-report.html"

    status = main(
        [
            "eval-hf",
            str(model_dir),
            "--wrapper",
            str(package_dir),
            "--input-ids-json",
            "[[1, 2, 3, 4]]",
            "--output",
            str(output),
            "--html-output",
            str(html_output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert status == 0
    assert payload["passed"] is True
    assert payload["sample_count"] == 1
    assert html_output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_eval_batch_cli_writes_mode_reports_and_comparison(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir, with_router=True)
    output_dir = tmp_path / "batch"
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "model": str(model_dir),
                "wrapper": str(package_dir),
                "output_dir": str(output_dir),
                "expert_modes": ["all", "default-pool", "router"],
                "input_ids": [[1, 2, 3, 4], [4, 3, 2, 1]],
                "write_html": True,
                "recovery_eval": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )

    status = main(["eval-batch", "--config", str(config_path)])

    manifest = json.loads((output_dir / "eval-batch-manifest.json").read_text(encoding="utf-8"))
    assert status == 0
    assert manifest["run_count"] == 3
    assert manifest["completed_report_count"] == 3
    assert [run["expert_mode"] for run in manifest["runs"]] == ["all", "default-pool", "router"]
    assert manifest["runs"][0]["status"] == "passed"
    assert manifest["comparison"]["status"] == "written"
    assert manifest["recovery_eval"]["enabled"] is True
    assert (output_dir / "eval-all.json").exists()
    assert (output_dir / "eval-default_pool.html").exists()
    assert (output_dir / "eval-compare.html").exists()


def test_evaluate_hf_dense_vs_carved_reports_routed_subset_tradeoff(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir, with_router=True)

    report = evaluate_hf_dense_vs_carved(
        model=model_dir,
        package_dir=package_dir,
        input_ids=[[1, 2, 3, 4], [4, 3, 2, 1]],
        expert_mode="router",
    )
    payload = report.to_dict()

    assert not report.passed
    assert report.max_abs_error > 0.0
    assert payload["active_experts"][0]["experts"] == [0]
    assert payload["active_experts"][2]["experts"] == [1]
    assert payload["samples"][0]["expert_mode"] == "router"
    assert payload["samples"][0]["active_experts"][0]["mode"] == "router"
    assert payload["samples"][0]["carved_vs_dense_latency_ratio"] is not None
    assert payload["samples"][0]["teacher_kl_loss"] > 0.0
    assert payload["summary"]["average_carved_nll_loss"] is not None
    assert payload["layer_attribution"][0]["dense_vs_all_max_abs_error"] <= 1e-6
    assert payload["layer_attribution"][0]["selected_vs_all_max_abs_error"] > 0.0
    assert payload["summary"]["worst_layer"] in [0, 1]
    assert payload["summary"]["worst_layer_selected_vs_all_max_abs_error"] > 0.0


def test_evaluate_hf_dense_vs_carved_validates_input_ids(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)

    with pytest.raises(EvaluationError, match="non-empty JSON list"):
        evaluate_hf_dense_vs_carved(
            model=model_dir,
            package_dir=package_dir,
            input_ids=[],
        )


def _write_wrapper_package(tmp_path: Path, model: Path, *, with_router: bool = False) -> Path:
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)
    package_dir = tmp_path / "wrapper"
    router_path = None
    if with_router:
        router_path = tmp_path / "router-plan.json"
        router_path.write_text(
            json.dumps(
                {
                    "strategy": "document_pool_then_token_router",
                    "expert_count": 3,
                    "pool_size": 1,
                    "default_pool": [2],
                    "documents": [
                        {"document_index": 0, "text_sha256": "", "experts": [0], "scores": [3.0, 1.0, 0.0]},
                        {"document_index": 1, "text_sha256": "", "experts": [1], "scores": [0.0, 3.0, 1.0]},
                    ],
                }
            ),
            encoding="utf-8",
        )
    export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        router_plan_path=router_path,
        output_dir=package_dir,
        copy_artifact=True,
    )
    return package_dir


def _write_manifest(tmp_path: Path, model: Path) -> Path:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "strategy": "carved_mlp",
                "experts": 3,
                "shared_ratio": 0.25,
                "moe_layers": [0, 1],
                "layout": {
                    "layers": [
                        {
                            "layer": layer,
                            "intermediate_size": 16,
                            "shared_channels": 4,
                            "expert_channels": [4, 4, 4],
                        }
                        for layer in [0, 1]
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = build_carve_manifest(model=str(model), recipe_path=recipe_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return manifest_path


def _write_tiny_llama_checkpoint(path: Path) -> Path:
    torch.manual_seed(5678)
    config = transformers.LlamaConfig(
        attention_bias=False,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
        vocab_size=32,
    )
    model = transformers.LlamaForCausalLM(config)
    model.save_pretrained(path, safe_serialization=True)
    return path
