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
    assert payload["memory"]["dense_parameter_count"] > 0
    assert payload["package"]["model_type"] == "moeforge_carved_moe"


def test_eval_hf_cli_writes_report(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)
    output = tmp_path / "eval-report.json"

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
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert status == 0
    assert payload["passed"] is True
    assert payload["sample_count"] == 1


def test_evaluate_hf_dense_vs_carved_validates_input_ids(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)

    with pytest.raises(EvaluationError, match="non-empty JSON list"):
        evaluate_hf_dense_vs_carved(
            model=model_dir,
            package_dir=package_dir,
            input_ids=[],
        )


def _write_wrapper_package(tmp_path: Path, model: Path) -> Path:
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)
    package_dir = tmp_path / "wrapper"
    export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
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
