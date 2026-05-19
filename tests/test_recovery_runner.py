from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.materialize import materialize_carve_manifest
from moeforge.recovery import write_recovery_plan
from moeforge.recovery_runner import export_recovered_wrapper, run_recovery, validate_recovered_wrapper
from moeforge.wrapper import export_wrapper_package

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")


def test_run_recovery_trains_tiny_wrapper_and_writes_checkpoints(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)
    recovery_dir = tmp_path / "recovery"
    config_path = tmp_path / "recovery.json"
    plan_path = tmp_path / "recovery-plan.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": str(model_dir),
                "student_model": str(model_dir),
                "wrapper": str(package_dir),
                "output_dir": str(recovery_dir),
                "train": {"input_ids": [[1, 2, 3, 4], [4, 3, 2, 1]]},
                "loss": {"teacher_kl_weight": 1.0, "logits_mse_weight": 0.1},
                "optimizer": {"learning_rate": 0.001},
                "schedule": {"steps": 2, "batch_size": 1, "save_every_steps": 1},
                "checkpoints": {"output_dir": "checkpoints"},
                "trainable": {"experts": True, "shared": False, "dense_backbone": False},
            }
        ),
        encoding="utf-8",
    )
    plan = write_recovery_plan(config_path=config_path, output_path=plan_path)

    report = run_recovery(plan_path=Path(str(plan["artifacts"]["plan_path"])))

    saved_report = json.loads((recovery_dir / "recovery-run-report.json").read_text(encoding="utf-8"))
    assert report["format"] == "moeforge_recovery_run"
    assert report["steps_completed"] == 2
    assert report["trainable_parameter_count"] > 0
    assert len(report["promoted_carved_parameters"]) > 0
    assert len(report["checkpoints"]) == 2
    assert Path(report["checkpoints"][0]["state_path"]).exists()
    assert report["losses"][0]["teacher_kl"] >= 0.0
    assert saved_report["replacement_report"]["replaced"][0]["module_path"] == "model.layers.0.mlp"

    recovered_dir = tmp_path / "recovered-wrapper"
    export_report = export_recovered_wrapper(
        checkpoint_path=recovery_dir / "checkpoints" / "checkpoint-step-2.json",
        wrapper_dir=package_dir,
        output_dir=recovered_dir,
    )
    recovered_config = json.loads((recovered_dir / "moeforge_config.json").read_text(encoding="utf-8"))
    assert export_report["updated_tensor_count"] > 0
    assert recovered_config["artifact_path"] == "recovered-carved-experts.safetensors"
    assert (recovered_dir / "recovered-carved-experts.safetensors").exists()
    assert (recovered_dir / "recovery-export-report.json").exists()

    validation = validate_recovered_wrapper(
        source_wrapper=package_dir,
        recovered_wrapper=recovered_dir,
        checkpoint_path=recovery_dir / "checkpoints" / "checkpoint-step-2.json",
        export_report_path=recovered_dir / "recovery-export-report.json",
        output_path=recovered_dir / "recovered-wrapper-validation.json",
    )
    assert validation["status"] == "validated"
    assert validation["tensor_comparison"]["updated_tensor_count"] == export_report["updated_tensor_count"]
    assert validation["tensor_comparison"]["source_tensor_count"] == validation["tensor_comparison"]["recovered_tensor_count"]
    assert validation["reload"]["loaded_layer_count"] == 2
    assert (recovered_dir / "recovered-wrapper-validation.json").exists()


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
    torch.manual_seed(9012)
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
