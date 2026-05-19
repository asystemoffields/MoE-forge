from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.materialize import materialize_carve_manifest
from moeforge.recovery import write_recovery_plan
from moeforge.recovery_runner import (
    _training_samples,
    export_recovered_wrapper,
    run_recovery,
    validate_recovered_wrapper,
)
from moeforge.wrapper import export_wrapper_package

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")
from safetensors.torch import load_file


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


def test_training_samples_tokenizes_text_file_manifest(tmp_path: Path) -> None:
    text_file = tmp_path / "train.txt"
    text_file.write_text("alpha beta\n\ngamma alpha", encoding="utf-8")
    config_path = tmp_path / "recovery.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": "tiny-teacher",
                "wrapper": "wrapper",
                "train": {"text_file": str(text_file), "sequence_length": 3},
            }
        ),
        encoding="utf-8",
    )
    plan = write_recovery_plan(config_path=config_path)

    input_ids, source = _training_samples(
        plan,
        model_ref="tiny-teacher",
        tokenizer_cls=_FakeTokenizer,
    )

    assert input_ids == [[11, 12], [13, 11]]
    assert source["kind"] == "text"
    assert source["token_counts"] == [2, 2]
    assert source["source"]["text_file"]["resolved_path"] == str(text_file)


def test_run_recovery_trains_and_exports_token_router(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir, token_router_top_k=2)
    recovery_dir = tmp_path / "router-recovery"
    config_path = tmp_path / "router-recovery.json"
    plan_path = tmp_path / "router-recovery-plan.json"
    config_path.write_text(
        json.dumps(
            {
                "teacher_model": str(model_dir),
                "student_model": str(model_dir),
                "wrapper": str(package_dir),
                "output_dir": str(recovery_dir),
                "train": {"input_ids": [[1, 2, 3, 4], [4, 3, 2, 1]]},
                "loss": {"teacher_kl_weight": 1.0, "logits_mse_weight": 0.05},
                "optimizer": {"learning_rate": 0.001},
                "schedule": {"steps": 2, "batch_size": 1, "save_every_steps": 2},
                "checkpoints": {"output_dir": "checkpoints"},
                "trainable": {"experts": False, "shared": False, "router": True, "dense_backbone": False},
            }
        ),
        encoding="utf-8",
    )
    plan = write_recovery_plan(config_path=config_path, output_path=plan_path)

    report = run_recovery(plan_path=Path(str(plan["artifacts"]["plan_path"])))

    assert report["promoted_carved_parameters"] == []
    assert len(report["promoted_router_parameters"]) == 4
    checkpoint_path = recovery_dir / "checkpoints" / "checkpoint-step-2.json"
    recovered_dir = tmp_path / "router-recovered-wrapper"
    export_report = export_recovered_wrapper(
        checkpoint_path=checkpoint_path,
        wrapper_dir=package_dir,
        output_dir=recovered_dir,
    )

    recovered_config = json.loads((recovered_dir / "moeforge_config.json").read_text(encoding="utf-8"))
    assert export_report["updated_tensor_count"] == 0
    assert export_report["updated_router_tensor_count"] == 4
    assert recovered_config["token_router_path"] == "learned-router.safetensors"
    assert (recovered_dir / "learned-router.safetensors").exists()

    validation = validate_recovered_wrapper(
        source_wrapper=package_dir,
        recovered_wrapper=recovered_dir,
        checkpoint_path=checkpoint_path,
        export_report_path=recovered_dir / "recovery-export-report.json",
    )
    assert validation["status"] == "validated"
    assert validation["checkpoint"]["promoted_router_parameter_count"] == 4


def test_export_recovered_wrapper_preserves_source_artifact_dtype(tmp_path: Path) -> None:
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama")
    package_dir = _write_wrapper_package(tmp_path, model_dir)
    source_artifact = package_dir / "carved-experts.safetensors"
    source_tensors = load_file(str(source_artifact), device="cpu")
    tensor_name = next(name for name in sorted(source_tensors) if ".experts." in name)
    source_tensor = source_tensors[tensor_name]
    state_path = tmp_path / "trainable-state.pt"
    checkpoint_path = tmp_path / "checkpoint.json"
    parameter_name = "promoted_expert_weight"
    checkpoint_tensor = (source_tensor + 0.25).to(torch.float16)
    torch.save({"trainable_state": {parameter_name: checkpoint_tensor}}, state_path)
    checkpoint_path.write_text(
        json.dumps(
            {
                "format": "moeforge_recovery_checkpoint",
                "step": 1,
                "metadata_path": str(checkpoint_path),
                "state_path": str(state_path),
                "trainable_parameter_count": checkpoint_tensor.numel(),
                "saved_tensor_count": 1,
                "promoted_carved_parameter_count": 1,
                "promoted_carved_parameters": [
                    {
                        "module": "model.layers.0.mlp",
                        "layer": 0,
                        "tensor": tensor_name,
                        "parameter": parameter_name,
                        "shape": list(checkpoint_tensor.shape),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    recovered_dir = tmp_path / "recovered-wrapper"
    export_report = export_recovered_wrapper(
        checkpoint_path=checkpoint_path,
        wrapper_dir=package_dir,
        output_dir=recovered_dir,
    )

    recovered = load_file(str(recovered_dir / "recovered-carved-experts.safetensors"), device="cpu")
    updated = export_report["updated_tensors"][0]
    assert recovered[tensor_name].dtype == source_tensor.dtype
    assert updated["source_dtype"] == "float32"
    assert updated["checkpoint_dtype"] == "float16"
    assert updated["export_dtype"] == "float32"
    assert updated["dtype_cast"] is True

    validation = validate_recovered_wrapper(
        source_wrapper=package_dir,
        recovered_wrapper=recovered_dir,
        checkpoint_path=checkpoint_path,
        export_report_path=recovered_dir / "recovery-export-report.json",
    )
    assert validation["status"] == "validated"
    assert validation["tensor_comparison"]["dtype_mismatches"] == []


def _write_wrapper_package(tmp_path: Path, model: Path, *, token_router_top_k: int | None = None) -> Path:
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)
    package_dir = tmp_path / "wrapper"
    export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        output_dir=package_dir,
        copy_artifact=True,
        token_router_top_k=token_router_top_k,
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


class _FakeTokenizer:
    vocab = {"alpha": 11, "beta": 12, "gamma": 13}

    @classmethod
    def from_pretrained(cls, model_ref: str) -> "_FakeTokenizer":
        assert model_ref == "tiny-teacher"
        return cls()

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        del truncation, return_attention_mask
        ids = [self.vocab[token] for token in text.split()]
        return {"input_ids": ids[:max_length]}
