from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.materialize import materialize_carve_manifest
from moeforge.runtime import dense_gated_mlp_forward
from moeforge.wrapper import WrapperError, export_wrapper_package, load_layer_runtime, load_router_plan, load_wrapper_config

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_export_wrapper_package_loads_runtime(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)
    router_path = tmp_path / "router-plan.json"
    router_path.write_text(
        json.dumps({"default_pool": [0, 1], "documents": [{"document_index": 0, "experts": [0]}]}),
        encoding="utf-8",
    )

    package_dir = tmp_path / "wrapper"
    config = export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        router_plan_path=router_path,
        output_dir=package_dir,
        copy_artifact=True,
    )
    config_path = package_dir / "moeforge_config.json"

    assert config.model_type == "moeforge_carved_moe"
    assert (package_dir / "carved-experts.safetensors").exists()
    assert (package_dir / "carve-manifest.json").exists()
    assert (package_dir / "router-plan.json").exists()
    assert load_wrapper_config(config_path).layers[0].layer == 0
    assert load_router_plan(config_path)["default_pool"] == [0, 1]

    source = safetensors_torch.load_file(str(model / "model.safetensors"))
    x = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    dense = dense_gated_mlp_forward(
        x=x,
        gate_weight=source["model.layers.0.mlp.gate_proj.weight"],
        up_weight=source["model.layers.0.mlp.up_proj.weight"],
        down_weight=source["model.layers.0.mlp.down_proj.weight"],
    )
    runtime = load_layer_runtime(config_path, layer=0)

    assert torch.allclose(runtime.forward_all(x), dense)
    assert runtime.forward_with_router(x, router_plan=load_router_plan(config_path), document_index=0).shape == dense.shape


def test_export_wrapper_package_can_reference_external_artifact(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)

    package_dir = tmp_path / "wrapper"
    export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        output_dir=package_dir,
        copy_artifact=False,
    )
    payload = json.loads((package_dir / "moeforge_config.json").read_text(encoding="utf-8"))

    assert Path(payload["artifact_path"]).is_absolute()
    assert not (package_dir / "carved-experts.safetensors").exists()


def test_export_wrapper_package_can_copy_source_model(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)

    package_dir = tmp_path / "wrapper"
    config = export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        output_dir=package_dir,
        copy_artifact=True,
        copy_source_model=True,
    )
    payload = json.loads((package_dir / "moeforge_config.json").read_text(encoding="utf-8"))
    hf_payload = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))

    assert config.source_model == "source-model"
    assert payload["source_model"] == "source-model"
    assert hf_payload["source_model"] == "source-model"
    assert (package_dir / "source-model" / "config.json").exists()
    assert (package_dir / "source-model" / "model.safetensors").exists()


def test_export_wrapper_package_can_reexport_from_package_paths(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
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

    export_wrapper_package(
        manifest_path=package_dir / "carve-manifest.json",
        artifact_path=package_dir / "carved-experts.safetensors",
        output_dir=package_dir,
        copy_artifact=True,
    )

    assert load_layer_runtime(package_dir / "moeforge_config.json", layer=0).expert_count == 2


def test_export_wrapper_package_validates_activation(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)

    with pytest.raises(WrapperError, match="unsupported activation"):
        export_wrapper_package(
            manifest_path=manifest_path,
            artifact_path=artifact_dir / "carved-experts.safetensors",
            output_dir=tmp_path / "wrapper",
            activation="relu",
        )


def _write_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_hidden_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "model.layers.0.mlp.up_proj.weight": torch.arange(100, 108, dtype=torch.float32).reshape(4, 2),
            "model.layers.0.mlp.down_proj.weight": torch.arange(200, 208, dtype=torch.float32).reshape(2, 4),
        },
        str(path / "model.safetensors"),
    )
    return path


def _write_manifest(tmp_path: Path, model: Path) -> Path:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "strategy": "carved_mlp",
                "experts": 2,
                "shared_ratio": 0.25,
                "moe_layers": [0],
                "layout": {
                    "layers": [
                        {
                            "layer": 0,
                            "intermediate_size": 4,
                            "shared_channels": 1,
                            "expert_channels": [2, 1],
                        }
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
