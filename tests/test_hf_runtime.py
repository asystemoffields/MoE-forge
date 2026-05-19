from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.hf_runtime import MoEForgeCarvedMLPModule, MoEForgeConfig, MoEForgeHFError
from moeforge.materialize import materialize_carve_manifest
from moeforge.runtime import dense_gated_mlp_forward
from moeforge.wrapper import export_wrapper_package

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_hf_config_loads_from_wrapper_package(tmp_path: Path) -> None:
    package_dir = _write_wrapper_package(tmp_path)
    payload = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))

    assert payload["model_type"] == "moeforge_carved_moe"
    assert payload["architectures"] == ["MoEForgeCarvedMLPModule"]
    assert payload["moeforge_wrapper_config"] == "moeforge_config.json"

    config = MoEForgeConfig.from_package(package_dir)

    assert config.layer_ids() == [0]
    assert config.expert_count == 2
    assert config.artifact_path == "carved-experts.safetensors"


def test_hf_config_supports_transformers_from_pretrained(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    package_dir = _write_wrapper_package(tmp_path)

    config = MoEForgeConfig.from_pretrained(package_dir)
    saved_dir = tmp_path / "saved-config"
    config.save_pretrained(saved_dir)
    loaded = MoEForgeConfig.from_pretrained(saved_dir)

    assert loaded.layer_ids() == [0]
    assert loaded.expert_count == 2


def test_hf_carved_mlp_module_matches_dense_and_router(tmp_path: Path) -> None:
    package_dir = _write_wrapper_package(tmp_path)
    module = MoEForgeCarvedMLPModule.from_package(package_dir)
    source = safetensors_torch.load_file(str(tmp_path / "model" / "model.safetensors"))
    x = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    dense = dense_gated_mlp_forward(
        x=x,
        gate_weight=source["model.layers.0.mlp.gate_proj.weight"],
        up_weight=source["model.layers.0.mlp.up_proj.weight"],
        down_weight=source["model.layers.0.mlp.down_proj.weight"],
    )

    assert torch.allclose(module.forward_all(x), dense)
    assert module.select_experts(document_index=0) == [0]
    assert torch.allclose(module.forward_with_router(x, document_index=0), module.forward_selected(x, experts=[0]))

    module = module.to(dtype=torch.float64)
    x64 = x.to(dtype=torch.float64)
    dense64 = dense_gated_mlp_forward(
        x=x64,
        gate_weight=source["model.layers.0.mlp.gate_proj.weight"].to(dtype=torch.float64),
        up_weight=source["model.layers.0.mlp.up_proj.weight"].to(dtype=torch.float64),
        down_weight=source["model.layers.0.mlp.down_proj.weight"].to(dtype=torch.float64),
    )

    assert torch.allclose(module.forward_all(x64), dense64)


def test_hf_module_requires_layer_for_multi_layer_package(tmp_path: Path) -> None:
    package_dir = _write_wrapper_package(tmp_path, layers=[0, 1])

    with pytest.raises(MoEForgeHFError, match="layer must be provided"):
        MoEForgeCarvedMLPModule.from_package(package_dir)

    assert MoEForgeCarvedMLPModule.from_package(package_dir, layer=1).layer == 1


def _write_wrapper_package(tmp_path: Path, *, layers: list[int] | None = None) -> Path:
    model = _write_checkpoint(tmp_path / "model", layers=layers or [0])
    manifest_path = _write_manifest(tmp_path, model, layers=layers or [0])
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)
    router_path = tmp_path / "router-plan.json"
    router_path.write_text(
        json.dumps({"default_pool": [0, 1], "documents": [{"document_index": 0, "experts": [0]}]}),
        encoding="utf-8",
    )

    package_dir = tmp_path / "wrapper"
    export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        router_plan_path=router_path,
        output_dir=package_dir,
        copy_artifact=True,
    )
    return package_dir


def _write_checkpoint(path: Path, *, layers: list[int]) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_hidden_layers": max(layers) + 1,
            }
        ),
        encoding="utf-8",
    )
    tensors = {}
    for layer in layers:
        offset = layer * 1000
        tensors[f"model.layers.{layer}.mlp.gate_proj.weight"] = (torch.arange(8, dtype=torch.float32) + offset).reshape(4, 2)
        tensors[f"model.layers.{layer}.mlp.up_proj.weight"] = (torch.arange(100, 108, dtype=torch.float32) + offset).reshape(4, 2)
        tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = (torch.arange(200, 208, dtype=torch.float32) + offset).reshape(2, 4)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    return path


def _write_manifest(tmp_path: Path, model: Path, *, layers: list[int]) -> Path:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "strategy": "carved_mlp",
                "experts": 2,
                "shared_ratio": 0.25,
                "moe_layers": layers,
                "layout": {
                    "layers": [
                        {
                            "layer": layer,
                            "intermediate_size": 4,
                            "shared_channels": 1,
                            "expert_channels": [2, 1],
                        }
                        for layer in layers
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
