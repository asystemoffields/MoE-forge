from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.hf_runtime import (
    MoEForgeCarvedMLPModule,
    MoEForgeConfig,
    MoEForgeForCausalLM,
    MoEForgeHFError,
    replace_hf_mlp_modules,
)
from moeforge.materialize import materialize_carve_manifest
from moeforge.runtime import dense_gated_mlp_forward
from moeforge.wrapper import export_wrapper_package

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_hf_config_loads_from_wrapper_package(tmp_path: Path) -> None:
    package_dir = _write_wrapper_package(tmp_path)
    payload = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))

    assert payload["model_type"] == "moeforge_carved_moe"
    assert payload["architectures"] == ["MoEForgeForCausalLM"]
    assert payload["auto_map"]["AutoModelForCausalLM"] == "modeling_moeforge.MoEForgeForCausalLM"
    assert payload["moeforge_wrapper_config"] == "moeforge_config.json"
    assert (package_dir / "configuration_moeforge.py").exists()
    assert (package_dir / "modeling_moeforge.py").exists()

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


def test_hf_carved_mlp_module_runs_learned_token_router(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    package_dir = _write_wrapper_package_from_checkpoint(
        tmp_path,
        model_dir,
        layers=[0],
        intermediate_size=16,
        shared_channels=4,
        expert_channels=[4, 4, 4],
        token_router_top_k=1,
    )
    module = MoEForgeCarvedMLPModule.from_package(package_dir, layer=0)
    reloaded = MoEForgeCarvedMLPModule.from_package(package_dir, layer=0)
    assert module.token_router_top_k == 1
    assert module.token_router is not None
    assert reloaded.token_router is not None
    assert float(module.token_router.weight.abs().max().item()) > 0.0
    assert torch.allclose(module.token_router.weight, reloaded.token_router.weight)
    module.token_router.bias.data[:] = torch.tensor([-20.0, 20.0, -20.0])
    x = torch.randn(2, 3, 8)

    routed = module(x)
    selected = module.forward_selected(x, experts=[1])

    assert torch.allclose(routed, selected, atol=1e-5)


def test_hf_token_router_top_k_all_preserves_carved_sum(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    package_dir = _write_wrapper_package_from_checkpoint(
        tmp_path,
        model_dir,
        layers=[0],
        intermediate_size=16,
        shared_channels=4,
        expert_channels=[4, 4, 4],
        token_router_top_k=3,
    )
    module = MoEForgeCarvedMLPModule.from_package(package_dir, layer=0)
    x = torch.randn(2, 3, 8)

    routed = module(x)

    assert torch.allclose(routed, module.forward_all(x), atol=1e-5)
    assert module.last_router_summary["routing_weighting"] == "binary_straight_through"


def test_replace_hf_mlp_modules_preserves_tiny_llama_outputs(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    package_dir = _write_wrapper_package_from_checkpoint(
        tmp_path,
        model_dir,
        layers=[0, 1],
        intermediate_size=16,
        shared_channels=4,
        expert_channels=[4, 4, 4],
        copy_source_model=True,
    )
    dense = transformers.LlamaForCausalLM.from_pretrained(model_dir)
    patched = transformers.LlamaForCausalLM.from_pretrained(model_dir)
    dense.eval()
    patched.eval()

    with torch.no_grad():
        hidden_states = torch.randn(2, 3, dense.config.hidden_size)
        for layer in [0, 1]:
            original = dense.model.layers[layer].mlp(hidden_states)
            replacement = MoEForgeCarvedMLPModule.from_package(package_dir, layer=layer)
            assert torch.allclose(replacement(hidden_states), original, atol=1e-6)

        report = replace_hf_mlp_modules(patched, package_dir)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        dense_logits = dense(input_ids=input_ids).logits
        patched_logits = patched(input_ids=input_ids).logits

    assert [item.layer for item in report.replaced] == [0, 1]
    assert report.replaced[0].module_path == "model.layers.0.mlp"
    assert report.replaced[0].original_class == "LlamaMLP"
    assert torch.allclose(patched_logits, dense_logits, atol=1e-5)


def test_auto_model_loads_wrapper_package_as_causal_lm(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    package_dir = _write_wrapper_package_from_checkpoint(
        tmp_path,
        model_dir,
        layers=[0, 1],
        intermediate_size=16,
        shared_channels=4,
        expert_channels=[4, 4, 4],
        copy_source_model=True,
    )
    dense = transformers.AutoModelForCausalLM.from_pretrained(model_dir)
    moe = transformers.AutoModelForCausalLM.from_pretrained(package_dir)
    dense.eval()
    moe.eval()

    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        dense_logits = dense(input_ids=input_ids).logits
        moe_logits = moe(input_ids=input_ids).logits

    assert isinstance(moe, MoEForgeForCausalLM)
    assert moe.config.model_type == "moeforge_carved_moe"
    assert moe.config.source_model == "source-model"
    assert [item.layer for item in moe.replacement_report.replaced] == [0, 1]
    assert torch.allclose(moe_logits, dense_logits, atol=1e-5)


def test_auto_model_loads_wrapper_package_with_remote_code_stubs(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    package_dir = _write_wrapper_package_from_checkpoint(
        tmp_path,
        model_dir,
        layers=[0, 1],
        intermediate_size=16,
        shared_channels=4,
        expert_channels=[4, 4, 4],
        copy_source_model=True,
    )

    moe = transformers.AutoModelForCausalLM.from_pretrained(package_dir, trust_remote_code=True)

    assert isinstance(moe, MoEForgeForCausalLM)
    assert [item.layer for item in moe.replacement_report.replaced] == [0, 1]


def test_auto_model_can_default_to_all_experts_with_router_packaged(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    model_dir = _write_tiny_llama_checkpoint(tmp_path / "tiny-llama", transformers=transformers)
    package_dir = _write_wrapper_package_from_checkpoint(
        tmp_path,
        model_dir,
        layers=[0, 1],
        intermediate_size=16,
        shared_channels=4,
        expert_channels=[4, 4, 4],
        copy_source_model=True,
        token_router_top_k=1,
        default_expert_mode="all",
    )
    dense = transformers.AutoModelForCausalLM.from_pretrained(model_dir)
    moe = transformers.AutoModelForCausalLM.from_pretrained(package_dir, trust_remote_code=True)
    routed = transformers.AutoModelForCausalLM.from_pretrained(
        package_dir,
        trust_remote_code=True,
        moeforge_expert_mode="learned-router",
    )
    dense.eval()
    moe.eval()
    routed.eval()

    with torch.no_grad():
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        dense_logits = dense(input_ids=input_ids).logits
        moe_logits = moe(input_ids=input_ids).logits

    assert moe.config.default_expert_mode == "all"
    assert moe.replacement_report.replaced[0].default_experts == [0, 1, 2]
    assert routed.replacement_report.replaced[0].default_experts is None
    assert torch.allclose(moe_logits, dense_logits, atol=1e-5)


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


def _write_wrapper_package_from_checkpoint(
    tmp_path: Path,
    model: Path,
    *,
    layers: list[int],
    intermediate_size: int,
    shared_channels: int,
    expert_channels: list[int],
    copy_source_model: bool = False,
    token_router_top_k: int | None = None,
    default_expert_mode: str | None = None,
) -> Path:
    manifest_path = _write_manifest(
        tmp_path,
        model,
        layers=layers,
        intermediate_size=intermediate_size,
        shared_channels=shared_channels,
        expert_channels=expert_channels,
    )
    artifact_dir = tmp_path / "artifact"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=artifact_dir)
    package_dir = tmp_path / "wrapper"
    export_wrapper_package(
        manifest_path=manifest_path,
        artifact_path=artifact_dir / "carved-experts.safetensors",
        output_dir=package_dir,
        copy_artifact=True,
        copy_source_model=copy_source_model,
        token_router_top_k=token_router_top_k,
        default_expert_mode=default_expert_mode,
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


def _write_manifest(
    tmp_path: Path,
    model: Path,
    *,
    layers: list[int],
    intermediate_size: int = 4,
    shared_channels: int = 1,
    expert_channels: list[int] | None = None,
) -> Path:
    expert_channels = expert_channels or [2, 1]
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "strategy": "carved_mlp",
                "experts": len(expert_channels),
                "shared_ratio": 0.25,
                "moe_layers": layers,
                "layout": {
                    "layers": [
                        {
                            "layer": layer,
                            "intermediate_size": intermediate_size,
                            "shared_channels": shared_channels,
                            "expert_channels": expert_channels,
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


def _write_tiny_llama_checkpoint(path: Path, *, transformers) -> Path:
    torch.manual_seed(1234)
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
