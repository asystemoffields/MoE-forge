from __future__ import annotations

from pathlib import Path

from moeforge.model_info import ModelInfo
from moeforge.planner import PlanOptions, plan_conversion


def test_balanced_plan_for_dense_model() -> None:
    info = ModelInfo(
        path=Path("model"),
        source_format="hf",
        architecture="LlamaForCausalLM",
        layer_count=32,
        hidden_size=4096,
        intermediate_size=11008,
        dense=True,
        adapter_family="llama",
        adapter={"supported_backends": ["carved_mlp", "sparse_upcycle", "adapter_moe"]},
    )

    recipe = plan_conversion(info, PlanOptions(goal="balanced"))

    assert recipe.strategy == "carved_mlp"
    assert recipe.adapter_family == "llama"
    assert recipe.experts == 8
    assert recipe.top_k == 2
    assert recipe.shared_ratio == 0.25
    assert recipe.moe_layers[0] == 8
    assert recipe.layout["active_fraction_mean"] is not None


def test_speed_plan_uses_top_one() -> None:
    info = ModelInfo(
        path=Path("model.gguf"),
        source_format="gguf",
        architecture="gemma4",
        layer_count=35,
        dense=True,
    )

    recipe = plan_conversion(info, PlanOptions(goal="speed", target="gguf"))

    assert recipe.top_k == 1
    assert recipe.export.quantize_after_export is True
    assert recipe.warnings


def test_layout_uses_per_layer_intermediate_sizes() -> None:
    info = ModelInfo(
        path=Path("model.gguf"),
        source_format="gguf",
        architecture="gemma4",
        layer_count=3,
        hidden_size=1536,
        intermediate_size=6144,
        intermediate_sizes=[6144, 12288, 12288],
        dense=True,
    )

    recipe = plan_conversion(info, PlanOptions(goal="balanced", experts=4, top_k=2, moe_layers="0:2"))

    layers = recipe.layout["layers"]
    assert layers[0]["intermediate_size"] == 6144
    assert layers[1]["intermediate_size"] == 12288
    assert layers[1]["active_channels_per_token"] > layers[0]["active_channels_per_token"]


def test_strategy_falls_back_to_supported_backend() -> None:
    info = ModelInfo(
        path=Path("gemma"),
        source_format="hf",
        architecture="Gemma4ForConditionalGeneration",
        layer_count=35,
        hidden_size=1536,
        intermediate_size=6144,
        dense=True,
        adapter_family="gemma",
        adapter={"supported_backends": ["carved_mlp", "adapter_moe"]},
        metadata={"checkpoint": {"has_weights": False}},
        warnings=["source warning"],
    )

    recipe = plan_conversion(info, PlanOptions(goal="quality"))

    assert recipe.strategy == "carved_mlp"
    assert any("sparse_upcycle" in warning for warning in recipe.warnings)
    assert any("no local checkpoint weights" in warning for warning in recipe.warnings)
    assert "source warning" in recipe.warnings
