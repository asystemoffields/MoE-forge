from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.materialize import materialize_carve_manifest
from moeforge.runtime import (
    CarvedGatedMLP,
    dense_gated_mlp_forward,
    reconstruct_weight_from_carved,
    verify_carved_artifact,
)

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_verify_carved_artifact_reconstructs_weights(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    output_dir = tmp_path / "out"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir)

    report = verify_carved_artifact(
        manifest_path=manifest_path,
        artifact_path=output_dir / "carved-experts.safetensors",
    )

    assert report.passed is True
    assert report.tensor_count == 3
    assert all(item.max_abs_error == 0.0 for item in report.tensors)


def test_reconstruct_weight_from_carved_matches_source(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    output_dir = tmp_path / "out"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir)

    carved = safetensors_torch.load_file(str(output_dir / "carved-experts.safetensors"))
    reconstructed = reconstruct_weight_from_carved(
        carved=carved,
        layer=0,
        role="down",
        source_shape=[2, 4],
        channel_axis=1,
        shared_channels=[0],
        expert_channels=[[1, 2], [3]],
    )

    assert reconstructed.tolist() == [[200.0, 201.0, 202.0, 203.0], [204.0, 205.0, 206.0, 207.0]]


def test_carved_gated_mlp_forward_all_matches_dense(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    output_dir = tmp_path / "out"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir)

    source = safetensors_torch.load_file(str(model / "model.safetensors"))
    x = torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float32)
    dense = dense_gated_mlp_forward(
        x=x,
        gate_weight=source["model.layers.0.mlp.gate_proj.weight"],
        up_weight=source["model.layers.0.mlp.up_proj.weight"],
        down_weight=source["model.layers.0.mlp.down_proj.weight"],
    )
    carved = CarvedGatedMLP.from_artifact(
        manifest_path=manifest_path,
        artifact_path=output_dir / "carved-experts.safetensors",
        layer=0,
    ).forward_all(x)

    assert torch.allclose(carved, dense)


def test_carved_gated_mlp_forward_selected_accepts_subset(tmp_path: Path) -> None:
    model = _write_checkpoint(tmp_path / "model")
    manifest_path = _write_manifest(tmp_path, model)
    output_dir = tmp_path / "out"
    materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir)

    x = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    runtime = CarvedGatedMLP.from_artifact(
        manifest_path=manifest_path,
        artifact_path=output_dir / "carved-experts.safetensors",
        layer=0,
    )

    selected = runtime.forward_selected(x, experts=[0])
    all_experts = runtime.forward_selected(x, experts=[0, 1])

    assert selected.shape == all_experts.shape
    assert not torch.allclose(selected, all_experts)


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
