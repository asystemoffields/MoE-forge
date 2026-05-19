from __future__ import annotations

import json
from pathlib import Path

import pytest

from moeforge.carve import build_carve_manifest
from moeforge.materialize import materialize_carve_manifest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def test_materialize_carve_manifest_writes_sliced_tensors(tmp_path: Path) -> None:
    model = _write_real_llama_checkpoint(tmp_path / "model")
    recipe_path = _write_recipe(tmp_path)
    manifest = build_carve_manifest(model=str(model), recipe_path=recipe_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    output_dir = tmp_path / "out"
    report = materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir)

    output = safetensors_torch.load_file(str(output_dir / "carved-experts.safetensors"))
    assert "moe.layers.0.mlp.shared.gate.weight" in output
    assert output["moe.layers.0.mlp.shared.gate.weight"].tolist() == [[0.0, 1.0]]
    assert output["moe.layers.0.mlp.experts.0.gate.weight"].tolist() == [[2.0, 3.0], [4.0, 5.0]]
    assert output["moe.layers.0.mlp.experts.1.gate.weight"].tolist() == [[6.0, 7.0]]
    assert output["moe.layers.0.mlp.shared.down.weight"].tolist() == [[200.0], [204.0]]
    assert output["moe.layers.0.mlp.experts.0.down.weight"].tolist() == [[201.0, 202.0], [205.0, 206.0]]
    assert len(report.tensors) == 9
    assert (output_dir / "carve-apply-report.json").exists()


def test_materialize_carve_manifest_dry_run_reports_shapes(tmp_path: Path) -> None:
    model = _write_real_llama_checkpoint(tmp_path / "model")
    recipe_path = _write_recipe(tmp_path)
    manifest = build_carve_manifest(model=str(model), recipe_path=recipe_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    output_dir = tmp_path / "dry"
    report = materialize_carve_manifest(manifest_path=manifest_path, output_dir=output_dir, dry_run=True)

    tensors = {item.name: item for item in report.tensors}
    assert tensors["moe.layers.0.mlp.shared.up.weight"].shape == [1, 2]
    assert tensors["moe.layers.0.mlp.experts.0.down.weight"].shape == [2, 2]
    assert (output_dir / "carve-apply-dry-run.json").exists()
    assert report.to_dict()["tensor_count"] == 9


def _write_real_llama_checkpoint(path: Path) -> Path:
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
    tensors = {
        "model.layers.0.mlp.gate_proj.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.mlp.up_proj.weight": torch.arange(100, 108, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.mlp.down_proj.weight": torch.arange(200, 208, dtype=torch.float32).reshape(2, 4),
    }
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    return path


def _write_recipe(tmp_path: Path) -> Path:
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
    return recipe_path
