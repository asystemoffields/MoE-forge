from __future__ import annotations

import json
import struct
from pathlib import Path

from moeforge.carve import build_carve_manifest


def test_build_carve_manifest_from_recipe_layout(tmp_path: Path) -> None:
    model = _write_tiny_llama_checkpoint(tmp_path / "model")
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
    payload = manifest.to_dict()

    layer = payload["layers"][0]
    assert layer["shared_channels"] == [0]
    assert layer["expert_channels"] == [[1, 2], [3]]
    assert layer["tensors"][0]["channel_axis"] == 0
    assert layer["tensors"][2]["role"] == "down"
    assert layer["tensors"][2]["channel_axis"] == 1


def test_build_carve_manifest_prefers_profile_assignment(tmp_path: Path) -> None:
    model = _write_tiny_llama_checkpoint(tmp_path / "model")
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
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "modules": {
                    "model.layers.0.mlp.gate_proj": {
                        "target": {"layer": 0, "role": "gate"},
                        "assignment": {
                            "available": True,
                            "width": 4,
                            "shared_channels": [3],
                            "experts": [
                                {"expert": 0, "channels": [0, 2]},
                                {"expert": 1, "channels": [1]},
                            ],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_carve_manifest(model=str(model), recipe_path=recipe_path, profile_path=profile_path)
    layer = manifest.to_dict()["layers"][0]

    assert layer["shared_channels"] == [3]
    assert layer["expert_channels"] == [[0, 2], [1]]


def _write_tiny_llama_checkpoint(path: Path) -> Path:
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
    _write_safetensors_stub(
        path / "model.safetensors",
        {
            "model.layers.0.mlp.gate_proj.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [0, 0]},
            "model.layers.0.mlp.up_proj.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [0, 0]},
            "model.layers.0.mlp.down_proj.weight": {"dtype": "F16", "shape": [2, 4], "data_offsets": [0, 0]},
        },
    )
    return path


def _write_safetensors_stub(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)

