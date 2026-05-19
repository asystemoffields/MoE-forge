from __future__ import annotations

import json
import struct
from pathlib import Path

from moeforge.inspectors import inspect_model
from moeforge.tensors import build_hf_tensor_index, read_safetensors_header


def test_read_safetensors_header(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors_stub(
        path,
        {
            "model.layers.0.mlp.gate_proj.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 0]},
        },
    )

    header = read_safetensors_header(path)

    assert header["model.layers.0.mlp.gate_proj.weight"]["shape"] == [8, 4]


def test_build_hf_tensor_index_from_safetensors_header(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors_stub(
        path,
        {
            "model.layers.0.mlp.gate_proj.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 0]},
            "model.layers.0.mlp.up_proj.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 0]},
        },
    )

    index = build_hf_tensor_index(tmp_path)

    assert index["model.layers.0.mlp.up_proj.weight"].file == "model.safetensors"
    assert index["model.layers.0.mlp.up_proj.weight"].shape == [8, 4]


def test_inspect_reports_available_ffn_tensor_map(tmp_path: Path) -> None:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _write_safetensors_stub(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.mlp.gate_proj.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 0]},
            "model.layers.0.mlp.up_proj.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 0]},
            "model.layers.0.mlp.down_proj.weight": {"dtype": "F16", "shape": [4, 8], "data_offsets": [0, 0]},
        },
    )

    info = inspect_model(tmp_path)

    tensor_map = info.metadata["ffn_tensor_map"]
    assert tensor_map["available"] is True
    assert tensor_map["mapped_layers"][0]["gate"]["shape"] == [8, 4]


def _write_safetensors_stub(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)

