from __future__ import annotations

import json
import struct
from pathlib import Path

from moeforge.inspectors import inspect_model
from moeforge.hf import HFModelRef, hf_config_cache_path, parse_hf_model_ref


def test_inspect_hf_config_gemma_style(tmp_path: Path) -> None:
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 35,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "max_position_embeddings": 131072,
            "vocab_size": 262144,
            "enable_moe_block": False,
            "use_double_wide_mlp": True,
            "num_experts": None,
            "top_k_experts": None,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"fake")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    info = inspect_model(tmp_path)

    assert info.source_format == "hf"
    assert info.architecture == "Gemma4ForConditionalGeneration"
    assert info.layer_count == 35
    assert info.hidden_size == 1536
    assert info.dense is True
    assert info.adapter_family == "gemma"
    assert info.adapter is not None
    assert info.metadata["checkpoint"]["has_weights"] is True
    assert info.metadata["checkpoint"]["format"] == "safetensors"
    assert info.metadata["checkpoint"]["weight_map_count"] == 1
    assert info.metadata["ffn_tensor_map"]["available"] is False
    assert info.warnings


def test_inspect_minimal_gguf(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_tiny_gguf(path)

    info = inspect_model(path)

    assert info.source_format == "gguf"
    assert info.architecture == "llama"
    assert info.layer_count == 4
    assert info.hidden_size == 128
    assert info.dense is True
    assert info.tensor_count == 0
    assert info.adapter_family == "llama"


def test_parse_hf_model_ref() -> None:
    assert parse_hf_model_ref("google/gemma-4-E2B-it") == HFModelRef("google/gemma-4-E2B-it")
    assert parse_hf_model_ref("hf:Qwen/Qwen2.5-0.5B@v1").revision == "v1"
    assert parse_hf_model_ref("C:/models/gemma") is None
    assert parse_hf_model_ref("./local/model") is None


def test_hf_config_cache_path_is_stable(tmp_path: Path) -> None:
    ref = HFModelRef("google/gemma-4-E2B-it", "main")

    path = hf_config_cache_path(ref, cache_dir=tmp_path)

    assert path == tmp_path / "hf" / "google--gemma-4-E2B-it" / "main" / "config.json"


def _write_tiny_gguf(path: Path) -> None:
    pairs = [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "Tiny"),
        ("llama.block_count", 4, 4),
        ("llama.embedding_length", 4, 128),
        ("llama.feed_forward_length", 9, (5, [256, 256, 512, 512])),
        ("llama.expert_count", 4, 0),
    ]
    with path.open("wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<IQQ", 3, 0, len(pairs)))
        for key, value_type, value in pairs:
            _write_string(fh, key)
            fh.write(struct.pack("<I", value_type))
            if value_type == 8:
                _write_string(fh, value)
            elif value_type == 4:
                fh.write(struct.pack("<I", value))
            elif value_type == 9:
                item_type, items = value
                fh.write(struct.pack("<IQ", item_type, len(items)))
                for item in items:
                    fh.write(struct.pack("<i", item))


def _write_string(fh, value: str) -> None:
    data = value.encode("utf-8")
    fh.write(struct.pack("<Q", len(data)))
    fh.write(data)
