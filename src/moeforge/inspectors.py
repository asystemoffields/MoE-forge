from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import adapter_summary, detect_adapter
from .checkpoints import inspect_hf_checkpoint_files
from .gguf import GGUFError, read_gguf_metadata
from .hf import HFRefError, download_hf_config, parse_hf_model_ref
from .model_info import ModelInfo
from .tensors import validate_ffn_tensor_map


def inspect_model(path: str | Path) -> ModelInfo:
    raw = str(path)
    candidate = Path(raw).expanduser()
    if not candidate.exists():
        ref = parse_hf_model_ref(raw)
        if ref:
            return inspect_hf_ref(ref)

    resolved = candidate.resolve()
    if resolved.is_dir():
        config_path = resolved / "config.json"
        if config_path.exists():
            return inspect_hf_config(config_path, model_path=resolved)
        raise FileNotFoundError(f"{resolved} does not contain config.json")

    if resolved.name == "config.json":
        return inspect_hf_config(resolved, model_path=resolved.parent)

    if resolved.suffix.lower() == ".gguf":
        return inspect_gguf(resolved)

    if not resolved.exists():
        raise FileNotFoundError(f"{raw} does not exist and is not a Hugging Face model id")
    raise ValueError(f"unsupported model input: {resolved}")


def inspect_hf_ref(ref) -> ModelInfo:
    try:
        config_path = download_hf_config(ref)
    except HFRefError:
        raise

    info = inspect_hf_config(config_path, model_path=Path(f"hf:{ref.display}"))
    info.metadata["hf_repo_id"] = ref.repo_id
    info.metadata["hf_revision"] = ref.revision
    info.metadata["cached_config_path"] = str(config_path)
    info.source_format = "hf_remote"
    return info


def inspect_hf_config(config_path: Path, *, model_path: Path | None = None) -> ModelInfo:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config

    architecture = _first(config.get("architectures")) or text.get("architectures")
    model_type = text.get("model_type") or config.get("model_type")
    expert_count = _first_int(
        text.get("num_experts"),
        text.get("num_local_experts"),
        text.get("n_routed_experts"),
        text.get("moe_num_experts"),
    )
    experts_used = _first_int(
        text.get("top_k_experts"),
        text.get("num_experts_per_tok"),
        text.get("num_experts_per_token"),
        text.get("moe_top_k"),
    )
    dense = expert_count in (None, 0)

    intermediate = _first_int(
        text.get("intermediate_size"),
        text.get("ffn_dim"),
        text.get("hidden_dim"),
    )
    intermediate_sizes = []
    if isinstance(text.get("intermediate_size"), list):
        intermediate_sizes = [int(item) for item in text["intermediate_size"]]
        intermediate = intermediate_sizes[0] if intermediate_sizes else None
    warnings = []
    if text.get("use_double_wide_mlp") and not intermediate_sizes:
        warnings.append(
            "Config reports use_double_wide_mlp, but does not provide per-layer intermediate sizes; layer layouts are approximate until tensor shapes are indexed."
        )

    metadata = {
        "config_path": str(config_path),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "enable_moe_block": text.get("enable_moe_block"),
        "use_double_wide_mlp": text.get("use_double_wide_mlp"),
        "layer_types": text.get("layer_types"),
        "checkpoint": inspect_hf_checkpoint_files(model_path or config_path.parent),
    }
    adapter = detect_adapter(
        architecture=str(architecture) if architecture else None,
        model_type=str(model_type) if model_type else None,
        metadata=metadata,
    )

    info = ModelInfo(
        path=model_path or config_path,
        source_format="hf",
        architecture=str(architecture) if architecture else None,
        model_type=str(model_type) if model_type else None,
        name=config.get("_name_or_path") or config.get("name_or_path"),
        layer_count=_first_int(text.get("num_hidden_layers"), text.get("n_layer"), text.get("num_layers")),
        hidden_size=_first_int(text.get("hidden_size"), text.get("n_embd"), text.get("d_model")),
        intermediate_size=intermediate,
        intermediate_sizes=intermediate_sizes,
        context_length=_first_int(
            text.get("max_position_embeddings"),
            text.get("seq_length"),
            text.get("n_positions"),
        ),
        vocab_size=_first_int(text.get("vocab_size"), config.get("vocab_size")),
        attention_heads=_first_int(text.get("num_attention_heads"), text.get("n_head")),
        kv_heads=_first_int(text.get("num_key_value_heads"), text.get("num_kv_heads")),
        expert_count=expert_count,
        experts_used=experts_used,
        dense=dense,
        adapter_family=adapter.family if adapter else None,
        adapter=adapter_summary(adapter),
        metadata=metadata,
        warnings=warnings,
    )
    info.metadata["ffn_tensor_map"] = validate_ffn_tensor_map(info)
    return info


def inspect_gguf(path: Path) -> ModelInfo:
    try:
        gguf = read_gguf_metadata(path, max_array_preview=256)
    except GGUFError:
        raise

    meta = gguf.metadata
    arch = _as_str(meta.get("general.architecture"))
    prefix = arch or ""

    expert_count = _first_int(
        meta.get(f"{prefix}.expert_count"),
        meta.get(f"{prefix}.n_expert"),
        meta.get("llm.expert_count"),
    )
    experts_used = _first_int(
        meta.get(f"{prefix}.expert_used_count"),
        meta.get(f"{prefix}.expert_used"),
        meta.get("llm.expert_used_count"),
    )

    intermediate_value = meta.get(f"{prefix}.feed_forward_length")
    intermediate_sizes = _array_to_ints(intermediate_value)
    intermediate = intermediate_sizes[0] if intermediate_sizes else _first_int(intermediate_value)

    selected = {
        key: _jsonable(meta[key])
        for key in sorted(meta)
        if key.startswith("general.")
        or key.startswith(f"{prefix}.")
        or key.startswith("pmra.")
    }

    metadata = {
        "gguf_version": gguf.version,
        "metadata_kv_count": gguf.metadata_kv_count,
        "selected": selected,
    }
    adapter = detect_adapter(
        architecture=arch,
        model_type=_as_str(meta.get("general.type")),
        metadata=metadata,
    )

    return ModelInfo(
        path=path,
        source_format="gguf",
        architecture=arch,
        model_type=_as_str(meta.get("general.type")),
        name=_as_str(meta.get("general.name")),
        size_label=_as_str(meta.get("general.size_label")),
        layer_count=_first_int(meta.get(f"{prefix}.block_count")),
        hidden_size=_first_int(meta.get(f"{prefix}.embedding_length")),
        intermediate_size=intermediate,
        intermediate_sizes=intermediate_sizes,
        context_length=_first_int(meta.get(f"{prefix}.context_length")),
        vocab_size=_array_length(meta.get("tokenizer.ggml.tokens")),
        attention_heads=_first_int(meta.get(f"{prefix}.attention.head_count")),
        kv_heads=_first_int(meta.get(f"{prefix}.attention.head_count_kv")),
        expert_count=expert_count,
        experts_used=experts_used,
        dense=expert_count in (None, 0),
        quantization=_as_str(meta.get("general.file_type")),
        tensor_count=gguf.tensor_count,
        adapter_family=adapter.family if adapter else None,
        adapter=adapter_summary(adapter),
        metadata=metadata,
    )


def _array_to_ints(value: Any) -> list[int]:
    if isinstance(value, dict) and value.get("kind") == "array":
        preview = value.get("preview") or []
        length = int(value.get("length") or len(preview))
        if len(preview) == length:
            return [int(item) for item in preview]
    if isinstance(value, list):
        return [int(item) for item in value]
    return []


def _array_length(value: Any) -> int | None:
    if isinstance(value, dict) and value.get("kind") == "array":
        return int(value["length"])
    if isinstance(value, list):
        return len(value)
    return None


def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _first_int(*values: Any) -> int | None:
    for value in values:
        value = _first(value)
        if value is None:
            continue
        if isinstance(value, dict) and value.get("kind") == "array":
            preview = value.get("preview") or []
            if preview:
                return _first_int(preview[0])
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return str(value)
