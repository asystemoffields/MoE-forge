from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelInfo:
    path: Path
    source_format: str
    architecture: str | None
    model_type: str | None = None
    name: str | None = None
    size_label: str | None = None
    parameter_count: int | None = None
    layer_count: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    intermediate_sizes: list[int] = field(default_factory=list)
    context_length: int | None = None
    vocab_size: int | None = None
    attention_heads: int | None = None
    kv_heads: int | None = None
    expert_count: int | None = None
    experts_used: int | None = None
    dense: bool | None = None
    quantization: str | None = None
    tensor_count: int | None = None
    adapter_family: str | None = None
    adapter: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source_format": self.source_format,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "name": self.name,
            "size_label": self.size_label,
            "parameter_count": self.parameter_count,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "intermediate_sizes": self.intermediate_sizes,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "attention_heads": self.attention_heads,
            "kv_heads": self.kv_heads,
            "expert_count": self.expert_count,
            "experts_used": self.experts_used,
            "dense": self.dense,
            "quantization": self.quantization,
            "tensor_count": self.tensor_count,
            "adapter_family": self.adapter_family,
            "adapter": self.adapter,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }
