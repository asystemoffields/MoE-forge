from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorPattern:
    gate: tuple[str, ...]
    up: tuple[str, ...]
    down: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureAdapter:
    family: str
    aliases: tuple[str, ...]
    model_type_patterns: tuple[str, ...]
    architecture_patterns: tuple[str, ...]
    ffn_kind: str
    hf_tensors: TensorPattern
    config_keys: dict[str, tuple[str, ...]]
    supported_backends: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hf_tensors"] = asdict(self.hf_tensors)
        return payload


ADAPTERS: tuple[ArchitectureAdapter, ...] = (
    ArchitectureAdapter(
        family="llama",
        aliases=("llama", "mistral", "mixtral-dense"),
        model_type_patterns=("llama", "mistral"),
        architecture_patterns=("Llama*ForCausalLM", "Mistral*ForCausalLM"),
        ffn_kind="gated_silu",
        hf_tensors=TensorPattern(
            gate=("model.layers.{layer}.mlp.gate_proj.weight",),
            up=("model.layers.{layer}.mlp.up_proj.weight",),
            down=("model.layers.{layer}.mlp.down_proj.weight",),
        ),
        config_keys={
            "layers": ("num_hidden_layers",),
            "hidden": ("hidden_size",),
            "intermediate": ("intermediate_size",),
            "experts": ("num_local_experts", "num_experts"),
            "top_k": ("num_experts_per_tok", "top_k_experts"),
        },
        supported_backends=("carved_mlp", "sparse_upcycle", "adapter_moe"),
        notes=("Covers Llama-style gated MLP checkpoints, including many Mistral dense models.",),
    ),
    ArchitectureAdapter(
        family="qwen2",
        aliases=("qwen", "qwen2", "qwen2.5", "qwen3"),
        model_type_patterns=("qwen2", "qwen3"),
        architecture_patterns=("Qwen2*ForCausalLM", "Qwen3*ForCausalLM"),
        ffn_kind="gated_silu",
        hf_tensors=TensorPattern(
            gate=("model.layers.{layer}.mlp.gate_proj.weight",),
            up=("model.layers.{layer}.mlp.up_proj.weight",),
            down=("model.layers.{layer}.mlp.down_proj.weight",),
        ),
        config_keys={
            "layers": ("num_hidden_layers",),
            "hidden": ("hidden_size",),
            "intermediate": ("intermediate_size",),
            "experts": ("num_experts", "num_experts_per_tok"),
            "top_k": ("num_experts_per_tok", "top_k_experts"),
        },
        supported_backends=("carved_mlp", "sparse_upcycle", "adapter_moe"),
        notes=("Qwen dense variants usually share Llama-style MLP tensor names.",),
    ),
    ArchitectureAdapter(
        family="gemma",
        aliases=("gemma", "gemma2", "gemma3", "gemma4"),
        model_type_patterns=("gemma", "gemma2", "gemma3", "gemma4", "gemma4_text"),
        architecture_patterns=("Gemma*ForCausalLM", "Gemma*ForConditionalGeneration"),
        ffn_kind="gated_gelu_tanh",
        hf_tensors=TensorPattern(
            gate=(
                "language_model.model.layers.{layer}.mlp.gate_proj.weight",
                "model.layers.{layer}.mlp.gate_proj.weight",
            ),
            up=(
                "language_model.model.layers.{layer}.mlp.up_proj.weight",
                "model.layers.{layer}.mlp.up_proj.weight",
            ),
            down=(
                "language_model.model.layers.{layer}.mlp.down_proj.weight",
                "model.layers.{layer}.mlp.down_proj.weight",
            ),
        ),
        config_keys={
            "layers": ("text_config.num_hidden_layers", "num_hidden_layers"),
            "hidden": ("text_config.hidden_size", "hidden_size"),
            "intermediate": ("text_config.intermediate_size", "intermediate_size"),
            "experts": ("text_config.num_experts", "num_experts"),
            "top_k": ("text_config.top_k_experts", "top_k_experts"),
        },
        supported_backends=("carved_mlp", "adapter_moe"),
        notes=(
            "Gemma checkpoint layouts vary across text-only and conditional-generation exports.",
            "Tensor mapping should be validated from the checkpoint index before surgery.",
        ),
    ),
    ArchitectureAdapter(
        family="phi",
        aliases=("phi", "phi3", "phi4"),
        model_type_patterns=("phi", "phi3", "phi4"),
        architecture_patterns=("Phi*ForCausalLM",),
        ffn_kind="gated_or_fused",
        hf_tensors=TensorPattern(
            gate=(),
            up=("model.layers.{layer}.mlp.gate_up_proj.weight",),
            down=("model.layers.{layer}.mlp.down_proj.weight",),
        ),
        config_keys={
            "layers": ("num_hidden_layers",),
            "hidden": ("hidden_size",),
            "intermediate": ("intermediate_size",),
            "experts": ("num_local_experts", "num_experts"),
            "top_k": ("num_experts_per_tok", "top_k_experts"),
        },
        supported_backends=("adapter_moe",),
        notes=("Fused gate/up tensors need a split-aware backend before carved-MLP is enabled.",),
    ),
)


def detect_adapter(
    *,
    architecture: str | None,
    model_type: str | None,
    metadata: dict[str, Any] | None = None,
) -> ArchitectureAdapter | None:
    candidates = [_normal(architecture), _normal(model_type)]
    metadata = metadata or {}

    for adapter in ADAPTERS:
        for value in candidates:
            if value and _matches_adapter(adapter, value):
                return adapter

    selected = metadata.get("selected")
    if isinstance(selected, dict):
        source_url = _normal(selected.get("general.source.url"))
        name = _normal(selected.get("general.name"))
        for adapter in ADAPTERS:
            for alias in adapter.aliases:
                if alias in source_url or alias in name:
                    return adapter

    return None


def adapter_summary(adapter: ArchitectureAdapter | None) -> dict[str, Any] | None:
    if adapter is None:
        return None
    return adapter.to_dict()


def _matches_adapter(adapter: ArchitectureAdapter, value: str) -> bool:
    for alias in adapter.aliases:
        if value == alias or value.startswith(alias):
            return True
    for pattern in adapter.model_type_patterns:
        if fnmatchcase(value, _normal(pattern)):
            return True
    for pattern in adapter.architecture_patterns:
        if fnmatchcase(value, _normal(pattern)):
            return True
    return False


def _normal(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("-", "_").lower()
