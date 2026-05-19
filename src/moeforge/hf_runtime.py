from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from itertools import chain
from pathlib import Path
from typing import Any

from .router import select_expert_pool
from .wrapper import WrapperConfig, load_wrapper_config

try:  # pragma: no cover - exercised through tests when transformers is installed
    from transformers import AutoConfig as _AutoConfig
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import PreTrainedModel as _PreTrainedModel
    from transformers import PretrainedConfig as _PretrainedConfig
    from transformers.generation import GenerationMixin as _GenerationMixin
except ImportError:  # pragma: no cover - optional dependency boundary
    _AutoConfig = None
    _AutoModelForCausalLM = None
    _GenerationMixin = object
    _PreTrainedModel = object
    _PretrainedConfig = object

try:  # pragma: no cover - exercised through tests when torch is installed
    import torch
except ImportError:  # pragma: no cover - optional dependency boundary
    torch = None

_TorchModule = torch.nn.Module if torch is not None else object
_HFModelBase = _PreTrainedModel if _PreTrainedModel is not object else _TorchModule
if _PreTrainedModel is not object and _GenerationMixin is not object and not issubclass(_PreTrainedModel, _GenerationMixin):
    class _MoEForgeModelBase(_PreTrainedModel, _GenerationMixin):  # type: ignore[misc]
        pass
else:
    _MoEForgeModelBase = _HFModelBase


class MoEForgeHFError(RuntimeError):
    """Raised when a Transformers-facing wrapper cannot be loaded or executed."""


@dataclass(slots=True)
class HFModuleReplacement:
    layer: int
    module_path: str
    original_class: str
    replacement_class: str
    device: str
    dtype: str | None
    default_experts: list[int] | None = None
    token_router_top_k: int | None = None


@dataclass(slots=True)
class HFReplacementReport:
    package_dir: str
    adapter_family: str | None
    replaced: list[HFModuleReplacement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MoEForgeConfig(_PretrainedConfig):
    model_type = "moeforge_carved_moe"

    def __init__(
        self,
        *,
        moeforge_wrapper_config: str = "moeforge_config.json",
        moeforge_format_version: int = 1,
        adapter_family: str | None = None,
        source_model: str = "",
        manifest_path: str = "carve-manifest.json",
        artifact_path: str = "carved-experts.safetensors",
        router_plan_path: str | None = None,
        token_router_top_k: int | None = None,
        token_router_path: str | None = None,
        activation: str = "silu",
        expert_count: int = 0,
        layers: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if _PretrainedConfig is not object:
            super().__init__(**kwargs)
        elif kwargs:
            self._extra_config = dict(kwargs)
        self.moeforge_wrapper_config = moeforge_wrapper_config
        self.moeforge_format_version = int(moeforge_format_version)
        self.adapter_family = adapter_family
        self.source_model = source_model
        self.manifest_path = manifest_path
        self.artifact_path = artifact_path
        self.router_plan_path = router_plan_path
        self.token_router_top_k = int(token_router_top_k) if token_router_top_k is not None else None
        self.token_router_path = token_router_path
        self.activation = activation
        self.expert_count = int(expert_count)
        self.layers = layers or []

    @classmethod
    def from_package(cls, package_dir: str | Path) -> "MoEForgeConfig":
        package = Path(package_dir)
        config_json = package / "config.json"
        if config_json.exists() and _PretrainedConfig is not object:
            return cls.from_pretrained(package)
        if config_json.exists():
            return cls(**_read_json(config_json))
        return cls.from_wrapper_config(load_wrapper_config(package / "moeforge_config.json"))

    @classmethod
    def from_wrapper_config(
        cls,
        wrapper_config: WrapperConfig,
        *,
        wrapper_config_file: str = "moeforge_config.json",
    ) -> "MoEForgeConfig":
        return cls(
            moeforge_wrapper_config=wrapper_config_file,
            moeforge_format_version=wrapper_config.format_version,
            adapter_family=wrapper_config.adapter_family,
            source_model=wrapper_config.source_model,
            manifest_path=wrapper_config.manifest_path,
            artifact_path=wrapper_config.artifact_path,
            router_plan_path=wrapper_config.router_plan_path,
            token_router_top_k=wrapper_config.token_router_top_k,
            token_router_path=wrapper_config.token_router_path,
            activation=wrapper_config.activation,
            expert_count=wrapper_config.expert_count,
            layers=[item.to_dict() if hasattr(item, "to_dict") else _layer_to_dict(item) for item in wrapper_config.layers],
        )

    def layer_ids(self) -> list[int]:
        return [int(item["layer"]) for item in self.layers]


class MoEForgeCarvedMLPModule(_TorchModule):
    def __init__(
        self,
        *,
        package_dir: str | Path,
        config: MoEForgeConfig,
        layer: int,
        tensors: dict[str, Any],
        router_plan: dict[str, Any] | None,
        router_tensors: dict[str, Any] | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover - optional dependency boundary
            raise MoEForgeHFError("HF runtime requires torch")
        super().__init__()
        self.config = config
        self.package_dir = Path(package_dir)
        self.layer = layer
        self.expert_count = config.expert_count
        self.activation = config.activation
        self.router_plan = router_plan
        self.default_experts: list[int] | None = None
        self.token_router_top_k = _normalized_top_k(config.token_router_top_k, expert_count=config.expert_count)
        self.token_router: Any | None = None
        self.last_router_summary: dict[str, Any] | None = None
        self._tensor_buffers: dict[str, str] = {}

        prefix = f"moe.layers.{layer}.mlp."
        for tensor_name, tensor in sorted(tensors.items()):
            if not tensor_name.startswith(prefix):
                continue
            buffer_name = _buffer_name(tensor_name)
            self.register_buffer(buffer_name, tensor)
            self._tensor_buffers[tensor_name] = buffer_name
        if not self._tensor_buffers:
            raise MoEForgeHFError(f"no carved tensors found for layer {layer}")
        self._init_token_router(router_tensors or {})

    @classmethod
    def from_package(
        cls,
        package_dir: str | Path,
        *,
        layer: int | None = None,
        config: MoEForgeConfig | None = None,
    ) -> "MoEForgeCarvedMLPModule":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise MoEForgeHFError("HF runtime requires torch and safetensors") from exc

        package = Path(package_dir)
        resolved_config = config or MoEForgeConfig.from_package(package)
        resolved_layer = _resolve_layer(layer, resolved_config)
        artifact_path = _resolve_package_path(package, resolved_config.artifact_path)
        tensors = load_file(str(artifact_path), device="cpu")
        router_plan = _read_json(_resolve_package_path(package, resolved_config.router_plan_path)) if resolved_config.router_plan_path else None
        router_tensors = (
            load_file(str(_resolve_package_path(package, resolved_config.token_router_path)), device="cpu")
            if resolved_config.token_router_path
            else None
        )
        return cls(
            package_dir=package,
            config=resolved_config,
            layer=resolved_layer,
            tensors=tensors,
            router_plan=router_plan,
            router_tensors=router_tensors,
        )

    def forward(
        self,
        hidden_states: Any,
        *,
        experts: list[int] | None = None,
        router_plan: dict[str, Any] | None = None,
        text: str | None = None,
        text_sha256: str | None = None,
        document_index: int | None = None,
    ) -> Any:
        if experts is None and _has_router_request(router_plan, text, text_sha256, document_index):
            experts = self.select_experts(
                router_plan=router_plan,
                text=text,
                text_sha256=text_sha256,
                document_index=document_index,
            )
        if experts is None:
            if self.default_experts is not None:
                experts = self.default_experts
            elif self.token_router is not None:
                return self.forward_token_router(hidden_states)
            else:
                experts = list(range(self.expert_count))
        return self.forward_selected(hidden_states, experts=experts)

    def forward_all(self, hidden_states: Any) -> Any:
        return self.forward_selected(hidden_states, experts=list(range(self.expert_count)))

    def forward_selected(self, hidden_states: Any, *, experts: list[int]) -> Any:
        if torch is None:  # pragma: no cover - optional dependency boundary
            raise MoEForgeHFError("HF runtime requires torch")
        output = self._group_contribution(hidden_states, kind="shared", expert=None)
        for expert in experts:
            if expert is not None and (expert < 0 or expert >= self.expert_count):
                raise MoEForgeHFError(f"expert index {expert} is out of range")
            contribution = self._group_contribution(hidden_states, kind="expert", expert=expert)
            if contribution is None:
                continue
            output = contribution if output is None else output + contribution

        if output is None:
            raise MoEForgeHFError(f"no runnable carved MLP tensors found for layer {self.layer}")
        return output

    def forward_token_router(self, hidden_states: Any) -> Any:
        if torch is None:  # pragma: no cover - optional dependency boundary
            raise MoEForgeHFError("HF runtime requires torch")
        if self.token_router is None or self.token_router_top_k is None:
            raise MoEForgeHFError("learned token router is not configured for this layer")
        logits = self.token_router(hidden_states)
        probabilities = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probabilities, k=self.token_router_top_k, dim=-1)
        self.last_router_summary = self._router_summary(top_values=top_values, top_indices=top_indices)
        output = self._group_contribution(hidden_states, kind="shared", expert=None)
        for expert in range(self.expert_count):
            mask = top_indices == expert
            if not bool(mask.any().detach().cpu().item()):
                continue
            weight = torch.where(mask, top_values, torch.zeros_like(top_values)).sum(dim=-1)
            contribution = self._group_contribution(hidden_states, kind="expert", expert=expert)
            if contribution is None:
                continue
            weighted = contribution * weight.unsqueeze(-1)
            output = weighted if output is None else output + weighted
        if output is None:
            raise MoEForgeHFError(f"token router found no runnable carved MLP tensors for layer {self.layer}")
        return output

    def forward_with_router(
        self,
        hidden_states: Any,
        *,
        router_plan: dict[str, Any] | None = None,
        text: str | None = None,
        text_sha256: str | None = None,
        document_index: int | None = None,
    ) -> Any:
        experts = self.select_experts(
            router_plan=router_plan,
            text=text,
            text_sha256=text_sha256,
            document_index=document_index,
        )
        return self.forward_selected(hidden_states, experts=experts)

    def select_experts(
        self,
        *,
        router_plan: dict[str, Any] | None = None,
        text: str | None = None,
        text_sha256: str | None = None,
        document_index: int | None = None,
    ) -> list[int]:
        plan = router_plan if router_plan is not None else self.router_plan
        if plan is None:
            raise MoEForgeHFError("router metadata is not available for this wrapper package")
        return select_expert_pool(
            plan,
            text=text,
            text_sha256=text_sha256,
            document_index=document_index,
        )

    def set_default_experts(self, experts: list[int] | None) -> None:
        if experts is None:
            self.default_experts = None
            return
        normalized = [int(expert) for expert in experts]
        for expert in normalized:
            if expert < 0 or expert >= self.expert_count:
                raise MoEForgeHFError(f"expert index {expert} is out of range")
        self.default_experts = normalized

    def _init_token_router(self, router_tensors: dict[str, Any]) -> None:
        if self.token_router_top_k is None:
            return
        hidden_size = self._infer_hidden_size()
        self.token_router = torch.nn.Linear(hidden_size, self.expert_count, bias=True)
        torch.nn.init.zeros_(self.token_router.weight)
        torch.nn.init.zeros_(self.token_router.bias)
        weight = router_tensors.get(_router_tensor_name(self.layer, "weight"))
        bias = router_tensors.get(_router_tensor_name(self.layer, "bias"))
        if weight is not None:
            if list(weight.shape) != list(self.token_router.weight.shape):
                raise MoEForgeHFError(f"token router weight shape mismatch for layer {self.layer}")
            self.token_router.weight.data.copy_(weight.to(dtype=self.token_router.weight.dtype))
        if bias is not None:
            if list(bias.shape) != list(self.token_router.bias.shape):
                raise MoEForgeHFError(f"token router bias shape mismatch for layer {self.layer}")
            self.token_router.bias.data.copy_(bias.to(dtype=self.token_router.bias.dtype))

    def _infer_hidden_size(self) -> int:
        for tensor_name in sorted(self._tensor_buffers):
            if tensor_name.endswith(".gate.weight") or tensor_name.endswith(".up.weight"):
                tensor = self._tensor(tensor_name)
                if tensor is not None and len(tensor.shape) == 2:
                    return int(tensor.shape[1])
        raise MoEForgeHFError(f"could not infer hidden size for token router in layer {self.layer}")

    def _group_contribution(self, hidden_states: Any, *, kind: str, expert: int | None) -> Any | None:
        prefix = self._prefix(kind, expert)
        gate = self._tensor(f"{prefix}.gate.weight")
        up = self._tensor(f"{prefix}.up.weight")
        down = self._tensor(f"{prefix}.down.weight")
        if gate is None or up is None or down is None:
            return None
        hidden = self._activate(torch.nn.functional.linear(hidden_states, gate))
        hidden = hidden * torch.nn.functional.linear(hidden_states, up)
        return torch.nn.functional.linear(hidden, down)

    def _router_summary(self, *, top_values: Any, top_indices: Any) -> dict[str, Any]:
        token_count = int(top_indices.shape[0] * top_indices.shape[1]) if len(top_indices.shape) >= 2 else int(top_indices.numel())
        expert_token_counts: dict[str, int] = {}
        expert_weight_sums: dict[str, float] = {}
        for expert in range(self.expert_count):
            mask = top_indices == expert
            count = int(mask.sum().detach().cpu().item())
            if count <= 0:
                continue
            weight_sum = float(torch.where(mask, top_values, torch.zeros_like(top_values)).sum().detach().cpu().item())
            expert_token_counts[str(expert)] = count
            expert_weight_sums[str(expert)] = weight_sum
        return {
            "layer": self.layer,
            "top_k": self.token_router_top_k,
            "token_count": token_count,
            "experts": [int(key) for key in sorted(expert_token_counts, key=int)],
            "expert_token_counts": expert_token_counts,
            "mean_selected_weight_by_expert": {
                key: expert_weight_sums[key] / expert_token_counts[key]
                for key in expert_token_counts
            },
        }

    def _prefix(self, kind: str, expert: int | None) -> str:
        if kind == "shared":
            return f"moe.layers.{self.layer}.mlp.shared"
        return f"moe.layers.{self.layer}.mlp.experts.{expert}"

    def _tensor(self, name: str) -> Any | None:
        buffer_name = self._tensor_buffers.get(name)
        if buffer_name is None:
            return None
        return getattr(self, buffer_name)

    def _activate(self, value: Any) -> Any:
        if torch is None:  # pragma: no cover - optional dependency boundary
            raise MoEForgeHFError("HF runtime requires torch")
        if self.activation == "silu":
            return torch.nn.functional.silu(value)
        if self.activation in {"gelu", "gelu_tanh"}:
            approximate = "tanh" if self.activation == "gelu_tanh" else "none"
            return torch.nn.functional.gelu(value, approximate=approximate)
        raise MoEForgeHFError(f"unsupported activation {self.activation}")


class MoEForgeForCausalLM(_MoEForgeModelBase):
    config_class = MoEForgeConfig
    base_model_prefix = "dense_model"

    def __init__(
        self,
        config: MoEForgeConfig,
        *,
        dense_model: Any,
        package_dir: str | Path,
        replacement_report: HFReplacementReport,
    ) -> None:
        if _PreTrainedModel is not object:
            super().__init__(config)
        else:  # pragma: no cover - only used without transformers installed
            super().__init__()
            self.config = config
        self.dense_model = dense_model
        self.package_dir = Path(package_dir)
        self.replacement_report = replacement_report

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        config: MoEForgeConfig | None = None,
        **kwargs: Any,
    ) -> "MoEForgeForCausalLM":
        if _AutoModelForCausalLM is None:
            raise MoEForgeHFError("MoEForgeForCausalLM.from_pretrained requires transformers")

        package = Path(pretrained_model_name_or_path)
        resolved_config = config or MoEForgeConfig.from_package(package)
        default_experts = kwargs.pop("moeforge_default_experts", None)
        source_model = _resolve_source_model_ref(package, resolved_config.source_model)
        source_kwargs = _source_model_kwargs(kwargs)
        dense_model = _AutoModelForCausalLM.from_pretrained(source_model, *model_args, **source_kwargs)
        replacement_report = replace_hf_mlp_modules(
            dense_model,
            package,
            config=resolved_config,
            default_experts=default_experts,
        )
        return cls(
            resolved_config,
            dense_model=dense_model,
            package_dir=package,
            replacement_report=replacement_report,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.dense_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.dense_model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self.dense_model, "prepare_inputs_for_generation"):
            raise MoEForgeHFError("base model does not implement prepare_inputs_for_generation")
        return self.dense_model.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self) -> Any:
        return self.dense_model.get_input_embeddings()

    def set_input_embeddings(self, value: Any) -> None:
        self.dense_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> Any:
        return self.dense_model.get_output_embeddings()

    def set_output_embeddings(self, value: Any) -> None:
        self.dense_model.set_output_embeddings(value)

    def resize_token_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        return self.dense_model.resize_token_embeddings(*args, **kwargs)


def register_transformers_auto_classes(*, exist_ok: bool = True) -> bool:
    if _AutoConfig is None or _AutoModelForCausalLM is None:
        return False
    _AutoConfig.register(MoEForgeConfig.model_type, MoEForgeConfig, exist_ok=exist_ok)
    _AutoModelForCausalLM.register(MoEForgeConfig, MoEForgeForCausalLM, exist_ok=exist_ok)
    return True


def hf_config_payload_from_wrapper(wrapper_config: WrapperConfig) -> dict[str, Any]:
    config = MoEForgeConfig.from_wrapper_config(wrapper_config)
    return {
        "activation": config.activation,
        "adapter_family": config.adapter_family,
        "architectures": ["MoEForgeForCausalLM"],
        "artifact_path": config.artifact_path,
        "expert_count": config.expert_count,
        "layers": config.layers,
        "library_name": "moe-forge",
        "manifest_path": config.manifest_path,
        "model_type": MoEForgeConfig.model_type,
        "moeforge_format_version": config.moeforge_format_version,
        "moeforge_wrapper_config": config.moeforge_wrapper_config,
        "router_plan_path": config.router_plan_path,
        "source_model": config.source_model,
        "token_router_path": config.token_router_path,
        "token_router_top_k": config.token_router_top_k,
    }


def replace_hf_mlp_modules(
    model: Any,
    package_dir: str | Path,
    *,
    layers: list[int] | None = None,
    config: MoEForgeConfig | None = None,
    default_experts: dict[int, list[int]] | list[int] | None = None,
) -> HFReplacementReport:
    if torch is None:  # pragma: no cover - optional dependency boundary
        raise MoEForgeHFError("HF runtime requires torch")

    package = Path(package_dir)
    resolved_config = config or MoEForgeConfig.from_package(package)
    layer_ids = resolved_config.layer_ids() if layers is None else layers
    report = HFReplacementReport(
        package_dir=str(package),
        adapter_family=resolved_config.adapter_family,
    )

    for layer in layer_ids:
        module_path = _resolve_mlp_module_path(model, layer=layer, adapter_family=resolved_config.adapter_family)
        parent_path, attribute = module_path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        original = getattr(parent, attribute)
        device, dtype = _module_device_dtype(original)
        replacement = MoEForgeCarvedMLPModule.from_package(package, layer=layer, config=resolved_config)
        layer_default_experts = _default_experts_for_layer(default_experts, layer=layer)
        replacement.set_default_experts(layer_default_experts)
        if dtype is None:
            replacement = replacement.to(device=device)
        else:
            replacement = replacement.to(device=device, dtype=dtype)
        setattr(parent, attribute, replacement)
        report.replaced.append(
            HFModuleReplacement(
                layer=layer,
                module_path=module_path,
                original_class=original.__class__.__name__,
                replacement_class=replacement.__class__.__name__,
                device=str(device),
                dtype=str(dtype) if dtype is not None else None,
                default_experts=layer_default_experts,
                token_router_top_k=replacement.token_router_top_k,
            )
        )

    return report


def _resolve_layer(layer: int | None, config: MoEForgeConfig) -> int:
    if layer is not None:
        if layer not in config.layer_ids():
            raise MoEForgeHFError(f"layer {layer} is not present in wrapper config")
        return layer
    layer_ids = config.layer_ids()
    if len(layer_ids) != 1:
        raise MoEForgeHFError("layer must be provided when the wrapper package contains multiple layers")
    return layer_ids[0]


def _resolve_mlp_module_path(model: Any, *, layer: int, adapter_family: str | None) -> str:
    candidates = _mlp_module_path_candidates(layer=layer, adapter_family=adapter_family)
    for candidate in candidates:
        try:
            model.get_submodule(candidate)
        except AttributeError:
            continue
        return candidate
    raise MoEForgeHFError(
        f"could not find an FFN module for layer {layer}; tried {', '.join(candidates)}"
    )


def _mlp_module_path_candidates(*, layer: int, adapter_family: str | None) -> list[str]:
    common = [
        f"model.layers.{layer}.mlp",
        f"language_model.model.layers.{layer}.mlp",
    ]
    if adapter_family == "gemma":
        return [common[1], common[0]]
    return common


def _module_device_dtype(module: Any) -> tuple[Any, Any | None]:
    tensors = chain(module.parameters(recurse=True), module.buffers(recurse=True))
    for tensor in tensors:
        dtype = tensor.dtype if tensor.is_floating_point() else None
        return tensor.device, dtype
    return torch.device("cpu"), None


def _default_experts_for_layer(
    default_experts: dict[int, list[int]] | list[int] | None,
    *,
    layer: int,
) -> list[int] | None:
    if default_experts is None:
        return None
    if isinstance(default_experts, dict):
        selected = default_experts.get(layer)
        return [int(expert) for expert in selected] if selected is not None else None
    return [int(expert) for expert in default_experts]


def _normalized_top_k(value: int | None, *, expert_count: int) -> int | None:
    if value is None:
        return None
    top_k = int(value)
    if top_k <= 0:
        raise MoEForgeHFError("token_router_top_k must be positive")
    return min(top_k, int(expert_count))


def _router_tensor_name(layer: int, field: str) -> str:
    return f"moe.layers.{int(layer)}.mlp.router.{field}"


def _resolve_source_model_ref(package_dir: Path, source_model: str) -> str:
    if not source_model:
        raise MoEForgeHFError("wrapper config is missing source_model")
    source_path = Path(source_model)
    if source_path.is_absolute() or source_path.exists():
        return str(source_path)
    package_relative = package_dir / source_path
    if package_relative.exists():
        return str(package_relative)
    return source_model


def _source_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "_from_auto",
        "adapter_kwargs",
        "config",
        "subfolder",
    }
    return {key: value for key, value in kwargs.items() if key not in ignored}


def _has_router_request(
    router_plan: dict[str, Any] | None,
    text: str | None,
    text_sha256: str | None,
    document_index: int | None,
) -> bool:
    return router_plan is not None or text is not None or text_sha256 is not None or document_index is not None


def _resolve_package_path(package_dir: Path, value: str | None) -> Path:
    if value is None:
        raise MoEForgeHFError("missing package path")
    path = Path(value)
    if path.is_absolute():
        return path
    return package_dir / path


def _buffer_name(tensor_name: str) -> str:
    return "tensor__" + tensor_name.replace(".", "__").replace("-", "_")


def _layer_to_dict(layer: Any) -> dict[str, Any]:
    return {
        "layer": int(layer.layer),
        "width": layer.width,
        "tensor_prefix": layer.tensor_prefix,
        "expert_count": int(layer.expert_count),
        "shared_channels": int(layer.shared_channels),
        "expert_channels": [int(value) for value in layer.expert_channels],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
