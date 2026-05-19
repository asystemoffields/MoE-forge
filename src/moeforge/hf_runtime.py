from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from itertools import chain
from pathlib import Path
from typing import Any

from .router import select_expert_pool
from .wrapper import WrapperConfig, load_wrapper_config

try:  # pragma: no cover - exercised through tests when transformers is installed
    from transformers import PretrainedConfig as _PretrainedConfig
except ImportError:  # pragma: no cover - optional dependency boundary
    _PretrainedConfig = object

try:  # pragma: no cover - exercised through tests when torch is installed
    import torch
except ImportError:  # pragma: no cover - optional dependency boundary
    torch = None

_TorchModule = torch.nn.Module if torch is not None else object


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
        return cls(
            package_dir=package,
            config=resolved_config,
            layer=resolved_layer,
            tensors=tensors,
            router_plan=router_plan,
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
            experts = self.default_experts or list(range(self.expert_count))
        return self.forward_selected(hidden_states, experts=experts)

    def forward_all(self, hidden_states: Any) -> Any:
        return self.forward_selected(hidden_states, experts=list(range(self.expert_count)))

    def forward_selected(self, hidden_states: Any, *, experts: list[int]) -> Any:
        if torch is None:  # pragma: no cover - optional dependency boundary
            raise MoEForgeHFError("HF runtime requires torch")
        output = None
        groups: list[tuple[str, int | None]] = [("shared", None)]
        groups.extend(("expert", expert) for expert in experts)
        for kind, expert in groups:
            if expert is not None and (expert < 0 or expert >= self.expert_count):
                raise MoEForgeHFError(f"expert index {expert} is out of range")
            prefix = self._prefix(kind, expert)
            gate = self._tensor(f"{prefix}.gate.weight")
            up = self._tensor(f"{prefix}.up.weight")
            down = self._tensor(f"{prefix}.down.weight")
            if gate is None or up is None or down is None:
                continue
            hidden = self._activate(torch.nn.functional.linear(hidden_states, gate))
            hidden = hidden * torch.nn.functional.linear(hidden_states, up)
            contribution = torch.nn.functional.linear(hidden, down)
            output = contribution if output is None else output + contribution

        if output is None:
            raise MoEForgeHFError(f"no runnable carved MLP tensors found for layer {self.layer}")
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


def hf_config_payload_from_wrapper(wrapper_config: WrapperConfig) -> dict[str, Any]:
    config = MoEForgeConfig.from_wrapper_config(wrapper_config)
    return {
        "activation": config.activation,
        "adapter_family": config.adapter_family,
        "architectures": ["MoEForgeCarvedMLPModule"],
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
