from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Iterable

from .adapters import ADAPTERS, ArchitectureAdapter
from .inspectors import inspect_model
from .layout import _parse_layer_spec


class ProfileError(RuntimeError):
    """Raised when activation profiling cannot proceed."""


@dataclass(slots=True)
class ProfileOptions:
    layers: str | None = None
    roles: tuple[str, ...] = ("gate", "up")
    max_samples: int = 32
    sequence_length: int = 512
    device: str = "auto"
    dtype: str = "auto"
    threshold: float = 0.0
    include_vectors: bool = False
    include_document_vectors: bool = False
    top_k_channels: int = 32
    document_top_k_channels: int = 8
    experts: int = 8
    shared_ratio: float = 0.25


@dataclass(slots=True)
class ChannelStats:
    count: int = 0
    width: int | None = None
    sum_abs: list[float] = field(default_factory=list)
    sum_sq: list[float] = field(default_factory=list)
    positive: list[int] = field(default_factory=list)
    active: list[int] = field(default_factory=list)
    threshold: float = 0.0

    def update(self, values: Any) -> None:
        rows = _to_channel_rows(values)
        if not rows:
            return
        width = len(rows[0])
        self._ensure_width(width)

        for row in rows:
            if len(row) != width:
                raise ProfileError("activation rows have inconsistent channel widths")
            self.count += 1
            for index, value in enumerate(row):
                number = float(value)
                abs_value = abs(number)
                self.sum_abs[index] += abs_value
                self.sum_sq[index] += number * number
                if number > 0:
                    self.positive[index] += 1
                if abs_value > self.threshold:
                    self.active[index] += 1

    def to_report(self, *, include_vectors: bool, top_k_channels: int) -> dict[str, Any]:
        if self.count == 0 or self.width is None:
            return {
                "count": 0,
                "width": self.width,
                "mean_abs_mean": None,
                "rms_mean": None,
                "active_rate_mean": None,
                "positive_rate_mean": None,
                "top_channels": [],
            }

        mean_abs = [value / self.count for value in self.sum_abs]
        rms = [(value / self.count) ** 0.5 for value in self.sum_sq]
        active_rate = [value / self.count for value in self.active]
        positive_rate = [value / self.count for value in self.positive]
        top_indices = sorted(range(len(mean_abs)), key=lambda item: mean_abs[item], reverse=True)[:top_k_channels]

        report = {
            "count": self.count,
            "width": self.width,
            "threshold": self.threshold,
            "mean_abs_mean": sum(mean_abs) / len(mean_abs),
            "rms_mean": sum(rms) / len(rms),
            "active_rate_mean": sum(active_rate) / len(active_rate),
            "positive_rate_mean": sum(positive_rate) / len(positive_rate),
            "top_channels": [
                {
                    "channel": index,
                    "mean_abs": mean_abs[index],
                    "rms": rms[index],
                    "active_rate": active_rate[index],
                    "positive_rate": positive_rate[index],
                }
                for index in top_indices
            ],
        }
        if include_vectors:
            report["vectors"] = {
                "mean_abs": mean_abs,
                "rms": rms,
                "active_rate": active_rate,
                "positive_rate": positive_rate,
            }
        return report

    def assign_experts(self, *, experts: int, shared_ratio: float) -> dict[str, Any]:
        if self.count == 0 or self.width is None:
            return {
                "available": False,
                "reason": "no activation statistics were collected",
            }
        if experts <= 0:
            raise ProfileError("experts must be positive")
        if not 0 <= shared_ratio < 1:
            raise ProfileError("shared_ratio must be in [0, 1)")

        mean_abs = [value / self.count for value in self.sum_abs]
        ranked = sorted(range(self.width), key=lambda index: mean_abs[index], reverse=True)
        shared_count = min(self.width, int(round(self.width * shared_ratio)))
        shared = ranked[:shared_count]
        routed = ranked[shared_count:]
        expert_channels: list[list[int]] = [[] for _ in range(experts)]
        expert_scores = [0.0 for _ in range(experts)]

        for channel in routed:
            expert = min(range(experts), key=lambda index: (expert_scores[index], len(expert_channels[index])))
            expert_channels[expert].append(channel)
            expert_scores[expert] += mean_abs[channel]

        return {
            "available": True,
            "method": "shared_top_mean_abs_then_greedy_balance",
            "width": self.width,
            "shared_ratio": shared_ratio,
            "shared_channels": sorted(shared),
            "experts": [
                {
                    "expert": index,
                    "channels": sorted(channels),
                    "channel_count": len(channels),
                    "score_sum": expert_scores[index],
                }
                for index, channels in enumerate(expert_channels)
            ],
        }

    def _ensure_width(self, width: int) -> None:
        if self.width is None:
            self.width = width
            self.sum_abs = [0.0] * width
            self.sum_sq = [0.0] * width
            self.positive = [0] * width
            self.active = [0] * width
        elif self.width != width:
            raise ProfileError(f"expected activation width {self.width}, got {width}")


@dataclass(slots=True)
class DocumentStats:
    index: int
    text_sha256: str
    char_count: int
    module_stats: dict[str, ChannelStats] = field(default_factory=dict)

    @classmethod
    def from_text(cls, *, index: int, text: str) -> "DocumentStats":
        return cls(
            index=index,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            char_count=len(text),
        )

    def update(self, module_name: str, values: Any, *, threshold: float) -> None:
        stats = self.module_stats.get(module_name)
        if stats is None:
            stats = ChannelStats(threshold=threshold)
            self.module_stats[module_name] = stats
        stats.update(values)

    def to_report(
        self,
        *,
        module_targets: dict[str, dict[str, Any]],
        include_vectors: bool,
        top_k_channels: int,
        experts: int,
        pool_size: int,
    ) -> dict[str, Any]:
        module_reports = {
            name: {
                **stats.to_report(include_vectors=include_vectors, top_k_channels=top_k_channels),
                "target": module_targets.get(name),
            }
            for name, stats in sorted(self.module_stats.items())
        }
        return {
            "index": self.index,
            "text_sha256": self.text_sha256,
            "char_count": self.char_count,
            "modules": module_reports,
            "expert_pool": recommend_document_expert_pool(
                module_reports=module_reports,
                experts=experts,
                pool_size=pool_size,
            ),
        }


@dataclass(slots=True)
class ActivationProfile:
    model: str
    adapter_family: str | None
    samples: int
    sequence_length: int
    module_stats: dict[str, ChannelStats] = field(default_factory=dict)
    module_targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    documents: list[DocumentStats] = field(default_factory=list)
    missing_modules: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _current_document_index: int | None = field(default=None, init=False)

    def begin_document(self, *, index: int, text: str) -> None:
        self.documents.append(DocumentStats.from_text(index=index, text=text))
        self._current_document_index = len(self.documents) - 1

    def end_document(self) -> None:
        self._current_document_index = None

    def update(self, module_name: str, values: Any, *, threshold: float) -> None:
        stats = self.module_stats.get(module_name)
        if stats is None:
            stats = ChannelStats(threshold=threshold)
            self.module_stats[module_name] = stats
        stats.update(values)
        if self._current_document_index is not None:
            self.documents[self._current_document_index].update(
                module_name,
                values,
                threshold=threshold,
            )

    def to_report(
        self,
        *,
        include_vectors: bool,
        top_k_channels: int,
        include_document_vectors: bool,
        document_top_k_channels: int,
        experts: int,
        shared_ratio: float,
        document_pool_size: int | None = None,
    ) -> dict[str, Any]:
        document_pool_size = document_pool_size or min(2, experts)
        return {
            "model": self.model,
            "adapter_family": self.adapter_family,
            "samples": self.samples,
            "sequence_length": self.sequence_length,
            "document_count": len(self.documents),
            "modules": {
                name: {
                    **stats.to_report(include_vectors=include_vectors, top_k_channels=top_k_channels),
                    "target": self.module_targets.get(name),
                    "assignment": stats.assign_experts(experts=experts, shared_ratio=shared_ratio),
                }
                for name, stats in sorted(self.module_stats.items())
            },
            "documents": [
                document.to_report(
                    module_targets=self.module_targets,
                    include_vectors=include_document_vectors,
                    top_k_channels=document_top_k_channels,
                    experts=experts,
                    pool_size=document_pool_size,
                )
                for document in self.documents
            ],
            "missing_modules": self.missing_modules,
            "warnings": self.warnings,
        }


def load_calibration_texts(
    *,
    text: str | None = None,
    text_file: Path | None = None,
    max_samples: int,
) -> list[str]:
    samples: list[str] = []
    if text:
        samples.append(text)
    if text_file:
        content = text_file.read_text(encoding="utf-8")
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        samples.extend(chunks)
    if not samples:
        samples = [
            "Mixture-of-Experts models route tokens through specialized feed-forward experts.",
            "A dense model can be profiled by measuring which MLP channels activate on calibration text.",
            "The goal is to preserve quality while reducing active compute.",
        ]
    return samples[:max_samples]


def resolve_profile_modules(
    *,
    adapter: ArchitectureAdapter,
    layer_count: int,
    layers: str | None,
    roles: Iterable[str],
) -> dict[str, dict[str, Any]]:
    selected_layers = _parse_layer_spec(layers, layer_count) if layers else list(range(layer_count))
    role_set = set(roles)
    targets: dict[str, dict[str, Any]] = {}
    for layer in selected_layers:
        for role, patterns in (
            ("gate", adapter.hf_tensors.gate),
            ("up", adapter.hf_tensors.up),
            ("down", adapter.hf_tensors.down),
        ):
            if role not in role_set:
                continue
            for pattern in patterns:
                module_name = _weight_pattern_to_module_name(pattern).format(layer=layer)
                targets[module_name] = {"layer": layer, "role": role, "pattern": pattern}
    return targets


def profile_hf_model(
    model_ref: str,
    texts: list[str],
    options: ProfileOptions,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise ProfileError("profiling requires optional dependencies: torch and transformers") from exc

    info = inspect_model(model_ref)
    if info.source_format == "gguf":
        raise ProfileError("activation profiling currently requires a Hugging Face model or model id")
    if info.adapter_family is None or info.layer_count is None:
        raise ProfileError("model needs a matched architecture adapter and layer count")

    adapter = _adapter_for_family(info.adapter_family)
    if adapter is None:
        raise ProfileError(f"no adapter registered for {info.adapter_family}")

    load_ref = _load_ref(model_ref)
    device = _resolve_device(options.device, torch)
    dtype = _resolve_dtype(options.dtype, torch)

    tokenizer = AutoTokenizer.from_pretrained(load_ref, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(load_ref, **model_kwargs)
    model.to(device)
    model.eval()

    module_targets = resolve_profile_modules(
        adapter=adapter,
        layer_count=info.layer_count,
        layers=options.layers,
        roles=options.roles,
    )
    named_modules = dict(model.named_modules())
    profile = ActivationProfile(
        model=model_ref,
        adapter_family=info.adapter_family,
        samples=min(len(texts), options.max_samples),
        sequence_length=options.sequence_length,
        module_targets=module_targets,
        warnings=list(info.warnings),
    )

    hooks = []
    for module_name, target in module_targets.items():
        module = named_modules.get(module_name)
        if module is None:
            profile.missing_modules.append({"module": module_name, **target})
            continue
        hooks.append(
            module.register_forward_hook(
                _make_hook(profile, module_name, threshold=options.threshold)
            )
        )

    if not hooks:
        raise ProfileError("none of the expected FFN modules were found on the loaded model")

    try:
        with torch.no_grad():
            for sample_index, sample in enumerate(texts[: options.max_samples]):
                inputs = tokenizer(
                    sample,
                    return_tensors="pt",
                    truncation=True,
                    max_length=options.sequence_length,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                profile.begin_document(index=sample_index, text=sample)
                try:
                    model(**inputs)
                finally:
                    profile.end_document()
    finally:
        for hook in hooks:
            hook.remove()

    return profile.to_report(
        include_vectors=options.include_vectors,
        top_k_channels=options.top_k_channels,
        include_document_vectors=options.include_document_vectors,
        document_top_k_channels=options.document_top_k_channels,
        experts=options.experts,
        shared_ratio=options.shared_ratio,
        document_pool_size=min(options.experts, max(1, options.top_k_channels // 8)),
    )


def recommend_document_expert_pool(
    *,
    module_reports: dict[str, dict[str, Any]],
    experts: int,
    pool_size: int,
) -> dict[str, Any]:
    if experts <= 0:
        raise ProfileError("experts must be positive")
    pool_size = max(1, min(pool_size, experts))
    scores = [0.0 for _ in range(experts)]
    for module in module_reports.values():
        top_channels = module.get("top_channels", [])
        if not isinstance(top_channels, list):
            continue
        for item in top_channels:
            if not isinstance(item, dict):
                continue
            channel = int(item.get("channel", 0))
            score = float(item.get("mean_abs", 0.0))
            scores[channel % experts] += score
    selected = sorted(range(experts), key=lambda index: scores[index], reverse=True)[:pool_size]
    return {
        "method": "top_channel_mod_expert_score",
        "experts": selected,
        "pool_size": pool_size,
        "scores": scores,
    }


def _make_hook(profile: ActivationProfile, module_name: str, *, threshold: float):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        profile.update(module_name, tensor, threshold=threshold)

    return hook


def _to_channel_rows(values: Any) -> list[list[float]]:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free tests pass list inputs
        torch = None

    if torch is not None and isinstance(values, torch.Tensor):
        tensor = values.detach().float().cpu()
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, tensor.shape[0])
        elif tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        return tensor.tolist()

    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        raise ProfileError("activation values must be tensor-like or list-like")
    if not values:
        return []
    if not isinstance(values[0], list):
        return [[float(item) for item in values]]
    return _flatten_nested_rows(values)


def _flatten_nested_rows(values: list[Any]) -> list[list[float]]:
    rows: list[list[float]] = []
    stack = list(values)
    while stack:
        item = stack.pop(0)
        if not item:
            continue
        if isinstance(item[0], list):
            stack = list(item) + stack
        else:
            rows.append([float(value) for value in item])
    return rows


def _adapter_for_family(family: str) -> ArchitectureAdapter | None:
    for adapter in ADAPTERS:
        if adapter.family == family:
            return adapter
    return None


def _weight_pattern_to_module_name(pattern: str) -> str:
    if pattern.endswith(".weight"):
        return pattern[: -len(".weight")]
    return pattern


def _load_ref(model_ref: str) -> str:
    if model_ref.startswith("hf:"):
        return model_ref[3:].split("@", 1)[0]
    return model_ref


def _resolve_device(device: str, torch: Any) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str, torch: Any) -> Any:
    if dtype == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ProfileError(f"unsupported dtype {dtype}")
    return mapping[dtype]
