from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .model_info import ModelInfo


@dataclass(slots=True)
class LayerLayout:
    layer: int
    intermediate_size: int
    shared_channels: int
    routed_channels: int
    expert_channels: list[int]
    active_channels_per_token: int


def build_layer_layouts(
    info: ModelInfo,
    layers: list[int] | str,
    *,
    experts: int,
    top_k: int,
    shared_ratio: float,
) -> list[LayerLayout]:
    if isinstance(layers, str):
        return []

    layouts = []
    for layer in layers:
        intermediate_size = _intermediate_for_layer(info, layer)
        shared_channels = _align(int(round(intermediate_size * shared_ratio)), 256)
        shared_channels = min(shared_channels, intermediate_size)
        routed_channels = max(0, intermediate_size - shared_channels)
        expert_channels = _split_evenly(routed_channels, experts, alignment=256)
        active_channels = shared_channels + sum(sorted(expert_channels, reverse=True)[:top_k])
        layouts.append(
            LayerLayout(
                layer=layer,
                intermediate_size=intermediate_size,
                shared_channels=shared_channels,
                routed_channels=routed_channels,
                expert_channels=expert_channels,
                active_channels_per_token=active_channels,
            )
        )
    return layouts


def _parse_layer_spec(value: str, layer_count: int | None = None) -> list[int]:
    normalized = value.strip().lower()
    if normalized in {"all", "*", "full"}:
        if layer_count is None:
            raise ValueError("layer count is required for all-layer selection")
        return list(range(layer_count))
    if ":" in value:
        start_raw, end_raw = value.split(":", 1)
        start = int(start_raw) if start_raw else 0
        end = int(end_raw) if end_raw else (layer_count - 1 if layer_count else start)
        return list(range(start, end + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def summarize_layouts(layouts: list[LayerLayout]) -> dict[str, Any]:
    if not layouts:
        return {"layers": [], "active_fraction_mean": None}

    layer_dicts = [asdict(layout) for layout in layouts]
    active_fractions = [
        layout.active_channels_per_token / layout.intermediate_size
        for layout in layouts
        if layout.intermediate_size
    ]
    return {
        "layers": layer_dicts,
        "active_fraction_mean": sum(active_fractions) / len(active_fractions),
        "active_fraction_min": min(active_fractions),
        "active_fraction_max": max(active_fractions),
    }


def _intermediate_for_layer(info: ModelInfo, layer: int) -> int:
    if info.intermediate_sizes and layer < len(info.intermediate_sizes):
        return info.intermediate_sizes[layer]
    if info.intermediate_size:
        return info.intermediate_size
    if info.hidden_size:
        return info.hidden_size * 4
    raise ValueError("cannot build layout without intermediate_size or hidden_size")


def _split_evenly(total: int, parts: int, *, alignment: int) -> list[int]:
    if parts <= 0:
        raise ValueError("experts must be positive")
    base = _align(total // parts, alignment)
    values = [base for _ in range(parts)]
    used = base * parts

    index = 0
    while used + alignment <= total:
        values[index % parts] += alignment
        used += alignment
        index += 1

    remainder = total - used
    if remainder > 0:
        values[-1] += remainder
    return values


def _align(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    if value < alignment:
        return value
    return max(alignment, (value // alignment) * alignment)
