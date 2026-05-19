from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .inspectors import inspect_model
from .layout import _parse_layer_spec


class CarveError(RuntimeError):
    """Raised when a carve manifest cannot be constructed."""


@dataclass(slots=True)
class TensorSlicePlan:
    role: str
    source_tensor: str
    source_file: str
    source_shape: list[int] | None
    channel_axis: int | None
    shared_channels: list[int]
    expert_channels: list[list[int]]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LayerCarvePlan:
    layer: int
    width: int | None
    shared_channels: list[int]
    expert_channels: list[list[int]]
    tensors: list[TensorSlicePlan]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CarveManifest:
    source_model: str
    adapter_family: str | None
    strategy: str
    experts: int
    shared_ratio: float
    layers: list[LayerCarvePlan]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_carve_manifest(
    *,
    model: str,
    recipe_path: Path,
    profile_path: Path | None = None,
) -> CarveManifest:
    info = inspect_model(model)
    if info.source_format != "hf":
        raise CarveError("carve manifests require a local Hugging Face checkpoint folder")
    if not isinstance(info.path, Path) or not info.path.is_dir():
        raise CarveError("carve manifests require a local Hugging Face checkpoint folder")

    recipe = _read_json(recipe_path)
    profile = _read_json(profile_path) if profile_path else None
    tensor_map = info.metadata.get("ffn_tensor_map")
    if not isinstance(tensor_map, dict) or not tensor_map.get("mapped_layers"):
        raise CarveError("inspection did not produce an FFN tensor map")

    layer_records = {
        int(record["layer"]): record
        for record in tensor_map.get("mapped_layers", [])
        if isinstance(record, dict) and "layer" in record
    }

    layer_ids = _recipe_layers(recipe, info.layer_count)
    experts = int(recipe.get("experts") or 0)
    if experts <= 0:
        raise CarveError("recipe must specify a positive expert count")
    shared_ratio = float(recipe.get("shared_ratio") or 0.0)

    warnings = list(info.warnings)
    if tensor_map.get("available") is False:
        warnings.append("FFN tensor map is incomplete; manifest includes the tensors that could be matched.")

    layers: list[LayerCarvePlan] = []
    for layer in layer_ids:
        tensor_record = layer_records.get(layer)
        if not tensor_record:
            layers.append(
                LayerCarvePlan(
                    layer=layer,
                    width=None,
                    shared_channels=[],
                    expert_channels=[],
                    tensors=[],
                    warnings=["no FFN tensor map record found for this layer"],
                )
            )
            continue

        assignment = _assignment_for_layer(
            layer=layer,
            profile=profile,
            recipe=recipe,
            tensor_record=tensor_record,
            experts=experts,
            shared_ratio=shared_ratio,
        )
        layer_warnings = list(assignment.get("warnings", []))
        shared_channels = assignment["shared_channels"]
        expert_channels = assignment["expert_channels"]
        width = assignment["width"]

        tensors = []
        for role in ("gate", "up", "down"):
            role_record = tensor_record.get(role)
            if not isinstance(role_record, dict):
                continue
            source_shape = role_record.get("shape")
            channel_axis = _channel_axis(role=role, shape=source_shape, width=width)
            tensor_warnings = []
            if channel_axis is None:
                tensor_warnings.append("could not infer channel axis from tensor shape")
            tensors.append(
                TensorSlicePlan(
                    role=role,
                    source_tensor=str(role_record.get("name")),
                    source_file=str(role_record.get("file")),
                    source_shape=[int(item) for item in source_shape] if isinstance(source_shape, list) else None,
                    channel_axis=channel_axis,
                    shared_channels=shared_channels,
                    expert_channels=expert_channels,
                    warnings=tensor_warnings,
                )
            )

        layers.append(
            LayerCarvePlan(
                layer=layer,
                width=width,
                shared_channels=shared_channels,
                expert_channels=expert_channels,
                tensors=tensors,
                warnings=layer_warnings,
            )
        )

    return CarveManifest(
        source_model=str(info.path),
        adapter_family=info.adapter_family,
        strategy=str(recipe.get("strategy") or "carved_mlp"),
        experts=experts,
        shared_ratio=shared_ratio,
        layers=layers,
        warnings=warnings,
    )


def _assignment_for_layer(
    *,
    layer: int,
    profile: dict[str, Any] | None,
    recipe: dict[str, Any],
    tensor_record: dict[str, Any],
    experts: int,
    shared_ratio: float,
) -> dict[str, Any]:
    profile_assignment = _profile_assignment_for_layer(layer, profile)
    if profile_assignment:
        return _normalize_assignment(profile_assignment, experts=experts)

    width = _infer_width_from_tensor_record(tensor_record)
    layout_assignment = _layout_assignment_for_layer(layer, recipe, experts=experts)
    if layout_assignment:
        if width is not None and layout_assignment["width"] != width:
            layout_assignment.setdefault("warnings", []).append(
                f"recipe layout width {layout_assignment['width']} differs from tensor width {width}"
            )
        return layout_assignment

    if width is None:
        return {
            "width": None,
            "shared_channels": [],
            "expert_channels": [[] for _ in range(experts)],
            "warnings": ["could not infer FFN width; channel assignment is empty"],
        }

    return _even_assignment(width=width, experts=experts, shared_ratio=shared_ratio)


def _profile_assignment_for_layer(layer: int, profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not profile:
        return None
    modules = profile.get("modules")
    if not isinstance(modules, dict):
        return None
    for module in modules.values():
        if not isinstance(module, dict):
            continue
        target = module.get("target")
        assignment = module.get("assignment")
        if (
            isinstance(target, dict)
            and int(target.get("layer", -1)) == layer
            and target.get("role") in {"gate", "up"}
            and isinstance(assignment, dict)
            and assignment.get("available")
        ):
            return assignment
    return None


def _normalize_assignment(assignment: dict[str, Any], *, experts: int) -> dict[str, Any]:
    shared = sorted({int(item) for item in assignment.get("shared_channels", [])})
    expert_channels = [[] for _ in range(experts)]
    for item in assignment.get("experts", []):
        if not isinstance(item, dict):
            continue
        index = int(item.get("expert", 0))
        if 0 <= index < experts:
            expert_channels[index] = sorted({int(channel) for channel in item.get("channels", [])})

    width = assignment.get("width")
    normalized = {
        "width": int(width) if width is not None else _width_from_channels(shared, expert_channels),
        "shared_channels": shared,
        "expert_channels": expert_channels,
        "warnings": [],
    }
    normalized["warnings"].extend(_validate_assignment(normalized["width"], shared, expert_channels))
    return normalized


def _layout_assignment_for_layer(layer: int, recipe: dict[str, Any], *, experts: int) -> dict[str, Any] | None:
    layout = recipe.get("layout")
    if not isinstance(layout, dict):
        return None
    for record in layout.get("layers", []):
        if not isinstance(record, dict) or int(record.get("layer", -1)) != layer:
            continue
        width = int(record["intermediate_size"])
        shared_count = int(record["shared_channels"])
        expert_counts = [int(item) for item in record.get("expert_channels", [])]
        shared = list(range(shared_count))
        expert_channels: list[list[int]] = []
        cursor = shared_count
        for count in expert_counts[:experts]:
            expert_channels.append(list(range(cursor, cursor + count)))
            cursor += count
        while len(expert_channels) < experts:
            expert_channels.append([])
        result = {
            "width": width,
            "shared_channels": shared,
            "expert_channels": expert_channels,
            "warnings": [],
        }
        result["warnings"].extend(_validate_assignment(width, shared, expert_channels))
        return result
    return None


def _even_assignment(*, width: int, experts: int, shared_ratio: float) -> dict[str, Any]:
    shared_count = int(round(width * shared_ratio))
    shared = list(range(shared_count))
    routed = list(range(shared_count, width))
    expert_channels = [[] for _ in range(experts)]
    for offset, channel in enumerate(routed):
        expert_channels[offset % experts].append(channel)
    result = {
        "width": width,
        "shared_channels": shared,
        "expert_channels": expert_channels,
        "warnings": ["used even assignment fallback"],
    }
    result["warnings"].extend(_validate_assignment(width, shared, expert_channels))
    return result


def _channel_axis(*, role: str, shape: Any, width: int | None) -> int | None:
    if width is None or not isinstance(shape, list):
        return None
    int_shape = [int(item) for item in shape]
    if role in {"gate", "up"} and int_shape and int_shape[0] == width:
        return 0
    if role == "down" and len(int_shape) > 1 and int_shape[1] == width:
        return 1
    matches = [index for index, size in enumerate(int_shape) if size == width]
    return matches[0] if len(matches) == 1 else None


def _infer_width_from_tensor_record(tensor_record: dict[str, Any]) -> int | None:
    for role in ("gate", "up"):
        role_record = tensor_record.get(role)
        shape = role_record.get("shape") if isinstance(role_record, dict) else None
        if isinstance(shape, list) and shape:
            return int(shape[0])
    down = tensor_record.get("down")
    down_shape = down.get("shape") if isinstance(down, dict) else None
    if isinstance(down_shape, list) and len(down_shape) > 1:
        return int(down_shape[1])
    return None


def _recipe_layers(recipe: dict[str, Any], layer_count: int | None) -> list[int]:
    layers = recipe.get("moe_layers")
    if isinstance(layers, list):
        return [int(item) for item in layers]
    if isinstance(layers, str) and layers != "middle_to_final_layers":
        return _parse_layer_spec(layers, layer_count)
    if layer_count is None:
        raise CarveError("recipe does not provide concrete layers and layer count is unknown")
    return list(range(layer_count // 4, layer_count))


def _validate_assignment(width: int | None, shared: list[int], expert_channels: list[list[int]]) -> list[str]:
    if width is None:
        return ["cannot validate assignment without width"]
    warnings = []
    all_channels = shared + [channel for expert in expert_channels for channel in expert]
    duplicates = sorted({channel for channel in all_channels if all_channels.count(channel) > 1})
    if duplicates:
        warnings.append(f"duplicate channel assignments: {duplicates[:10]}")
    out_of_range = [channel for channel in all_channels if channel < 0 or channel >= width]
    if out_of_range:
        warnings.append(f"out-of-range channel assignments: {out_of_range[:10]}")
    missing_count = width - len(set(all_channels))
    if missing_count:
        warnings.append(f"{missing_count} channels are not assigned")
    return warnings


def _width_from_channels(shared: list[int], expert_channels: list[list[int]]) -> int | None:
    channels = shared + [channel for expert in expert_channels for channel in expert]
    if not channels:
        return None
    return max(channels) + 1


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

