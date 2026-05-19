from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


class MaterializeError(RuntimeError):
    """Raised when a carve manifest cannot be materialized."""


@dataclass(slots=True)
class MaterializedTensor:
    name: str
    source_tensor: str
    source_file: str
    role: str
    layer: int
    kind: str
    expert: int | None
    shape: list[int] | None


@dataclass(slots=True)
class MaterializeReport:
    manifest: str
    output_dir: str
    dry_run: bool
    tensors: list[MaterializedTensor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tensor_count"] = len(self.tensors)
        return payload


def materialize_carve_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> MaterializeReport:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_model = Path(str(manifest.get("source_model", "")))
    if not source_model.is_dir():
        raise MaterializeError("manifest source_model must be a local checkpoint folder")

    report = MaterializeReport(
        manifest=str(manifest_path),
        output_dir=str(output_dir),
        dry_run=dry_run,
        warnings=list(manifest.get("warnings", [])),
    )
    output_tensors: dict[str, Any] = {}

    if dry_run:
        for plan in _iter_tensor_outputs(manifest):
            report.tensors.append(_planned_tensor_record(plan, shape=_planned_shape(plan)))
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "carve-apply-dry-run.json"
        report.output_files = [str(report_path)]
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise MaterializeError("carve-apply requires torch and safetensors") from exc

    file_cache: dict[str, dict[str, Any]] = {}
    for plan in _iter_tensor_outputs(manifest):
        source_file = source_model / plan["source_file"]
        if not source_file.exists():
            raise MaterializeError(f"source tensor file not found: {source_file}")
        tensors = file_cache.get(str(source_file))
        if tensors is None:
            tensors = load_file(str(source_file), device="cpu")
            file_cache[str(source_file)] = tensors

        source_tensor = tensors.get(plan["source_tensor"])
        if source_tensor is None:
            raise MaterializeError(f"source tensor not found: {plan['source_tensor']}")
        axis = plan["channel_axis"]
        if axis is None:
            raise MaterializeError(f"cannot slice {plan['source_tensor']}: channel axis is unknown")

        channels = plan["channels"]
        if not channels:
            continue
        max_channel = max(channels)
        if max_channel >= source_tensor.shape[axis]:
            raise MaterializeError(
                f"channel {max_channel} is out of range for {plan['source_tensor']} axis {axis}"
            )

        index = torch.tensor(channels, dtype=torch.long)
        sliced = source_tensor.index_select(axis, index).contiguous()
        output_tensors[plan["output_tensor"]] = sliced
        report.tensors.append(_planned_tensor_record(plan, shape=list(sliced.shape)))

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_file = output_dir / "carved-experts.safetensors"
    metadata = {
        "moe_forge": "carved_experts",
        "source_model": str(source_model),
        "manifest": str(manifest_path),
    }
    save_file(output_tensors, str(tensor_file), metadata=metadata)
    report_path = output_dir / "carve-apply-report.json"
    report.output_files = [str(tensor_file), str(report_path)]
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _iter_tensor_outputs(manifest: dict[str, Any]):
    for layer in manifest.get("layers", []):
        if not isinstance(layer, dict):
            continue
        layer_id = int(layer.get("layer", -1))
        for tensor in layer.get("tensors", []):
            if not isinstance(tensor, dict):
                continue
            role = str(tensor.get("role"))
            shared_channels = [int(item) for item in tensor.get("shared_channels", [])]
            if shared_channels:
                yield {
                    "layer": layer_id,
                    "role": role,
                    "kind": "shared",
                    "expert": None,
                    "channels": shared_channels,
                    "source_tensor": str(tensor.get("source_tensor")),
                    "source_file": str(tensor.get("source_file")),
                    "source_shape": tensor.get("source_shape"),
                    "channel_axis": tensor.get("channel_axis"),
                    "output_tensor": f"moe.layers.{layer_id}.mlp.shared.{role}.weight",
                }
            for expert_index, channels in enumerate(tensor.get("expert_channels", [])):
                channels = [int(item) for item in channels]
                if not channels:
                    continue
                yield {
                    "layer": layer_id,
                    "role": role,
                    "kind": "expert",
                    "expert": expert_index,
                    "channels": channels,
                    "source_tensor": str(tensor.get("source_tensor")),
                    "source_file": str(tensor.get("source_file")),
                    "source_shape": tensor.get("source_shape"),
                    "channel_axis": tensor.get("channel_axis"),
                    "output_tensor": f"moe.layers.{layer_id}.mlp.experts.{expert_index}.{role}.weight",
                }


def _planned_tensor_record(plan: dict[str, Any], *, shape: list[int] | None) -> MaterializedTensor:
    return MaterializedTensor(
        name=plan["output_tensor"],
        source_tensor=plan["source_tensor"],
        source_file=plan["source_file"],
        role=plan["role"],
        layer=plan["layer"],
        kind=plan["kind"],
        expert=plan["expert"],
        shape=shape,
    )


def _planned_shape(plan: dict[str, Any]) -> list[int] | None:
    source_shape = plan.get("source_shape")
    axis = plan.get("channel_axis")
    if not isinstance(source_shape, list) or axis is None:
        return None
    shape = [int(item) for item in source_shape]
    shape[int(axis)] = len(plan["channels"])
    return shape
