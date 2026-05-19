from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any

from .adapters import ArchitectureAdapter
from .model_info import ModelInfo


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    file: str
    dtype: str | None = None
    shape: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "dtype": self.dtype,
            "shape": self.shape,
        }


def build_hf_tensor_index(model_path: Path) -> dict[str, TensorRecord]:
    if not model_path.is_dir():
        return {}

    index_records = _records_from_index_files(model_path)
    header_records = _records_from_safetensors_headers(model_path)

    merged = dict(index_records)
    for name, record in header_records.items():
        previous = merged.get(name)
        if previous is None:
            merged[name] = record
        else:
            merged[name] = TensorRecord(
                name=name,
                file=previous.file,
                dtype=record.dtype or previous.dtype,
                shape=record.shape or previous.shape,
            )
    return merged


def validate_ffn_tensor_map(info: ModelInfo) -> dict[str, Any]:
    if info.source_format != "hf" or info.adapter is None:
        return {"available": False, "reason": "requires a local HF checkpoint with a matched adapter"}

    model_path = info.path
    if not isinstance(model_path, Path) or not model_path.is_dir():
        return {"available": False, "reason": "requires a local HF checkpoint folder"}

    adapter = _adapter_from_info(info)
    if adapter is None:
        return {"available": False, "reason": "adapter metadata is incomplete"}

    tensor_index = build_hf_tensor_index(model_path)
    if not tensor_index:
        return {"available": False, "reason": "no readable safetensors or index file found"}

    layers = info.layer_count or 0
    missing: list[dict[str, Any]] = []
    mapped_layers: list[dict[str, Any]] = []
    for layer in range(layers):
        layer_record: dict[str, Any] = {"layer": layer}
        for role, patterns in (
            ("gate", adapter.hf_tensors.gate),
            ("up", adapter.hf_tensors.up),
            ("down", adapter.hf_tensors.down),
        ):
            if not patterns:
                continue
            match = _first_matching_tensor(tensor_index, patterns, layer)
            if match is None:
                missing.append({"layer": layer, "role": role, "patterns": list(patterns)})
            else:
                layer_record[role] = match.to_dict()
        mapped_layers.append(layer_record)

    return {
        "available": not missing,
        "tensor_count": len(tensor_index),
        "mapped_layers": mapped_layers,
        "missing": missing,
    }


def _records_from_index_files(model_path: Path) -> dict[str, TensorRecord]:
    records: dict[str, TensorRecord] = {}
    for path in sorted(model_path.glob("*.index.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            continue
        for tensor_name, file_name in weight_map.items():
            records[str(tensor_name)] = TensorRecord(name=str(tensor_name), file=str(file_name))
    return records


def _records_from_safetensors_headers(model_path: Path) -> dict[str, TensorRecord]:
    records: dict[str, TensorRecord] = {}
    for path in sorted(model_path.glob("*.safetensors")):
        try:
            header = read_safetensors_header(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for tensor_name, payload in header.items():
            if tensor_name == "__metadata__" or not isinstance(payload, dict):
                continue
            records[str(tensor_name)] = TensorRecord(
                name=str(tensor_name),
                file=path.name,
                dtype=_optional_str(payload.get("dtype")),
                shape=[int(item) for item in payload.get("shape", [])],
            )
    return records


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        raw_length = fh.read(8)
        if len(raw_length) != 8:
            raise ValueError("safetensors file is too small")
        header_length = struct.unpack("<Q", raw_length)[0]
        header_bytes = fh.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("safetensors header is truncated")
    return json.loads(header_bytes.decode("utf-8"))


def _adapter_from_info(info: ModelInfo) -> ArchitectureAdapter | None:
    from .adapters import ADAPTERS

    for adapter in ADAPTERS:
        if adapter.family == info.adapter_family:
            return adapter
    return None


def _first_matching_tensor(
    tensor_index: dict[str, TensorRecord],
    patterns: tuple[str, ...],
    layer: int,
) -> TensorRecord | None:
    for pattern in patterns:
        name = pattern.format(layer=layer)
        record = tensor_index.get(name)
        if record is not None:
            return record
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)

