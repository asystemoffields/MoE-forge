from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Any


class GGUFError(ValueError):
    """Raised when a file cannot be parsed as GGUF metadata."""


_TYPE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


@dataclass(slots=True)
class GGUFMetadata:
    path: Path
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata: dict[str, Any]


def read_gguf_metadata(path: Path, *, max_array_preview: int = 8) -> GGUFMetadata:
    with path.open("rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            preview = magic.decode("ascii", errors="replace")
            raise GGUFError(f"{path} does not start with GGUF magic bytes; found {preview!r}")

        version = _read_struct(fh, "<I")
        tensor_count = _read_struct(fh, "<Q")
        metadata_kv_count = _read_struct(fh, "<Q")

        metadata: dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            key = _read_string(fh)
            value_type = _read_struct(fh, "<I")
            metadata[key] = _read_value(fh, value_type, max_array_preview=max_array_preview)

    return GGUFMetadata(
        path=path,
        version=version,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
        metadata=metadata,
    )


def _read_value(fh: BinaryIO, value_type: int, *, max_array_preview: int) -> Any:
    if value_type in _SCALAR_FORMATS:
        return _read_struct(fh, _SCALAR_FORMATS[value_type])
    if value_type == 8:
        return _read_string(fh)
    if value_type == 9:
        return _read_array(fh, max_array_preview=max_array_preview)
    raise GGUFError(f"unsupported GGUF metadata value type {value_type}")


def _read_array(fh: BinaryIO, *, max_array_preview: int) -> dict[str, Any]:
    item_type = _read_struct(fh, "<I")
    length = _read_struct(fh, "<Q")
    preview = []
    for index in range(length):
        if index < max_array_preview:
            preview.append(_read_value(fh, item_type, max_array_preview=max_array_preview))
        else:
            _skip_value(fh, item_type)
    return {
        "kind": "array",
        "item_type": _TYPE_NAMES.get(item_type, f"type_{item_type}"),
        "length": length,
        "preview": preview,
    }


def _skip_value(fh: BinaryIO, value_type: int) -> None:
    if value_type in _SCALAR_FORMATS:
        fh.seek(struct.calcsize(_SCALAR_FORMATS[value_type]), 1)
        return
    if value_type == 8:
        length = _read_struct(fh, "<Q")
        fh.seek(length, 1)
        return
    if value_type == 9:
        item_type = _read_struct(fh, "<I")
        length = _read_struct(fh, "<Q")
        for _ in range(length):
            _skip_value(fh, item_type)
        return
    raise GGUFError(f"unsupported GGUF metadata value type {value_type}")


def _read_string(fh: BinaryIO) -> str:
    length = _read_struct(fh, "<Q")
    data = fh.read(length)
    if len(data) != length:
        raise GGUFError("unexpected end of file while reading GGUF string")
    return data.decode("utf-8", errors="replace")


def _read_struct(fh: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = fh.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file while reading GGUF value")
    return struct.unpack(fmt, data)[0]

