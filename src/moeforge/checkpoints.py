from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_hf_checkpoint_files(model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        return {
            "has_weights": False,
            "format": None,
            "files": [],
            "index_files": [],
            "total_size_bytes": None,
        }

    safetensors = sorted(model_path.glob("*.safetensors"))
    bins = sorted(model_path.glob("*.bin"))
    index_files = sorted(model_path.glob("*.index.json"))

    files = safetensors or bins
    fmt = "safetensors" if safetensors else ("bin" if bins else None)
    total_size = sum(item.stat().st_size for item in files) if files else None
    weight_map_count = _weight_map_count(index_files)

    return {
        "has_weights": bool(files),
        "format": fmt,
        "files": [item.name for item in files],
        "index_files": [item.name for item in index_files],
        "total_size_bytes": total_size,
        "weight_map_count": weight_map_count,
    }


def _weight_map_count(index_files: list[Path]) -> int | None:
    count = 0
    found = False
    for path in index_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weight_map = payload.get("weight_map")
        if isinstance(weight_map, dict):
            count += len(weight_map)
            found = True
    return count if found else None

