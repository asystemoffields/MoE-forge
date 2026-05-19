from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any

from .runtime import CarvedGatedMLP


class WrapperError(RuntimeError):
    """Raised when a wrapper package cannot be exported or loaded."""


@dataclass(slots=True)
class LayerWrapperConfig:
    layer: int
    width: int | None
    tensor_prefix: str
    expert_count: int
    shared_channels: int
    expert_channels: list[int]


@dataclass(slots=True)
class WrapperConfig:
    format_version: int
    model_type: str
    adapter_family: str | None
    source_model: str
    manifest_path: str
    artifact_path: str
    router_plan_path: str | None
    token_router_top_k: int | None
    token_router_path: str | None
    activation: str
    expert_count: int
    layers: list[LayerWrapperConfig]
    warnings: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_wrapper_package(
    *,
    manifest_path: Path,
    artifact_path: Path,
    output_dir: Path,
    router_plan_path: Path | None = None,
    activation: str = "silu",
    copy_artifact: bool = False,
    copy_source_model: bool = False,
    token_router_top_k: int | None = None,
) -> WrapperConfig:
    manifest = _read_json(manifest_path)
    if not artifact_path.exists():
        raise WrapperError(f"artifact does not exist: {artifact_path}")
    if router_plan_path is not None and not router_plan_path.exists():
        raise WrapperError(f"router plan does not exist: {router_plan_path}")
    if activation not in {"silu", "gelu", "gelu_tanh"}:
        raise WrapperError(f"unsupported activation: {activation}")
    if token_router_top_k is not None and token_router_top_k <= 0:
        raise WrapperError("token_router_top_k must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = output_dir / "carve-manifest.json"
    _copy_if_different(manifest_path, local_manifest)

    local_router: Path | None = None
    if router_plan_path is not None:
        local_router = output_dir / "router-plan.json"
        _copy_if_different(router_plan_path, local_router)

    if copy_artifact:
        local_artifact = output_dir / artifact_path.name
        _copy_if_different(artifact_path, local_artifact)
        artifact_ref = local_artifact.name
    else:
        artifact_ref = str(artifact_path.resolve())

    source_model_ref = str(manifest.get("source_model", ""))
    if copy_source_model:
        source_model_ref = _copy_source_model(source_model_ref, output_dir=output_dir)

    config = WrapperConfig(
        format_version=1,
        model_type="moeforge_carved_moe",
        adapter_family=manifest.get("adapter_family"),
        source_model=source_model_ref,
        manifest_path=local_manifest.name,
        artifact_path=artifact_ref,
        router_plan_path=local_router.name if local_router else None,
        token_router_top_k=token_router_top_k,
        token_router_path=None,
        activation=activation,
        expert_count=int(manifest.get("experts") or 0),
        layers=_layer_configs(manifest),
        warnings=list(manifest.get("warnings", [])),
        references=[
            "https://github.com/asystemoffields/MoE-forge",
        ],
    )
    config_path = output_dir / "moeforge_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_hf_config(output_dir, config)
    _write_wrapper_readme(output_dir, config)
    return config


def load_wrapper_config(config_path: Path) -> WrapperConfig:
    payload = _read_json(config_path)
    return WrapperConfig(
        format_version=int(payload["format_version"]),
        model_type=str(payload["model_type"]),
        adapter_family=payload.get("adapter_family"),
        source_model=str(payload.get("source_model", "")),
        manifest_path=str(payload["manifest_path"]),
        artifact_path=str(payload["artifact_path"]),
        router_plan_path=payload.get("router_plan_path"),
        token_router_top_k=int(payload["token_router_top_k"]) if payload.get("token_router_top_k") is not None else None,
        token_router_path=payload.get("token_router_path"),
        activation=str(payload.get("activation", "silu")),
        expert_count=int(payload["expert_count"]),
        layers=[
            LayerWrapperConfig(
                layer=int(item["layer"]),
                width=item.get("width"),
                tensor_prefix=str(item["tensor_prefix"]),
                expert_count=int(item["expert_count"]),
                shared_channels=int(item["shared_channels"]),
                expert_channels=[int(value) for value in item.get("expert_channels", [])],
            )
            for item in payload.get("layers", [])
        ],
        warnings=[str(item) for item in payload.get("warnings", [])],
        references=[str(item) for item in payload.get("references", [])],
    )


def load_layer_runtime(config_path: Path, *, layer: int) -> CarvedGatedMLP:
    config = load_wrapper_config(config_path)
    package_dir = config_path.parent
    artifact_path = _resolve_package_path(package_dir, config.artifact_path)
    manifest_path = _resolve_package_path(package_dir, config.manifest_path)
    if not any(item.layer == layer for item in config.layers):
        raise WrapperError(f"layer {layer} is not present in wrapper config")
    return CarvedGatedMLP.from_artifact(
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        layer=layer,
        activation=config.activation,
    )


def load_router_plan(config_path: Path) -> dict[str, Any] | None:
    config = load_wrapper_config(config_path)
    if config.router_plan_path is None:
        return None
    return _read_json(_resolve_package_path(config_path.parent, config.router_plan_path))


def _layer_configs(manifest: dict[str, Any]) -> list[LayerWrapperConfig]:
    expert_count = int(manifest.get("experts") or 0)
    layers = []
    for layer in manifest.get("layers", []):
        if not isinstance(layer, dict):
            continue
        layers.append(
            LayerWrapperConfig(
                layer=int(layer.get("layer")),
                width=layer.get("width"),
                tensor_prefix=f"moe.layers.{int(layer.get('layer'))}.mlp",
                expert_count=expert_count,
                shared_channels=len(layer.get("shared_channels", [])),
                expert_channels=[len(item) for item in layer.get("expert_channels", [])],
            )
        )
    return layers


def _resolve_package_path(package_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return package_dir / path


def _copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _copy_source_model(source_model: str, *, output_dir: Path) -> str:
    source = Path(source_model)
    if not source.is_dir():
        raise WrapperError(f"copy_source_model requires a local source model directory: {source_model}")
    destination = output_dir / "source-model"
    if source.resolve() == destination.resolve():
        return destination.name
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination.name


def _write_wrapper_readme(output_dir: Path, config: WrapperConfig) -> None:
    lines = [
        "# MoE Forge Wrapper Package",
        "",
        "This directory contains metadata for loading carved MoE Forge FFN artifacts.",
        "",
        "Files:",
        "",
        "- `config.json`: Transformers-compatible MoE Forge config",
        "- `moeforge_config.json`: wrapper configuration",
        "- `carve-manifest.json`: tensor slicing manifest",
    ]
    if config.router_plan_path:
        lines.append("- `router-plan.json`: document-pool router metadata")
    lines.extend(
        [
            "",
            f"Activation: `{config.activation}`",
            f"Expert count: `{config.expert_count}`",
            f"Layers: `{', '.join(str(item.layer) for item in config.layers)}`",
            f"Source model: `{config.source_model}`",
            f"Token router top-k: `{config.token_router_top_k}`",
            "",
            "Load as a Transformers causal LM:",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM",
            "",
            "model = AutoModelForCausalLM.from_pretrained(\".\", trust_remote_code=True)",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hf_config(output_dir: Path, config: WrapperConfig) -> None:
    from .hf_runtime import hf_config_payload_from_wrapper

    payload = hf_config_payload_from_wrapper(config)
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_hf_remote_code_stubs(output_dir)


def _write_hf_remote_code_stubs(output_dir: Path) -> None:
    (output_dir / "configuration_moeforge.py").write_text(
        "\n".join(
            [
                '"""Transformers AutoConfig entrypoint for MoE Forge packages."""',
                "",
                "from moeforge.hf_runtime import MoEForgeConfig",
                "",
                "__all__ = [\"MoEForgeConfig\"]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "modeling_moeforge.py").write_text(
        "\n".join(
            [
                '"""Transformers AutoModel entrypoint for MoE Forge packages."""',
                "",
                "from moeforge.hf_runtime import MoEForgeForCausalLM",
                "",
                "__all__ = [\"MoEForgeForCausalLM\"]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
