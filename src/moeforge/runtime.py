from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


class RuntimeErrorMoEForge(RuntimeError):
    """Raised when a carved artifact cannot be loaded or executed."""


@dataclass(slots=True)
class TensorParity:
    source_tensor: str
    layer: int
    role: str
    shape: list[int]
    max_abs_error: float
    mean_abs_error: float
    allclose: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VerifyReport:
    manifest: str
    artifact: str
    tensor_count: int
    passed: bool
    tensors: list[TensorParity] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_carved_artifact(
    *,
    manifest_path: Path,
    artifact_path: Path,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> VerifyReport:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeErrorMoEForge("verification requires torch and safetensors") from exc

    manifest = _read_manifest(manifest_path)
    source_model = Path(str(manifest.get("source_model", "")))
    if not source_model.is_dir():
        raise RuntimeErrorMoEForge("manifest source_model must be a local checkpoint folder")
    if not artifact_path.exists():
        raise RuntimeErrorMoEForge(f"artifact not found: {artifact_path}")

    carved = load_file(str(artifact_path), device="cpu")
    source_cache: dict[str, dict[str, Any]] = {}
    report = VerifyReport(
        manifest=str(manifest_path),
        artifact=str(artifact_path),
        tensor_count=0,
        passed=True,
        warnings=list(manifest.get("warnings", [])),
    )

    for layer in manifest.get("layers", []):
        if not isinstance(layer, dict):
            continue
        for tensor_plan in layer.get("tensors", []):
            if not isinstance(tensor_plan, dict):
                continue
            source_file = source_model / str(tensor_plan.get("source_file"))
            if str(source_file) not in source_cache:
                source_cache[str(source_file)] = load_file(str(source_file), device="cpu")
            source = source_cache[str(source_file)].get(str(tensor_plan.get("source_tensor")))
            if source is None:
                raise RuntimeErrorMoEForge(f"source tensor missing: {tensor_plan.get('source_tensor')}")
            reconstructed = reconstruct_weight_from_carved(
                carved=carved,
                layer=int(layer.get("layer")),
                role=str(tensor_plan.get("role")),
                source_shape=list(source.shape),
                channel_axis=int(tensor_plan.get("channel_axis")),
                shared_channels=[int(item) for item in tensor_plan.get("shared_channels", [])],
                expert_channels=[
                    [int(channel) for channel in expert]
                    for expert in tensor_plan.get("expert_channels", [])
                ],
            )
            diff = (reconstructed - source).abs()
            allclose = bool(torch.allclose(reconstructed, source, atol=atol, rtol=rtol))
            report.tensors.append(
                TensorParity(
                    source_tensor=str(tensor_plan.get("source_tensor")),
                    layer=int(layer.get("layer")),
                    role=str(tensor_plan.get("role")),
                    shape=list(source.shape),
                    max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
                    mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
                    allclose=allclose,
                )
            )
            report.passed = report.passed and allclose

    report.tensor_count = len(report.tensors)
    return report


def reconstruct_weight_from_carved(
    *,
    carved: dict[str, Any],
    layer: int,
    role: str,
    source_shape: list[int],
    channel_axis: int,
    shared_channels: list[int],
    expert_channels: list[list[int]],
) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeErrorMoEForge("reconstruction requires torch") from exc

    dtype = _first_tensor_dtype(carved)
    output = torch.zeros(tuple(source_shape), dtype=dtype)

    shared_name = f"moe.layers.{layer}.mlp.shared.{role}.weight"
    if shared_channels:
        shared = carved.get(shared_name)
        if shared is None:
            raise RuntimeErrorMoEForge(f"missing carved tensor: {shared_name}")
        _assign_channels(output, shared, channel_axis=channel_axis, channels=shared_channels)

    for expert, channels in enumerate(expert_channels):
        if not channels:
            continue
        name = f"moe.layers.{layer}.mlp.experts.{expert}.{role}.weight"
        value = carved.get(name)
        if value is None:
            raise RuntimeErrorMoEForge(f"missing carved tensor: {name}")
        _assign_channels(output, value, channel_axis=channel_axis, channels=channels)

    return output


class CarvedGatedMLP:
    """Executable all-experts gated MLP built from carved tensors."""

    def __init__(
        self,
        *,
        layer: int,
        tensors: dict[str, Any],
        expert_count: int,
        activation: str = "silu",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeErrorMoEForge("runtime requires torch") from exc

        self.torch = torch
        self.layer = layer
        self.tensors = tensors
        self.expert_count = expert_count
        self.activation = activation

    @classmethod
    def from_artifact(
        cls,
        *,
        manifest_path: Path,
        artifact_path: Path,
        layer: int,
        activation: str = "silu",
    ) -> "CarvedGatedMLP":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeErrorMoEForge("runtime requires safetensors") from exc

        manifest = _read_manifest(manifest_path)
        return cls(
            layer=layer,
            tensors=load_file(str(artifact_path), device="cpu"),
            expert_count=int(manifest.get("experts") or 0),
            activation=activation,
        )

    def forward_all(self, x: Any) -> Any:
        return self.forward_selected(x, experts=list(range(self.expert_count)))

    def forward_selected(self, x: Any, *, experts: list[int]) -> Any:
        torch = self.torch
        output = None

        groups: list[tuple[str, int | None]] = [("shared", None)]
        groups.extend(("expert", index) for index in experts)
        for kind, expert in groups:
            if expert is not None and (expert < 0 or expert >= self.expert_count):
                raise RuntimeErrorMoEForge(f"expert index {expert} is out of range")
            prefix = self._prefix(kind, expert)
            gate = self.tensors.get(f"{prefix}.gate.weight")
            up = self.tensors.get(f"{prefix}.up.weight")
            down = self.tensors.get(f"{prefix}.down.weight")
            if gate is None or up is None or down is None:
                continue
            hidden = self._activate(torch.nn.functional.linear(x, gate)) * torch.nn.functional.linear(x, up)
            contribution = torch.nn.functional.linear(hidden, down)
            output = contribution if output is None else output + contribution

        if output is None:
            raise RuntimeErrorMoEForge(f"no runnable carved MLP tensors found for layer {self.layer}")
        return output

    def _prefix(self, kind: str, expert: int | None) -> str:
        if kind == "shared":
            return f"moe.layers.{self.layer}.mlp.shared"
        return f"moe.layers.{self.layer}.mlp.experts.{expert}"

    def _activate(self, value: Any) -> Any:
        torch = self.torch
        if self.activation == "silu":
            return torch.nn.functional.silu(value)
        if self.activation in {"gelu", "gelu_tanh"}:
            return torch.nn.functional.gelu(value, approximate="tanh" if self.activation == "gelu_tanh" else "none")
        raise RuntimeErrorMoEForge(f"unsupported activation {self.activation}")


def dense_gated_mlp_forward(
    *,
    x: Any,
    gate_weight: Any,
    up_weight: Any,
    down_weight: Any,
    activation: str = "silu",
) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeErrorMoEForge("runtime requires torch") from exc

    if activation == "silu":
        activated = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_weight))
    elif activation in {"gelu", "gelu_tanh"}:
        activated = torch.nn.functional.gelu(
            torch.nn.functional.linear(x, gate_weight),
            approximate="tanh" if activation == "gelu_tanh" else "none",
        )
    else:
        raise RuntimeErrorMoEForge(f"unsupported activation {activation}")
    return torch.nn.functional.linear(activated * torch.nn.functional.linear(x, up_weight), down_weight)


def _assign_channels(output: Any, value: Any, *, channel_axis: int, channels: list[int]) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeErrorMoEForge("channel assignment requires torch") from exc

    index = torch.tensor(channels, dtype=torch.long)
    output.index_copy_(channel_axis, index, value)


def _first_tensor_dtype(tensors: dict[str, Any]) -> Any:
    for value in tensors.values():
        return value.dtype
    raise RuntimeErrorMoEForge("carved artifact contains no tensors")


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
